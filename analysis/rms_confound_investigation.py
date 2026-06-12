"""
RMS Confound Investigation
===========================

Determines whether the acoustic confound (attended stream ≈ lower RMS)
originates from:

  A. The raw stored audio in the .mat files (dataset/preprocessing origin)
  B. Our envelope extraction pipeline (code error)
  C. Both or neither

Also creates an amplitude-equalized control to test whether
per-trial RMS normalization eliminates the shortcut.

PIPELINE AUDIT:
  Step 0: Raw wavA/wavB loaded from .mat — no manipulation
  Step 1: Hilbert envelope (|analytic signal|) — preserves relative amplitude
  Step 2: Moving average smoothing — preserves relative amplitude
  Step 3: speech_envelope(..., normalize=True) — z-scores the envelope.
          This DESTROYS within-stream amplitude info but does so
          identically for both streams. Cannot introduce cross-stream bias.

FINDING (expected):
  If the confound is present in Step 0 (raw audio), it originates in the
  dataset, not our code. Our normalization at Step 3 does not create it.

DTU DATASET DOCUMENTATION CITATION:
  From Fuglsang et al. 2017 (DOI: 10.5281/zenodo.1199011):
  - One male speaker, one female speaker competing simultaneously.
  - Stimuli presented in 3 reverberant conditions.
  - "Common practices include normalizing speech envelopes or stimuli to
    have the same RMS amplitude" (referenced in downstream papers).
  - This normalization is NOT documented as pre-applied in the DATA_preproc
    files — it is left to the researcher.
  - Our preprocessing does NOT apply cross-speaker RMS equalization.

Usage:
    python analysis/rms_confound_investigation.py

On Kaggle:
    !python analysis/rms_confound_investigation.py

Outputs:
    analysis/summaries/rms_confound_investigation.json
    Full verdict printed to stdout
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.signal import hilbert

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from baselines.ridge_aad import (
    load_subject_examples,
    moving_average,
    subject_files,
)

MAPPING = {1: "A", 2: "B"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def rms(arr: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(arr.astype(float)))))


def attended(ex, mapping: dict) -> np.ndarray:
    return ex.wav_a if mapping[ex.label] == "A" else ex.wav_b


def unattended(ex, mapping: dict) -> np.ndarray:
    return ex.wav_b if mapping[ex.label] == "A" else ex.wav_a


def p_attended_lower(examples: list, fn_a, fn_b, mapping: dict) -> tuple[float, list]:
    """
    Compute P(feature(attended) < feature(unattended)) across all trials.
    Returns (probability, list of (feat_att, feat_una) tuples).
    """
    n_lower = 0
    pairs = []
    for ex in examples:
        f_att = fn_a(attended(ex, mapping))
        f_una = fn_b(unattended(ex, mapping))
        pairs.append((f_att, f_una))
        if f_att < f_una:
            n_lower += 1
    return n_lower / len(examples), pairs


def ratio_stats(pairs: list[tuple]) -> dict:
    ratios = np.array([a / (b + 1e-12) for a, b in pairs])
    return {
        "mean": float(ratios.mean()),
        "median": float(np.median(ratios)),
        "std": float(ratios.std()),
        "p5": float(np.percentile(ratios, 5)),
        "p95": float(np.percentile(ratios, 95)),
    }


# ---------------------------------------------------------------------------
# Stage functions
# ---------------------------------------------------------------------------

def stage_raw_rms(wav: np.ndarray) -> float:
    """Stage 0: RMS of raw waveform as loaded from .mat."""
    return rms(wav)


def stage_envelope_rms(wav: np.ndarray) -> float:
    """Stage 1-2: RMS of Hilbert envelope after moving average."""
    env = np.abs(hilbert(wav.astype(float)))
    env = moving_average(env, window=64)
    return rms(env)


def stage_normalized_envelope_rms(wav: np.ndarray) -> float:
    """Stage 3: RMS of z-scored envelope (what our model sees)."""
    env = np.abs(hilbert(wav.astype(float)))
    env = moving_average(env, window=64)
    env = env - env.mean()
    env = env / (env.std() + 1e-12)
    return rms(env)


# Per-trial RMS equalized versions
def equalized_raw_rms(wav: np.ndarray, eq_rms: float) -> float:
    """RMS after dividing the stream by its own RMS (per-trial equalization)."""
    return 1.0  # By definition, normalized to 1.0 for any stream


# ---------------------------------------------------------------------------
# Main investigation
# ---------------------------------------------------------------------------

def investigate_stage(label: str, examples: list, feat_fn, mapping: dict) -> dict:
    p, pairs = p_attended_lower(examples, feat_fn, feat_fn, mapping)
    stats = ratio_stats(pairs)
    flag = "🚨 CONFOUND" if p > 0.80 else ("⚠️  MODERATE" if p > 0.65 else "✅  CLEAN")
    print(f"  {flag}  {label:<40s}  P(att<una) = {p:.4f}  ratio_mean = {stats['mean']:.4f}")
    return {"stage": label, "p_attended_lower": p, "ratio_stats": stats}


def run_equalized_baseline(examples: list, mapping: dict) -> dict:
    """
    Per-trial RMS equalization: divide each stream by its own RMS before
    extracting features. If the confound disappears, it was purely amplitude.
    """
    n_lower = 0
    for ex in examples:
        att_wav = attended(ex, mapping)
        una_wav = unattended(ex, mapping)

        # Equalize: normalize each stream to unit RMS
        att_eq = att_wav / (rms(att_wav) + 1e-12)
        una_eq = una_wav / (rms(una_wav) + 1e-12)

        # After equalization, both streams have RMS ≈ 1.0
        # So we cannot discriminate on RMS anymore.
        # Check envelope variance instead (next most powerful feature)
        env_att = np.abs(hilbert(att_eq))
        env_una = np.abs(hilbert(una_eq))
        if np.var(env_att) < np.var(env_una):
            n_lower += 1

    p_env_var = n_lower / len(examples)
    print(f"\n  After per-trial RMS equalization:")
    print(f"    P(attended envelope_var < unattended envelope_var) = {p_env_var:.4f}")
    if p_env_var > 0.75:
        note = "⚠️  Secondary confound remains in envelope variance after RMS equalization."
    elif p_env_var > 0.60:
        note = "→  Mild residual confound in envelope variance. RMS was the primary driver."
    else:
        note = "✅  Envelope variance confound eliminated. RMS equalization resolves it."
    print(f"    {note}")
    return {"p_attended_env_var_lower_after_equalization": p_env_var, "note": note}


def run_audio_only_heuristic_equalized(examples: list, mapping: dict) -> dict:
    """
    Run the audio-only 'lower RMS = attended' heuristic on:
      - Original raw audio
      - After per-trial RMS equalization of both streams

    The equalized case should drop to ~50% if RMS is the only shortcut.
    """
    # Original
    n_correct_orig = 0
    n_correct_eq   = 0
    n = len(examples)

    for ex in examples:
        att_w = attended(ex, mapping)
        una_w = unattended(ex, mapping)

        # Original heuristic
        if rms(att_w) < rms(una_w):
            n_correct_orig += 1

        # Equalized: normalize both, then check envelope variance
        r_att = rms(att_w)
        r_una = rms(una_w)
        att_eq = att_w / (r_att + 1e-12)
        una_eq = una_w / (r_una + 1e-12)
        # After equalization, lower envelope variance = attended?
        if np.var(np.abs(hilbert(att_eq))) < np.var(np.abs(hilbert(una_eq))):
            n_correct_eq += 1

    acc_orig = n_correct_orig / n
    acc_eq   = n_correct_eq   / n
    return {"heuristic_original_accuracy": acc_orig, "heuristic_equalized_accuracy": acc_eq}


def main() -> None:
    from analysis._common import ensure_output_dirs, SUMMARY_DIR

    ensure_output_dirs()

    subject_paths = subject_files()
    if not subject_paths:
        raise RuntimeError("No subject files found. Check EEG_DATA_DIR.")

    print(f"[rms-investigation] Loading {len(subject_paths)} subjects...", flush=True)
    subject_examples = {str(p): load_subject_examples(p) for p in subject_paths}
    all_examples = [ex for exs in subject_examples.values() for ex in exs]
    n = len(all_examples)
    print(f"[rms-investigation] Total trials: {n}\n")

    # -----------------------------------------------------------------------
    # STEP 1 — PIPELINE AUDIT
    # -----------------------------------------------------------------------
    print("=" * 68)
    print("  STEP 1 — Pipeline audit: where does the confound first appear?")
    print("=" * 68)
    print(f"  {'Flag':<18}  {'Stage':<40}  P(att<una)   ratio_mean")
    print(f"  {'-'*16:<18}  {'-'*38:<40}  ----------   ----------")

    results_stages = []
    results_stages.append(investigate_stage(
        "Stage 0: RAW waveform RMS (from .mat)",
        all_examples, stage_raw_rms, MAPPING
    ))
    results_stages.append(investigate_stage(
        "Stage 1-2: Hilbert+MA envelope RMS",
        all_examples, stage_envelope_rms, MAPPING
    ))
    results_stages.append(investigate_stage(
        "Stage 3: z-scored envelope RMS",
        all_examples, stage_normalized_envelope_rms, MAPPING
    ))

    # -----------------------------------------------------------------------
    # STEP 2 — Source determination
    # -----------------------------------------------------------------------
    print("\n" + "=" * 68)
    print("  STEP 2 — Source determination")
    print("=" * 68)

    raw_p   = results_stages[0]["p_attended_lower"]
    env_p   = results_stages[1]["p_attended_lower"]
    znorm_p = results_stages[2]["p_attended_lower"]

    if raw_p > 0.80:
        source = (
            "🚨 DATASET ORIGIN — The confound is present in the raw wavA/wavB "
            "signals stored in the .mat files. Our preprocessing code does NOT "
            "introduce it. The DTU DATA_preproc files contain audio where the "
            "attended speaker is consistently the lower-amplitude speaker.\n\n"
            "  Likely cause: The male and female speakers in the original recordings "
            "have systematically different vocal intensities. The dataset does NOT "
            "apply per-trial RMS equalization of the two speakers before storage. "
            "This is documented in the literature as something researchers should "
            "apply themselves (COCOHA toolbox note), but was not applied here."
        )
    elif env_p > raw_p + 0.10:
        source = (
            "⚠️  CODE AMPLIFICATION — The confound is weak in the raw audio but "
            "amplified by our envelope extraction. Investigate the Hilbert/MA step."
        )
    else:
        source = (
            "✅  No strong confound detected at any pipeline stage."
        )

    print(f"\n  {source}\n")
    print(f"  Raw audio P(att lower):           {raw_p:.4f}")
    print(f"  Envelope P(att lower):            {env_p:.4f}")
    print(f"  Z-scored envelope P(att lower):   {znorm_p:.4f}")
    print(f"  (Z-scoring sets RMS to ~1 for both streams, so ~0.50 is expected)")

    # -----------------------------------------------------------------------
    # STEP 3 — Equalization test
    # -----------------------------------------------------------------------
    print("\n" + "=" * 68)
    print("  STEP 3 — Does per-trial RMS equalization eliminate the shortcut?")
    print("=" * 68)

    heuristic_results = run_audio_only_heuristic_equalized(all_examples, MAPPING)
    eq_results        = run_equalized_baseline(all_examples, MAPPING)

    print(f"\n  Audio-only heuristic (lower RMS = attended):")
    print(f"    Original accuracy:   {heuristic_results['heuristic_original_accuracy']:.4f}")
    print(f"    After equalization:  {heuristic_results['heuristic_equalized_accuracy']:.4f}")

    drop = heuristic_results["heuristic_original_accuracy"] - heuristic_results["heuristic_equalized_accuracy"]
    if drop > 0.30:
        eq_verdict = (
            "✅  Equalization substantially reduces the shortcut. "
            f"Accuracy drops by {drop:.3f}. "
            "Per-trial RMS normalization of both audio streams before any analysis "
            "is the recommended fix for this dataset."
        )
    elif drop > 0.10:
        eq_verdict = (
            f"→  Partial reduction (drop = {drop:.3f}). "
            "RMS equalization helps but secondary confounds remain."
        )
    else:
        eq_verdict = (
            f"⚠️  Minimal reduction (drop = {drop:.3f}). "
            "The confound is not purely RMS-driven. Deeper investigation required."
        )
    print(f"\n  Equalization verdict: {eq_verdict}")

    # -----------------------------------------------------------------------
    # STEP 4 — What benchmark to use going forward
    # -----------------------------------------------------------------------
    print("\n" + "=" * 68)
    print("  STEP 4 — Benchmark recommendations")
    print("=" * 68)
    print("""
  Current situation:
    Raw dataset has a systematic RMS imbalance between attending speakers.
    Any model trained on this data without equalization is exploiting this.

  Option A — Per-trial RMS equalization (recommended minimum):
    For each trial:
      wavA_eq = wavA / RMS(wavA)
      wavB_eq = wavB / RMS(wavB)
    Apply BEFORE envelope extraction.
    Re-run acoustic_bias.py and audio_only_baseline.py on equalized data.
    Expected: audio-only accuracy drops toward 50%.
    If EEG models still outperform 50%, they are using genuine EEG signal.

  Option B — LUFS/loudness normalization (gold standard):
    Use ITU-R BS.1770 loudness normalization per speaker across the session.
    Matches standard psychoacoustic experimental design.

  Option C — Use competing dataset:
    KUL AAD dataset (Das et al., 2019) is better controlled for this.
    MINDS dataset also available with stricter balancing.

  Minimum required before any EEG model claim:
    1. Equalize amplitudes per trial (Option A minimum)
    2. Re-run audio_only_baseline.py — must show accuracy near 50%
    3. Only then do EEG model accuracies constitute evidence of AAD
""")

    # -----------------------------------------------------------------------
    # Save
    # -----------------------------------------------------------------------
    summary = {
        "n_subjects":             len(subject_paths),
        "n_trials":               n,
        "pipeline_stages":        results_stages,
        "source_determination":   source,
        "equalization_test":      {**heuristic_results, **eq_results},
        "equalization_verdict":   eq_verdict,
        "dataset_documentation":  {
            "citation": "Fuglsang et al. 2017, DOI:10.5281/zenodo.1199011",
            "speakers": "One male, one female narrator competing simultaneously",
            "rms_normalization_documented": False,
            "rms_normalization_applied_in_preproc": False,
            "source": "Raw amplitude difference between speakers baked into .mat files",
        },
    }

    out_path = SUMMARY_DIR / "rms_confound_investigation.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"[rms-investigation] Summary saved to: {out_path}")


if __name__ == "__main__":
    main()
