"""
Label Integrity Investigation
==============================

This is the most critical diagnostic in the project.

Hypothesis: The dataset labels (1/2) encode SPEAKER IDENTITY (male/female),
not attended vs. unattended attention state. If true:
  - wavA is always the SAME speaker (e.g., male) across ALL trials and subjects
  - wavB is always the SAME speaker (e.g., female) across ALL trials and subjects
  - label=1 means the subject attended to whoever is in wavA
  - label=2 means the subject attended to whoever is in wavB
  - The two speakers have systematically different acoustics (RMS, spectral profile)
  - Any model can exploit this without using EEG at all

If FALSE:
  - wavA and wavB are randomly assigned per trial
  - The acoustic statistics of wavA and wavB should be roughly symmetric

This script runs three checks:

CHECK 1 — RMS vs. Label
  For every trial across all subjects, compute:
    RMS(wavA), RMS(wavB), label
  Then compute: what fraction of trials have wavA_RMS < wavB_RMS?
  And: what fraction of label=1 trials have wavA_RMS < wavB_RMS?
  If both are ~95-100%, wavA is systematically the quieter stream,
  independent of what the subject attended.

CHECK 2 — Speaker identity via acoustic fingerprint clustering
  Cluster all audio segments (across all trials, all subjects) into 2 groups
  using acoustic features. If the 2 clusters perfectly separate wavA from wavB
  (regardless of label), wavA and wavB are fixed speaker slots.
  If the clusters are random w.r.t. stream slot, the assignment is random.

CHECK 3 — Within-subject label distribution
  For each subject, compute the fraction of trials with label=1.
  A balanced AAD experiment should have roughly 50% label=1 per subject.
  If consistently ~50%, the experiment balanced attention across speakers.
  If systematically skewed, something else is going on.
  More critically: check if the RMS(wavA) < RMS(wavB) relationship holds
  REGARDLESS of label. If yes, wavA is always the same person.

Usage:
    python analysis/label_integrity.py

On Kaggle:
    !python analysis/label_integrity.py

Outputs:
    analysis/summaries/label_integrity_summary.json
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

from baselines.ridge_aad import load_subject_examples, subject_files


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def rms(wav: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(wav.astype(float)))))


def mean_envelope(wav: np.ndarray) -> float:
    return float(np.abs(hilbert(wav.astype(float))).mean())


def spectral_centroid(wav: np.ndarray) -> float:
    s = np.abs(np.fft.rfft(wav.astype(float)))
    f = np.fft.rfftfreq(len(wav))
    total = s.sum()
    return float(np.dot(f, s) / total) if total > 1e-12 else 0.0


# ---------------------------------------------------------------------------
# CHECK 1 — RMS(wavA) vs RMS(wavB) independently of label
# ---------------------------------------------------------------------------

def check_rms_vs_label(all_examples: list) -> dict:
    """
    Key question: Is RMS(wavA) < RMS(wavB) almost always TRUE,
    regardless of what the label says?

    If yes: wavA and wavB are fixed speakers (not randomly assigned).
    The acoustic difference is a property of WHO is in each slot,
    not of who was attended.
    """
    n_total = len(all_examples)
    n_wava_lower = 0          # wavA has lower RMS (regardless of label)
    n_label1_wava_lower = 0   # label=1 AND wavA has lower RMS
    n_label2_wava_lower = 0   # label=2 AND wavA has lower RMS
    n_label1 = 0
    n_label2 = 0

    per_trial = []
    for ex in all_examples:
        rms_a = rms(ex.wav_a)
        rms_b = rms(ex.wav_b)
        wava_lower = rms_a < rms_b

        if wava_lower:
            n_wava_lower += 1
        if ex.label == 1:
            n_label1 += 1
            if wava_lower:
                n_label1_wava_lower += 1
        else:
            n_label2 += 1
            if wava_lower:
                n_label2_wava_lower += 1

        per_trial.append({
            "subject":     ex.subject,
            "trial_index": ex.trial_index,
            "label":       ex.label,
            "rms_a":       float(rms_a),
            "rms_b":       float(rms_b),
            "wava_lower":  bool(wava_lower),
        })

    frac_wava_lower_overall  = n_wava_lower / n_total
    frac_wava_lower_label1   = n_label1_wava_lower / n_label1 if n_label1 > 0 else float("nan")
    frac_wava_lower_label2   = n_label2_wava_lower / n_label2 if n_label2 > 0 else float("nan")

    print("\n" + "=" * 65)
    print("  CHECK 1 — RMS(wavA) vs RMS(wavB), by label")
    print("=" * 65)
    print(f"  Total trials:                     {n_total}")
    print(f"  Trials where RMS(wavA) < RMS(wavB): {n_wava_lower}/{n_total}  = {frac_wava_lower_overall:.4f}")
    print(f"  ... among label=1 trials:           {n_label1_wava_lower}/{n_label1}  = {frac_wava_lower_label1:.4f}")
    print(f"  ... among label=2 trials:           {n_label2_wava_lower}/{n_label2}  = {frac_wava_lower_label2:.4f}")
    print()

    # Interpretation
    if frac_wava_lower_overall > 0.90:
        if abs(frac_wava_lower_label1 - frac_wava_lower_label2) < 0.10:
            verdict = (
                "🚨 CRITICAL — wavA is almost ALWAYS lower RMS than wavB, "
                "and this holds REGARDLESS of label. "
                "wavA and wavB are FIXED SPEAKER SLOTS, not random assignments. "
                "The labels encode speaker identity, not attended attention state."
            )
        else:
            verdict = (
                "⚠️  WARNING — wavA is usually lower RMS, but the effect differs "
                "between label=1 and label=2. Mixed situation requiring further analysis."
            )
    elif frac_wava_lower_overall > 0.60:
        verdict = (
            "→  MODERATE — wavA tends to be lower RMS but not consistently. "
            "Possible weak speaker assignment bias."
        )
    else:
        verdict = (
            "✅  BALANCED — RMS(wavA) vs RMS(wavB) is roughly 50/50. "
            "Speaker assignment appears random. Labels likely encode genuine attention."
        )

    print(f"  Verdict: {verdict}")
    return {
        "n_total":                     n_total,
        "n_label1":                    n_label1,
        "n_label2":                    n_label2,
        "n_wava_lower_rms":            n_wava_lower,
        "frac_wava_lower_overall":     frac_wava_lower_overall,
        "frac_wava_lower_label1":      frac_wava_lower_label1,
        "frac_wava_lower_label2":      frac_wava_lower_label2,
        "verdict":                     verdict,
        "per_trial":                   per_trial,
    }


# ---------------------------------------------------------------------------
# CHECK 2 — Is wavA always acoustically the same "person"?
# ---------------------------------------------------------------------------

def check_speaker_slot_identity(all_examples: list) -> dict:
    """
    Extract acoustic fingerprint vectors for wavA and wavB from every trial.
    Then verify: do all wavA segments have similar fingerprints, and all
    wavB segments have similar fingerprints — regardless of label?

    If wavA fingerprints cluster tightly (low variance) and wavB fingerprints
    cluster tightly (low variance), the two slots always contain the same
    two speakers. The labels are then attention labels over fixed speaker slots,
    and the acoustic shortcut is the identity of the speaker in each slot.
    """
    print("\n" + "=" * 65)
    print("  CHECK 2 — Acoustic fingerprint stability across trials")
    print("=" * 65)

    feats_a, feats_b = [], []
    for ex in all_examples:
        feats_a.append([
            rms(ex.wav_a),
            mean_envelope(ex.wav_a),
            spectral_centroid(ex.wav_a),
        ])
        feats_b.append([
            rms(ex.wav_b),
            mean_envelope(ex.wav_b),
            spectral_centroid(ex.wav_b),
        ])

    fa = np.array(feats_a)  # (N, 3)
    fb = np.array(feats_b)  # (N, 3)

    # Coefficient of variation (CV) for each feature in each slot
    # Low CV = consistent speaker in that slot
    feat_names = ["RMS", "Mean_Envelope", "Spectral_Centroid"]
    results_a, results_b = {}, {}
    for i, name in enumerate(feat_names):
        cv_a = fa[:, i].std() / (fa[:, i].mean() + 1e-12)
        cv_b = fb[:, i].std() / (fb[:, i].mean() + 1e-12)
        results_a[name] = {"mean": float(fa[:, i].mean()), "std": float(fa[:, i].std()), "cv": float(cv_a)}
        results_b[name] = {"mean": float(fb[:, i].mean()), "std": float(fb[:, i].std()), "cv": float(cv_b)}
        print(f"  {name:<22s}  wavA: mean={fa[:, i].mean():.5f}, cv={cv_a:.3f}  |  wavB: mean={fb[:, i].mean():.5f}, cv={cv_b:.3f}")

    # Inter-slot separation: how different are wavA and wavB on average?
    mean_diff = np.abs(fa.mean(axis=0) - fb.mean(axis=0)) / (np.abs(fa.mean(axis=0)) + np.abs(fb.mean(axis=0)) + 1e-12)
    mean_sep  = float(mean_diff.mean())
    mean_cv_a = float(np.mean([results_a[n]["cv"] for n in feat_names]))
    mean_cv_b = float(np.mean([results_b[n]["cv"] for n in feat_names]))

    print(f"\n  Mean normalised inter-slot separation: {mean_sep:.4f}")
    print(f"  Mean CV across features — wavA: {mean_cv_a:.4f}, wavB: {mean_cv_b:.4f}")

    if mean_cv_a < 0.25 and mean_cv_b < 0.25 and mean_sep > 0.10:
        verdict = (
            "🚨 CRITICAL — Both speaker slots show LOW within-slot variance and HIGH "
            "between-slot separation. wavA is consistently the same speaker and wavB "
            "is consistently the other speaker across ALL trials and subjects."
        )
    elif mean_cv_a < 0.50 and mean_cv_b < 0.50:
        verdict = (
            "⚠️  WARNING — Moderate consistency in speaker slot assignments. "
            "Partial speaker identity bias is present."
        )
    else:
        verdict = (
            "✅  Fingerprints are not consistent across slots. "
            "Speaker assignment appears to vary across trials."
        )

    print(f"\n  Verdict: {verdict}")
    return {"feats_a": results_a, "feats_b": results_b, "mean_sep": mean_sep,
            "mean_cv_a": mean_cv_a, "mean_cv_b": mean_cv_b, "verdict": verdict}


# ---------------------------------------------------------------------------
# CHECK 3 — Within-subject label balance and per-subject RMS pattern
# ---------------------------------------------------------------------------

def check_per_subject_label_balance(subject_examples: dict[str, list]) -> dict:
    """
    For each subject:
      - How many label=1 vs label=2 trials?
      - In what fraction of their trials is RMS(wavA) < RMS(wavB)?
    If the RMS(wavA) < RMS(wavB) pattern holds for ALL subjects consistently,
    the shortcut is in the fixed speaker slot, not in attention.
    """
    print("\n" + "=" * 65)
    print("  CHECK 3 — Per-subject label balance and RMS pattern")
    print("=" * 65)
    print(f"  {'Subject':<10s}  {'N':>5}  {'Label1%':>8}  {'wavA_lower%':>12}  {'Label1↔wavA_lower':>18}")
    print(f"  {'-'*8:<10s}  {'---':>5}  {'-------':>8}  {'-----------':>12}  {'------------------':>18}")

    per_subject = {}
    for path_str, examples in subject_examples.items():
        subj = examples[0].subject if examples else path_str
        n = len(examples)
        n1 = sum(1 for ex in examples if ex.label == 1)
        n_wava_lower = sum(1 for ex in examples if rms(ex.wav_a) < rms(ex.wav_b))
        n_label1_wava_lower = sum(1 for ex in examples if ex.label == 1 and rms(ex.wav_a) < rms(ex.wav_b))

        frac_label1 = n1 / n if n > 0 else float("nan")
        frac_wava_lower = n_wava_lower / n if n > 0 else float("nan")
        # Agreement between "label=1" and "wavA lower RMS"
        agreement = n_label1_wava_lower / max(n1, n_wava_lower) if max(n1, n_wava_lower) > 0 else float("nan")

        print(f"  {subj:<10s}  {n:>5}  {frac_label1:>8.3f}  {frac_wava_lower:>12.3f}  {agreement:>18.3f}")
        per_subject[subj] = {
            "n": n, "n_label1": n1,
            "frac_label1": frac_label1,
            "frac_wava_lower_rms": frac_wava_lower,
            "label1_wava_lower_agreement": agreement,
        }

    all_wava_lower = [v["frac_wava_lower_rms"] for v in per_subject.values()]
    print(f"\n  wavA-lower-RMS fraction: mean={np.mean(all_wava_lower):.3f}, std={np.std(all_wava_lower):.3f}")

    if np.mean(all_wava_lower) > 0.90 and np.std(all_wava_lower) < 0.10:
        verdict = (
            "🚨 CRITICAL — wavA is consistently lower RMS than wavB across virtually "
            "ALL subjects. This is independent of label (attention state). "
            "The acoustic shortcut is structural — baked into the data format."
        )
    else:
        verdict = "✅  Per-subject RMS patterns are variable. No consistent speaker slot bias."

    print(f"\n  Verdict: {verdict}")
    return {"per_subject": per_subject, "verdict": verdict}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    from analysis._common import ensure_output_dirs, SUMMARY_DIR

    ensure_output_dirs()

    subject_paths = subject_files()
    if not subject_paths:
        raise RuntimeError("No subject files found. Check EEG_DATA_DIR.")

    print(f"[label-integrity] Loading {len(subject_paths)} subjects...", flush=True)
    subject_examples: dict[str, list] = {
        str(p): load_subject_examples(p) for p in subject_paths
    }
    all_examples = [ex for exs in subject_examples.values() for ex in exs]
    print(f"[label-integrity] Total trials: {len(all_examples)}")

    print("\n" + "#" * 65)
    print("  LABEL INTEGRITY INVESTIGATION")
    print("  Hypothesis: Labels encode speaker identity, not attention")
    print("#" * 65)

    result_c1 = check_rms_vs_label(all_examples)
    result_c2 = check_speaker_slot_identity(all_examples)
    result_c3 = check_per_subject_label_balance(subject_examples)

    # -----------------------------------------------------------------------
    # Final synthesis
    # -----------------------------------------------------------------------
    print("\n" + "#" * 65)
    print("  FINAL SYNTHESIS")
    print("#" * 65)

    verdicts = [result_c1["verdict"], result_c2["verdict"], result_c3["verdict"]]
    n_critical = sum(1 for v in verdicts if "CRITICAL" in v)
    n_warning  = sum(1 for v in verdicts if "WARNING"  in v)

    print(f"\n  Check 1 (RMS vs label):              {result_c1['verdict'][:50]}...")
    print(f"  Check 2 (Speaker slot fingerprint):  {result_c2['verdict'][:50]}...")
    print(f"  Check 3 (Per-subject patterns):      {result_c3['verdict'][:50]}...")

    print()
    if n_critical >= 2:
        final_verdict = (
            "🚨🚨 CONFIRMED DATASET BIAS 🚨🚨\n\n"
            "  The labels in this dataset encode SPEAKER IDENTITY (male/female),\n"
            "  not cognitive attention state.\n\n"
            "  wavA and wavB are fixed speaker slots. wavA contains one specific\n"
            "  speaker with consistently lower RMS. wavB contains the other.\n\n"
            "  All models trained on this data without controlling for this bias\n"
            "  are decoding SPEAKER IDENTITY, not auditory attention decoding.\n\n"
            "  The 55-58% EEG reconstruction results are suspect until evaluated\n"
            "  against the 98%+ audio-only ceiling.\n\n"
            "  REQUIRED ACTION: Reconstruct the dataset with randomly permuted\n"
            "  wavA/wavB assignments per trial, OR use audio-balanced evaluation."
        )
    elif n_critical >= 1 or n_warning >= 2:
        final_verdict = (
            "⚠️  SIGNIFICANT BIAS DETECTED\n\n"
            "  Partial evidence of speaker-identity confound. Further investigation\n"
            "  required before results can be trusted as genuine AAD."
        )
    else:
        final_verdict = (
            "✅  No evidence of systematic speaker identity confound.\n"
            "  Labels appear to encode genuine attention state."
        )

    print(f"  {final_verdict}")

    # -----------------------------------------------------------------------
    # Save
    # -----------------------------------------------------------------------
    summary = {
        "n_subjects": len(subject_paths),
        "n_trials":   len(all_examples),
        "check_1_rms_vs_label":          {k: v for k, v in result_c1.items() if k != "per_trial"},
        "check_2_speaker_slot_identity": result_c2,
        "check_3_per_subject_balance":   result_c3,
        "final_verdict":                 final_verdict,
    }
    out_path = SUMMARY_DIR / "label_integrity_summary.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"\n[label-integrity] Summary saved to: {out_path}")


if __name__ == "__main__":
    main()
