"""
Acoustic Bias Analysis
======================

Quantifies the acoustic difference between attended and unattended speech
streams across the full dataset.

Measures (per trial):
  - RMS energy
  - Envelope variance
  - Spectral centroid (frequency-domain)
  - Dynamic range (95th - 5th percentile of absolute amplitude)

Outputs:
  - analysis/summaries/acoustic_bias_summary.json
  - Prints a statistical summary to stdout

Usage:
    python analysis/acoustic_bias.py

On Kaggle:
    !python analysis/acoustic_bias.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.io import loadmat
from scipy.signal import hilbert, butter, filtfilt

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from baselines.ridge_aad import load_subject_examples, subject_files


# ---------------------------------------------------------------------------
# Feature extractors
# ---------------------------------------------------------------------------

def rms_energy(wav: np.ndarray) -> float:
    """Root-mean-square energy of the signal."""
    return float(np.sqrt(np.mean(np.square(wav.astype(float)))))


def envelope_variance(wav: np.ndarray) -> float:
    """Variance of the Hilbert envelope."""
    env = np.abs(hilbert(wav.astype(float)))
    return float(np.var(env))


def spectral_centroid(wav: np.ndarray, fs: int = 16000) -> float:
    """
    Frequency-domain spectral centroid.
    wav is the raw audio waveform at the original sample rate.
    NOTE: The dataset stores audio already at a lower rate — we compute
    relative centroid (0..1 of Nyquist), so the absolute value of fs
    cancels out in interpretation. What matters is the DIFFERENCE between
    attended and unattended.
    """
    spectrum = np.abs(np.fft.rfft(wav.astype(float)))
    freqs = np.fft.rfftfreq(len(wav), d=1.0 / fs)
    total_power = spectrum.sum()
    if total_power < 1e-12:
        return 0.0
    centroid = float(np.dot(freqs, spectrum) / total_power)
    # Normalise by Nyquist so the measure is comparable regardless of fs
    return centroid / (fs / 2.0)


def dynamic_range(wav: np.ndarray) -> float:
    """95th percentile minus 5th percentile of |amplitude|."""
    abs_wav = np.abs(wav.astype(float))
    return float(np.percentile(abs_wav, 95) - np.percentile(abs_wav, 5))


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def analyse(
    subject_paths: list[Path] | None = None,
    *,
    audio_fs: int = 8000,       # Audio sample rate stored in the .mat files
) -> dict:
    paths = subject_paths or subject_files()
    if not paths:
        raise RuntimeError("No subject files found. Check your EEG_DATA_DIR.")

    records = []

    for path in paths:
        subject = path.stem.split("_")[0]
        print(f"[acoustic-bias] Processing {subject}...", flush=True)
        examples = load_subject_examples(path)

        for ex in examples:
            attended_wav   = ex.wav_a if ex.label == 1 else ex.wav_b
            unattended_wav = ex.wav_b if ex.label == 1 else ex.wav_a

            records.append({
                "subject":            subject,
                "trial_index":        ex.trial_index,
                "label":              ex.label,
                # --- attended ---
                "att_rms":            rms_energy(attended_wav),
                "att_env_var":        envelope_variance(attended_wav),
                "att_spectral_cent":  spectral_centroid(attended_wav, fs=audio_fs),
                "att_dyn_range":      dynamic_range(attended_wav),
                # --- unattended ---
                "una_rms":            rms_energy(unattended_wav),
                "una_env_var":        envelope_variance(unattended_wav),
                "una_spectral_cent":  spectral_centroid(unattended_wav, fs=audio_fs),
                "una_dyn_range":      dynamic_range(unattended_wav),
            })

    # -----------------------------------------------------------------------
    # Statistics
    # -----------------------------------------------------------------------
    def col(key: str) -> np.ndarray:
        return np.array([r[key] for r in records])

    att_rms   = col("att_rms")
    una_rms   = col("una_rms")
    att_env   = col("att_env_var")
    una_env   = col("una_env_var")
    att_cent  = col("att_spectral_cent")
    una_cent  = col("una_spectral_cent")
    att_dyn   = col("att_dyn_range")
    una_dyn   = col("una_dyn_range")
    diff_rms  = att_rms - una_rms
    diff_env  = att_env - una_env
    diff_cent = att_cent - una_cent
    diff_dyn  = att_dyn - una_dyn

    def describe(arr: np.ndarray, label: str) -> dict:
        return {
            "label":  label,
            "mean":   float(arr.mean()),
            "std":    float(arr.std()),
            "median": float(np.median(arr)),
            "p5":     float(np.percentile(arr, 5)),
            "p95":    float(np.percentile(arr, 95)),
            "n":      int(len(arr)),
        }

    summary = {
        "n_subjects": len(paths),
        "n_trials":   len(records),
        "rms_energy": {
            "attended":   describe(att_rms,   "attended_rms"),
            "unattended": describe(una_rms,   "unattended_rms"),
            "diff_att_minus_una": describe(diff_rms, "diff_rms"),
        },
        "envelope_variance": {
            "attended":   describe(att_env,   "attended_env_var"),
            "unattended": describe(una_env,   "unattended_env_var"),
            "diff_att_minus_una": describe(diff_env, "diff_env_var"),
        },
        "spectral_centroid": {
            "attended":   describe(att_cent,  "attended_spectral_cent"),
            "unattended": describe(una_cent,  "unattended_spectral_cent"),
            "diff_att_minus_una": describe(diff_cent, "diff_spectral_cent"),
        },
        "dynamic_range": {
            "attended":   describe(att_dyn,   "attended_dyn_range"),
            "unattended": describe(una_dyn,   "unattended_dyn_range"),
            "diff_att_minus_una": describe(diff_dyn, "diff_dyn_range"),
        },
    }

    # -----------------------------------------------------------------------
    # Console report
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  ACOUSTIC BIAS ANALYSIS")
    print(f"  {len(paths)} subjects, {len(records)} trials")
    print("=" * 60)

    metrics = [
        ("RMS Energy",         "rms_energy",        diff_rms),
        ("Envelope Variance",  "envelope_variance",  diff_env),
        ("Spectral Centroid",  "spectral_centroid",  diff_cent),
        ("Dynamic Range",      "dynamic_range",      diff_dyn),
    ]

    for label, key, diff_arr in metrics:
        att_mean = summary[key]["attended"]["mean"]
        una_mean = summary[key]["unattended"]["mean"]
        d_mean   = diff_arr.mean()
        d_std    = diff_arr.std()
        # Cohen's d: effect size
        pooled_std = np.sqrt((att_rms.std() ** 2 + una_rms.std() ** 2) / 2) if key == "rms_energy" else (diff_arr.std() + 1e-12)
        cohens_d = abs(d_mean) / (d_std + 1e-12)

        # One-sample t-test: is the mean difference significantly != 0?
        t_stat = d_mean / (d_std / np.sqrt(len(diff_arr)) + 1e-12)

        sign = ">" if d_mean > 0 else "<"
        print(f"\n  {label}")
        print(f"    Attended mean:   {att_mean:.6f}")
        print(f"    Unattended mean: {una_mean:.6f}")
        print(f"    Diff (att-una):  {d_mean:.6f}  ± {d_std:.6f}")
        print(f"    Cohen's d:       {cohens_d:.3f}")
        print(f"    t-statistic:     {t_stat:.3f}")
        bias_level = "⚠️  STRONG BIAS" if abs(cohens_d) > 0.5 else ("→  MODERATE BIAS" if abs(cohens_d) > 0.2 else "✅  WEAK/NO BIAS")
        print(f"    Verdict:         {bias_level}")

        # Store in summary
        summary[key]["cohens_d"]  = float(cohens_d)
        summary[key]["t_stat"]    = float(t_stat)

    print("\n" + "=" * 60)

    # -----------------------------------------------------------------------
    # Strong bias warning
    # -----------------------------------------------------------------------
    strong = [k for k, _, d in metrics if abs(d.mean()) / (d.std() + 1e-12) > 0.3]
    if strong:
        print("\n  ⚠️  POTENTIAL SHORTCUT DETECTED")
        print("  The following features differ systematically between")
        print("  attended and unattended streams:")
        for m in strong:
            print(f"    - {m}")
        print("\n  A contrastive model exploiting these differences would")
        print("  achieve high 'accuracy' without using EEG at all.")
    else:
        print("\n  ✅  No strong acoustic bias detected.")
        print("  The attended/unattended streams appear acoustically balanced.")
    print("=" * 60)

    return summary, records


def main() -> None:
    from analysis._common import ensure_output_dirs, SUMMARY_DIR

    ensure_output_dirs()
    out_path = SUMMARY_DIR / "acoustic_bias_summary.json"

    summary, _ = analyse()

    out_path.write_text(json.dumps(summary, indent=2))
    print(f"\n[acoustic-bias] Summary saved to: {out_path}")


if __name__ == "__main__":
    main()
