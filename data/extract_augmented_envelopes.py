"""
extract_augmented_envelopes.py
==============================
Extracts augmented gammatone audio features from raw WAV files using a
two-cutoff strategy:

  Branch A (8 Hz):  existing baseline — slow speech tracking envelope
  Branch B (30 Hz): faster amplitude modulation — preserves onset transients
                    in the 8–30 Hz band that the 8 Hz lowpass destroys.

For each WAV file, three feature streams are saved:
  "baseline"  — shape (28, T_64)  8 Hz lowpass envelope, downsampled to 64 Hz
  "delta"     — shape (28, T_64)  30 Hz lowpass envelope, first temporal diff
  "onset"     — shape (28, T_64)  max(delta, 0)

All streams are computed from the SAME compressed filterbank output.
Only the lowpass cutoff differs between baseline and delta/onset.

Mathematical justification:
  - delta/onset at 64 Hz from 8 Hz envelope == linear transform of baseline.
    A CNN kernel already learns this. Zero new information.
  - delta/onset from 30 Hz branch: captures AM in 8-30 Hz range that the
    8 Hz lowpass discards. Genuine new information for the model.

Output:
  audio_features_augmented.pkl
  dict: { wav_filename: {"baseline": arr, "delta": arr, "onset": arr} }

Usage (Kaggle):
  python data/extract_augmented_envelopes.py \\
    --audio_dir /kaggle/input/datasets/lokeshgile/eeg-audio \\
    --out_file  /kaggle/working/audio_features_augmented.pkl
"""

import argparse
import pickle
import warnings
import numpy as np
from pathlib import Path
from scipy.io import wavfile
from scipy.signal import butter, filtfilt, resample, gammatone, lfilter

warnings.filterwarnings("ignore")

# ── Constants ──────────────────────────────────────────────────────────────────
NUM_BANDS   = 28
LOW_FREQ    = 50       # Hz — lowest gammatone centre frequency
HIGH_FREQ   = 8000     # Hz — highest gammatone centre frequency
COMPRESS    = 0.6      # power-law compression exponent
LP_BASELINE = 8.0      # Hz — lowpass cutoff for the envelope (existing pipeline)
LP_DELTA    = 30.0     # Hz — lowpass cutoff for the onset/delta branch
TARGET_FS   = 64       # Hz — target sample rate after downsampling
LP_ORDER    = 3        # Butterworth order for both lowpass filters


def erb_space(low_freq: float, high_freq: float, num_bands: int) -> np.ndarray:
    """Returns ERB-spaced centre frequencies (same as existing pipeline)."""
    erb_low  = 21.4 * np.log10(4.37 * low_freq  / 1000 + 1)
    erb_high = 21.4 * np.log10(4.37 * high_freq / 1000 + 1)
    erb_pts  = np.linspace(erb_low, erb_high, num_bands)
    return (10 ** (erb_pts / 21.4) - 1) / 4.37 * 1000


def make_lowpass(cutoff_hz: float, fs: int, order: int = LP_ORDER):
    """Design a Butterworth lowpass filter."""
    nyq = 0.5 * fs
    return butter(order, cutoff_hz / nyq, btype="low")


def extract_augmented_envelopes(
    wav_path: Path,
    num_bands:   int   = NUM_BANDS,
    low_freq:    float = LOW_FREQ,
    high_freq:   float = HIGH_FREQ,
    target_fs:   int   = TARGET_FS,
) -> dict:
    """
    Returns dict with keys "baseline", "delta", "onset",
    each shape (num_bands, T_target_fs).
    """
    fs, data = wavfile.read(str(wav_path))
    if data.ndim > 1:
        data = data.mean(axis=1)            # mix to mono

    # Clamp high_freq to Nyquist
    high_freq = min(high_freq, fs / 2 - 100)

    cfs = erb_space(low_freq, high_freq, num_bands)

    # Design both lowpass filters at the original WAV sample rate
    b_lp_base, a_lp_base = make_lowpass(LP_BASELINE, fs)
    b_lp_delta, a_lp_delta = make_lowpass(LP_DELTA,    fs)

    audio = data.astype(np.float64)
    n_out = int(len(audio) * target_fs / fs)  # expected output length

    bands_baseline = []
    bands_fast     = []  # 30 Hz lowpass — used for delta & onset

    for cf in cfs:
        # ── Step 1: Gammatone filter (identical to existing pipeline) ──────
        b_gt, a_gt = gammatone(cf, "fir", fs=fs)
        filtered    = lfilter(b_gt, a_gt, audio)

        # ── Step 2: Rectification + power-law compression ─────────────────
        compressed = np.abs(filtered) ** COMPRESS

        # ── Step 3A: 8 Hz lowpass (baseline branch) ────────────────────────
        env_base = filtfilt(b_lp_base, a_lp_base, compressed)
        env_base_ds = resample(env_base, n_out)
        bands_baseline.append(env_base_ds)

        # ── Step 3B: 30 Hz lowpass (delta/onset branch) ───────────────────
        env_fast = filtfilt(b_lp_delta, a_lp_delta, compressed)
        env_fast_ds = resample(env_fast, n_out)
        bands_fast.append(env_fast_ds)

    baseline = np.vstack(bands_baseline)   # (28, T)
    fast     = np.vstack(bands_fast)       # (28, T)

    # ── Step 4: First temporal difference of the 30 Hz branch ────────────
    # prepend first sample to keep shape (28, T) — no boundary artefact
    delta = np.diff(fast, axis=1, prepend=fast[:, :1])   # (28, T)
    onset = np.maximum(delta, 0.0)                         # (28, T)

    return {
        "baseline": baseline.astype(np.float32),
        "delta":    delta.astype(np.float32),
        "onset":    onset.astype(np.float32),
    }


def main(audio_dir: Path, out_file: Path) -> None:
    out_file.parent.mkdir(parents=True, exist_ok=True)
    wav_files = sorted(audio_dir.glob("*.wav"))

    if not wav_files:
        raise FileNotFoundError(
            f"No .wav files found in {audio_dir}. "
            "Check the --audio_dir argument."
        )

    print(f"Found {len(wav_files)} WAV files in {audio_dir}")
    print(f"Extracting: baseline (8 Hz), delta (30 Hz diff), onset (max(delta,0))")
    print(f"Output → {out_file}")
    print("─" * 60)

    results = {}
    errors  = []

    for i, wav_path in enumerate(wav_files):
        try:
            feats = extract_augmented_envelopes(wav_path)
            results[wav_path.name] = feats

            if (i + 1) % 20 == 0 or i == 0 or (i + 1) == len(wav_files):
                b_shape = feats["baseline"].shape
                d_range = (feats["delta"].min(), feats["delta"].max())
                o_frac  = (feats["onset"] > 0).mean()
                print(
                    f"  [{i+1:4d}/{len(wav_files)}] {wav_path.name:40s} "
                    f"shape={b_shape}  delta∈[{d_range[0]:+.3f},{d_range[1]:+.3f}]  "
                    f"onset_density={o_frac:.2f}"
                )
        except Exception as exc:
            print(f"  [ERROR] {wav_path.name}: {exc}")
            errors.append(wav_path.name)

    print("─" * 60)
    print(f"Processed {len(results)}/{len(wav_files)} files successfully.")

    if errors:
        print(f"Failed files ({len(errors)}): {errors}")

    # ── Sanity check: correlation between baseline and delta streams ───────
    # If two-cutoff strategy is working, correlation should be near-zero.
    sample_key = next(iter(results))
    sample     = results[sample_key]
    flat_base  = sample["baseline"].ravel()
    flat_delta = sample["delta"].ravel()
    corr       = float(np.corrcoef(flat_base, flat_delta)[0, 1])
    print(f"\nSanity check on '{sample_key}':")
    print(f"  corr(baseline, delta) = {corr:.4f}  (target: near 0.0)")
    if abs(corr) > 0.5:
        print("  WARNING: high correlation — two-cutoff strategy may not be working.")
    else:
        print("  OK: delta stream is sufficiently decorrelated from baseline.")

    print(f"\nSaving augmented features → {out_file} ...")
    with open(out_file, "wb") as f:
        pickle.dump(results, f, protocol=4)
    size_mb = out_file.stat().st_size / 1e6
    print(f"Saved. File size: {size_mb:.1f} MB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract augmented gammatone features from raw WAV files."
    )
    parser.add_argument(
        "--audio_dir",
        type=str,
        default="/kaggle/input/datasets/lokeshgile/eeg-audio",
        help="Directory containing raw .wav files",
    )
    parser.add_argument(
        "--out_file",
        type=str,
        default="/kaggle/working/audio_features_augmented.pkl",
        help="Output pickle path for augmented features",
    )
    args = parser.parse_args()
    main(Path(args.audio_dir), Path(args.out_file))
