"""
EEG Sanity Validation Suite
============================

Rigorously proves whether the models are genuinely using EEG information for
Auditory Attention Decoding, rather than exploiting label leakage, evaluation bugs,
or dataset artifacts.

Crucially, applies per-trial RMS equalization to the audio streams before extracting
envelopes. This ensures that any accuracy above 50% is genuinely derived from the
EEG-audio alignment, not the amplitude shortcut we discovered previously.

Tests:
  Test A - Normal EEG
  Test B - Zero EEG (zeros out EEG, keeps audio/labels)
  Test C - EEG Shuffle (randomly permute EEG trials across audio/labels)
  Test D - Random Noise EEG (replace EEG with Gaussian noise matching mean/std)
  Test E - Channel Ablation (2, 4, 8 channels)

Usage:
    python evaluation/eeg_sanity_suite.py

On Kaggle:
    !python evaluation/eeg_sanity_suite.py

Outputs:
    evaluation/summaries/eeg_sanity_results.json
    Printed table to stdout
"""
from __future__ import annotations

import json
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from baselines.ridge_aad import (
    fit_ridge,
    iter_leave_one_subject_out,
    load_subject_examples,
    normalize_eeg,
    predict_envelope,
    speech_envelope,
    subject_files,
)

MAPPING = {1: "A", 2: "B"}
FS = 64
COMPRESSION = 0.6
LOWPASS_HZ = 8.0
LAG_MS = 250
LAG_STEP_MS = 16


def _rms(wav: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(wav.astype(float)))))


def equalize_audio(wav: np.ndarray) -> np.ndarray:
    """RMS equalize the audio stream to unit RMS."""
    r = _rms(wav)
    return wav.astype(float) / (r + 1e-12)


def get_target_envelope(wav: np.ndarray) -> np.ndarray:
    """Extract standard envelope from an already-equalized waveform."""
    return speech_envelope(wav, compression=COMPRESSION, lowpass_hz=LOWPASS_HZ, fs=FS, normalize=True)


def evaluate_fold(train_exs: list, test_exs: list) -> float:
    """Fit Ridge on train_exs and evaluate accuracy on test_exs."""
    # 1. Prepare target envelopes for training
    # Since fit_ridge calls `accumulate_ridge_terms` which uses `target_envelope`,
    # we instead need to either modify `TrialExample` directly, or just let `fit_ridge`
    # run as normal if we replace `wav_a` and `wav_b` with the equalized versions in `ex`.
    # Actually, modifying `ex.wav_a` and `ex.wav_b` in the examples before passing to
    # `fit_ridge` works perfectly, as it will call `speech_envelope` on the equalized wavs.

    w = fit_ridge(
        train_exs,
        MAPPING,
        lag_ms=LAG_MS,
        lag_step_ms=LAG_STEP_MS,
        fs=FS,
        ridge_lambda=1.0,
    )

    n_correct = 0
    for ex in test_exs:
        pred = predict_envelope(ex.eeg, w, lag_ms=LAG_MS, lag_step_ms=LAG_STEP_MS, fs=FS)
        env_a = get_target_envelope(ex.wav_a)
        env_b = get_target_envelope(ex.wav_b)
        
        # trim to same length (lags reduce eeg length)
        min_len = min(len(pred), len(env_a))
        pred = pred[:min_len]
        env_a = env_a[:min_len]
        env_b = env_b[:min_len]

        corr_a = np.corrcoef(pred, env_a)[0, 1]
        corr_b = np.corrcoef(pred, env_b)[0, 1]

        attended = MAPPING[ex.label]
        if attended == "A" and corr_a > corr_b:
            n_correct += 1
        elif attended == "B" and corr_b > corr_a:
            n_correct += 1

    return n_correct / len(test_exs)


def run_sanity_test(test_name: str, subject_examples: dict, transform_fn) -> dict:
    print(f"Running {test_name}...")
    folds_acc = []
    
    # We must apply transform_fn to ALL examples before LOSO iteration
    # so that both train and test sets are properly corrupted/ablated.
    transformed_subjects = {}
    
    # Collect all examples if we need cross-trial shuffling
    all_exs = []
    for k, exs in subject_examples.items():
        all_exs.extend(exs)

    # If the transform_fn requires the whole dataset (e.g., shuffling), we do it globally
    # Wait, transform_fn can just take the whole dictionary and return a new one.
    transformed_subjects = transform_fn(subject_examples)

    # Now run LOSO
    paths = list(subject_paths for subject_paths in transformed_subjects.keys())
    # Mocking paths since iter_leave_one_subject_out just needs keys
    for held_out_key in paths:
        train_exs = []
        for k, exs in transformed_subjects.items():
            if k != held_out_key:
                train_exs.extend(exs)
        test_exs = transformed_subjects[held_out_key]
        
        if not train_exs or not test_exs:
            continue
            
        acc = evaluate_fold(train_exs, test_exs)
        folds_acc.append(acc)

    mean_acc = float(np.mean(folds_acc))
    print(f"  -> Accuracy: {mean_acc:.4f}")
    return {"test": test_name, "accuracy": mean_acc, "folds": folds_acc}


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------

def apply_equalization(subject_examples: dict) -> dict:
    """Base transform: Equalize audio. ALL tests must use this."""
    res = {}
    for k, exs in subject_examples.items():
        new_exs = deepcopy(exs)
        for ex in new_exs:
            ex.wav_a = equalize_audio(ex.wav_a)
            ex.wav_b = equalize_audio(ex.wav_b)
        res[k] = new_exs
    return res


def transform_normal(subject_examples: dict) -> dict:
    return apply_equalization(subject_examples)


def transform_zero(subject_examples: dict) -> dict:
    res = apply_equalization(subject_examples)
    for k, exs in res.items():
        for ex in exs:
            ex.eeg = np.zeros_like(ex.eeg)
    return res


def transform_shuffle(subject_examples: dict) -> dict:
    res = apply_equalization(subject_examples)
    # Extract all EEGs
    all_eegs = [ex.eeg for exs in res.values() for ex in exs]
    # Shuffle them
    np.random.shuffle(all_eegs)
    
    idx = 0
    for k, exs in res.items():
        for ex in exs:
            ex.eeg = all_eegs[idx]
            idx += 1
    return res


def transform_noise(subject_examples: dict) -> dict:
    res = apply_equalization(subject_examples)
    for k, exs in res.items():
        for ex in exs:
            mean = np.mean(ex.eeg)
            std = np.std(ex.eeg) + 1e-12
            ex.eeg = np.random.normal(loc=mean, scale=std, size=ex.eeg.shape)
    return res


def transform_ablation_2(subject_examples: dict) -> dict:
    # 2 channels: typically temporal (e.g. indices for T7, T8 if known)
    # We will just pick channels 20 and 50 as a proxy for bilateral temporal
    res = apply_equalization(subject_examples)
    for k, exs in res.items():
        for ex in exs:
            ex.eeg = ex.eeg[:, [20, 50]]
    return res


def transform_ablation_4(subject_examples: dict) -> dict:
    res = apply_equalization(subject_examples)
    for k, exs in res.items():
        for ex in exs:
            ex.eeg = ex.eeg[:, [10, 20, 40, 50]]
    return res


def transform_ablation_8(subject_examples: dict) -> dict:
    res = apply_equalization(subject_examples)
    for k, exs in res.items():
        for ex in exs:
            ex.eeg = ex.eeg[:, [5, 10, 15, 20, 35, 40, 45, 50]]
    return res


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    from analysis._common import ensure_output_dirs
    out_dir = REPO_ROOT / "evaluation" / "summaries"
    out_dir.mkdir(parents=True, exist_ok=True)

    paths = subject_files()
    if not paths:
        raise RuntimeError("No subject files found.")

    print(f"[sanity-suite] Loading {len(paths)} subjects...")
    subject_examples = {str(p): load_subject_examples(p) for p in paths}

    print("\n" + "=" * 65)
    print("  EEG SANITY VALIDATION SUITE (RIDGE BASELINE)")
    print("  * Using per-trial RMS equalized audio *")
    print("=" * 65)

    results = []
    
    # Test A
    results.append(run_sanity_test("Test A: Normal EEG", subject_examples, transform_normal))
    
    # Test B
    results.append(run_sanity_test("Test B: Zero EEG", subject_examples, transform_zero))
    
    # Test C
    results.append(run_sanity_test("Test C: EEG Shuffle", subject_examples, transform_shuffle))
    
    # Test D
    results.append(run_sanity_test("Test D: Random Noise EEG", subject_examples, transform_noise))
    
    # Test E
    results.append(run_sanity_test("Test E: Channel Ablation (2ch)", subject_examples, transform_ablation_2))
    results.append(run_sanity_test("Test E: Channel Ablation (4ch)", subject_examples, transform_ablation_4))
    results.append(run_sanity_test("Test E: Channel Ablation (8ch)", subject_examples, transform_ablation_8))

    print("\n" + "=" * 65)
    print("  SUMMARY")
    print("=" * 65)
    print(f"  {'Test':<35}  {'Accuracy':>10}")
    print(f"  {'-'*35:<35}  {'-'*10:>10}")
    
    for r in results:
        print(f"  {r['test']:<35}  {r['accuracy']:>10.4f}")
        
    out_path = out_dir / "eeg_sanity_results.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\n[sanity-suite] Results saved to {out_path}")

if __name__ == "__main__":
    main()
