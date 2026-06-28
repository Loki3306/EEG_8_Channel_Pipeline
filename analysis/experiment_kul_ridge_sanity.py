import os
import sys
import numpy as np
import copy
import random
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from data.kul_cached_dataset import KULCachedLoader
from baselines.ridge_aad import TrialExample, lagged_eeg_matrix, feature_statistics, standardize_features
from evaluation.aad_metrics import safe_corr

FS = 64
RIDGE_LAMBDA = 100.0
LAGS = 32
LAG_STEP_MS = 16
WINDOW_SEC = 10
SUBJECTS_TO_EVAL = ["S1", "S5", "S11", "S13"]

def evaluate_trial_majority_vote(predicted: np.ndarray, wav_a: np.ndarray, wav_b: np.ndarray, window_seconds: int, fs: int = 64):
    window_samples = window_seconds * fs
    if window_seconds <= 0 or window_samples >= predicted.size:
        corr_a = safe_corr(predicted, wav_a)
        corr_b = safe_corr(predicted, wav_b)
        if corr_a == corr_b:
            correct = np.random.rand() > 0.5
        else:
            correct = corr_a > corr_b
        return correct
        
    correct_windows = 0
    total_windows = 0
    
    for start in range(0, predicted.size - window_samples + 1, window_samples):
        stop = start + window_samples
        corr_a = safe_corr(predicted[start:stop], wav_a[start:stop])
        corr_b = safe_corr(predicted[start:stop], wav_b[start:stop])
        
        if corr_a == corr_b:
            if np.random.rand() > 0.5:
                correct_windows += 1
        elif corr_a > corr_b:
            correct_windows += 1
        total_windows += 1
        
    if total_windows == 0:
        return False
        
    return (correct_windows > total_windows / 2.0)

def run_sanity_checks():
    print("Loading KUL Cache...")
    loader = KULCachedLoader(REPO_ROOT / "data" / "processed_kul")
    all_subject_data = loader.load_all()
    subject_paths = sorted(all_subject_data.keys())
    
    subject_examples = {}
    for sub_id in subject_paths:
        examples = []
        for t in all_subject_data[sub_id]:
            ex = TrialExample(
                subject=sub_id,
                trial_index=t["meta"].get("TrialID", 0),
                eeg=t["eeg"].numpy().T,
                wav_a=t["audio_a"].numpy().mean(axis=0),
                wav_b=t["audio_b"].numpy().mean(axis=0),
                label=1
            )
            examples.append(ex)
        subject_examples[sub_id] = examples
        
    results = {sub: {} for sub in SUBJECTS_TO_EVAL}
    
    for held_out in SUBJECTS_TO_EVAL:
        if held_out not in subject_paths:
            print(f"Skipping {held_out}, not in cache.")
            continue
            
        print(f"\n========================================")
        print(f"Sanity Check Fold: {held_out}")
        print(f"========================================")
        
        fold_train_examples = []
        other_subjects = [s for s in subject_paths if s != held_out]
        for other_id in other_subjects:
            fold_train_examples.extend(subject_examples[other_id])
                
        print("1. Computing feature statistics...")
        feature_mean, feature_std = feature_statistics(
            fold_train_examples, lags=LAGS, lag_ms=None, lag_step_ms=LAG_STEP_MS, channel_ids=None
        )
        
        feature_count = feature_mean.shape[0]
        base_xtx = np.zeros((feature_count, feature_count), dtype=float)
        base_xty = np.zeros(feature_count, dtype=float)
        
        rand_xtx = np.zeros((feature_count, feature_count), dtype=float)
        rand_xty = np.zeros(feature_count, dtype=float)
        
        print("2. Accumulating Ridge Matrices...")
        for example in fold_train_examples:
            x = lagged_eeg_matrix(example.eeg, lags=LAGS, lag_ms=None, lag_step_ms=LAG_STEP_MS)
            x = standardize_features(x, feature_mean, feature_std)
            xtx = x.T @ x
            
            # Baseline
            base_xtx += xtx
            base_xty += x.T @ example.wav_a
            
            # Random Label Training (Permuted EEG-Audio Pairing)
            rand_xtx += xtx
            random_example = random.choice(fold_train_examples)
            # Match length
            min_len = min(x.shape[0], len(random_example.wav_a))
            rand_xty += x[:min_len].T @ random_example.wav_a[:min_len]
                
        print("3. Solving Ridge...")
        base_weights = np.linalg.solve(base_xtx + RIDGE_LAMBDA * np.eye(feature_count), base_xty)
        rand_weights = np.linalg.solve(rand_xtx + RIDGE_LAMBDA * np.eye(feature_count), rand_xty)
        
        print("4. Evaluating Sanity Conditions...")
        test_examples = subject_examples[held_out]
        
        cond_correct = {
            "Baseline": 0,
            "Zero EEG": 0,
            "Mismatched EEG": 0,
            "Shuffle EEG": 0,
            "Mismatched Audio": 0,
            "Shift 5s": 0,
            "Shift 10s": 0,
            "Shift 20s": 0,
            "Swap Labels": 0,
            "Random Train": 0,
        }
        num_trials = len(test_examples)
        
        for i, example in enumerate(test_examples):
            # Compute normal X
            x = lagged_eeg_matrix(example.eeg, lags=LAGS, lag_ms=None, lag_step_ms=LAG_STEP_MS)
            x = standardize_features(x, feature_mean, feature_std)
            
            # Baseline Pred
            pred_base = x @ base_weights
            if pred_base.std() > 0:
                pred_base = (pred_base - pred_base.mean()) / (pred_base.std() + 1e-12)
            else:
                pred_base = np.zeros_like(pred_base)
            
            # Zero EEG
            x_zero = np.zeros_like(x)
            pred_zero = x_zero @ base_weights
            if pred_zero.std() > 0:
                pred_zero = (pred_zero - pred_zero.mean()) / (pred_zero.std() + 1e-12)
            else:
                pred_zero = np.zeros_like(pred_zero)
            
            # Mismatched EEG
            other_subj = random.choice(other_subjects)
            other_ex = random.choice(subject_examples[other_subj])
            x_mismatch = lagged_eeg_matrix(other_ex.eeg, lags=LAGS, lag_ms=None, lag_step_ms=LAG_STEP_MS)
            x_mismatch = standardize_features(x_mismatch, feature_mean, feature_std)
            pred_mism_eeg = x_mismatch @ base_weights
            if pred_mism_eeg.std() > 0:
                pred_mism_eeg = (pred_mism_eeg - pred_mism_eeg.mean()) / (pred_mism_eeg.std() + 1e-12)
            else:
                pred_mism_eeg = np.zeros_like(pred_mism_eeg)
            
            # Shuffle EEG
            shuffled_eeg = np.copy(example.eeg)
            for ch in range(shuffled_eeg.shape[1]):
                np.random.shuffle(shuffled_eeg[:, ch])
            x_shuff = lagged_eeg_matrix(shuffled_eeg, lags=LAGS, lag_ms=None, lag_step_ms=LAG_STEP_MS)
            x_shuff = standardize_features(x_shuff, feature_mean, feature_std)
            pred_shuff = x_shuff @ base_weights
            if pred_shuff.std() > 0:
                pred_shuff = (pred_shuff - pred_shuff.mean()) / (pred_shuff.std() + 1e-12)
            else:
                pred_shuff = np.zeros_like(pred_shuff)
            
            # Random Train Pred
            pred_rtrain = x @ rand_weights
            if pred_rtrain.std() > 0:
                pred_rtrain = (pred_rtrain - pred_rtrain.mean()) / (pred_rtrain.std() + 1e-12)
            else:
                pred_rtrain = np.zeros_like(pred_rtrain)
            
            # Evaluations
            if evaluate_trial_majority_vote(pred_base, example.wav_a, example.wav_b, WINDOW_SEC): cond_correct["Baseline"] += 1
            if evaluate_trial_majority_vote(pred_zero, example.wav_a, example.wav_b, WINDOW_SEC): cond_correct["Zero EEG"] += 1
            if evaluate_trial_majority_vote(pred_mism_eeg, example.wav_a, example.wav_b, WINDOW_SEC): cond_correct["Mismatched EEG"] += 1
            if evaluate_trial_majority_vote(pred_shuff, example.wav_a, example.wav_b, WINDOW_SEC): cond_correct["Shuffle EEG"] += 1
            
            # Mismatched Audio
            other_subj = random.choice(other_subjects)
            other_audio = random.choice(subject_examples[other_subj])
            if evaluate_trial_majority_vote(pred_base, other_audio.wav_a, other_audio.wav_b, WINDOW_SEC): cond_correct["Mismatched Audio"] += 1
            
            # Circular Shifts
            shift_5 = 5 * FS
            if evaluate_trial_majority_vote(pred_base, np.roll(example.wav_a, shift_5), np.roll(example.wav_b, shift_5), WINDOW_SEC): cond_correct["Shift 5s"] += 1
            shift_10 = 10 * FS
            if evaluate_trial_majority_vote(pred_base, np.roll(example.wav_a, shift_10), np.roll(example.wav_b, shift_10), WINDOW_SEC): cond_correct["Shift 10s"] += 1
            shift_20 = 20 * FS
            if evaluate_trial_majority_vote(pred_base, np.roll(example.wav_a, shift_20), np.roll(example.wav_b, shift_20), WINDOW_SEC): cond_correct["Shift 20s"] += 1
            
            # Swap Labels
            if evaluate_trial_majority_vote(pred_base, example.wav_b, example.wav_a, WINDOW_SEC): cond_correct["Swap Labels"] += 1
            
            # Random Train
            if evaluate_trial_majority_vote(pred_rtrain, example.wav_a, example.wav_b, WINDOW_SEC): cond_correct["Random Train"] += 1
            
        for k in cond_correct:
            results[held_out][k] = (cond_correct[k] / num_trials) * 100.0
            
    # Print Table
    cols = ["Baseline", "Zero EEG", "Mismatched EEG", "Shuffle EEG", "Mismatched Audio", "Shift 5s", "Shift 10s", "Shift 20s", "Swap Labels", "Random Train"]
    print("\n" + "="*145)
    print(f"{'Subject':<8} | " + " | ".join([f"{c:<10}" for c in cols]))
    print("="*145)
    
    metrics = {k: [] for k in cols}
    for sub in SUBJECTS_TO_EVAL:
        r = results[sub]
        for k in metrics:
            metrics[k].append(r[k])
        print(f"{sub:<8} | " + " | ".join([f"{r[c]:<10.1f}" for c in cols]))
        
    print("-" * 145)
    print(f"{'Mean':<8} | " + " | ".join([f"{np.mean(metrics[c]):<10.1f}" for c in cols]))
    print(f"{'Std':<8} | " + " | ".join([f"{np.std(metrics[c]):<10.1f}" for c in cols]))
    print("="*145)
    
if __name__ == "__main__":
    run_sanity_checks()
