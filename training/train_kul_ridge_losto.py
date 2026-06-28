import os
import sys
import csv
import json
import numpy as np
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from data.kul_cached_dataset import KULCachedLoader
from baselines.ridge_aad import TrialExample, lagged_eeg_matrix, feature_statistics, standardize_features
from training.loso_ridge_runner import evaluate_trial_windows
from evaluation.story_split import iter_leave_one_story_out
from evaluation.aad_metrics import TrialScore, summarize_trials, safe_corr

import matplotlib.pyplot as plt

FS = 64
RIDGE_LAMBDA = 100.0
LAGS = 32
LAG_STEP_MS = 16

def evaluate_trial_majority_vote(predicted: np.ndarray, wav_a: np.ndarray, wav_b: np.ndarray, window_seconds: int, fs: int = 64):
    if window_seconds <= 0:
        corr_a = safe_corr(predicted, wav_a)
        corr_b = safe_corr(predicted, wav_b)
        return corr_a > corr_b, 1, 1 if corr_a > corr_b else 0
        
    window_samples = window_seconds * fs
    if window_samples >= predicted.size:
        corr_a = safe_corr(predicted, wav_a)
        corr_b = safe_corr(predicted, wav_b)
        return corr_a > corr_b, 1, 1 if corr_a > corr_b else 0
        
    correct_windows = 0
    total_windows = 0
    
    for start in range(0, predicted.size - window_samples + 1, window_samples):
        stop = start + window_samples
        corr_a = safe_corr(predicted[start:stop], wav_a[start:stop])
        corr_b = safe_corr(predicted[start:stop], wav_b[start:stop])
        if corr_a > corr_b:
            correct_windows += 1
        total_windows += 1
        
    if total_windows == 0:
        return False, 0, 0
        
    trial_correct = (correct_windows > total_windows / 2.0)
    return trial_correct, total_windows, correct_windows

def run_ridge_losto_window(window_sec, all_subject_data):
    print(f"\n==================================================")
    print(f"Running LOStO Ridge for Window: {window_sec}s")
    print(f"==================================================")
    
    per_story = []
    
    global_legacy_trials = 0
    global_legacy_correct = 0
    
    global_mv_trials = 0
    global_mv_correct = 0
    global_mv_windows_total = 0
    global_mv_windows_correct = 0
        
    # We iterate over story folds
    for fold_index, (held_out_story, train_records, test_records) in enumerate(iter_leave_one_story_out(all_subject_data), start=1):
        
        # 1. Convert train_records to TrialExamples
        fold_train_examples = []
        for rec in train_records:
            t = rec["trial"]
            ex = TrialExample(
                subject=rec["sub_id"],
                trial_index=t["meta"].get("TrialID", 0),
                eeg=t["eeg"].numpy().T,
                wav_a=t["audio_a"].numpy().mean(axis=0),
                wav_b=t["audio_b"].numpy().mean(axis=0),
                label=1  # We use label=1 because wav_a is always the target in our cache.
            )
            fold_train_examples.append(ex)
                
        # 2. Compute feature statistics for training
        feature_mean, feature_std = feature_statistics(
            fold_train_examples, 
            lags=LAGS, 
            lag_ms=None, 
            lag_step_ms=LAG_STEP_MS, 
            channel_ids=None
        )
        
        feature_count = feature_mean.shape[0]
        train_xtx = np.zeros((feature_count, feature_count), dtype=float)
        train_xty = np.zeros(feature_count, dtype=float)
        
        # 3. Accumulate XT*X and XT*Y
        for i, example in enumerate(fold_train_examples, start=1):
            eeg = example.eeg
            x = lagged_eeg_matrix(eeg, lags=LAGS, lag_ms=None, lag_step_ms=LAG_STEP_MS)
            x = standardize_features(x, feature_mean, feature_std)
            
            y = example.wav_a 
            
            train_xtx += x.T @ x
            train_xty += x.T @ y
            
        # 4. Solve Ridge
        weights = np.linalg.solve(train_xtx + RIDGE_LAMBDA * np.eye(train_xtx.shape[0], dtype=float), train_xty)
        
        # 5. Evaluate Held-out Story
        test_examples = []
        for rec in test_records:
            t = rec["trial"]
            ex = TrialExample(
                subject=rec["sub_id"],
                trial_index=t["meta"].get("TrialID", 0),
                eeg=t["eeg"].numpy().T,
                wav_a=t["audio_a"].numpy().mean(axis=0),
                wav_b=t["audio_b"].numpy().mean(axis=0),
                label=1
            )
            test_examples.append(ex)
            
        legacy_trial_correct = 0
        mv_trial_correct = 0
        mv_windows_correct = 0
        mv_windows_total = 0
        
        for example in test_examples:
            # Predict
            eeg = example.eeg
            x = lagged_eeg_matrix(eeg, lags=LAGS, lag_ms=None, lag_step_ms=LAG_STEP_MS)
            x = standardize_features(x, feature_mean, feature_std)
            pred = x @ weights
            pred = pred - pred.mean()
            pred = pred / (pred.std() + 1e-12)
            
            # Legacy Eval (Averaging Correlations)
            corr_a, corr_b = evaluate_trial_windows(pred, example.wav_a, example.wav_b, window_seconds=window_sec)
            if corr_a > corr_b:
                legacy_trial_correct += 1
                
            # Majority Vote Eval
            trial_ok, n_win, c_win = evaluate_trial_majority_vote(pred, example.wav_a, example.wav_b, window_sec)
            if trial_ok:
                mv_trial_correct += 1
            mv_windows_total += n_win
            mv_windows_correct += c_win
            
        total_trials = len(test_examples)
        
        legacy_acc = legacy_trial_correct / total_trials if total_trials > 0 else 0
        mv_trial_acc = mv_trial_correct / total_trials if total_trials > 0 else 0
        mv_win_acc = mv_windows_correct / mv_windows_total if mv_windows_total > 0 else 0
        
        per_story.append({
            "held_out_story": held_out_story,
            "legacy_trial_acc": legacy_acc,
            "mv_trial_acc": mv_trial_acc,
            "mv_win_acc": mv_win_acc,
        })
        
        global_legacy_trials += total_trials
        global_legacy_correct += legacy_trial_correct
        global_mv_trials += total_trials
        global_mv_correct += mv_trial_correct
        global_mv_windows_total += mv_windows_total
        global_mv_windows_correct += mv_windows_correct
        
        print(f"    Fold {held_out_story}: Legacy Trial={legacy_acc*100:.1f}% | MV Trial={mv_trial_acc*100:.1f}% | MV Win={mv_win_acc*100:.1f}%")
        
    summary = {
        "legacy_trial_acc": global_legacy_correct / global_legacy_trials if global_legacy_trials > 0 else 0,
        "mv_trial_acc": global_mv_correct / global_mv_trials if global_mv_trials > 0 else 0,
        "mv_win_acc": global_mv_windows_correct / global_mv_windows_total if global_mv_windows_total > 0 else 0,
    }
    
    print(f"Done Window {window_sec}s | Legacy Trial={summary['legacy_trial_acc']*100:.1f}% | MV Trial={summary['mv_trial_acc']*100:.1f}% | MV Win={summary['mv_win_acc']*100:.1f}%")
    return summary, per_story

def main():
    print("Loading KUL Cache...")
    loader = KULCachedLoader(REPO_ROOT / "data" / "processed_kul")
    try:
        all_subject_data = loader.load_all()
    except FileNotFoundError:
        print("KUL Cache not found. Please run preprocessing/build_kul_cache.py")
        return
        
    out_dir = REPO_ROOT / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    windows = [2, 5, 10, 15, 30]
    
    window_results = {}
    
    for w in windows:
        summary, per_story = run_ridge_losto_window(w, all_subject_data)
        window_results[w] = summary
        
        # Save fold-level results
        with open(out_dir / f"kul_ridge_losto_window_{w}.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["held_out_story", "legacy_trial_acc", "mv_trial_acc", "mv_win_acc"])
            writer.writeheader()
            for row in per_story:
                writer.writerow(row)
                
    # Save overall summary
    summary_path = out_dir / "kul_ridge_losto_summary.csv"
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["Window", "Legacy_Trial_Acc", "MV_Trial_Acc", "MV_Window_Acc"])
        writer.writeheader()
        
        for w in windows:
            writer.writerow({
                "Window": w,
                "Legacy_Trial_Acc": window_results[w]["legacy_trial_acc"],
                "MV_Trial_Acc": window_results[w]["mv_trial_acc"],
                "MV_Window_Acc": window_results[w]["mv_win_acc"],
            })
            
    # Plot
    plt.figure(figsize=(10, 6))
    x = windows
    y_legacy = [window_results[w]["legacy_trial_acc"] * 100 for w in windows]
    y_mv = [window_results[w]["mv_trial_acc"] * 100 for w in windows]
    y_win = [window_results[w]["mv_win_acc"] * 100 for w in windows]
    
    plt.plot(x, y_legacy, '-o', label='Legacy Trial Acc (Correlation Avg)', linewidth=2)
    plt.plot(x, y_mv, '-s', label='MV Trial Acc (Majority Vote)', linewidth=2)
    plt.plot(x, y_win, '--^', label='MV Window Acc (Thresholded)', linewidth=2, color='gray')
    
    plt.axhline(50, color='r', linestyle='--', label='Chance (50%)')
    plt.xlabel("Window Size (s)", fontsize=12)
    plt.ylabel("Accuracy (%)", fontsize=12)
    plt.title("KUL LOStO Ridge AAD Baseline", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.ylim(30, 100)
    plt.tight_layout()
    plt.savefig(out_dir / "kul_ridge_losto.png", dpi=300)
    print(f"\nSaved summary plot to {out_dir / 'kul_ridge_losto.png'}")
    
if __name__ == "__main__":
    main()
