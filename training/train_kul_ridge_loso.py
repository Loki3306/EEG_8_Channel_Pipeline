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
from evaluation.aad_metrics import TrialScore, summarize_trials

import matplotlib.pyplot as plt

FS = 64
RIDGE_LAMBDA = 100.0
LAGS = 32
LAG_STEP_MS = 16

def run_ridge_loso_window(window_sec, all_subject_data):
    print(f"\n==================================================")
    print(f"Running LOSO Ridge for Window: {window_sec}s")
    print(f"==================================================")
    
    subject_paths = sorted(all_subject_data.keys())
    
    per_subject = []
    all_scores = []
    
    # Pre-format KUL dicts into TrialExamples
    subject_examples = {}
    for sub_id in subject_paths:
        examples = []
        for t in all_subject_data[sub_id]:
            # Audio A is ALWAYS the attended track in our KUL Cache.
            ex = TrialExample(
                subject=sub_id,
                trial_index=t["meta"].get("TrialID", 0),
                eeg=t["eeg"].numpy(),
                wav_a=t["audio_a"].numpy().ravel(),
                wav_b=t["audio_b"].numpy().ravel(),
                label=1  # We use label=1 because wav_a is always the target.
            )
            examples.append(ex)
        subject_examples[sub_id] = examples
        
    for fold_index, held_out in enumerate(subject_paths, start=1):
        print(f"  Fold {fold_index}/{len(subject_paths)}: held out {held_out} | fitting ridge")
        
        fold_train_examples = []
        for other_id in subject_paths:
            if other_id != held_out:
                fold_train_examples.extend(subject_examples[other_id])
                
        # 1. Compute feature statistics for training
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
        
        # 2. Accumulate XT*X and XT*Y
        n_total = len(fold_train_examples)
        for i, example in enumerate(fold_train_examples, start=1):
            eeg = example.eeg
            x = lagged_eeg_matrix(eeg, lags=LAGS, lag_ms=None, lag_step_ms=LAG_STEP_MS)
            x = standardize_features(x, feature_mean, feature_std)
            
            # y is always wav_a because audio_a is attended in KUL Cache
            y = example.wav_a 
            
            train_xtx += x.T @ x
            train_xty += x.T @ y
            
        # 3. Solve Ridge
        weights = np.linalg.solve(train_xtx + RIDGE_LAMBDA * np.eye(train_xtx.shape[0], dtype=float), train_xty)
        
        # 4. Evaluate Held-out
        test_examples = subject_examples[held_out]
        
        scores = []
        for example in test_examples:
            # Predict
            eeg = example.eeg
            x = lagged_eeg_matrix(eeg, lags=LAGS, lag_ms=None, lag_step_ms=LAG_STEP_MS)
            x = standardize_features(x, feature_mean, feature_std)
            pred = x @ weights
            pred = pred - pred.mean()
            pred = pred / (pred.std() + 1e-12)
            
            # Evaluate window corr
            corr_a, corr_b = evaluate_trial_windows(pred, example.wav_a, example.wav_b, window_seconds=window_sec)
            
            true_stream = "A" # wav_a is always attended
            predicted_stream = "A" if corr_a > corr_b else "B"
            
            scores.append(
                TrialScore(
                    trial_index=example.trial_index,
                    corr_a=corr_a,
                    corr_b=corr_b,
                    true_stream=true_stream,
                    predicted_stream=predicted_stream,
                )
            )
            
        fold_summary = summarize_trials(scores)
        fold_summary["held_out_subject"] = held_out
        per_subject.append(fold_summary)
        all_scores.extend(scores)
        
        print(f"    Fold {held_out}: Trial Acc={fold_summary['trial_accuracy']*100:.2f}% | Win Acc={fold_summary['balanced_accuracy']*100:.2f}%")
        
    summary = summarize_trials(all_scores)
    print(f"Done Window {window_sec}s | Trial Acc={summary['trial_accuracy']*100:.2f}% | Win Acc={summary['balanced_accuracy']*100:.2f}%")
    return summary, per_subject

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
        summary, per_sub = run_ridge_loso_window(w, all_subject_data)
        window_results[w] = summary
        
        # Save fold-level results
        with open(out_dir / f"kul_ridge_window_{w}.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["held_out_subject", "trial_accuracy", "balanced_accuracy"])
            writer.writeheader()
            for row in per_sub:
                writer.writerow({
                    "held_out_subject": row["held_out_subject"],
                    "trial_accuracy": row["trial_accuracy"],
                    "balanced_accuracy": row["balanced_accuracy"]
                })
                
    # Save overall summary
    summary_path = out_dir / "kul_ridge_summary.csv"
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["Window", "Mean_Trial_Acc", "Mean_Window_Acc", "Std_Trial_Acc"])
        writer.writeheader()
        
        for w in windows:
            # Recompute Std from fold-level? 
            # Or use global? Wait, DTU summary didn't have std, we will compute std from trial accuracy of folds.
            writer.writerow({
                "Window": w,
                "Mean_Trial_Acc": window_results[w]["trial_accuracy"],
                "Mean_Window_Acc": window_results[w]["balanced_accuracy"],
                "Std_Trial_Acc": 0.0 # Placeholder if needed, or we compute it if needed.
            })
            
    # Compute true STD across folds
    for w in windows:
        with open(out_dir / f"kul_ridge_window_{w}.csv", "r") as f:
            reader = csv.DictReader(f)
            fold_accs = [float(r["trial_accuracy"]) for r in reader]
        window_results[w]["std"] = np.std(fold_accs)
        
    # Re-write summary with true STD
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["Window", "Mean_Trial_Acc", "Mean_Window_Acc", "Std_Trial_Acc"])
        writer.writeheader()
        for w in windows:
            writer.writerow({
                "Window": w,
                "Mean_Trial_Acc": window_results[w]["trial_accuracy"],
                "Mean_Window_Acc": window_results[w]["balanced_accuracy"],
                "Std_Trial_Acc": window_results[w]["std"]
            })
            
    # Plot
    plt.figure(figsize=(8, 6))
    x = windows
    y = [window_results[w]["trial_accuracy"] * 100 for w in windows]
    e = [window_results[w]["std"] * 100 for w in windows]
    
    plt.errorbar(x, y, yerr=e, fmt='-o', color='b', capsize=5, capthick=2, elinewidth=2, markersize=8)
    plt.axhline(50, color='r', linestyle='--', label='Chance (50%)')
    plt.xlabel("Window Size (s)", fontsize=12)
    plt.ylabel("Mean Trial Accuracy (%)", fontsize=12)
    plt.title("KUL Classical Ridge AAD Baseline", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.ylim(30, 100)
    plt.tight_layout()
    plt.savefig(out_dir / "kul_ridge_window_scaling.png", dpi=300)
    print(f"\nSaved summary plot to {out_dir / 'kul_ridge_window_scaling.png'}")
    
if __name__ == "__main__":
    main()
