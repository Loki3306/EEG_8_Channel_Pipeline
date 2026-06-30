import os
import sys
import csv
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from data.kul_cached_dataset import KULCachedLoader
from baselines.ridge_aad import TrialExample, lagged_eeg_matrix, feature_statistics, standardize_features
from training.loso_ridge_runner import evaluate_trial_windows
from evaluation.aad_metrics import safe_corr
from models.eegnet_ridge import ResidualEEGNetRidge

FS = 64
RIDGE_LAMBDA = 100.0
LAGS = 32
LAG_STEP_MS = 16

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 10
LR = 1e-4

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

def run_residual_loso_window(window_sec, all_subject_data):
    print(f"\n==================================================")
    print(f"Running LOSO Residual Neural Ridge for Window: {window_sec}s")
    print(f"==================================================")
    
    subject_paths = sorted(all_subject_data.keys())
    per_subject = []
    
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
        
    global_legacy_trials = 0
    global_legacy_correct = 0
    
    global_mv_trials = 0
    global_mv_correct = 0
    global_mv_windows_total = 0
    global_mv_windows_correct = 0
        
    for fold_index, held_out in enumerate(subject_paths, start=1):
        print(f"  Fold {fold_index}/{len(subject_paths)}: held out {held_out}")
        
        fold_train_examples = []
        for other_id in subject_paths:
            if other_id != held_out:
                fold_train_examples.extend(subject_examples[other_id])
                
        # 1. Feature Stats for Ridge
        feature_mean, feature_std = feature_statistics(fold_train_examples, lags=LAGS, lag_ms=None, lag_step_ms=LAG_STEP_MS, channel_ids=None)
        feature_count = feature_mean.shape[0]
        
        train_xtx = np.zeros((feature_count, feature_count), dtype=float)
        train_xty = np.zeros(feature_count, dtype=float)
        
        # 2. Accumulate XT*X for base Ridge initialization (using standard uncentered Y as baseline does)
        for i, example in enumerate(fold_train_examples):
            x = lagged_eeg_matrix(example.eeg, lags=LAGS, lag_ms=None, lag_step_ms=LAG_STEP_MS)
            x = standardize_features(x, feature_mean, feature_std)
            
            y = example.wav_a
            
            train_xtx += x.T @ x
            train_xty += x.T @ y
            
        # 3. Solve Ridge exactly as baseline
        weights = np.linalg.solve(train_xtx + RIDGE_LAMBDA * np.eye(feature_count, dtype=float), train_xty)
        
        # 4. Initialize Neural Residual Model
        model = ResidualEEGNetRidge(ridge_feature_count=feature_count, in_channels=8).to(DEVICE)
        model.load_ridge_weights(weights)
        
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)
        criterion = nn.MSELoss()
        
        # 5. Train Neural Residual Network
        print(f"    Training Neural Residual...")
        model.train()
        
        accumulation_steps = 16
        
        for epoch in range(EPOCHS):
            epoch_loss = 0.0
            epoch_ridge_loss = 0.0
            np.random.shuffle(fold_train_examples)
            optimizer.zero_grad()
            
            for step, example in enumerate(fold_train_examples):
                x_lagged = lagged_eeg_matrix(example.eeg, lags=LAGS, lag_step_ms=LAG_STEP_MS)
                x_lagged = standardize_features(x_lagged, feature_mean, feature_std)
                
                # Raw EEG for EEGNet
                x_raw = example.eeg.T # [Channels, Time]
                
                # To tensors
                x_lagged_t = torch.FloatTensor(x_lagged).unsqueeze(0).to(DEVICE) # [1, Time, Features]
                x_raw_t = torch.FloatTensor(x_raw).unsqueeze(0).to(DEVICE) # [1, Channels, Time]
                
                # Target y (full length, aligned with x_lagged which pads zeros)
                y_t = torch.FloatTensor(example.wav_a).unsqueeze(0).to(DEVICE) # [1, Time]
                
                pred = model(x_raw_t, x_lagged_t)
                
                # For logging, monitor Ridge-only loss
                with torch.no_grad():
                    base_pred = model.base_ridge(x_lagged_t).squeeze(-1)
                    r_loss = criterion(base_pred[:, :y_t.size(1)], y_t[:, :base_pred.size(1)])
                    epoch_ridge_loss += r_loss.item()
                
                # Sequence length matching
                min_len = min(pred.size(1), y_t.size(1))
                loss = criterion(pred[:, :min_len], y_t[:, :min_len])
                loss = loss / accumulation_steps
                loss.backward()
                
                if (step + 1) % accumulation_steps == 0 or (step + 1) == len(fold_train_examples):
                    # Gradient clipping for stability
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    optimizer.zero_grad()
                
                epoch_loss += loss.item() * accumulation_steps
                
            if (epoch+1) % 5 == 0:
                print(f"      Epoch {epoch+1}/{EPOCHS}, Hybrid Loss: {epoch_loss/len(fold_train_examples):.4f}, Ridge Loss: {epoch_ridge_loss/len(fold_train_examples):.4f}")
        
        # 6. Evaluate Held-out
        model.eval()
        test_examples = subject_examples[held_out]
        
        legacy_trial_correct = 0
        mv_trial_correct = 0
        mv_windows_correct = 0
        mv_windows_total = 0
        
        with torch.no_grad():
            for example in test_examples:
                x_lagged = lagged_eeg_matrix(example.eeg, lags=LAGS, lag_step_ms=LAG_STEP_MS)
                x_lagged = standardize_features(x_lagged, feature_mean, feature_std)
                x_raw = example.eeg.T
                
                x_lagged_t = torch.FloatTensor(x_lagged).unsqueeze(0).to(DEVICE)
                x_raw_t = torch.FloatTensor(x_raw).unsqueeze(0).to(DEVICE)
                
                pred = model(x_raw_t, x_lagged_t).squeeze(0).cpu().numpy()
                
                wav_a = example.wav_a
                wav_b = example.wav_b
                
                min_len = min(pred.size, wav_a.size)
                pred = pred[:min_len]
                wav_a = wav_a[:min_len]
                wav_b = wav_b[:min_len]
                
                pred = pred - pred.mean()
                pred = pred / (pred.std() + 1e-12)
                
                # Eval
                corr_a, corr_b = evaluate_trial_windows(pred, wav_a, wav_b, window_seconds=window_sec)
                if corr_a > corr_b:
                    legacy_trial_correct += 1
                    
                trial_ok, n_win, c_win = evaluate_trial_majority_vote(pred, wav_a, wav_b, window_sec)
                if trial_ok:
                    mv_trial_correct += 1
                mv_windows_total += n_win
                mv_windows_correct += c_win
                
        total_trials = len(test_examples)
        
        legacy_acc = legacy_trial_correct / total_trials if total_trials > 0 else 0
        mv_trial_acc = mv_trial_correct / total_trials if total_trials > 0 else 0
        mv_win_acc = mv_windows_correct / mv_windows_total if mv_windows_total > 0 else 0
        
        per_subject.append({
            "held_out_subject": held_out,
            "legacy_trial_acc": legacy_acc,
            "mv_trial_acc": mv_trial_acc,
            "mv_win_acc": mv_win_acc,
            "trials_total": total_trials,
            "mv_windows_total": mv_windows_total
        })
        
        print(f"    Legacy Trial Acc: {legacy_acc*100:.1f}%")
        print(f"    Majority Vote Trial Acc: {mv_trial_acc*100:.1f}%")
        print(f"    Window ({window_sec}s) Acc: {mv_win_acc*100:.1f}%")
        
        global_legacy_trials += total_trials
        global_legacy_correct += legacy_trial_correct
        global_mv_trials += total_trials
        global_mv_correct += mv_trial_correct
        global_mv_windows_total += mv_windows_total
        global_mv_windows_correct += mv_windows_correct
        
    global_legacy_acc = global_legacy_correct / global_legacy_trials
    global_mv_trial_acc = global_mv_correct / global_mv_trials
    global_mv_win_acc = global_mv_windows_correct / global_mv_windows_total
    
    summary = {
        "window_sec": window_sec,
        "global_legacy_acc": global_legacy_acc,
        "global_mv_trial_acc": global_mv_trial_acc,
        "global_mv_win_acc": global_mv_win_acc,
    }
    return summary, per_subject

def main():
    print("Loading KUL dataset...")
    if Path("/kaggle/input/datasets/lowk1ee/kul-preprocessed-cache/data/processed_kul").exists():
        cache_dir = Path("/kaggle/input/datasets/lowk1ee/kul-preprocessed-cache/data/processed_kul")
    elif Path("/kaggle/input/kul-preprocessed-cache/data/processed_kul").exists():
        cache_dir = Path("/kaggle/input/kul-preprocessed-cache/data/processed_kul")
    else:
        cache_dir = REPO_ROOT / "data" / "processed_kul"
        
    loader = KULCachedLoader(cache_dir=cache_dir)
    all_subject_data = loader.load_all()
    
    windows_to_test = [60, 30, 10]
    out_dir = REPO_ROOT / "results" / "kul_residual_eegnet_loso"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    results = []
    
    for w in windows_to_test:
        summary, per_story = run_residual_loso_window(w, all_subject_data)
        results.append(summary)
        
        with open(out_dir / f"kul_residual_eegnet_window_{w}.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=per_story[0].keys())
            writer.writeheader()
            writer.writerows(per_story)
            
    summary_path = out_dir / "kul_residual_eegnet_summary.csv"
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
        
    print(f"\nSaved final results to {out_dir}")

if __name__ == "__main__":
    main()
