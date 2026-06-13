"""
Subject Adaptation Screening for EEGNet AAD.

Protocol:
1. For each held-out subject in the screening set (S1, S7, S8, S14):
   a. Train EEGNet on all OTHER screening subjects (LOSO pretrain)
   b. Evaluate the pretrained model on ALL held-out trials (baseline)
   c. Take 1 trial from the held-out subject as calibration data
   d. Freeze all layers except the output projection layer
   e. Fine-tune on the calibration trial for N epochs
   f. Evaluate on the REMAINING held-out trials (adapted)
2. Report both baseline and adapted accuracy side-by-side.

This allows us to measure the exact gain from subject adaptation
before committing to a full 18-subject LOSO run.
"""
import argparse
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pathlib import Path
from copy import deepcopy
from scipy.signal import butter, filtfilt

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from models.eegnet import EEGNet
from baselines.ridge_aad import load_subject_examples, subject_files, iter_leave_one_subject_out

SCREENING_SUBJECTS = ["S1_data_preproc", "S7_data_preproc", "S8_data_preproc", "S14_data_preproc"]

FS = 64
DECISION_WINDOW_SEC = 10
CHANNELS = [13, 46, 43, 23, 50, 0, 52, 14]

def butter_bandpass_filter(data, lowcut, highcut, fs, order=2, axis=0):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    y = filtfilt(b, a, data, axis=axis)
    return y

def normalize_array(arr):
    arr = arr - arr.mean(axis=0, keepdims=True)
    scale = arr.std(axis=0, keepdims=True) + 1e-12
    return arr / scale

class PearsonMSELoss(nn.Module):
    def __init__(self, alpha=0.1):
        super().__init__()
        self.alpha = alpha
        self.mse = nn.MSELoss()

    def forward(self, pred, target):
        pred_mean = pred.mean(dim=2, keepdim=True)
        target_mean = target.mean(dim=2, keepdim=True)
        pred_std = pred.std(dim=2, keepdim=True) + 1e-8
        target_std = target.std(dim=2, keepdim=True) + 1e-8
        
        cov = ((pred - pred_mean) * (target - target_mean)).mean(dim=2)
        corr = cov / (pred_std.squeeze(2) * target_std.squeeze(2))
        
        pearson_loss = 1 - corr.mean()
        mse_loss = self.mse(pred, target)
        
        return pearson_loss + self.alpha * mse_loss

def prepare_dataset(examples, lowcut, highcut):
    X = []
    Y = []
    Y_A = []
    Y_B = []
    
    for ex in examples:
        eeg = ex.eeg[:, CHANNELS].T
        eeg = butter_bandpass_filter(eeg, lowcut, highcut, FS, axis=1)
        wav_a = butter_bandpass_filter(ex.wav_a.reshape(-1, 1), lowcut, highcut, FS, axis=0).ravel()
        wav_b = butter_bandpass_filter(ex.wav_b.reshape(-1, 1), lowcut, highcut, FS, axis=0).ravel()
        
        x_norm = normalize_array(eeg.T).T
        env_a = normalize_array(wav_a.reshape(-1, 1)).ravel()
        env_b = normalize_array(wav_b.reshape(-1, 1)).ravel()
        
        target_env = env_a
        
        min_len = min(x_norm.shape[1], len(target_env))
        x_norm = x_norm[:, :min_len]
        target_env = target_env[:min_len]
        env_a = env_a[:min_len]
        env_b = env_b[:min_len]
        
        X.append(x_norm)
        Y.append(target_env)
        Y_A.append(env_a)
        Y_B.append(env_b)
        
    return X, Y, Y_A, Y_B

def evaluate_windows(pred, env_a, env_b, window_samples):
    num_correct = 0.0
    num_total = 0
    start = 0
    while start + window_samples <= len(pred):
        end = start + window_samples
        p = pred[start:end]
        ea = env_a[start:end]
        eb = env_b[start:end]
        
        std_p = np.std(p)
        if std_p < 1e-12:
            ca = 0.0
            cb = 0.0
        else:
            ca = np.corrcoef(p, ea)[0, 1]
            cb = np.corrcoef(p, eb)[0, 1]
            
        if ca > cb:
            num_correct += 1.0
        elif ca == cb:
            num_correct += 0.5
                
        num_total += 1
        start += window_samples
    return num_correct, num_total

def evaluate_model(model, X, Y_A, Y_B, device):
    model.eval()
    window_samples = DECISION_WINDOW_SEC * FS
    n_correct = 0.0
    n_total = 0
    
    with torch.no_grad():
        for i in range(len(X)):
            x = torch.FloatTensor(X[i]).unsqueeze(0).to(device)
            pred = model(x).squeeze(0).squeeze(0).cpu().numpy()
            
            ea = Y_A[i]
            eb = Y_B[i]
                
            nc, nt = evaluate_windows(pred, ea, eb, window_samples)
            n_correct += nc
            n_total += nt
            
    return n_correct, n_total

def freeze_except_output(model):
    """Freeze all layers except the output projection."""
    for name, param in model.named_parameters():
        if 'output_proj' in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
    
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"    Frozen: {total - trainable} params | Trainable: {trainable} params (output_proj only)")

def fine_tune_on_calibration(model, X_cal, Y_cal, device, ft_epochs=10, ft_lr=1e-4):
    """Fine-tune the unfrozen layers on calibration data."""
    freeze_except_output(model)
    
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), 
                           lr=ft_lr, weight_decay=1e-4)
    criterion = PearsonMSELoss(alpha=0.1)
    
    for epoch in range(ft_epochs):
        model.train()
        epoch_loss = 0.0
        for i in range(len(X_cal)):
            x = torch.FloatTensor(X_cal[i]).unsqueeze(0).to(device)
            y = torch.FloatTensor(Y_cal[i]).unsqueeze(0).unsqueeze(0).to(device)
            
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
    
    return model

def run_screening(lowcut, highcut, num_cal_trials, ft_epochs, ft_lr):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Band: {lowcut}-{highcut} Hz")
    print(f"Calibration trials per subject: {num_cal_trials}")
    print(f"Fine-tuning epochs: {ft_epochs}, LR: {ft_lr}")
    print("=" * 60)
    
    all_paths = subject_files()
    if not all_paths:
        print("No subjects found.")
        return
        
    paths = [p for p in all_paths if p.stem in SCREENING_SUBJECTS]
    print(f"Screening subjects: {[p.stem for p in paths]}")
    
    subject_examples = {str(p): load_subject_examples(p) for p in paths}
    folds = list(iter_leave_one_subject_out(paths))
    
    baseline_accs = []
    adapted_accs = []
    
    for held_out_path, train_paths in folds:
        held_out_key = str(held_out_path)
        print(f"\n{'='*60}")
        print(f"Held-out subject: {held_out_path.stem}")
        print(f"{'='*60}")
        
        # ---- Phase 1: LOSO Pretrain ----
        train_exs = []
        for p in train_paths:
            train_exs.extend(subject_examples[str(p)])
        all_test_exs = subject_examples[held_out_key]
        
        np.random.seed(42)
        np.random.shuffle(train_exs)
        val_split = int(0.1 * len(train_exs))
        val_exs = train_exs[:val_split]
        train_exs = train_exs[val_split:]
        
        X_tr, Y_tr, YA_tr, YB_tr = prepare_dataset(train_exs, lowcut, highcut)
        X_va, Y_va, YA_va, YB_va = prepare_dataset(val_exs, lowcut, highcut)
        
        model = EEGNet(in_channels=8).to(device)
        optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        criterion = PearsonMSELoss(alpha=0.1)
        
        best_val_acc = 0.0
        best_weights = deepcopy(model.state_dict())
        patience = 15
        epochs_no_improve = 0
        
        for epoch in range(100):
            model.train()
            train_loss = 0.0
            for i in range(len(X_tr)):
                x = torch.FloatTensor(X_tr[i]).unsqueeze(0).to(device)
                y = torch.FloatTensor(Y_tr[i]).unsqueeze(0).unsqueeze(0).to(device)
                
                optimizer.zero_grad()
                pred = model(x)
                loss = criterion(pred, y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                
            nc_va, nt_va = evaluate_model(model, X_va, YA_va, YB_va, device)
            val_acc = nc_va / max(nt_va, 1)
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_weights = deepcopy(model.state_dict())
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                
            if (epoch + 1) % 5 == 0 or epochs_no_improve >= patience:
                print(f"  Epoch {epoch+1:02d}/100 | Loss: {train_loss:.4f} | Val: {val_acc*100:.1f}% | Pat: {epochs_no_improve}/{patience}")
                
            if epochs_no_improve >= patience:
                break
                
        model.load_state_dict(best_weights)
        pretrained_weights = deepcopy(model.state_dict())
        
        # ---- Phase 2: Evaluate baseline (no adaptation) on ALL test trials ----
        X_all_test, Y_all_test, YA_all_test, YB_all_test = prepare_dataset(all_test_exs, lowcut, highcut)
        
        nc_base, nt_base = evaluate_model(model, X_all_test, YA_all_test, YB_all_test, device)
        baseline_acc = nc_base / max(nt_base, 1)
        print(f"\n  BASELINE (no adaptation): {baseline_acc*100:.2f}%")
        baseline_accs.append(baseline_acc)
        
        # ---- Phase 3: Split test trials into calibration + evaluation ----
        np.random.seed(123)  # Fixed seed for reproducibility
        n_test = len(all_test_exs)
        perm = np.random.permutation(n_test)
        cal_indices = perm[:num_cal_trials]
        eval_indices = perm[num_cal_trials:]
        
        cal_exs = [all_test_exs[i] for i in cal_indices]
        eval_exs = [all_test_exs[i] for i in eval_indices]
        
        print(f"  Calibration trials: {len(cal_exs)} | Evaluation trials: {len(eval_exs)}")
        
        X_cal, Y_cal, YA_cal, YB_cal = prepare_dataset(cal_exs, lowcut, highcut)
        X_eval, Y_eval, YA_eval, YB_eval = prepare_dataset(eval_exs, lowcut, highcut)
        
        # ---- Phase 4: Fine-tune on calibration data ----
        model.load_state_dict(pretrained_weights)  # Reset to pretrained
        model = fine_tune_on_calibration(model, X_cal, Y_cal, device, ft_epochs, ft_lr)
        
        # ---- Phase 5: Evaluate adapted model on remaining trials ----
        nc_adapt, nt_adapt = evaluate_model(model, X_eval, YA_eval, YB_eval, device)
        adapted_acc = nc_adapt / max(nt_adapt, 1)
        print(f"  ADAPTED  ({num_cal_trials} trial cal): {adapted_acc*100:.2f}%")
        
        # Also evaluate baseline on the same eval subset for fair comparison
        model.load_state_dict(pretrained_weights)
        nc_base_fair, nt_base_fair = evaluate_model(model, X_eval, YA_eval, YB_eval, device)
        baseline_fair_acc = nc_base_fair / max(nt_base_fair, 1)
        print(f"  BASELINE (same eval set):  {baseline_fair_acc*100:.2f}%")
        print(f"  GAIN: {(adapted_acc - baseline_fair_acc)*100:+.2f}%")
        
        adapted_accs.append(adapted_acc)
    
    # ---- Summary ----
    mean_baseline = np.mean(baseline_accs)
    mean_adapted = np.mean(adapted_accs)
    
    print("\n" + "=" * 60)
    print("SUBJECT ADAPTATION SCREENING RESULTS")
    print("=" * 60)
    print(f"  Baseline LOSO (all trials):    {mean_baseline*100:.2f}%")
    print(f"  Adapted  ({num_cal_trials}-trial cal):      {mean_adapted*100:.2f}%")
    print(f"  Mean Gain:                     {(mean_adapted - mean_baseline)*100:+.2f}%")
    print("=" * 60)
    print(f"\n  Per-subject baseline: {[f'{a*100:.1f}%' for a in baseline_accs]}")
    print(f"  Per-subject adapted:  {[f'{a*100:.1f}%' for a in adapted_accs]}")
    
    if mean_adapted > 0.76:
        print("\n  >> VERDICT: PROMISING. Screening exceeds 76% threshold.")
        print("  >> Recommend full 18-subject LOSO with adaptation.")
    elif mean_adapted > mean_baseline:
        print(f"\n  >> VERDICT: MARGINAL GAIN ({(mean_adapted - mean_baseline)*100:+.2f}%).")
        print("  >> Consider testing with more calibration trials or unfreezing more layers.")
    else:
        print("\n  >> VERDICT: NO GAIN. Subject adaptation does not help with current settings.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Subject Adaptation Screening for EEGNet")
    parser.add_argument("--lowcut", type=float, default=1.0, help="Lowcut frequency for bandpass filter")
    parser.add_argument("--highcut", type=float, default=6.0, help="Highcut frequency for bandpass filter")
    parser.add_argument("--num-cal-trials", type=int, default=1, help="Number of calibration trials per subject")
    parser.add_argument("--ft-epochs", type=int, default=10, help="Number of fine-tuning epochs")
    parser.add_argument("--ft-lr", type=float, default=1e-4, help="Fine-tuning learning rate")
    args = parser.parse_args()
    
    run_screening(args.lowcut, args.highcut, args.num_cal_trials, args.ft_epochs, args.ft_lr)
