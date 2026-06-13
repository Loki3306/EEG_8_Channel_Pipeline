import sys
import argparse
import itertools
import json
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pathlib import Path
from copy import deepcopy
from scipy.signal import butter, filtfilt

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from models.vlaai_lite import VLAAILite
from baselines.ridge_aad import load_subject_examples, subject_files, iter_leave_one_subject_out

FS = 64
DECISION_WINDOW_SEC = 10
BP_LOWCUT = 1.0
BP_HIGHCUT = 8.0
CHANNELS = [47, 12, 49, 31, 17, 53, 30, 37]

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

def prepare_dataset(examples):
    X, Y, Y_A, Y_B = [], [], [], []
    for ex in examples:
        eeg = ex.eeg[:, CHANNELS].T
        eeg = butter_bandpass_filter(eeg, BP_LOWCUT, BP_HIGHCUT, FS, axis=1)
        wav_a = butter_bandpass_filter(ex.wav_a.reshape(-1, 1), BP_LOWCUT, BP_HIGHCUT, FS, axis=0).ravel()
        wav_b = butter_bandpass_filter(ex.wav_b.reshape(-1, 1), BP_LOWCUT, BP_HIGHCUT, FS, axis=0).ravel()
        
        x_norm = normalize_array(eeg.T).T
        env_a = normalize_array(wav_a.reshape(-1, 1)).ravel()
        env_b = normalize_array(wav_b.reshape(-1, 1)).ravel()
        
        target_env = env_a
        min_len = min(x_norm.shape[1], len(target_env))
        
        X.append(x_norm[:, :min_len])
        Y.append(target_env[:min_len])
        Y_A.append(env_a[:min_len])
        Y_B.append(env_b[:min_len])
        
    return X, Y, Y_A, Y_B

def evaluate_windows(pred, env_a, env_b, window_samples):
    num_correct, num_total, start = 0.0, 0, 0
    while start + window_samples <= len(pred):
        end = start + window_samples
        p = pred[start:end]
        ea = env_a[start:end]
        eb = env_b[start:end]
        
        if np.std(p) < 1e-12:
            ca, cb = 0.0, 0.0
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

def evaluate_model(model, X, Y_A, Y_B, device, random_eeg=False, shuffle_eeg=False):
    model.eval()
    window_samples = DECISION_WINDOW_SEC * FS
    n_correct, n_total = 0.0, 0
    
    np.random.seed(42)
    shuffle_indices = np.random.permutation(len(X))
    while np.any(shuffle_indices == np.arange(len(X))) and len(X) > 1:
        shuffle_indices = np.random.permutation(len(X))
    
    with torch.no_grad():
        for i in range(len(X)):
            if shuffle_eeg:
                shuf_idx = shuffle_indices[i]
                x_np = X[shuf_idx]
                mlen = min(x_np.shape[1], len(Y_A[i]))
                x_np = x_np[:, :mlen]
            else:
                x_np = X[i]
                
            x = torch.FloatTensor(x_np).unsqueeze(0).to(device)
            if random_eeg:
                x = torch.randn_like(x)
            
            pred = model(x).squeeze(0).squeeze(0).cpu().numpy()
            
            if shuffle_eeg:
                ea = Y_A[i][:mlen]
                eb = Y_B[i][:mlen]
                pred = pred[:mlen]
            else:
                ea = Y_A[i]
                eb = Y_B[i]
                
            nc, nt = evaluate_windows(pred, ea, eb, window_samples)
            n_correct += nc
            n_total += nt
            
    return n_correct, n_total

def run_experiment(lr, wd, width_mult, device, folds_data):
    spatial_dim = 32 * width_mult
    temporal_dim = 64 * width_mult
    
    all_normal_accs = []
    all_random_accs = []
    all_shuffle_accs = []
    
    for held_out_key, train_exs, val_exs, test_exs in folds_data:
        # Note: We reuse prepared dataset splits loaded in memory to save massive processing time
        X_tr, Y_tr, YA_tr, YB_tr = train_exs
        X_va, Y_va, YA_va, YB_va = val_exs
        X_te, Y_te, YA_te, YB_te = test_exs
        
        model = VLAAILite(in_channels=8, spatial_dim=spatial_dim, temporal_dim=temporal_dim).to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
        criterion = PearsonMSELoss(alpha=0.1)
        
        best_val_acc = 0.0
        best_weights = deepcopy(model.state_dict())
        patience = 5
        epochs_no_improve = 0
        
        for epoch in range(50):
            model.train()
            for i in range(len(X_tr)):
                x = torch.FloatTensor(X_tr[i]).unsqueeze(0).to(device)
                y = torch.FloatTensor(Y_tr[i]).unsqueeze(0).unsqueeze(0).to(device)
                
                optimizer.zero_grad()
                loss = criterion(model(x), y)
                loss.backward()
                optimizer.step()
                
            nc_va, nt_va = evaluate_model(model, X_va, YA_va, YB_va, device)
            val_acc = nc_va / max(nt_va, 1)
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_weights = deepcopy(model.state_dict())
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                
            if epochs_no_improve >= patience:
                break
                
        model.load_state_dict(best_weights)
        
        nc_norm, nt_norm = evaluate_model(model, X_te, YA_te, YB_te, device)
        nc_rand, nt_rand = evaluate_model(model, X_te, YA_te, YB_te, device, random_eeg=True)
        nc_shuf, nt_shuf = evaluate_model(model, X_te, YA_te, YB_te, device, shuffle_eeg=True)
        
        all_normal_accs.append(nc_norm / max(nt_norm, 1))
        all_random_accs.append(nc_rand / max(nt_rand, 1))
        all_shuffle_accs.append(nc_shuf / max(nt_shuf, 1))
        
    return {
        "lr": lr, "wd": wd, "width_mult": width_mult,
        "normal_acc": np.mean(all_normal_accs),
        "random_acc": np.mean(all_random_accs),
        "shuffle_acc": np.mean(all_shuffle_accs)
    }

def train_sweep():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    paths = subject_files()
    if not paths:
        return
        
    print("Pre-loading and preparing all subjects (this takes a moment)...")
    subject_examples = {str(p): load_subject_examples(p) for p in paths}
    
    # Pre-process folds so we don't repeat the slow bandpass filters for every grid parameter
    folds = list(iter_leave_one_subject_out(paths))
    folds_data = []
    for held_out_path, train_paths in folds:
        held_out_key = str(held_out_path)
        train_exs = []
        for p in train_paths:
            train_exs.extend(subject_examples[str(p)])
            
        test_exs = subject_examples[held_out_key]
        
        np.random.seed(42)
        np.random.shuffle(train_exs)
        val_split = int(0.1 * len(train_exs))
        val_exs = train_exs[:val_split]
        train_exs = train_exs[val_split:]
        
        # Store already prepared datasets
        folds_data.append((
            held_out_key,
            prepare_dataset(train_exs),
            prepare_dataset(val_exs),
            prepare_dataset(test_exs)
        ))
    
    print("Pre-processing complete. Starting sweep...")
    
    # Targeted Grid Search
    lrs = [1e-4, 3e-4, 1e-3]
    wds = [1e-4]
    widths = [1]
    
    combinations = list(itertools.product(lrs, wds, widths))
    results = []
    
    results_file = Path("vlaai_sweep_results.json")
    
    for idx, (lr, wd, width_mult) in enumerate(combinations):
        print(f"\n[{idx+1}/{len(combinations)}] Running config: LR={lr}, WD={wd}, Width={width_mult}x")
        res = run_experiment(lr, wd, width_mult, device, folds_data)
        
        print(f" -> Normal : {res['normal_acc']*100:.2f}%")
        print(f" -> Random : {res['random_acc']*100:.2f}%")
        print(f" -> Shuffle: {res['shuffle_acc']*100:.2f}%")
        
        results.append(res)
        
        # Save intermediate results
        with open(results_file, "w") as f:
            json.dump(results, f, indent=4)
            
    print(f"\nSweep complete! Best configuration saved to {results_file}.")

if __name__ == "__main__":
    train_sweep()
