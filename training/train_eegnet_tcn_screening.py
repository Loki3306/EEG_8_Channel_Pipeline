import argparse
import sys
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

from models.eegnet_tcn import EEGNetTCN
from baselines.ridge_aad import load_subject_examples, subject_files, iter_leave_one_subject_out

SCREENING_SUBJECTS = ["S3_data_preproc", "S13_data_preproc", "S15_data_preproc", "S16_data_preproc"]
FS = 64
DECISION_WINDOW_SEC = 10

class PearsonMSELoss(nn.Module):
    def __init__(self, alpha=0.1):
        super().__init__()
        self.alpha = alpha
        self.mse = nn.MSELoss()

    def forward(self, pred, target):
        pred_flat = pred.reshape(pred.shape[0], -1)
        target_flat = target.reshape(target.shape[0], -1)
        
        p_mean = pred_flat.mean(dim=1, keepdim=True)
        t_mean = target_flat.mean(dim=1, keepdim=True)
        
        p_std = torch.sqrt(pred_flat.var(dim=1, keepdim=True, unbiased=False) + 1e-8)
        t_std = torch.sqrt(target_flat.var(dim=1, keepdim=True, unbiased=False) + 1e-8)
        
        cov = ((pred_flat - p_mean) * (target_flat - t_mean)).mean(dim=1, keepdim=True)
        corr = cov / (p_std * t_std)
        
        pearson_loss = 1 - corr.mean()
        mse_loss = self.mse(pred, target)
        
        return pearson_loss + self.alpha * mse_loss

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

def get_mapping_data():
    return None

def prepare_dataset(examples, channels, lowcut, highcut, subject_id, mapping):
    X = []
    Y = []
    Y_A = []
    Y_B = []
    
    sub_key = subject_id.replace("_data_preproc", "")
    
    for i, ex in enumerate(examples):
        eeg = ex.eeg[:, channels].T
        eeg = butter_bandpass_filter(eeg, lowcut, highcut, FS, axis=1)
        x_norm = normalize_array(eeg.T).T
        
        env_a_full = ex.env_a.T
        env_b_full = ex.env_b.T
        
        if len(env_a_full.shape) == 1:
            env_a_full = env_a_full.reshape(1, -1)
        if len(env_b_full.shape) == 1:
            env_b_full = env_b_full.reshape(1, -1)
            
        target_env = env_a_full
        
        min_len = min(x_norm.shape[1], target_env.shape[1])
        x_norm = x_norm[:, :min_len]
        target_env = target_env[:, :min_len]
        env_a = env_a_full[:, :min_len]
        env_b = env_b_full[:, :min_len]
        
        target_env = normalize_array(target_env.T).T
        env_a = normalize_array(env_a.T).T
        env_b = normalize_array(env_b.T).T
        
        X.append(x_norm)
        Y.append(target_env)
        Y_A.append(env_a)
        Y_B.append(env_b)
        
    return X, Y, Y_A, Y_B

def evaluate_windows(pred, env_a, env_b, window_samples):
    num_correct = 0.0
    num_total = 0
    start = 0
    while start + window_samples <= pred.shape[1]:
        end = start + window_samples
        p = pred[0, start:end]
        ea = env_a[0, start:end]
        eb = env_b[0, start:end]
        
        std_p = np.std(p)
        if np.isnan(std_p) or std_p < 1e-12:
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

def evaluate_model(model, X, Y_A, Y_B, device, zero_eeg=False, shuffle_labels=False):
    model.eval()
    window_samples = DECISION_WINDOW_SEC * FS
    n_correct = 0.0
    n_total = 0
    
    np.random.seed(42)
    shuffle_indices = np.random.permutation(len(X))
    while np.any(shuffle_indices == np.arange(len(X))):
        shuffle_indices = np.random.permutation(len(X))
    
    with torch.no_grad():
        for i in range(len(X)):
            x_np = X[i]
            x = torch.FloatTensor(x_np).unsqueeze(0).to(device)
            
            if zero_eeg:
                x = torch.zeros_like(x)
            
            pred = model(x).squeeze(0).cpu().numpy() # [1, Time]
            
            if shuffle_labels:
                shuf_idx = shuffle_indices[i]
                mlen = min(pred.shape[1], Y_A[shuf_idx].shape[1])
                ea = Y_A[shuf_idx][:, :mlen]
                eb = Y_B[shuf_idx][:, :mlen]
                pred = pred[:, :mlen]
            else:
                ea = Y_A[i]
                eb = Y_B[i]
                
            nc, nt = evaluate_windows(pred, ea, eb, window_samples)
            n_correct += nc
            n_total += nt
            
    return n_correct, n_total

def train_eegnet_tcn_screening(channels, lowcut, highcut):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} | EEGNet+TCN | Channels: {channels} | Band: {lowcut}-{highcut} Hz\n")
    
    mapping = get_mapping_data()
    
    all_paths = subject_files()
    if not all_paths:
        print("No subjects found.")
        return
        
    paths = [p for p in all_paths if p.stem in SCREENING_SUBJECTS]
    subject_examples = {str(p): load_subject_examples(p) for p in paths}
    folds = list(iter_leave_one_subject_out(paths))
    
    all_accs_norm = []
    all_accs_zero = []
    all_accs_shuf = []
    
    for held_out_path, train_paths in folds:
        held_out_key = str(held_out_path)
        print(f"\nEvaluating fold with held-out subject: {held_out_path.stem}")
        
        train_exs = []
        for p in train_paths:
            train_exs.extend(subject_examples[str(p)])
            
        test_exs = subject_examples[held_out_key]
        
        np.random.seed(42)
        np.random.shuffle(train_exs)
        val_split = int(0.1 * len(train_exs))
        val_exs = train_exs[:val_split]
        train_exs = train_exs[val_split:]
        
        X_tr, Y_tr, YA_tr, YB_tr = [], [], [], []
        for p in train_paths:
            tX, tY, tYA, tYB = prepare_dataset(subject_examples[str(p)], channels, lowcut, highcut, p.stem, mapping)
            X_tr.extend(tX); Y_tr.extend(tY); YA_tr.extend(tYA); YB_tr.extend(tYB)
            
        X_va, Y_va, YA_va, YB_va = X_tr[:val_split], Y_tr[:val_split], YA_tr[:val_split], YB_tr[:val_split]
        X_tr, Y_tr, YA_tr, YB_tr = X_tr[val_split:], Y_tr[val_split:], YA_tr[val_split:], YB_tr[val_split:]
        
        X_te, Y_te, YA_te, YB_te = prepare_dataset(test_exs, channels, lowcut, highcut, held_out_path.stem, mapping)
        
        model = EEGNetTCN(in_channels=len(channels), num_classes=1).to(device)
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
                y = torch.FloatTensor(Y_tr[i]).unsqueeze(0).to(device) # shape: [1, 1, Time]
                
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
                
            print(f"  Epoch {epoch+1:02d}/100 | Train Loss: {train_loss:.4f} | Val Acc: {val_acc*100:.2f}% | Patience: {epochs_no_improve}/15")
                
            if epochs_no_improve >= patience:
                break
                
        model.load_state_dict(best_weights)
        
        nc_norm, nt_norm = evaluate_model(model, X_te, YA_te, YB_te, device, zero_eeg=False, shuffle_labels=False)
        acc_norm = nc_norm / max(nt_norm, 1)
        
        nc_zero, nt_zero = evaluate_model(model, X_te, YA_te, YB_te, device, zero_eeg=True, shuffle_labels=False)
        acc_zero = nc_zero / max(nt_zero, 1)
        
        nc_shuf, nt_shuf = evaluate_model(model, X_te, YA_te, YB_te, device, zero_eeg=False, shuffle_labels=True)
        acc_shuf = nc_shuf / max(nt_shuf, 1)
        
        print(f"  -> Accuracy Normal  : {acc_norm*100:.2f}%")
        print(f"  -> Accuracy Zero EEG: {acc_zero*100:.2f}%")
        print(f"  -> Accuracy Shuffled: {acc_shuf*100:.2f}%")
        
        all_accs_norm.append(acc_norm)
        all_accs_zero.append(acc_zero)
        all_accs_shuf.append(acc_shuf)
        
    final_acc_norm = np.mean(all_accs_norm)
    final_acc_zero = np.mean(all_accs_zero)
    final_acc_shuf = np.mean(all_accs_shuf)
    
    print("\n" + "="*50)
    print(f"[EEGNET+TCN SCREENING RESULTS]")
    print("="*50)
    print(f" Accuracy Normal  : {final_acc_norm*100:.2f}%")
    print(f" Accuracy Zero EEG: {final_acc_zero*100:.2f}%")
    print(f" Accuracy Shuffled: {final_acc_shuf*100:.2f}%")
    print("="*50)

def main():
    parser = argparse.ArgumentParser(description="Train EEGNet+TCN Screening")
    parser.add_argument("--channels", type=int, nargs='+', default=[13, 46, 43, 23, 50, 0, 52, 14],
                        help="List of channel indices to use (default: Top 8)")
    parser.add_argument("--lowcut", type=float, default=1.0, help="Lowcut frequency")
    parser.add_argument("--highcut", type=float, default=6.0, help="Highcut frequency")
    args = parser.parse_args()
    
    train_eegnet_tcn_screening(args.channels, args.lowcut, args.highcut)

if __name__ == "__main__":
    main()
