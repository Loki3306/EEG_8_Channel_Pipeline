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

FS = 64
DECISION_WINDOW_SEC = 10
BP_LOWCUT = 1.0
BP_HIGHCUT = 8.0
# Use the data-driven discovered 8-channel set: C5, FCz, FC6, P9, C6, Fp1, TP8, T7
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
        # pred, target: [Batch, 1, Time]
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
        
        # Bandpass filter
        eeg = butter_bandpass_filter(eeg, lowcut, highcut, FS, axis=1)
        wav_a = butter_bandpass_filter(ex.wav_a.reshape(-1, 1), lowcut, highcut, FS, axis=0).ravel()
        wav_b = butter_bandpass_filter(ex.wav_b.reshape(-1, 1), lowcut, highcut, FS, axis=0).ravel()
        
        # Trial-level normalization
        x_norm = normalize_array(eeg.T).T  # scale over time
        env_a = normalize_array(wav_a.reshape(-1, 1)).ravel()
        env_b = normalize_array(wav_b.reshape(-1, 1)).ravel()
        
        # Target is ALWAYS env_a
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

def evaluate_model(model, X, Y_A, Y_B, device, zero_eeg=False, shuffle_eeg=False, random_eeg=False):
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
            if shuffle_eeg:
                shuf_idx = shuffle_indices[i]
                x_np = X[shuf_idx]
                mlen = min(x_np.shape[1], len(Y_A[i]))
                x_np = x_np[:, :mlen]
            else:
                x_np = X[i]
                
            x = torch.FloatTensor(x_np).unsqueeze(0).to(device)
            
            if zero_eeg:
                x = torch.zeros_like(x)
            elif random_eeg:
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

def train_eegnet_loso(lowcut, highcut):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} | Running EEGNet LOSO | Band: {lowcut}-{highcut} Hz")
    
    paths = subject_files()
    if not paths:
        print("No subjects found.")
        return
        
    subject_examples = {str(p): load_subject_examples(p) for p in paths}
    folds = list(iter_leave_one_subject_out(paths))
    
    all_normal_accs = []
    all_random_accs = []
    all_shuffle_accs = []
    
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
        
        X_tr, Y_tr, YA_tr, YB_tr = prepare_dataset(train_exs, lowcut, highcut)
        X_va, Y_va, YA_va, YB_va = prepare_dataset(val_exs, lowcut, highcut)
        X_te, Y_te, YA_te, YB_te = prepare_dataset(test_exs, lowcut, highcut)
        
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
                
            print(f"  Epoch {epoch+1:02d}/100 | Train Loss: {train_loss:.4f} | Val Acc: {val_acc*100:.2f}% | Patience: {epochs_no_improve}/15")
            
                
            if epochs_no_improve >= patience:
                break
                
        model.load_state_dict(best_weights)
        
        # Test Evaluation
        nc_norm, nt_norm = evaluate_model(model, X_te, YA_te, YB_te, device, zero_eeg=False)
        nc_rand, nt_rand = evaluate_model(model, X_te, YA_te, YB_te, device, random_eeg=True)
        nc_shuf, nt_shuf = evaluate_model(model, X_te, YA_te, YB_te, device, shuffle_eeg=True)
        
        normal_acc = nc_norm / max(nt_norm, 1)
        rand_acc = nc_rand / max(nt_rand, 1)
        shuffle_acc = nc_shuf / max(nt_shuf, 1)
        
        print(f"  -> Accuracy Normal : {normal_acc*100:.2f}%")
        print(f"  -> Accuracy Random : {rand_acc*100:.2f}%")
        print(f"  -> Accuracy Shuffle: {shuffle_acc*100:.2f}%")
        
        all_normal_accs.append(normal_acc)
        all_random_accs.append(rand_acc)
        all_shuffle_accs.append(shuffle_acc)
        
    final_normal = np.mean(all_normal_accs)
    final_random = np.mean(all_random_accs)
    final_shuffle = np.mean(all_shuffle_accs)
    
    print("\n" + "="*50)
    print("EEGNET 8-CHANNEL LOSO RESULTS")
    print("="*50)
    print(f" Normal EEG  : {final_normal*100:.2f}%")
    print(f" Random EEG  : {final_random*100:.2f}%")
    print(f" Shuffle EEG : {final_shuffle*100:.2f}%")
    print("="*50)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--lowcut", type=float, default=1.0, help="Lowcut frequency for bandpass filter")
    parser.add_argument("--highcut", type=float, default=8.0, help="Highcut frequency for bandpass filter")
    args = parser.parse_args()
    
    train_eegnet_loso(args.lowcut, args.highcut)
