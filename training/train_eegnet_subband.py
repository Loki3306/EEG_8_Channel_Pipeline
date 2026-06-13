import argparse
import sys
import json
import pickle
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
NUM_BANDS = 8

class SubbandEEGNet(nn.Module):
    def __init__(self, in_channels=8, F1=8, D=2, F2=16, kernel_length=64, num_bands=8):
        super().__init__()
        # Use standard EEGNet, but change output projection
        self.base = EEGNet(in_channels=in_channels, F1=F1, D=D, F2=F2, kernel_length=kernel_length)
        # Override output projection
        self.base.output_proj = nn.Conv1d(F2, num_bands, kernel_size=1)

    def forward(self, x):
        return self.base(x)

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

def normalize_array_global(arr):
    # arr shape: (Time, Channels)
    # Zero-mean each band independently
    arr = arr - arr.mean(axis=0, keepdims=True)
    # Divide by global standard deviation to preserve relative variance
    scale = arr.std() + 1e-12
    return arr / scale

class PearsonMSELoss(nn.Module):
    def __init__(self, alpha=0.1):
        super().__init__()
        self.alpha = alpha
        self.mse = nn.MSELoss()

    def forward(self, pred, target):
        # Flatten over channels and time to calculate a single global Pearson correlation.
        # This naturally weights bands by their true variance.
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

def get_mapping_data():
    map_file = REPO_ROOT / "data" / "audio_mapping.json"
    env_file = REPO_ROOT / "data" / "subband_envelopes.pkl"
    with open(map_file, 'r') as f:
        mapping = json.load(f)
    with open(env_file, 'rb') as f:
        envelopes = pickle.load(f)
    return mapping, envelopes

def prepare_dataset(examples, channels, lowcut, highcut, subject_id, mapping, envelopes):
    X = []
    Y = []
    Y_A = []
    Y_B = []
    
    # We strip "_data_preproc" from subject_id for lookup
    sub_key = subject_id.replace("_data_preproc", "")
    
    for i, ex in enumerate(examples):
        eeg = ex.eeg[:, channels].T
        # Bandpass filter EEG
        eeg = butter_bandpass_filter(eeg, lowcut, highcut, FS, axis=1)
        x_norm = normalize_array(eeg.T).T  # scale over time
        
        trial_key = f"trial_{i}"
        
        if sub_key in mapping and trial_key in mapping[sub_key]:
            fname_a = mapping[sub_key][trial_key]["wavA"]["filename"]
            fname_b = mapping[sub_key][trial_key]["wavB"]["filename"]
            
            env_a_full = envelopes[fname_a] # shape: (8, Time)
            env_b_full = envelopes[fname_b] # shape: (8, Time)
        else:
            print(f"Warning: Missing mapping for {sub_key} {trial_key}")
            continue
            
        # Target is ALWAYS env_a
        target_env = env_a_full
        
        min_len = min(x_norm.shape[1], target_env.shape[1])
        x_norm = x_norm[:, :min_len]
        target_env = target_env[:, :min_len]
        env_a = env_a_full[:, :min_len]
        env_b = env_b_full[:, :min_len]
        
        # Normalize using global variance so we don't artificially boost high-frequency noise bands
        target_env = normalize_array_global(target_env.T).T
        env_a = normalize_array_global(env_a.T).T
        env_b = normalize_array_global(env_b.T).T
        
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
        # pred is [8, Time]
        p = pred[:, start:end]
        ea = env_a[:, start:end]
        eb = env_b[:, start:end]
        
        std_p = np.std(p)
        if np.isnan(std_p) or std_p < 1e-12:
            ca = 0.0
            cb = 0.0
        else:
            # Flatten to calculate correlation over all 8 bands jointly
            ca = np.corrcoef(p.ravel(), ea.ravel())[0, 1]
            cb = np.corrcoef(p.ravel(), eb.ravel())[0, 1]
            
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
                mlen = min(x_np.shape[1], Y_A[i].shape[1])
                x_np = x_np[:, :mlen]
            else:
                x_np = X[i]
                
            x = torch.FloatTensor(x_np).unsqueeze(0).to(device)
            
            if zero_eeg:
                x = torch.zeros_like(x)
            elif random_eeg:
                x = torch.randn_like(x)
            
            pred = model(x).squeeze(0).cpu().numpy() # [8, Time]
            
            if shuffle_eeg:
                ea = Y_A[i][:, :mlen]
                eb = Y_B[i][:, :mlen]
                pred = pred[:, :mlen]
            else:
                ea = Y_A[i]
                eb = Y_B[i]
                
            nc, nt = evaluate_windows(pred, ea, eb, window_samples)
            n_correct += nc
            n_total += nt
            
    return n_correct, n_total

def train_eegnet_subband_screening(channels, lowcut, highcut):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} | Subband EEGNet | Channels: {len(channels)} | Band: {lowcut}-{highcut} Hz")
    
    mapping, envelopes = get_mapping_data()
    
    all_paths = subject_files()
    if not all_paths:
        print("No subjects found.")
        return
        
    paths = [p for p in all_paths if p.stem in SCREENING_SUBJECTS]
    subject_examples = {str(p): load_subject_examples(p) for p in paths}
    folds = list(iter_leave_one_subject_out(paths))
    
    all_normal_accs = []
    
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
            tX, tY, tYA, tYB = prepare_dataset(subject_examples[str(p)], channels, lowcut, highcut, p.stem, mapping, envelopes)
            X_tr.extend(tX); Y_tr.extend(tY); YA_tr.extend(tYA); YB_tr.extend(tYB)
            
        # Resplit correctly
        X_va, Y_va, YA_va, YB_va = X_tr[:val_split], Y_tr[:val_split], YA_tr[:val_split], YB_tr[:val_split]
        X_tr, Y_tr, YA_tr, YB_tr = X_tr[val_split:], Y_tr[val_split:], YA_tr[val_split:], YB_tr[val_split:]
        
        X_te, Y_te, YA_te, YB_te = prepare_dataset(test_exs, channels, lowcut, highcut, held_out_path.stem, mapping, envelopes)
        
        model = SubbandEEGNet(in_channels=len(channels)).to(device)
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
                y = torch.FloatTensor(Y_tr[i]).unsqueeze(0).to(device) # shape: [1, 8, Time]
                
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
        normal_acc = nc_norm / max(nt_norm, 1)
        
        print(f"  -> Accuracy Normal : {normal_acc*100:.2f}%")
        all_normal_accs.append(normal_acc)
        
    final_normal = np.mean(all_normal_accs)
    
    print("\n" + "="*50)
    print(f"EEGNET SUBBAND {len(channels)}-CHANNEL SCREENING RESULTS")
    print("="*50)
    print(f" Normal EEG  : {final_normal*100:.2f}%")
    print("="*50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lowcut", type=float, default=1.0, help="Lowcut frequency for bandpass filter")
    parser.add_argument("--highcut", type=float, default=6.0, help="Highcut frequency for bandpass filter")
    parser.add_argument("--channels", type=int, nargs='+', default=[13, 46, 43, 23, 50, 0, 52, 14],
                        help="List of channel indices to use (default: Top 8)")
    args = parser.parse_args()
    
    train_eegnet_subband_screening(args.channels, args.lowcut, args.highcut)
