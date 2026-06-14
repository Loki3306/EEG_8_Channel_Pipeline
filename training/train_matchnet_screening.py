import argparse
import sys
import json
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from pathlib import Path
from copy import deepcopy
from scipy.signal import butter, filtfilt

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from models.matchnet import ContrastiveMatchNet, contrastive_loss
from baselines.ridge_aad import load_subject_examples, subject_files, iter_leave_one_subject_out

SCREENING_SUBJECTS = ["S3_data_preproc", "S13_data_preproc", "S15_data_preproc", "S16_data_preproc"]

FS = 64
DECISION_WINDOW_SEC = 10
TRAIN_WINDOW_SEC = 5
TRAIN_HOP_SEC = 2
NUM_BANDS = 28

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
    arr = arr - arr.mean(axis=0, keepdims=True)
    scale = arr.std() + 1e-12
    return arr / scale

def get_mapping_data():
    map_file = REPO_ROOT / "data" / "audio_mapping.json"
    env_file = REPO_ROOT / "data" / "gammatone_envelopes.pkl"
    with open(map_file, 'r') as f:
        mapping = json.load(f)
    with open(env_file, 'rb') as f:
        envelopes = pickle.load(f)
    return mapping, envelopes

def prepare_dataset(examples, channels, lowcut, highcut, subject_id, mapping, envelopes):
    X = []
    Y_A = []
    Y_B = []
    
    sub_key = subject_id.replace("_data_preproc", "")
    
    for i, ex in enumerate(examples):
        eeg = ex.eeg[:, channels].T
        eeg = butter_bandpass_filter(eeg, lowcut, highcut, FS, axis=1)
        x_norm = normalize_array(eeg.T).T 
        
        trial_key = f"trial_{i}"
        
        if sub_key in mapping and trial_key in mapping[sub_key]:
            fname_a = mapping[sub_key][trial_key]["wavA"]["filename"]
            fname_b = mapping[sub_key][trial_key]["wavB"]["filename"]
            env_a_full = envelopes[fname_a] 
            env_b_full = envelopes[fname_b] 
        else:
            print(f"Warning: Missing mapping for {sub_key} {trial_key}")
            continue
            
        min_len = min(x_norm.shape[1], env_a_full.shape[1])
        x_norm = x_norm[:, :min_len]
        env_a = env_a_full[:, :min_len]
        env_b = env_b_full[:, :min_len]
        
        env_a = normalize_array_global(env_a.T).T
        env_b = normalize_array_global(env_b.T).T
        
        X.append(x_norm)
        Y_A.append(env_a)
        Y_B.append(env_b)
        
    return X, Y_A, Y_B

def chunk_trial(x, ya, yb, window_sec, hop_sec):
    """Chunks a single trial into smaller overlapping windows for training."""
    win_samples = int(window_sec * FS)
    hop_samples = int(hop_sec * FS)
    
    chunks_x, chunks_ya, chunks_yb = [], [], []
    start = 0
    while start + win_samples <= x.shape[1]:
        end = start + win_samples
        chunks_x.append(x[:, start:end])
        chunks_ya.append(ya[:, start:end])
        chunks_yb.append(yb[:, start:end])
        start += hop_samples
        
    return chunks_x, chunks_ya, chunks_yb

def evaluate_model(model, X, Y_A, Y_B, device, zero_eeg=False, shuffle_labels=False):
    """
    Evaluates the model using 10-second non-overlapping windows.
    Decision rule: cosine_similarity(Z_eeg, Z_A) > cosine_similarity(Z_eeg, Z_B)
    """
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
            
            if shuffle_labels:
                shuf_idx = shuffle_indices[i]
                ya_np = Y_A[shuf_idx]
                yb_np = Y_B[shuf_idx]
            else:
                ya_np = Y_A[i]
                yb_np = Y_B[i]
            
            
            start = 0
            while start + window_samples <= x_np.shape[1]:
                end = start + window_samples
                x_chunk = torch.FloatTensor(x_np[:, start:end]).unsqueeze(0).to(device)
                
                if zero_eeg:
                    x_chunk = torch.zeros_like(x_chunk)
                    
                ya_chunk = torch.FloatTensor(ya_np[:, start:end]).unsqueeze(0).to(device)
                yb_chunk = torch.FloatTensor(yb_np[:, start:end]).unsqueeze(0).to(device)
                
                z_eeg, z_a, z_b = model(x_chunk, ya_chunk, yb_chunk)
                
                sim_a = F.cosine_similarity(z_eeg, z_a, dim=1).mean().item()
                sim_b = F.cosine_similarity(z_eeg, z_b, dim=1).mean().item()
                
                if sim_a > sim_b:
                    n_correct += 1.0
                elif sim_a == sim_b:
                    n_correct += 0.5
                    
                n_total += 1
                start += window_samples
                
    return n_correct, n_total

def train_matchnet_loso(eeg_model, channels, lowcut, highcut):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} | MatchNet ({eeg_model}) | Channels: {channels}")
    
    mapping, envelopes = get_mapping_data()
    
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
        
        # Prepare datasets
        X_tr_full, YA_tr_full, YB_tr_full = [], [], []
        for p in train_paths:
            if str(p) in [e.subject for e in val_exs]: continue # Skip validation mixing roughly
            tX, tYA, tYB = prepare_dataset(subject_examples[str(p)], channels, lowcut, highcut, p.stem, mapping, envelopes)
            X_tr_full.extend(tX); YA_tr_full.extend(tYA); YB_tr_full.extend(tYB)
            
        # Re-extract correctly without overlap
        X_tr_full, YA_tr_full, YB_tr_full = [], [], []
        X_va_full, YA_va_full, YB_va_full = [], [], []
        
        for p in train_paths:
            tX, tYA, tYB = prepare_dataset(subject_examples[str(p)], channels, lowcut, highcut, p.stem, mapping, envelopes)
            # 90/10 split at trial level
            v_split_idx = int(0.1 * len(tX))
            X_va_full.extend(tX[:v_split_idx])
            YA_va_full.extend(tYA[:v_split_idx])
            YB_va_full.extend(tYB[:v_split_idx])
            X_tr_full.extend(tX[v_split_idx:])
            YA_tr_full.extend(tYA[v_split_idx:])
            YB_tr_full.extend(tYB[v_split_idx:])

        X_te_full, YA_te_full, YB_te_full = prepare_dataset(test_exs, channels, lowcut, highcut, held_out_path.stem, mapping, envelopes)
        
        # Chunk training data
        X_tr, YA_tr, YB_tr = [], [], []
        for i in range(len(X_tr_full)):
            cx, cya, cyb = chunk_trial(X_tr_full[i], YA_tr_full[i], YB_tr_full[i], TRAIN_WINDOW_SEC, TRAIN_HOP_SEC)
            X_tr.extend(cx); YA_tr.extend(cya); YB_tr.extend(cyb)
            
        # Model
        model = ContrastiveMatchNet(eeg_model_type=eeg_model, eeg_channels=len(channels), audio_channels=NUM_BANDS, latent_dim=64).to(device)
        optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        
        best_val_acc = 0.0
        best_weights = deepcopy(model.state_dict())
        patience = 10
        epochs_no_improve = 0
        batch_size = 16
        
        print(f"Training on {len(X_tr)} chunks ({TRAIN_WINDOW_SEC}s)...")
        
        for epoch in range(100):
            model.train()
            train_loss = 0.0
            
            # Shuffle chunks
            perm = np.random.permutation(len(X_tr))
            
            for i in range(0, len(X_tr), batch_size):
                batch_idx = perm[i:i+batch_size]
                bx = torch.FloatTensor(np.stack([X_tr[j] for j in batch_idx])).to(device)
                bya = torch.FloatTensor(np.stack([YA_tr[j] for j in batch_idx])).to(device)
                byb = torch.FloatTensor(np.stack([YB_tr[j] for j in batch_idx])).to(device)
                
                optimizer.zero_grad()
                z_eeg, z_a, z_b = model(bx, bya, byb)
                loss, sa, sb = contrastive_loss(z_eeg, z_a, z_b, margin=0.1)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                
            nc_va, nt_va = evaluate_model(model, X_va_full, YA_va_full, YB_va_full, device)
            val_acc = nc_va / max(nt_va, 1)
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_weights = deepcopy(model.state_dict())
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                
            print(f"  Epoch {epoch+1:02d}/100 | Train Loss: {train_loss:.4f} | Val Acc: {val_acc*100:.2f}% | Patience: {epochs_no_improve}/10")
                
            if epochs_no_improve >= patience:
                break
                
        model.load_state_dict(best_weights)
        
        nc_norm, nt_norm = evaluate_model(model, X_te_full, YA_te_full, YB_te_full, device, zero_eeg=False, shuffle_labels=False)
        acc_norm = nc_norm / max(nt_norm, 1)
        
        nc_zero, nt_zero = evaluate_model(model, X_te_full, YA_te_full, YB_te_full, device, zero_eeg=True, shuffle_labels=False)
        acc_zero = nc_zero / max(nt_zero, 1)
        
        nc_shuf, nt_shuf = evaluate_model(model, X_te_full, YA_te_full, YB_te_full, device, zero_eeg=False, shuffle_labels=True)
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
    print(f"[MATCHNET ({eeg_model.upper()}) LOSO SCREENING RESULTS]")
    print("="*50)
    print(f" Accuracy Normal  : {final_acc_norm*100:.2f}%")
    print(f" Accuracy Zero EEG: {final_acc_zero*100:.2f}%")
    print(f" Accuracy Shuffled: {final_acc_shuf*100:.2f}%")
    print("="*50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Contrastive MatchNet")
    parser.add_argument("--model", type=str, default="eegnet", choices=["eegnet", "atcnet", "eegnet_tcn"], help="Base EEG encoder")
    parser.add_argument("--channels", type=int, nargs='+', default=[13, 46, 43, 23, 50, 0, 52, 14])
    parser.add_argument("--lowcut", type=float, default=1.0)
    parser.add_argument("--highcut", type=float, default=6.0)
    args = parser.parse_args()
    
    train_matchnet_loso(args.model, args.channels, args.lowcut, args.highcut)
