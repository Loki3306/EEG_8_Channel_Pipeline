import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
import numpy as np
from pathlib import Path
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.match_mismatch_net import MatchMismatchNet

# -------------------------------------------------------------------------
# CONSTANTS
# -------------------------------------------------------------------------
SR = 128
WIN_SEC = 2.0
WIN_SAMPLES = int(WIN_SEC * SR)

# The user's specific hypothesis: exclude +/- 1.0s around every switch
EXCLUSION_SEC = 1.0
EXCLUSION_SAMPLES = int(EXCLUSION_SEC * SR)

BATCH_SIZE = 32
EPOCHS = 30
LR = 1e-3

def get_attended_speaker_at_time(start_idx, switch_points):
    current_spk = 'L'
    for spk, idx in switch_points:
        if idx <= start_idx:
            current_spk = spk
        else:
            break
    return current_spk

def extract_match_mismatch_windows(trial):
    eeg = trial['eeg'].numpy()
    env_l = trial['env_l'].numpy()
    env_r = trial['env_r'].numpy()
    switch_points = trial['meta']['switch_points']
    
    # Normalize EEG and Envelopes
    eeg = (eeg - np.mean(eeg, axis=1, keepdims=True)) / (np.std(eeg, axis=1, keepdims=True) + 1e-8)
    env_l = (env_l - np.mean(env_l)) / (np.std(env_l) + 1e-8)
    env_r = (env_r - np.mean(env_r)) / (np.std(env_r) + 1e-8)
    
    T = eeg.shape[1]
    
    X_eeg = []
    X_audio = []
    Y_labels = []
    
    for start in range(0, T - WIN_SAMPLES + 1, int(0.5 * SR)): # 0.5s stride to maximize usable windows
        end = start + WIN_SAMPLES
        
        # Check overlap with ANY switch's symmetric exclusion zone
        overlap = False
        for spk, sw_idx in switch_points:
            gap_start = sw_idx - EXCLUSION_SAMPLES
            gap_end = sw_idx + EXCLUSION_SAMPLES
            
            if max(start, gap_start) < min(end, gap_end):
                overlap = True
                break
                
        if overlap:
            continue
            
        att_spk = get_attended_speaker_at_time(start, switch_points)
        
        eeg_win = eeg[:, start:end]
        att_audio = env_l[start:end] if att_spk == 'L' else env_r[start:end]
        unatt_audio = env_r[start:end] if att_spk == 'L' else env_l[start:end]
        
        # Positive Pair (Match)
        X_eeg.append(eeg_win)
        X_audio.append(att_audio)
        Y_labels.append(1.0)
        
        # Negative Pair (Mismatch)
        X_eeg.append(eeg_win)
        X_audio.append(unatt_audio)
        Y_labels.append(0.0)
        
    if len(X_eeg) == 0:
        return None, None, None
        
    return np.stack(X_eeg), np.stack(X_audio), np.array(Y_labels)

def main():
    cache_path = Path('/kaggle/working/eeg_cache/S1_processed.pt')
    if not cache_path.exists():
        print("Cache not found. Please run generate_aasd_cache.py first.")
        return
        
    print("\n=======================================================")
    print(" PHASE 54: TRUE AAD SYMMETRIC EXCLUSION (±1.0s)")
    print("=======================================================\n")
    
    print("Loading S1 Cache...")
    cached = torch.load(cache_path, map_location='cpu', weights_only=False)
    trials = cached['raw']
    
    print(f"Extracting strictly bounded {WIN_SEC}s windows...")
    trial_eeg = []
    trial_audio = []
    trial_labels = []
    
    total_windows = 0
    for tr in trials:
        X_e, X_a, Y = extract_match_mismatch_windows(tr)
        if X_e is not None:
            trial_eeg.append(X_e)
            trial_audio.append(X_a)
            trial_labels.append(Y)
            total_windows += len(Y)
            
    print(f"Extracted {total_windows} completely stable True AAD pairs.")
    if total_windows == 0:
        print("ERROR: Window length + Exclusion Zone left no stable data!")
        return
            
    kf = KFold(n_splits=5, shuffle=False)
    fold_aurocs = []
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on Device: {device}\n")
    
    for fold, (train_idx, test_idx) in enumerate(kf.split(trial_eeg)):
        X_train_e = np.concatenate([trial_eeg[i] for i in train_idx], axis=0)
        X_train_a = np.concatenate([trial_audio[i] for i in train_idx], axis=0)
        Y_train = np.concatenate([trial_labels[i] for i in train_idx], axis=0)
        
        X_test_e = np.concatenate([trial_eeg[i] for i in test_idx], axis=0)
        X_test_a = np.concatenate([trial_audio[i] for i in test_idx], axis=0)
        Y_test = np.concatenate([trial_labels[i] for i in test_idx], axis=0)
        
        train_ds = TensorDataset(torch.from_numpy(X_train_e).float(), torch.from_numpy(X_train_a).float(), torch.from_numpy(Y_train).float())
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
        
        test_ds = TensorDataset(torch.from_numpy(X_test_e).float(), torch.from_numpy(X_test_a).float(), torch.from_numpy(Y_test).float())
        test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
        
        model = MatchMismatchNet(samples=WIN_SAMPLES).to(device)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
        
        best_val_auc = 0.50
        
        for epoch in range(EPOCHS):
            model.train()
            for b_e, b_a, b_y in train_loader:
                b_e, b_a, b_y = b_e.to(device), b_a.to(device), b_y.to(device)
                optimizer.zero_grad()
                logits = model(b_e, b_a)
                loss = criterion(logits, b_y)
                loss.backward()
                optimizer.step()
                
            model.eval()
            all_preds = []
            all_labels = []
            with torch.no_grad():
                for b_e, b_a, b_y in test_loader:
                    b_e, b_a = b_e.to(device), b_a.to(device)
                    logits = model(b_e, b_a)
                    probs = torch.sigmoid(logits).cpu().numpy()
                    all_preds.extend(probs)
                    all_labels.extend(b_y.numpy())
                    
            try:
                val_auc = roc_auc_score(all_labels, all_preds)
            except ValueError:
                val_auc = 0.50
                
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                
        print(f"  Fold {fold+1} Best AUROC: {best_val_auc:.4f}")
        fold_aurocs.append(best_val_auc)
        
    avg_auc = np.mean(fold_aurocs)
    print("\n=======================================================")
    print(f" AVERAGE 5-FOLD AUROC (STABLE TRUE AAD): {avg_auc:.4f}")
    print("=======================================================\n")

if __name__ == "__main__":
    main()
