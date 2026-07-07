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
COGNITIVE_GAP_SEC = 1.5
GAP_SAMPLES = int(COGNITIVE_GAP_SEC * SR)
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
    
    for start in range(0, T - WIN_SAMPLES + 1, WIN_SAMPLES):
        end = start + WIN_SAMPLES
        
        # Check overlap with ANY cognitive gap
        overlap = False
        for spk, sw_idx in switch_points:
            gap_start = sw_idx
            gap_end = sw_idx + GAP_SAMPLES
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
        
    print("Loading S1 Cache...")
    cached = torch.load(cache_path, map_location='cpu', weights_only=False)
    trials = cached['raw']
    
    print(f"Extracting Match/Mismatch Pairs (Excluding {COGNITIVE_GAP_SEC}s Gaps)...")
    trial_eeg = []
    trial_aud = []
    trial_y = []
    
    for tr in trials:
        e, a, y = extract_match_mismatch_windows(tr)
        trial_eeg.append(e)
        trial_aud.append(a)
        trial_y.append(y)
        
    kf = KFold(n_splits=5, shuffle=False)
    fold_aurocs = []
    
    print("\n=======================================================")
    print(" PHASE 50: TWO-STREAM MATCH-MISMATCH AAD ON S1")
    print("=======================================================\n")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on Device: {device}\n")
    
    for fold, (train_idx, test_idx) in enumerate(kf.split(trials)):
        print(f"--- FOLD {fold+1}/5 ---")
        
        X_train_e = np.concatenate([trial_eeg[i] for i in train_idx if trial_eeg[i] is not None], axis=0)
        X_train_a = np.concatenate([trial_aud[i] for i in train_idx if trial_aud[i] is not None], axis=0)
        y_train = np.concatenate([trial_y[i] for i in train_idx if trial_y[i] is not None], axis=0)
        
        X_test_e = np.concatenate([trial_eeg[i] for i in test_idx if trial_eeg[i] is not None], axis=0)
        X_test_a = np.concatenate([trial_aud[i] for i in test_idx if trial_aud[i] is not None], axis=0)
        y_test = np.concatenate([trial_y[i] for i in test_idx if trial_y[i] is not None], axis=0)
        
        train_ds = TensorDataset(torch.from_numpy(X_train_e).float(), torch.from_numpy(X_train_a).float(), torch.from_numpy(y_train).float())
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
        
        test_ds = TensorDataset(torch.from_numpy(X_test_e).float(), torch.from_numpy(X_test_a).float(), torch.from_numpy(y_test).float())
        test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
        
        model = MatchMismatchNet(in_channels=60, samples=WIN_SAMPLES).to(device)
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
    print(f" AVERAGE 5-FOLD AUROC: {avg_auc:.4f}")
    print("=======================================================\n")

if __name__ == "__main__":
    main()
