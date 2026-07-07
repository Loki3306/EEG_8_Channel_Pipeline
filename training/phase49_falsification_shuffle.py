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

# Add parent to path for relative imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.eegnet_classifier import EEGNetClassifier

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

def extract_windows_for_trial(trial):
    eeg = trial['eeg'].numpy() # (60, Time)
    switch_points = trial['meta']['switch_points']
    
    # Z-Score the EEG per trial
    eeg = (eeg - np.mean(eeg, axis=1, keepdims=True)) / (np.std(eeg, axis=1, keepdims=True) + 1e-8)
    
    T = eeg.shape[1]
    
    X_list = []
    y_list = []
    
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
        label = 1.0 if att_spk == 'L' else 0.0
        
        X_list.append(eeg[:, start:end])
        y_list.append(label)
        
    if len(X_list) == 0:
        return None, None
        
    return np.stack(X_list), np.array(y_list)

def main():
    cache_path = Path('/kaggle/working/eeg_cache/S1_processed.pt')
    if not cache_path.exists():
        print("Cache not found. Please run generate_aasd_cache.py first.")
        return
        
    print("Loading S1 Cache...")
    cached = torch.load(cache_path, map_location='cpu', weights_only=False)
    trials = cached['raw']
    
    print(f"Slicing into {WIN_SEC}s Windows (Excluding {COGNITIVE_GAP_SEC}s Gaps)...")
    trial_Xs = []
    trial_ys = []
    
    for tr in trials:
        X, y = extract_windows_for_trial(tr)
        trial_Xs.append(X)
        trial_ys.append(y)
        
    # =====================================================================
    # FALSIFICATION TEST: LABEL SHUFFLING (PERMUTATION TEST)
    # =====================================================================
    print("\n[!!!] EXECUTING FALSIFICATION TEST: SHUFFLING ALL LABELS [!!!]")
    for i in range(len(trial_ys)):
        if trial_ys[i] is not None:
            # We randomly shuffle the 1s and 0s for this trial.
            # This completely destroys the relationship between the EEG and Audio.
            np.random.shuffle(trial_ys[i])
    print("[!!!] LABELS SHUFFLED. EXPECTED AUROC MUST COLLAPSE TO 0.50 [!!!]\n")
        
    kf = KFold(n_splits=5, shuffle=False)
    fold_aurocs = []
    
    print("=======================================================")
    print(" PHASE 49 FALSIFICATION: EEGNET WITH SHUFFLED LABELS")
    print("=======================================================\n")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on Device: {device}\n")
    
    for fold, (train_idx, test_idx) in enumerate(kf.split(trials)):
        print(f"--- FOLD {fold+1}/5 ---")
        
        X_train = np.concatenate([trial_Xs[i] for i in train_idx if trial_Xs[i] is not None], axis=0)
        y_train = np.concatenate([trial_ys[i] for i in train_idx if trial_ys[i] is not None], axis=0)
        
        X_test = np.concatenate([trial_Xs[i] for i in test_idx if trial_Xs[i] is not None], axis=0)
        y_test = np.concatenate([trial_ys[i] for i in test_idx if trial_ys[i] is not None], axis=0)
        
        train_ds = TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).float())
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
        
        test_ds = TensorDataset(torch.from_numpy(X_test).float(), torch.from_numpy(y_test).float())
        test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
        
        model = EEGNetClassifier(in_channels=60, samples=WIN_SAMPLES).to(device)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
        
        # Test Epoch 0 Baseline
        model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x = batch_x.to(device)
                logits = model(batch_x)
                probs = torch.sigmoid(logits).cpu().numpy()
                all_preds.extend(probs)
                all_labels.extend(batch_y.numpy())
        try:
            epoch_0_auc = roc_auc_score(all_labels, all_preds)
        except ValueError:
            epoch_0_auc = 0.50
        print(f"  Epoch 0 (Untrained) AUROC: {epoch_0_auc:.4f}")
        
        best_val_auc = epoch_0_auc
        
        for epoch in range(EPOCHS):
            model.train()
            train_loss = 0.0
            
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                
                optimizer.zero_grad()
                logits = model(batch_x)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()
                
            model.eval()
            all_preds = []
            all_labels = []
            with torch.no_grad():
                for batch_x, batch_y in test_loader:
                    batch_x = batch_x.to(device)
                    logits = model(batch_x)
                    probs = torch.sigmoid(logits).cpu().numpy()
                    all_preds.extend(probs)
                    all_labels.extend(batch_y.numpy())
                    
            try:
                val_auc = roc_auc_score(all_labels, all_preds)
            except ValueError:
                val_auc = 0.50
                
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                
        print(f"  Fold {fold+1} Best Trained AUROC: {best_val_auc:.4f}")
        fold_aurocs.append(best_val_auc)
        
    avg_auc = np.mean(fold_aurocs)
    print("\n=======================================================")
    print(f" AVERAGE 5-FOLD AUROC (SHUFFLED LABELS): {avg_auc:.4f}")
    if 0.45 <= avg_auc <= 0.55:
        print(" SUCCESS! MODEL FAILED TO LEARN (AS EXPECTED). NO CHEATING DETECTED.")
    else:
        print(" WARNING! MODEL ACHIEVED HIGH AUROC DESPITE RANDOM LABELS. DATA LEAKAGE DETECTED!")
    print("=======================================================\n")

if __name__ == "__main__":
    main()
