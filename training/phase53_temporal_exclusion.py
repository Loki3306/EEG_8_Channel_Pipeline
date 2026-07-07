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

# Windows starting before this many seconds after a switch are "Transition"
TRANSITION_SEC = 3.5 
TRANSITION_SAMPLES = int(TRANSITION_SEC * SR)

BATCH_SIZE = 32
EPOCHS = 30
LR = 1e-3

def get_attended_speaker_and_time_since_switch(start_idx, switch_points):
    current_spk = 'L'
    last_sw_idx = 0
    for spk, idx in switch_points:
        if idx <= start_idx:
            current_spk = spk
            last_sw_idx = idx
        else:
            break
    return current_spk, start_idx - last_sw_idx

def extract_condition_windows(trial):
    eeg = trial['eeg'].numpy()
    env_l = trial['env_l'].numpy()
    env_r = trial['env_r'].numpy()
    switch_points = trial['meta']['switch_points']
    
    # Normalize
    eeg = (eeg - np.mean(eeg, axis=1, keepdims=True)) / (np.std(eeg, axis=1, keepdims=True) + 1e-8)
    env_l = (env_l - np.mean(env_l)) / (np.std(env_l) + 1e-8)
    env_r = (env_r - np.mean(env_r)) / (np.std(env_r) + 1e-8)
    
    T = eeg.shape[1]
    
    stable_eeg, stable_audio, stable_labels = [], [], []
    trans_eeg, trans_audio, trans_labels = [], [], []
    
    for start in range(0, T - WIN_SAMPLES + 1, WIN_SAMPLES):
        end = start + WIN_SAMPLES
        
        # Exclude immediate cognitive gap (1.5s)
        overlap = False
        for spk, sw_idx in switch_points:
            gap_start = sw_idx
            gap_end = sw_idx + GAP_SAMPLES
            if max(start, gap_start) < min(end, gap_end):
                overlap = True
                break
                
        if overlap:
            continue
            
        att_spk, samples_since_sw = get_attended_speaker_and_time_since_switch(start, switch_points)
        
        eeg_win = eeg[:, start:end]
        att_audio = env_l[start:end] if att_spk == 'L' else env_r[start:end]
        unatt_audio = env_r[start:end] if att_spk == 'L' else env_l[start:end]
        
        is_stable = samples_since_sw >= TRANSITION_SAMPLES
        
        if is_stable:
            stable_eeg.extend([eeg_win, eeg_win])
            stable_audio.extend([att_audio, unatt_audio])
            stable_labels.extend([1.0, 0.0])
        else:
            trans_eeg.extend([eeg_win, eeg_win])
            trans_audio.extend([att_audio, unatt_audio])
            trans_labels.extend([1.0, 0.0])
            
    res = {}
    if len(stable_eeg) > 0:
        res['stable'] = (np.stack(stable_eeg), np.stack(stable_audio), np.array(stable_labels))
    else:
        res['stable'] = (None, None, None)
        
    if len(trans_eeg) > 0:
        res['trans'] = (np.stack(trans_eeg), np.stack(trans_audio), np.array(trans_labels))
    else:
        res['trans'] = (None, None, None)
        
    return res

def run_evaluation(trials, condition_name):
    print(f"\n=======================================================")
    print(f" EVALUATING CONDITION: {condition_name.upper()}")
    print(f"=======================================================")
    
    trial_eeg = []
    trial_audio = []
    trial_labels = []
    
    for tr in trials:
        res = extract_condition_windows(tr)
        X_e, X_a, Y = res[condition_name]
        trial_eeg.append(X_e)
        trial_audio.append(X_a)
        trial_labels.append(Y)
        
    kf = KFold(n_splits=5, shuffle=False)
    fold_aurocs = []
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    for fold, (train_idx, test_idx) in enumerate(kf.split(trials)):
        train_e = [trial_eeg[i] for i in train_idx if trial_eeg[i] is not None]
        if not train_e: continue
        X_train_e = np.concatenate(train_e, axis=0)
        X_train_a = np.concatenate([trial_audio[i] for i in train_idx if trial_audio[i] is not None], axis=0)
        Y_train = np.concatenate([trial_labels[i] for i in train_idx if trial_labels[i] is not None], axis=0)
        
        test_e = [trial_eeg[i] for i in test_idx if trial_eeg[i] is not None]
        if not test_e: continue
        X_test_e = np.concatenate(test_e, axis=0)
        X_test_a = np.concatenate([trial_audio[i] for i in test_idx if trial_audio[i] is not None], axis=0)
        Y_test = np.concatenate([trial_labels[i] for i in test_idx if trial_labels[i] is not None], axis=0)
        
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
        
    avg_auc = np.mean(fold_aurocs) if fold_aurocs else 0.50
    print(f"\n  -> {condition_name.upper()} Average AUROC: {avg_auc:.4f}\n")
    return avg_auc

def main():
    cache_path = Path('/kaggle/working/eeg_cache/S1_processed.pt')
    if not cache_path.exists():
        print("Cache not found. Please run generate_aasd_cache.py first.")
        return
        
    print("\n=======================================================")
    print(" PHASE 53: TEMPORAL EXCLUSION (LABEL UNCERTAINTY)")
    print("=======================================================\n")
    
    print("Loading S1 Cache...")
    cached = torch.load(cache_path, map_location='cpu', weights_only=False)
    trials = cached['raw']
    
    print(f"Splitting Windows into:")
    print(f"  TRANSITION: 1.5s - {TRANSITION_SEC}s post-switch")
    print(f"  STABLE: > {TRANSITION_SEC}s post-switch")
    
    auc_trans = run_evaluation(trials, 'trans')
    auc_stable = run_evaluation(trials, 'stable')
    
    print("=======================================================")
    print(" PHASE 53 FINAL COMPARISON")
    print("=======================================================")
    print(f"  TRANSITION WINDOWS (<3.5s): {auc_trans:.4f} AUROC")
    print(f"  STABLE WINDOWS (>3.5s)    : {auc_stable:.4f} AUROC")
    print("=======================================================\n")

if __name__ == "__main__":
    main()
