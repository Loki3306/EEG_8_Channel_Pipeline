import os
import sys
import numpy as np
import scipy.io
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, accuracy_score
import time
from pathlib import Path
import random

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.aad_eegnet import AAD_EEGNet

# We will train on all 60 channels to verify if AASD contains the signal.
# Removing PHYSICAL_8_CHANNELS slicing.
MAX_LAG = 24
WINDOW_LEN = 64 * 5
HOP_LEN = 64 * 1
EPOCHS = 15
BATCH_SIZE = 16
LR = 1e-3
WEIGHT_DECAY = 1e-2

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def safe_pearson(x, y):
    if np.var(x) < 1e-8 or np.var(y) < 1e-8:
        return 0.0
    return np.corrcoef(x, y)[0, 1]

def safe_corr_torch(x, y):
    # x, y shape: [batch, time]
    vx = x - torch.mean(x, dim=1, keepdim=True)
    vy = y - torch.mean(y, dim=1, keepdim=True)
    cost = torch.sum(vx * vy, dim=1)
    denom = torch.sqrt(torch.sum(vx ** 2, dim=1) * torch.sum(vy ** 2, dim=1))
    
    # Avoid division by zero
    corr = torch.where(denom < 1e-8, torch.zeros_like(cost), cost / denom)
    return corr

def build_ground_truth_envelope(trial):
    eeg_full = trial['eeg'].numpy()
    env_l_raw = trial['env_l'].numpy()
    env_r_raw = trial['env_r'].numpy()
    switch_points = trial['meta']['switch_points']
    
    env_l = (env_l_raw - env_l_raw.mean()) / (env_l_raw.std() + 1e-8)
    env_r = (env_r_raw - env_r_raw.mean()) / (env_r_raw.std() + 1e-8)
    
    att = np.zeros(eeg_full.shape[1], dtype=np.float32)
    unatt = np.zeros(eeg_full.shape[1], dtype=np.float32)
    if len(switch_points) == 0: switch_points = [('R', 0)]
    initial_state = 'R' if (switch_points[0][1] > 0 and switch_points[0][0] == 'L') else switch_points[0][0]
    if switch_points[0][1] > 0:
        initial_state = 'R' if switch_points[0][0] == 'L' else 'L'
        
    current_state = initial_state
    prev_idx = 0
    for state, idx_64 in switch_points:
        idx_64 = min(idx_64, eeg_full.shape[1])
        if idx_64 > prev_idx:
            if current_state == 'L':
                att[prev_idx:idx_64] = env_l[prev_idx:idx_64]
                unatt[prev_idx:idx_64] = env_r[prev_idx:idx_64]
            else:
                att[prev_idx:idx_64] = env_r[prev_idx:idx_64]
                unatt[prev_idx:idx_64] = env_l[prev_idx:idx_64]
        prev_idx, current_state = idx_64, state
        
    if current_state == 'L':
        att[prev_idx:] = env_l[prev_idx:]
        unatt[prev_idx:] = env_r[prev_idx:]
    else:
        att[prev_idx:] = env_r[prev_idx:]
        unatt[prev_idx:] = env_l[prev_idx:]
        
    return att, unatt

def simulate_trial_unsupervised(model, trial, device):
    """
    Simulate online processing. Extract predictions and test correlation windows.
    """
    eeg_full = trial['eeg'].numpy()
    env_l_raw = trial['env_l'].numpy()
    env_r_raw = trial['env_r'].numpy()
    
    eeg = (eeg_full - eeg_full.mean(axis=1, keepdims=True)) / (eeg_full.std(axis=1, keepdims=True) + 1e-8)
    env_l = (env_l_raw - env_l_raw.mean()) / (env_l_raw.std() + 1e-8)
    env_r = (env_r_raw - env_r_raw.mean()) / (env_r_raw.std() + 1e-8)
    
    att, unatt = build_ground_truth_envelope(trial)
    
    eeg_60ch = torch.from_numpy(eeg).float().unsqueeze(0).to(device)
    with torch.no_grad():
        pred = model(eeg_60ch).squeeze(0).cpu().numpy()
        
    num_windows = (eeg.shape[1] - WINDOW_LEN) // HOP_LEN + 1
    if num_windows <= 0: return [], []
    
    true_att_corr, true_unatt_corr = [], []
    for i in range(num_windows):
        start = i * HOP_LEN
        end = start + WINDOW_LEN
        
        pred_w = pred[start:end]
        att_w = att[start:end]
        unatt_w = unatt[start:end]
        
        true_att_corr.append(safe_pearson(pred_w, att_w))
        true_unatt_corr.append(safe_pearson(pred_w, unatt_w))
            
    return true_att_corr, true_unatt_corr

def compute_trial_auroc(sa_list, su_list):
    y_true = [1] * len(sa_list) + [0] * len(su_list)
    y_scores = sa_list + su_list
    if len(y_true) < 2 or len(np.unique(y_true)) < 2: return 0.5
    return roc_auc_score(y_true, y_scores)

class AADDataset(torch.utils.data.Dataset):
    def __init__(self, trials, max_len=3600):
        self.trials = trials
        self.max_len = max_len
        
    def __len__(self):
        return len(self.trials)
        
    def __getitem__(self, idx):
        tr = self.trials[idx]
        eeg_full = tr['eeg'].numpy()
        eeg = (eeg_full - eeg_full.mean(axis=1, keepdims=True)) / (eeg_full.std(axis=1, keepdims=True) + 1e-8)
        att, _ = build_ground_truth_envelope(tr)
        
        # Truncate or pad to max_len (usually 60s * 64Hz = 3840)
        # To batch cleanly, we can truncate all to 3600 (56 seconds)
        T = eeg.shape[1]
        if T >= self.max_len:
            eeg = eeg[:, :self.max_len]
            att = att[:self.max_len]
        else:
            eeg = np.pad(eeg, ((0, 0), (0, self.max_len - T)))
            att = np.pad(att, (0, self.max_len - T))
            
        return torch.from_numpy(eeg).float(), torch.from_numpy(att).float()

def run_native_eegnet(cache_dir, subject_ids, device):
    print("\n=======================================================")
    print(" PHASE 47: NATIVE AASD TRAINING WITH EEGNET (LOSO) ")
    print("=======================================================")
    
    # Pre-load all data into memory
    print("Loading all subjects into memory...")
    subject_data = {}
    for subj in subject_ids:
        cached = torch.load(cache_dir / f"{subj}_processed.pt", map_location='cpu', weights_only=False)
        subject_data[subj] = cached['raw']
    print("Data loaded.\n")
    
    fold_aurocs = []
    fold_importances = []
    
    for test_subj in subject_ids:
        print(f"\n>> Leaving out Test Subject: {test_subj}")
        
        train_trials = []
        for subj, trials in subject_data.items():
            if subj != test_subj:
                train_trials.extend(trials)
                
        test_trials = subject_data[test_subj]
        
        train_ds = AADDataset(train_trials)
        train_loader = torch.utils.data.DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
        
        model = AAD_EEGNet(in_channels=60, max_lag=MAX_LAG).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.0)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
        
        mse_criterion = torch.nn.MSELoss()
        
        model.train()
        print("   Training Model...")
        for epoch in range(EPOCHS):
            epoch_loss = 0.0
            epoch_corr = 0.0
            for X, y in train_loader:
                X = X.to(device)
                y = y.to(device)
                
                optimizer.zero_grad()
                pred = model(X)
                
                # Align Y for temporal convolution delay and skip initial padded frames
                y_aligned = y[:, MAX_LAG:]
                pred_aligned = pred[:, MAX_LAG:]
                
                # Loss is MSE + 0.1 * (1 - Pearson Correlation)
                corr = safe_corr_torch(pred_aligned, y_aligned).mean()
                mse = mse_criterion(pred_aligned, y_aligned)
                loss = mse + 0.1 * (1.0 - corr)
                
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                epoch_corr += corr.item()
                
            scheduler.step()
            if (epoch + 1) % 5 == 0 or epoch == EPOCHS - 1:
                avg_corr = epoch_corr / len(train_loader)
                print(f"      Epoch {epoch+1:02d}/{EPOCHS} | Loss: {epoch_loss/len(train_loader):.4f} | Corr: {avg_corr:.4f}")
                
        # Evaluate Zero-Shot on Test Subject
        model.eval()
        zs_sa, zs_su = [], []
        for tr in test_trials:
            sa, su = simulate_trial_unsupervised(model, tr, device)
            zs_sa.extend(sa)
            zs_su.extend(su)
            
        zs_auroc = compute_trial_auroc(zs_sa, zs_su)
        print(f"   [Zero-Shot (Generalized) AUROC on {test_subj}]: {zs_auroc:.4f}")
        fold_aurocs.append(zs_auroc)
        
        # Extract channel importance
        fold_importances.append(model.get_channel_importance())
        
    print("\n=======================================================")
    print(" SUMMARY OF ZERO-SHOT GENERALIZATION (ALL 60 CHANNELS) ")
    print("=======================================================")
    print(f"Mean AUROC across {len(subject_ids)} subjects: {np.mean(fold_aurocs):.4f}")
    
    # Average channel importance
    if len(fold_importances) > 0:
        mean_importance = np.mean(fold_importances, axis=0)
        top_channels = np.argsort(mean_importance)[::-1]
        print("\nTop 10 Most Important Channels (Indices):")
        for rank, ch_idx in enumerate(top_channels[:10]):
            print(f"  #{rank+1}: Channel {ch_idx} (Weight: {mean_importance[ch_idx]:.4f})")

def main():
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Kaggle vs Local Paths
    cache_dir = Path('/kaggle/working/eeg_cache') if Path('/kaggle/working').exists() else REPO_ROOT / 'processed_eeg'
    
    if not cache_dir.exists():
        print(f"Cache directory {cache_dir} not found. Run Phase 41 first to generate cache.")
        return
        
    subject_ids = []
    for pt_file in cache_dir.glob("*_processed.pt"):
        subject_ids.append(pt_file.name.split('_')[0])
    subject_ids.sort()
        
    run_native_eegnet(cache_dir, subject_ids, device)

if __name__ == "__main__":
    main()
