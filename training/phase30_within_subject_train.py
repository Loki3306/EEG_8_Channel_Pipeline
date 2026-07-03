import os
import sys
import numpy as np
import scipy.io
import scipy.signal
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
import glob
from pathlib import Path
import time
import random
import argparse
import torch.nn.functional as F

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Adjust path based on where it's executed
if not os.path.exists('models') and os.path.exists('../models'):
    sys.path.insert(0, '..')

from models.matchnet import ContrastiveMatchNet, infonce_loss

class WindowedDataset(Dataset):
    def __init__(self, trials, window_len=128, hop_len=64, censor_margin=256):
        self.windows = []
        for trial in trials:
            eeg = trial['eeg']
            env_l = trial['env_l'].unsqueeze(0) # [1, T]
            env_r = trial['env_r'].unsqueeze(0) # [1, T]
            att = trial['att'].unsqueeze(0)     # [1, T]
            unatt = trial['unatt'].unsqueeze(0) # [1, T]
            switch_points = trial.get('meta', {}).get('switch_points', [])
            
            for start in range(0, eeg.shape[1] - window_len + 1, hop_len):
                end = start + window_len
                
                # CENSORING LOGIC: Drop window if it overlaps [s_idx, s_idx + censor_margin]
                is_censored = False
                for state, s_idx in switch_points:
                    if s_idx > 0:
                        if (start < s_idx + censor_margin) and (end > s_idx):
                            is_censored = True
                            break
                            
                if is_censored:
                    continue
                    
                w_eeg = eeg[:, start:end]
                w_att = att[:, start:end]
                w_unatt = unatt[:, start:end]
                
                # Normalization
                w_eeg_mean = w_eeg.mean(dim=1, keepdim=True)
                w_eeg_std = w_eeg.std(dim=1, keepdim=True) + 1e-8
                w_eeg = (w_eeg - w_eeg_mean) / w_eeg_std
                
                w_att_mean = w_att.mean(dim=1, keepdim=True)
                w_att_std = w_att.std(dim=1, keepdim=True) + 1e-8
                w_att = (w_att - w_att_mean) / w_att_std
                
                w_unatt_mean = w_unatt.mean(dim=1, keepdim=True)
                w_unatt_std = w_unatt.std(dim=1, keepdim=True) + 1e-8
                w_unatt = (w_unatt - w_unatt_mean) / w_unatt_std
                
                # Determine state for evaluation
                mid_point = start + window_len // 2
                current_state = switch_points[0][0]
                for state, s_idx in switch_points:
                    if mid_point >= s_idx: current_state = state
                    
                self.windows.append({
                    'eeg': w_eeg,
                    'att': w_att,
                    'unatt': w_unatt,
                    'env_l': env_l[:, start:end],
                    'env_r': env_r[:, start:end],
                    'state': current_state
                })
                
    def __len__(self):
        return len(self.windows)
        
    def __getitem__(self, idx):
        return self.windows[idx]

def load_aasd_subject_trials(mat_path, b, a, sel_idx, audio_dir):
    mat = scipy.io.loadmat(mat_path, squeeze_me=True, struct_as_record=False)
    eeg_var = [k for k in mat.keys() if not k.startswith('__')][0]
    data_all, events = mat[eeg_var].data, mat[eeg_var].event
    
    trials = []
    for epoch_idx in range(1, 61):
        audio_marker_val = None
        for ev in events:
            if len(ev) >= 5:
                t_str, epoch_val = str(ev[0]).strip(), str(ev[4]).strip()
                if epoch_val == str(epoch_idx) and t_str.isdigit() and 11 <= int(t_str) <= 70:
                    audio_marker_val = int(t_str)
                    break
        if audio_marker_val is None: continue
            
        npz_path = os.path.join(audio_dir, f"{audio_marker_val}.npz")
        if not os.path.exists(npz_path): continue
            
        epoch_start_lat_128 = (epoch_idx - 1) * 7680 + 1
        switch_points = []
        for ev in events:
            if len(ev) >= 5:
                t_str, epoch_val = str(ev[0]).strip(), str(ev[4]).strip()
                if epoch_val == str(epoch_idx) and t_str in ['179', '184', '254', '255']:
                    abs_lat = float(ev[1])
                    rel_lat_128 = abs_lat - epoch_start_lat_128
                    idx_64 = max(0, int(rel_lat_128 / 2.0) - 4) # Hardware lag (-62ms -> 4 samples)
                    switch_points.append(('R' if t_str in ['179', '254'] else 'L', idx_64))
        switch_points.sort(key=lambda x: x[1])
        
        trial_eeg = data_all[:, :, epoch_idx - 1]
        trial_eeg = trial_eeg - trial_eeg.mean(axis=0, keepdims=True)
        trial_eeg_filt = scipy.signal.filtfilt(b, a, trial_eeg, axis=1)
        trial_eeg_8 = scipy.signal.resample_poly(trial_eeg_filt, 1, 2, axis=1)[sel_idx, 4:] # Hardware lag
        
        audio_data = np.load(npz_path)
        env_l, env_r = audio_data['env_l'][:-4], audio_data['env_r'][:-4]
        
        min_len = min(trial_eeg_8.shape[1], len(env_l))
        trial_eeg_8 = trial_eeg_8[:, :min_len]
        env_l, env_r = env_l[:min_len], env_r[:min_len]
        
        att, unatt = np.zeros_like(env_l), np.zeros_like(env_r)
        if len(switch_points) == 0: switch_points = [('R', 0)]
        
        current_state = switch_points[0][0]
        prev_idx = 0
        for state, idx_64 in switch_points:
            if idx_64 > prev_idx:
                if current_state == 'R':
                    att[prev_idx:idx_64], unatt[prev_idx:idx_64] = env_r[prev_idx:idx_64], env_l[prev_idx:idx_64]
                else:
                    att[prev_idx:idx_64], unatt[prev_idx:idx_64] = env_l[prev_idx:idx_64], env_r[prev_idx:idx_64]
            prev_idx, current_state = idx_64, state
            
        if current_state == 'R':
            att[prev_idx:], unatt[prev_idx:] = env_r[prev_idx:], env_l[prev_idx:]
        else:
            att[prev_idx:], unatt[prev_idx:] = env_l[prev_idx:], env_r[prev_idx:]
            
        trials.append({
            'meta': {'switch_points': switch_points},
            'eeg': torch.FloatTensor(trial_eeg_8),
            'env_l': torch.FloatTensor(env_l),
            'env_r': torch.FloatTensor(env_r),
            'att': torch.FloatTensor(att),
            'unatt': torch.FloatTensor(unatt)
        })
    return trials

def evaluate_model(model, val_loader, device):
    model.eval()
    all_margins = []
    all_labels = []
    trial_preds = []
    trial_labels = []
    
    with torch.no_grad():
        for batch in val_loader:
            eeg = batch['eeg'].to(device)
            att = batch['att'].to(device)
            unatt = batch['unatt'].to(device)
            states = batch['state']
            
            z_eeg, z_a, z_b = model(eeg, att, unatt)
            
            sim_att = F.cosine_similarity(z_eeg, z_a, dim=1).mean(dim=1)
            sim_unatt = F.cosine_similarity(z_eeg, z_b, dim=1).mean(dim=1)
            margin = (sim_att - sim_unatt).cpu().numpy()
            
            for i in range(len(margin)):
                all_margins.append(margin[i])
                all_labels.append(1) # We always mapped z_a to attended in forward pass
                
                pred = 1 if margin[i] > 0 else 0
                trial_preds.append(pred)
                trial_labels.append(1)

    auc = roc_auc_score(all_labels + [0]*len(all_labels), all_margins + [-m for m in all_margins])
    bacc = balanced_accuracy_score(trial_labels + [0]*len(trial_labels), trial_preds + [1-p for p in trial_preds])
    return auc, bacc, np.mean(all_margins)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type=str, default="S18", help="Target subject for within-subject evaluation")
    parser.add_argument("--split_mode", type=str, default="random", choices=["random", "chronological"], help="How to split trials")
    parser.add_argument("--censor_margin", type=int, default=256, help="Samples to censor after switch (256=4s)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--epochs", type=int, default=30, help="Training epochs")
    args = parser.parse_args()

    seed_everything(args.seed)
    print("="*50)
    print("=== PHASE 30.0 WITHIN-SUBJECT SANITY BENCHMARK ===")
    print(f"=== SUBJECT: {args.subject} | SPLIT: {args.split_mode} | SEED: {args.seed} ===")
    print(f"=== CENSOR MARGIN: {args.censor_margin} samples ({args.censor_margin/64:.1f}s) ===")
    print("="*50 + "\n")
    
    mat_path = glob.glob(f'/kaggle/input/datasets/lokeshgile/aasd-processed-eeg/Processed EEG/*/{args.subject}.mat')
    if not mat_path:
        mat_path = glob.glob(f'data/*/{args.subject}.mat')
    
    if not mat_path:
        print(f"ERROR: Could not find {args.subject}.mat. Please run on Kaggle.")
        return
        
    mat_path = mat_path[0]
    audio_dir = '/kaggle/input/datasets/lokeshgile/aasd-audio-gammatones'
    if not os.path.exists(audio_dir):
        audio_dir = 'data/audio_features'
        
    fs_eeg = 256
    nyq = 0.5 * fs_eeg
    b, a = scipy.signal.butter(4, [1.0/nyq, 8.0/nyq], btype='band')
    target_channels = ['T7', 'C2', 'FT8', 'P7', 'CPz', 'Fp1', 'TP8', 'C3']
    channel_names = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'FC5', 'FC1', 'FC2', 'FC6', 'T7', 'C3', 'Cz', 'C4', 'T8', 'CP5', 'CP1', 'CP2', 'CP6', 'P7', 'P3', 'Pz', 'P4', 'P8', 'PO9', 'O1', 'Oz', 'O2', 'PO10', 'AF7', 'AF3', 'AF4', 'AF8', 'F5', 'F1', 'F2', 'F6', 'FT7', 'FC3', 'FC4', 'FT8', 'C5', 'C1', 'C2', 'C6', 'TP7', 'CP3', 'CPz', 'CP4', 'TP8', 'P5', 'P1', 'P2', 'P6', 'PO7', 'PO3', 'POz', 'PO4', 'PO8', 'O9', 'O10', 'Iz', 'Cz']
    sel_idx = [channel_names.index(tc) for tc in target_channels]
    
    print(f"[INFO] Loading Subject {args.subject}...")
    t0 = time.time()
    trials = load_aasd_subject_trials(mat_path, b, a, sel_idx, audio_dir)
    print(f"[INFO] Loaded {len(trials)} trials in {time.time()-t0:.1f}s")
    
    if len(trials) == 0:
        print("ERROR: No trials loaded.")
        return
        
    if args.split_mode == 'random':
        random.shuffle(trials)
        train_trials = trials[:40]
        test_trials = trials[40:]
    else: # chronological
        train_trials = trials[:40]
        test_trials = trials[40:]
        
    print(f"[INFO] Train Trials: {len(train_trials)} | Test Trials: {len(test_trials)}")
    
    train_ds = WindowedDataset(train_trials, censor_margin=args.censor_margin)
    test_ds = WindowedDataset(test_trials, censor_margin=args.censor_margin)
    
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)
    print(f"[INFO] Train Windows: {len(train_ds)} | Test Windows: {len(test_ds)}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] Using device: {device}")
    
    # 50k Parameter ContrastiveMatchNet with 1 audio channel
    model = ContrastiveMatchNet(eeg_model_type="eegnet", eeg_channels=8, audio_channels=1, latent_dim=64).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    
    print("\n[INFO] Training...")
    best_auc = 0.0
    
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        t0 = time.time()
        
        for batch in train_loader:
            eeg = batch['eeg'].to(device)
            att = batch['att'].to(device)
            unatt = batch['unatt'].to(device)
            
            optimizer.zero_grad()
            z_eeg, z_a, z_b = model(eeg, att, unatt)
            
            loss, sa, sb = infonce_loss(z_eeg, z_a, z_b, temperature=0.1)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
            
        train_loss /= len(train_loader)
        
        auc, bacc, mean_margin = evaluate_model(model, test_loader, device)
        
        new_best = ""
        if auc > best_auc:
            best_auc = auc
            new_best = " [*]"
            
        print(f"Epoch {epoch:02d}/{args.epochs} | Loss: {train_loss:.4f} | Margin: {mean_margin:+.4f} | AUROC: {auc:.3f} | BAcc: {bacc:.3f} | Time: {time.time()-t0:.1f}s{new_best}")

    print(f"\n[INFO] Training complete. Best Stable AUROC: {best_auc:.3f}")

if __name__ == "__main__":
    main()
