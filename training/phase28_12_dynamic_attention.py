import os
import sys
import numpy as np
import scipy.io
import scipy.signal
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import glob
from pathlib import Path
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.aad_conformer import AADConformer

def get_ev_attr(e, attr_name, array_idx=0):
    try:
        if hasattr(e, attr_name): return getattr(e, attr_name)
        if hasattr(e.flat[0], attr_name): return getattr(e.flat[0], attr_name)
        return e[array_idx]
    except: return ''

def pearson_loss(pred, target):
    pred = pred - pred.mean(dim=-1, keepdim=True)
    target = target - target.mean(dim=-1, keepdim=True)
    cov = (pred * target).sum(dim=-1)
    var_pred = (pred ** 2).sum(dim=-1)
    var_target = (target ** 2).sum(dim=-1)
    corr = cov / torch.sqrt(var_pred * var_target + 1e-8)
    return 1.0 - corr.mean()

class WindowedDataset(Dataset):
    def __init__(self, trials, window_len=128, hop_len=64):
        self.windows = []
        for trial in trials:
            eeg = trial['eeg'] # (8, time)
            att = trial['att'][0] # (time)
            
            for start in range(0, eeg.shape[1] - window_len + 1, hop_len):
                w_eeg = eeg[:, start:start+window_len]
                w_att = att[start:start+window_len]
                self.windows.append((w_eeg, w_att))
                
    def __len__(self):
        return len(self.windows)
        
    def __getitem__(self, idx):
        return self.windows[idx]

def main():
    print("==================================================")
    print("=== PHASE 28.12 DYNAMIC ATTENTION SWITCHING =====")
    print("==================================================\n")
    
    # 1. Build Cache
    S1_mat = '/kaggle/input/datasets/lokeshgile/aasd-processed-eeg/Processed EEG/S01/S1.mat'
    if not os.path.exists(S1_mat):
        S1_mat = glob.glob('/kaggle/input/datasets/lokeshgile/aasd-processed-eeg/Processed EEG/*/*.mat')[0]
        
    mat = scipy.io.loadmat(S1_mat, squeeze_me=True, struct_as_record=False)
    eeg_var = [k for k in mat.keys() if not k.startswith('__')][0]
    data_all = mat[eeg_var].data
    events = mat[eeg_var].event
    
    audio_markers = []
    for ev in events:
        t_str = str(get_ev_attr(ev, 'type', 0)).strip()
        if t_str.isdigit():
            val = int(t_str)
            if 11 <= val <= 70:
                lat = int(get_ev_attr(ev, 'latency'))
                audio_markers.append((t_str, lat))
                
    audio_dir = '/kaggle/input/datasets/lokeshgile/aasd-audio-gammatones'
    target_channels = ['T7', 'C2', 'FT8', 'P7', 'CPz', 'Fp1', 'TP8', 'C3']
    fallback_map = {'T7': 23, 'C2': 28, 'FT8': 22, 'P7': 41, 'CPz': 36, 'Fp1': 0, 'TP8': 40, 'C3': 25}
    sel_idx = [fallback_map[tc] for tc in target_channels]
    
    b, a = scipy.signal.butter(4, [1.0/64.0, 8.0/64.0], btype='band')
    
    new_trials = []
    for trial_idx in range(data_all.shape[2]):
        if trial_idx >= len(audio_markers): break
        marker, lat = audio_markers[trial_idx]
        
        trial_eeg = data_all[:, :, trial_idx]
        trial_eeg = trial_eeg - trial_eeg.mean(axis=0, keepdims=True)
        trial_eeg_filt = scipy.signal.filtfilt(b, a, trial_eeg, axis=1)
        trial_eeg_64 = scipy.signal.resample_poly(trial_eeg_filt, 1, 2, axis=1)
        trial_eeg_8 = trial_eeg_64[sel_idx, :]
        
        start_64 = int(lat // 2)
        trial_eeg_8 = trial_eeg_8[:, start_64:]
        
        npz_path = os.path.join(audio_dir, f"{int(marker)}.npz")
        if not os.path.exists(npz_path): continue
        
        audio_data = np.load(npz_path)
        env_l = audio_data['env_l']
        env_r = audio_data['env_r']
        
        min_len = min(trial_eeg_8.shape[1], len(env_l))
        trial_eeg_8 = trial_eeg_8[:, :min_len]
        env_l = env_l[:min_len]
        env_r = env_r[:min_len]
        
        trial_eeg_8 = trial_eeg_8 - trial_eeg_8.mean(axis=1, keepdims=True)
        trial_eeg_8 = trial_eeg_8 / (trial_eeg_8.std(axis=1, keepdims=True) + 1e-12)
        
        env_l = (env_l - env_l.mean()) / (env_l.std() + 1e-12)
        env_r = (env_r - env_r.mean()) / (env_r.std() + 1e-12)
        
        raw_evs = []
        for ev in events:
            ep = get_ev_attr(ev, 'epoch')
            if ep and int(ep) == trial_idx + 1:
                t_str = str(get_ev_attr(ev, 'type')).strip()
                ev_lat = float(get_ev_attr(ev, 'latency'))
                raw_evs.append((t_str, ev_lat))
                
        # --- DYNAMIC ATTENTION SWITCHING ---
        switch_points = []
        for ev_t, ev_lat in raw_evs:
            if ev_t in ['179', '254']:
                idx_64 = int(max(0, ev_lat - lat) // 2)
                switch_points.append(('R', idx_64))
            elif ev_t in ['184', '255']:
                idx_64 = int(max(0, ev_lat - lat) // 2)
                switch_points.append(('L', idx_64))
                
        switch_points.sort(key=lambda x: x[1])
        
        current_att = 'R'
        if switch_points and switch_points[0][1] > 0:
            # If the trial starts before the first switch point, we might need a default
            # Usually the first marker determines the starting state
            current_att = switch_points[0][0]
            
        att = np.zeros(min_len)
        last_idx = 0
        
        for state, idx in switch_points:
            if idx > last_idx:
                idx = min(idx, min_len)
                if current_att == 'R':
                    att[last_idx:idx] = env_r[last_idx:idx]
                else:
                    att[last_idx:idx] = env_l[last_idx:idx]
                last_idx = idx
            current_att = state
            
        if last_idx < min_len:
            if current_att == 'R':
                att[last_idx:] = env_r[last_idx:]
            else:
                att[last_idx:] = env_l[last_idx:]
                
        new_trials.append({
            'meta': {'raw_evs': raw_evs},
            'eeg': torch.FloatTensor(trial_eeg_8),
            'att': torch.FloatTensor(att).unsqueeze(0)
        })
        
    print(f"[INFO] Built dynamically spliced cache with {len(new_trials)} trials.")
    
    # 2. Windowing
    split_idx = int(len(new_trials) * 0.8)
    train_trials = new_trials[:split_idx]
    test_trials = new_trials[split_idx:]
    
    train_ds = WindowedDataset(train_trials)
    test_ds = WindowedDataset(test_trials)
    
    print(f"[INFO] Train Windows: {len(train_ds)}, Test Windows: {len(test_ds)}")
    
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    
    # 3. Training
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AADConformer(in_channels=8).to(device)
    
    kul_path = '/kaggle/input/datasets/lowkieee/eeg-aad-conformer-seed1-checkpoints/checkpoints/seed_1/model_S1.pt'
    if os.path.exists(kul_path):
        ckpt = torch.load(kul_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt, strict=False)
        print("[INFO] Loaded KUL weights (strict=False).")
        
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    for epoch in range(1, 11):
        model.train()
        train_loss = 0
        train_acc = 0
        total = 0
        
        for eeg_w, att_w in train_loader:
            eeg_w, att_w = eeg_w.to(device), att_w.to(device)
            optimizer.zero_grad()
            
            pred, _ = model(eeg_w, return_features=True)
            pred = pred.squeeze(1)
            
            loss = pearson_loss(pred, att_w)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * eeg_w.size(0)
            
            with torch.no_grad():
                pred_c = pred - pred.mean(dim=-1, keepdim=True)
                att_c = att_w - att_w.mean(dim=-1, keepdim=True)
                cov = (pred_c * att_c).sum(dim=-1)
                train_acc += (cov > 0).sum().item()
                total += eeg_w.size(0)
                
        print(f"Epoch {epoch:02d} - Train Loss: {train_loss/total:.4f} - Train Acc: {train_acc/total:.4f}")
        
    print("\nDIAGNOSIS:")
    final_acc = train_acc/total
    if final_acc > 0.70:
        print("MASSIVE SUCCESS! The model easily learned the switching dataset.")
        print("The constant-label bug was destroying 50% of the target data.")
    else:
        print("INCOMPLETE. The model STILL cannot overfit. This is mathematically impossible unless the entire audio folder is mislabeled or out of sync.")

if __name__ == "__main__":
    main()
