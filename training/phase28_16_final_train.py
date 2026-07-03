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

def cross_corr(x, y):
    if len(x) == 0 or len(y) == 0: return 0.0
    x = (x - np.mean(x)) / (np.std(x) + 1e-8)
    y = (y - np.mean(y)) / (np.std(y) + 1e-8)
    c = np.corrcoef(x, y)[0, 1]
    return c if not np.isnan(c) else 0.0

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

def evaluate_60s_trials(model, trials, device):
    model.eval()
    corrs_att = []
    corrs_unatt = []
    
    with torch.no_grad():
        for trial in trials:
            eeg = trial['eeg'].unsqueeze(0).to(device)
            att = trial['att'].numpy()[0]
            unatt = trial['unatt'].numpy()[0]
            
            pred, _ = model(eeg, return_features=True)
            pred = pred.squeeze().cpu().numpy()
            
            min_len = min(len(pred), len(att))
            pred = pred[:min_len]
            att = att[:min_len]
            unatt = unatt[:min_len]
            
            c_att = cross_corr(pred, att)
            c_unatt = cross_corr(pred, unatt)
            corrs_att.append(c_att)
            corrs_unatt.append(c_unatt)
            
    return np.mean(corrs_att), np.mean(corrs_unatt)

def main():
    print("==================================================")
    print("=== PHASE 28.16 FINAL DYNAMIC TRAINING ==========")
    print("==================================================\n")
    
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
    
    # 1-8 Hz filter (Nyquist = 64.0 Hz)
    b, a = scipy.signal.butter(4, [1.0/64.0, 8.0/64.0], btype='band')
    
    new_trials = []
    for trial_idx in range(data_all.shape[2]):
        if trial_idx >= len(audio_markers): break
        marker, trial_start_lat = audio_markers[trial_idx]
        
        trial_eeg = data_all[:, :, trial_idx]
        trial_eeg = trial_eeg - trial_eeg.mean(axis=0, keepdims=True)
        trial_eeg_filt = scipy.signal.filtfilt(b, a, trial_eeg, axis=1)
        trial_eeg_64 = scipy.signal.resample_poly(trial_eeg_filt, 1, 2, axis=1)
        trial_eeg_8 = trial_eeg_64[sel_idx, :]
        
        # Discard everything before the audio marker starts playing
        start_64 = int(trial_start_lat // 2)
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
        
        # FATAL BUG FIXED HERE: Extract raw_evs matching epoch = int(marker)
        raw_evs = []
        for ev in events:
            ep = str(get_ev_attr(ev, 'epoch')).strip()
            if ep and int(ep) == int(marker):
                t_str = str(get_ev_attr(ev, 'type')).strip()
                ev_lat = float(get_ev_attr(ev, 'latency'))
                raw_evs.append((t_str, ev_lat))
                
        # Calculate dynamic attention timeline based on switch events
        # Initialize arrays for spliced targets
        att = np.zeros_like(env_l)
        unatt = np.zeros_like(env_r)
        
        # Find all switch events and their relative latencies
        switch_points = []
        for ev_t, ev_lat in raw_evs:
            if ev_t in ['179', '184', '254', '255']:
                rel_samples = ev_lat - trial_start_lat
                idx_64 = max(0, int(rel_samples // 2))
                
                # '179' or '254' = Right ear, '184' or '255' = Left ear
                if ev_t in ['179', '254']:
                    switch_points.append(('R', idx_64))
                else:
                    switch_points.append(('L', idx_64))
                    
        # Ensure switch points are sorted chronologically
        switch_points.sort(key=lambda x: x[1])
        
        # If no switches found, assume 'R' (or whatever default, but there should be switches now)
        if len(switch_points) == 0:
            print(f"[WARNING] Trial {trial_idx+1} has no switch points! Defaulting to R.")
            switch_points = [('R', 0)]
            
        # Splice the envelopes based on the timeline
        prev_idx = 0
        current_state = switch_points[0][0] # State at t=0
        
        for state, idx_64 in switch_points:
            # We don't splice backwards
            if idx_64 > prev_idx:
                if current_state == 'R':
                    att[prev_idx:idx_64] = env_r[prev_idx:idx_64]
                    unatt[prev_idx:idx_64] = env_l[prev_idx:idx_64]
                else:
                    att[prev_idx:idx_64] = env_l[prev_idx:idx_64]
                    unatt[prev_idx:idx_64] = env_r[prev_idx:idx_64]
            prev_idx = idx_64
            current_state = state
            
        # Fill the remainder of the trial
        if current_state == 'R':
            att[prev_idx:] = env_r[prev_idx:]
            unatt[prev_idx:] = env_l[prev_idx:]
        else:
            att[prev_idx:] = env_l[prev_idx:]
            unatt[prev_idx:] = env_r[prev_idx:]
            
        new_trials.append({
            'meta': {'raw_evs': raw_evs, 'switch_points': switch_points},
            'eeg': torch.FloatTensor(trial_eeg_8),
            'att': torch.FloatTensor(att).unsqueeze(0),
            'unatt': torch.FloatTensor(unatt).unsqueeze(0)
        })
        
    print(f"[INFO] Built truly dynamic cache with {len(new_trials)} trials.")
    if len(new_trials) > 0:
        print(f"Trial 1 Switch Points: {new_trials[0]['meta']['switch_points']}")
    
    # 2. Windowing
    split_idx = int(len(new_trials) * 0.8)
    train_trials = new_trials[:split_idx]
    test_trials = new_trials[split_idx:]
    
    train_ds = WindowedDataset(train_trials)
    
    print(f"[INFO] Train Windows: {len(train_ds)}, Test Trials: {len(test_trials)}")
    
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    
    # 3. Training
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AADConformer(in_channels=8).to(device)
    
    kul_path = '/kaggle/input/datasets/lowkieee/eeg-aad-conformer-seed1-checkpoints/checkpoints/seed_1/model_S1.pt'
    if os.path.exists(kul_path):
        ckpt = torch.load(kul_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt, strict=False)
        print("[INFO] Loaded KUL weights (strict=False).")
        
    optimizer = optim.Adam(model.parameters(), lr=1e-4) # Higher LR for faster convergence
    
    # Evaluate Zero-Shot
    c_att, c_unatt = evaluate_60s_trials(model, test_trials, device)
    print(f"\n[ZERO-SHOT] Test 60s Corrs - Attended: {c_att:.4f} | Unattended: {c_unatt:.4f}\n")
    
    for epoch in range(1, 51):
        model.train()
        train_loss = 0
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
            total += eeg_w.size(0)
            
        if epoch % 5 == 0:
            c_att_train, c_unatt_train = evaluate_60s_trials(model, train_trials, device)
            c_att_test, c_unatt_test = evaluate_60s_trials(model, test_trials, device)
            
            mean_corr_2s = 1.0 - (train_loss/total)
            print(f"Epoch {epoch:02d} - 2s Corr: {mean_corr_2s:.4f} | Train 60s (Att: {c_att_train:.3f}, Unatt: {c_unatt_train:.3f}) | Test 60s (Att: {c_att_test:.3f}, Unatt: {c_unatt_test:.3f})")
        
    print("\nDIAGNOSIS:")
    if c_att_test > c_unatt_test and c_att_test > 0.05:
        print("MASSIVE SUCCESS! The model successfully learned to decode auditory attention on AASD!")
    else:
        print("INCOMPLETE. The model is STILL failing. This would imply the dataset itself is fatally corrupted.")

if __name__ == "__main__":
    main()
