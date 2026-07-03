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
            audio_l = trial['audio_l'][0] # (time)
            audio_r = trial['audio_r'][0] # (time)
            
            # Find attended speaker
            initial_att = 'R'
            for ev_t, ev_lat in trial['meta']['raw_evs']:
                if ev_t in ['179', '254']: initial_att = 'R'; break
                if ev_t in ['184', '255']: initial_att = 'L'; break
                
            att = audio_r if initial_att == 'R' else audio_l
            
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
    print("=== PHASE 28.9 FAST TRAINING ON NEW CACHE =======")
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
    b, a = scipy.signal.butter(4, [1.0/32.0, 8.0/32.0], btype='band')
    
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
        
        # Z-score EEG so it matches KUL expectation
        trial_eeg_8 = trial_eeg_8 - trial_eeg_8.mean(axis=1, keepdims=True)
        trial_eeg_8 = trial_eeg_8 / (trial_eeg_8.std(axis=1, keepdims=True) + 1e-12)
        
        raw_evs = []
        for ev in events:
            ep = get_ev_attr(ev, 'epoch')
            if ep and int(ep) == trial_idx + 1:
                t_str = str(get_ev_attr(ev, 'type')).strip()
                ev_lat = float(get_ev_attr(ev, 'latency'))
                raw_evs.append((t_str, ev_lat))
                
        new_trials.append({
            'meta': {'raw_evs': raw_evs},
            'eeg': torch.FloatTensor(trial_eeg_8),
            'audio_l': torch.FloatTensor(env_l).unsqueeze(0),
            'audio_r': torch.FloatTensor(env_r).unsqueeze(0)
        })
        
    print(f"[INFO] Built corrected cache with {len(new_trials)} trials.")
    
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
    
    for epoch in range(1, 6):
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
            
            # Fast accuracy approx
            with torch.no_grad():
                pred_c = pred - pred.mean(dim=-1, keepdim=True)
                att_c = att_w - att_w.mean(dim=-1, keepdim=True)
                cov = (pred_c * att_c).sum(dim=-1)
                train_acc += (cov > 0).sum().item()
                total += eeg_w.size(0)
                
        print(f"Epoch {epoch:02d} - Train Loss: {train_loss/total:.4f} - Train Acc: {train_acc/total:.4f}")
        
    print("\nDIAGNOSIS:")
    final_acc = train_acc/total
    if final_acc > 0.60:
        print("SUCCESS! The model is rapidly learning the corrected dataset.")
        print("The epoched latency bug was indeed the fatal flaw.")
    else:
        print("INCOMPLETE. The model is still struggling. Could the target label (179/184) mapping be inverted?")

if __name__ == "__main__":
    main()
