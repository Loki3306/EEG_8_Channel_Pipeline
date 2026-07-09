import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score
import numpy as np
from pathlib import Path
import sys
import os
import random
from scipy.signal import resample_poly

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.aad_tcn import DeepMatchMismatchTCN

# -------------------------------------------------------------------------
# CONSTANTS
# -------------------------------------------------------------------------
SR_ORIGINAL = 128
SR = 50 # NATIVE WAVLM FREQUENCY
WIN_SEC = 2.0
HOP_SEC = 0.5
EXCLUSION_SEC = 1.5   
SEQ_SEC = 3.5         

WIN_SAMPLES = int(WIN_SEC * SR)
HOP_SAMPLES = int(HOP_SEC * SR)
EXCLUSION_SAMPLES = int(EXCLUSION_SEC * SR)
SEQ_SAMPLES = int(SEQ_SEC * SR)

EAR_CHANNEL_INDICES = [23, 31, 32, 40, 14, 22, 41, 49]
BATCH_SIZE = 32
TRAIN_EPOCHS = 12
TRAIN_LR = 1e-3

class MatchMismatchWavLMDataset(Dataset):
    def __init__(self, sequences):
        self.seqs = sequences
    def __len__(self): return len(self.seqs)
    def __getitem__(self, idx):
        e, a, y = self.seqs[idx]
        return e, a, y[-1]

def extract_match_mismatch_sequences_wavlm(trials):
    sequences = []
    seq_hop = int(0.5 * SR) 
    
    for tr in trials:
        eeg = tr['eeg'] # [C, T]
        wavlm_l = tr['wavlm_l'] # [T, 768]
        wavlm_r = tr['wavlm_r'] # [T, 768]
        sp = tr['meta']['switch_points']
        T = eeg.shape[1]
        
        # Scale switch points using time in seconds to prevent precision loss
        scaled_sp = [(spk, round((idx / SR_ORIGINAL) * SR)) for spk, idx in sp]
        
        boundaries = [0]
        boundaries.extend([idx for spk, idx in scaled_sp])
        boundaries = sorted(set(boundaries))
        if boundaries[-1] != T: boundaries.append(T)
            
        for i in range(len(boundaries) - 1):
            start_idx = boundaries[i]
            end_idx = boundaries[i+1]
            current_spk = 'L'
            for spk, idx in scaled_sp:
                if idx <= start_idx: current_spk = spk
                else: break
                
            safe_start = start_idx + EXCLUSION_SAMPLES
            safe_end = end_idx
            
            if safe_end - safe_start >= SEQ_SAMPLES:
                for seq_start in range(safe_start, safe_end - SEQ_SAMPLES + 1, seq_hop):
                    e_seq = eeg[:, seq_start:seq_start + SEQ_SAMPLES]
                    wl_seq = wavlm_l[seq_start:seq_start + SEQ_SAMPLES, :]
                    wr_seq = wavlm_r[seq_start:seq_start + SEQ_SAMPLES, :]
                    
                    if e_seq.shape[1] != SEQ_SAMPLES or wl_seq.shape[0] != SEQ_SAMPLES:
                        continue
                    
                    # Unfold EEG: [TimeWindows, Channels, WindowSamples]
                    e = torch.from_numpy(e_seq.copy()).unfold(-1, WIN_SAMPLES, HOP_SAMPLES).permute(1, 0, 2)
                    
                    # Unfold WavLM: [TimeWindows, WindowSamples, 768]
                    wl_tensor = torch.from_numpy(wl_seq.copy()).transpose(0, 1)
                    wl_windows = wl_tensor.unfold(-1, WIN_SAMPLES, HOP_SAMPLES).permute(1, 0, 2).transpose(1, 2)
                    
                    wr_tensor = torch.from_numpy(wr_seq.copy()).transpose(0, 1)
                    wr_windows = wr_tensor.unfold(-1, WIN_SAMPLES, HOP_SAMPLES).permute(1, 0, 2).transpose(1, 2)
                    
                    num_windows = e.shape[0]
                    
                    # Create the MATCH pair
                    if current_spk == 'L':
                        match_aud = wl_windows
                        mismatch_aud = wr_windows
                    else:
                        match_aud = wr_windows
                        mismatch_aud = wl_windows
                        
                    y_match = torch.full((num_windows,), 1.0, dtype=torch.float32)
                    sequences.append((e, match_aud, y_match))
                    
                    # Create the MISMATCH pair (Negative sampling)
                    y_mismatch = torch.full((num_windows,), 0.0, dtype=torch.float32)
                    sequences.append((e, mismatch_aud, y_mismatch))
                    
    return sequences

def main():
    possible_paths = [
        Path('/kaggle/input/datasets/lowkieee/wavlm-cache/kaggle/working/wavlm_cache'),
        Path('/kaggle/input/wavlm-cache/kaggle/working/wavlm_cache'),
        Path('/kaggle/input/aasd-universal-cache-v1/kaggle/working/wavlm_cache'),
        Path('/kaggle/input/wavlm-cache-full/kaggle/working/wavlm_cache'),
        Path('/kaggle/working/wavlm_cache')
    ]
    
    cache_dir = Path('/kaggle/working/wavlm_cache')
    for p in possible_paths:
        if p.exists() and len(list(p.glob('*_wavlm.pt'))) > 0:
            cache_dir = p
            break
            
    print(f"\n=======================================================")
    print(f" PHASE 106: WAVLM 50Hz NATIVE LAG SWEEP")
    print(f" No Interpolation. Strict Phase Alignment.")
    print(f"=======================================================\n", flush=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on Device: {device}\n", flush=True)
    
    cache_files = sorted(list(cache_dir.glob('*_wavlm.pt')))
    
    # We only test a subset of interesting subjects to keep runtime manageable
    TARGET_SUBJECTS = ['S10', 'S11', 'S12', 'S13']
    filtered_files = [f for f in cache_files if f.stem.split('_')[0] in TARGET_SUBJECTS]
    
    # Lags in samples at 50Hz (1 sample = 20ms)
    LAGS = [-4, -2, 0, 2, 4] # -80ms to +80ms in 40ms steps
    
    results = {subj: {} for subj in TARGET_SUBJECTS}
    
    for cache_file in filtered_files:
        subj_name = cache_file.stem.split('_')[0]
        print(f"\n=======================================================")
        print(f" SUBJECT {subj_name}")
        print(f"=======================================================", flush=True)
        
        cached = torch.load(cache_file, map_location='cpu', weights_only=False)['raw']
        
        # Pre-process raw data ONCE per subject
        raw_trials = []
        for i in range(len(cached)):
            tr = cached[i]
            eeg_128 = tr['eeg'].numpy()[EAR_CHANNEL_INDICES, :] 
            wavlm_l = tr['wavlm_l'].float().numpy() # [T, 768] (Native 50Hz)
            wavlm_r = tr['wavlm_r'].float().numpy()
            
            # Downsample EEG from 128Hz to exactly match WavLM 50Hz length ratio
            # Use resample_poly instead of FFT-based resample to preserve phase and prevent ringing
            eeg_50 = resample_poly(eeg_128, 50, 128, axis=1)
            
            # Normalize
            eeg_50 = (eeg_50 - np.mean(eeg_50, axis=1, keepdims=True)) / (np.std(eeg_50, axis=1, keepdims=True) + 1e-8)
            wavlm_l = (wavlm_l - np.mean(wavlm_l, axis=0, keepdims=True)) / (np.std(wavlm_l, axis=0, keepdims=True) + 1e-8)
            wavlm_r = (wavlm_r - np.mean(wavlm_r, axis=0, keepdims=True)) / (np.std(wavlm_r, axis=0, keepdims=True) + 1e-8)
            
            raw_trials.append({
                'eeg': eeg_50,
                'wavlm_l': wavlm_l,
                'wavlm_r': wavlm_r,
                'meta': tr['meta']
            })
            
        # TRIAL-LEVEL SPLIT (Consistent across all lags)
        random.seed(42)
        indices = list(range(len(raw_trials)))
        random.shuffle(indices)
        
        split_idx = int(len(raw_trials) * 0.8)
        train_indices = indices[:split_idx]
        eval_indices = indices[split_idx:]
        
        # Now sweep the lags
        for lag in LAGS:
            lag_ms = lag * 20
            print(f"\n--- Testing Lag: {lag_ms} ms ({lag} samples) ---", flush=True)
            
            lag_trials = []
            for tr_idx in indices:
                tr = raw_trials[tr_idx]
                e = tr['eeg']
                wl = tr['wavlm_l']
                wr = tr['wavlm_r']
                
                # Apply lag
                # POSITIVE LAG: audio is shifted LEFT (audio occurs EARLIER than EEG)
                # NEGATIVE LAG: audio is shifted RIGHT (audio occurs LATER than EEG)
                if lag > 0:
                    e_aligned = e[:, lag:]
                    wl_aligned = wl[:-lag, :]
                    wr_aligned = wr[:-lag, :]
                elif lag < 0:
                    abs_lag = abs(lag)
                    e_aligned = e[:, :-abs_lag]
                    wl_aligned = wl[abs_lag:, :]
                    wr_aligned = wr[abs_lag:, :]
                else:
                    e_aligned = e
                    wl_aligned = wl
                    wr_aligned = wr
                    
                min_len = min(e_aligned.shape[1], wl_aligned.shape[0])
                
                lag_trials.append({
                    'eeg': e_aligned[:, :min_len],
                    'wavlm_l': wl_aligned[:min_len, :],
                    'wavlm_r': wr_aligned[:min_len, :],
                    'meta': tr['meta'],
                    'original_idx': tr_idx
                })
                
            train_trials = [t for t in lag_trials if t['original_idx'] in train_indices]
            eval_trials = [t for t in lag_trials if t['original_idx'] in eval_indices]
            
            calib_pool = extract_match_mismatch_sequences_wavlm(train_trials)
            eval_set = extract_match_mismatch_sequences_wavlm(eval_trials)
            random.shuffle(calib_pool)
            
            if len(eval_set) == 0 or len(calib_pool) == 0:
                print("Not enough sequences extracted.")
                continue
                
            train_loader = DataLoader(MatchMismatchWavLMDataset(calib_pool), batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, num_workers=2)
            eval_loader = DataLoader(MatchMismatchWavLMDataset(eval_set), batch_size=BATCH_SIZE, shuffle=False, pin_memory=True, num_workers=2)
            
            model = DeepMatchMismatchTCN(
                eeg_channels=8, 
                latent_dim=64, 
                tcn_channels=[64, 64, 64], 
                kernel_size=2, 
                dropout=0.2,
                encoder_type='wavlm'
            ).to(device)
            
            optimizer = optim.Adam(model.parameters(), lr=TRAIN_LR, weight_decay=1e-4)
            criterion = nn.BCEWithLogitsLoss()
            scaler = torch.amp.GradScaler(device.type, enabled=(device.type == 'cuda'))
            
            best_auc = 0
            for epoch in range(TRAIN_EPOCHS):
                model.train()
                for b_e, b_a, b_y in train_loader:
                    b_e = b_e.to(device, non_blocking=True).float()
                    b_a = b_a.to(device, non_blocking=True).float()
                    b_y = b_y.to(device, non_blocking=True).float()
                    if b_e.size(0) == 1: continue 
                    optimizer.zero_grad()
                    
                    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=(device.type == 'cuda')):
                        logits, _ = model(b_e, b_a)
                        loss = criterion(logits, b_y)
                        
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                    
                model.eval()
                all_preds, all_labels = [], []
                with torch.no_grad():
                    for b_e, b_a, b_y in eval_loader:
                        b_e = b_e.to(device, non_blocking=True).float()
                        b_a = b_a.to(device, non_blocking=True).float()
                        
                        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=(device.type == 'cuda')):
                            logits, _ = model(b_e, b_a)
                            
                        all_preds.extend(torch.sigmoid(logits).cpu().numpy().flatten())
                        all_labels.extend(b_y.numpy().flatten())
                        
                if len(np.unique(all_labels)) > 1:
                    auc = roc_auc_score(all_labels, all_preds)
                    best_auc = max(best_auc, auc)
                    
            print(f"Lag {lag_ms} ms Final Best AUROC: {best_auc:.4f}")
            results[subj_name][lag_ms] = best_auc
            
    print("\n\n=======================================================")
    print(" PHASE 106 SWEEP RESULTS (Best AUROC)")
    print("=======================================================")
    print(f"{'Subject':<10} " + " ".join([f"{lag:6d}ms" for lag in [l*20 for l in LAGS]]))
    for subj in TARGET_SUBJECTS:
        if subj in results and len(results[subj]) > 0:
            row = f"{subj:<10} "
            for lag in [l*20 for l in LAGS]:
                row += f"{results[subj].get(lag, 0.0):.4f}   "
            print(row)

if __name__ == "__main__":
    main()
