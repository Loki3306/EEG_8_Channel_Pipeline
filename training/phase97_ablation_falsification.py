import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pathlib import Path
from scipy import signal
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.aad_tcn import TCNAADModel

# Constants
SR = 64
WIN_LEN = 3.5
HOP_LEN = 0.5
WIN_SAMPLES = int(WIN_LEN * SR)
HOP_SAMPLES = int(HOP_LEN * SR)

TRAIN_EPOCHS = 10
TRAIN_LR = 1e-3
BATCH_SIZE = 32

SUBJECT = 13

# HPF > 16.0 Hz
LOWCUT = 16.0
HIGHCUT = None

def apply_modulation_filter(env, lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    if lowcut is None and highcut is not None:
        b, a = signal.butter(order, highcut / nyq, btype='low')
    elif highcut is None and lowcut is not None:
        b, a = signal.butter(order, lowcut / nyq, btype='high')
    else:
        low = lowcut / nyq
        high = highcut / nyq
        b, a = signal.butter(order, [low, high], btype='band')
        
    filtered = signal.filtfilt(b, a, env, axis=1)
    return filtered

def extract_sequences(trials):
    b_e, b_a, b_b, b_y = [], [], [], []
    for tr in trials:
        eeg = tr['eeg'].numpy()
        env_l = tr['env_l'].numpy()
        env_r = tr['env_r'].numpy()
        
        # Normalize BEFORE filtering
        eeg = (eeg - np.mean(eeg, axis=1, keepdims=True)) / (np.std(eeg, axis=1, keepdims=True) + 1e-8)
        env_l = (env_l - np.mean(env_l, axis=1, keepdims=True)) / (np.std(env_l, axis=1, keepdims=True) + 1e-8)
        env_r = (env_r - np.mean(env_r, axis=1, keepdims=True)) / (np.std(env_r, axis=1, keepdims=True) + 1e-8)

        # Apply Temporal Modulation Filter
        env_l = apply_modulation_filter(env_l, LOWCUT, HIGHCUT, SR)
        env_r = apply_modulation_filter(env_r, LOWCUT, HIGHCUT, SR)
        
        min_len = min(eeg.shape[1], env_l.shape[1])
        eeg = eeg[:, :min_len]
        env_l = env_l[:, :min_len]
        env_r = env_r[:, :min_len]
        
        boundaries = tr['stable_attention_boundaries']
        
        for start_sec, end_sec in boundaries:
            start_samp = int(start_sec * SR)
            end_samp = int(end_sec * SR)
            
            if end_samp > min_len: end_samp = min_len
            
            seq_len = end_samp - start_samp
            if seq_len < WIN_SAMPLES: continue
            
            seq_hop = int(0.5 * SR) 
            
            for seq_start in range(start_samp, end_samp - WIN_SAMPLES + 1, seq_hop):
                SEQ_SAMPLES = WIN_SAMPLES
                e_seq = eeg[:, seq_start:seq_start + SEQ_SAMPLES]
                al_seq = env_l[:, seq_start:seq_start + SEQ_SAMPLES]
                ar_seq = env_r[:, seq_start:seq_start + SEQ_SAMPLES]
                
                e = torch.from_numpy(e_seq.copy()).unfold(-1, WIN_SAMPLES, HOP_SAMPLES).permute(1, 0, 2)
                al = torch.from_numpy(al_seq.copy()).unfold(-1, WIN_SAMPLES, HOP_SAMPLES).permute(1, 0, 2)
                ar = torch.from_numpy(ar_seq.copy()).unfold(-1, WIN_SAMPLES, HOP_SAMPLES).permute(1, 0, 2)
                
                label = 1.0 if tr['attended_ear'] == 'L' else 0.0
                num_windows = e.shape[0]
                y = torch.full((num_windows,), label, dtype=torch.float32)
                
                b_e.append(e)
                b_a.append(al)
                b_b.append(ar)
                b_y.append(y)
                
    return (
        torch.cat(b_e, dim=0),
        torch.cat(b_a, dim=0),
        torch.cat(b_b, dim=0),
        torch.cat(b_y, dim=0)
    )

def main():
    possible_paths = [
        Path('/kaggle/input/datasets/lowkieee/multiband-cache/kaggle/working/multiband_cache'),
        Path('/kaggle/input/multiband-cache/kaggle/working/multiband_cache'),
        Path('/kaggle/input/aasd-universal-cache-v1/kaggle/working/multiband_cache'),
        Path('/kaggle/working/multiband_cache')
    ]
    
    cache_dir = Path('/kaggle/working/multiband_cache')
    for p in possible_paths:
        if p.exists():
            cache_dir = p
            break
            
    # We will just evaluate S13
    cache_file = cache_dir / f"S{SUBJECT}_multiband.pt"
    if not cache_file.exists():
        print(f"Missing cache file! {cache_file}")
        return
        
    print(f"\n=======================================================")
    print(f" PHASE 97: ABLATION FALSIFICATION SUITE")
    print(f" Testing S{SUBJECT} on HPF > 16.0 Hz")
    print(f"=======================================================\n")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on Device: {device}\n")
    
    data = torch.load(cache_file, map_location='cpu')
    if isinstance(data, list):
        trials = data
    else:
        trials = data['trials']
    
    split_idx = int(len(trials) * 0.8)
    train_trials = trials[:split_idx]
    eval_trials = trials[split_idx:]
    
    print("Extracting Training Sequences...")
    tr_e, tr_a, tr_b, tr_y = extract_sequences(train_trials)
    train_loader = DataLoader(TensorDataset(tr_e, tr_a, tr_b, tr_y), batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    
    print("Extracting Evaluation Sequences...")
    ev_e, ev_a, ev_b, ev_y = extract_sequences(eval_trials)
    eval_loader = DataLoader(TensorDataset(ev_e, ev_a, ev_b, ev_y), batch_size=BATCH_SIZE, shuffle=False)
    
    model = TCNAADModel(encoder_type='baseline', audio_channels=16, use_wavlm=False, attention_type='none').to(device)
    optimizer = optim.Adam(model.parameters(), lr=TRAIN_LR, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()
    scaler = torch.amp.GradScaler('cuda')
    
    print(f"\nTraining Baseline Model ({TRAIN_EPOCHS} Epochs)...")
    for epoch in range(TRAIN_EPOCHS):
        model.train()
        for b_e, b_a, b_b, b_y in train_loader:
            b_e = b_e.to(device, non_blocking=True).float()
            b_a = b_a.to(device, non_blocking=True).float()
            b_b = b_b.to(device, non_blocking=True).float()
            b_y = b_y.to(device, non_blocking=True).float()
            if b_e.size(0) == 1: continue 
            optimizer.zero_grad()
            
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                logits, _ = model(b_e, b_a, b_b)
                loss = criterion(logits, b_y)
                
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
    print("\n--- Training Complete. Running Ablations ---\n")
    model.eval()
    
    ablations = [
        "Baseline",
        "Zero EEG",
        "Shuffled EEG",
        "Zero Audio",
        "Shuffled Audio",
        "Swapped Audio"
    ]
    
    for ablation in ablations:
        all_preds, all_labels = [], []
        with torch.no_grad():
            for b_e, b_a, b_b, b_y in eval_loader:
                
                # Apply Ablation Logic
                if ablation == "Zero EEG":
                    b_e = torch.zeros_like(b_e)
                elif ablation == "Shuffled EEG":
                    idx = torch.randperm(b_e.size(0))
                    b_e = b_e[idx]
                elif ablation == "Zero Audio":
                    b_a = torch.zeros_like(b_a)
                    b_b = torch.zeros_like(b_b)
                elif ablation == "Shuffled Audio":
                    idx = torch.randperm(b_a.size(0))
                    b_a = b_a[idx]
                    b_b = b_b[idx]
                elif ablation == "Swapped Audio":
                    tmp = b_a.clone()
                    b_a = b_b
                    b_b = tmp

                b_e = b_e.to(device, non_blocking=True).float()
                b_a = b_a.to(device, non_blocking=True).float()
                b_b = b_b.to(device, non_blocking=True).float()
                
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    logits, _ = model(b_e, b_a, b_b)
                    
                all_preds.extend(torch.sigmoid(logits).cpu().numpy().flatten())
                all_labels.extend(b_y.numpy().flatten())
        
        auc = roc_auc_score(all_labels, all_preds)
        print(f"[{ablation:15s}] AUROC : {auc:.4f}")

if __name__ == '__main__':
    main()
