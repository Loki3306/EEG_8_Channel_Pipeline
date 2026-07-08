import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score
import numpy as np
from scipy import signal
from pathlib import Path
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.aad_tcn import TCNAADModel

# -------------------------------------------------------------------------
# CONSTANTS
# -------------------------------------------------------------------------
SR = 128
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
TRAIN_EPOCHS = 10
TRAIN_LR = 1e-3

# We evaluate S01 (Slow) and S16 (Fast) to verify the heterogeneous spectral hypothesis
EVAL_SUBJECTS = [1, 16] 

# Modulation Bands (Hz)
MODULATION_BANDS = [
    # Cumulative Low-Pass
    ("LPF < 2.0 Hz", None, 2.0),
    ("LPF < 4.0 Hz", None, 4.0),
    ("LPF < 8.0 Hz", None, 8.0),
    ("LPF < 16.0 Hz", None, 16.0),
    # Cumulative High-Pass
    ("HPF > 2.0 Hz", 2.0, None),
    ("HPF > 4.0 Hz", 4.0, None),
    ("HPF > 8.0 Hz", 8.0, None),
    ("HPF > 16.0 Hz", 16.0, None),
    # Band-Pass Sweeps
    ("Band 0.5-2.0 Hz", 0.5, 2.0),
    ("Band 2.0-4.0 Hz", 2.0, 4.0),
    ("Band 4.0-8.0 Hz", 4.0, 8.0),
    ("Band 8.0-16.0 Hz", 8.0, 16.0),
    ("Band 16.0-32.0 Hz", 16.0, 32.0)
]

def apply_modulation_filter(env, lowcut, highcut, fs, order=4):
    """
    Applies a zero-phase Butterworth bandpass filter across the time dimension
    to isolate specific temporal modulation frequencies in the audio envelope.
    env: numpy array of shape [Channels, Time]
    """
    nyq = 0.5 * fs
    if lowcut is None and highcut is not None:
        b, a = signal.butter(order, highcut / nyq, btype='low')
    elif highcut is None and lowcut is not None:
        b, a = signal.butter(order, lowcut / nyq, btype='high')
    else:
        low = lowcut / nyq
        high = highcut / nyq
        b, a = signal.butter(order, [low, high], btype='band')
        
    # Use filtfilt for zero-phase distortion (preserves phase alignment for AAD)
    filtered = signal.filtfilt(b, a, env, axis=1)
    return filtered

class SpectralDataset(Dataset):
    def __init__(self, sequences):
        self.seqs = sequences
    def __len__(self): return len(self.seqs)
    def __getitem__(self, idx):
        e, a, b, y = self.seqs[idx]
        return e, a, b, y[-1]

def extract_sequences(trials):
    sequences = []
    seq_hop = int(0.5 * SR) 
    
    for tr in trials:
        eeg = tr['eeg']
        env_l = tr['env_l'] # [16, Time]
        env_r = tr['env_r']
        sp = tr['meta']['switch_points']
        T = eeg.shape[1]
        
        boundaries = [0]
        boundaries.extend([idx for spk, idx in sp])
        boundaries = sorted(set(boundaries))
        if boundaries[-1] != T: boundaries.append(T)
            
        for i in range(len(boundaries) - 1):
            start_idx = boundaries[i]
            end_idx = boundaries[i+1]
            current_spk = 'L'
            for spk, idx in sp:
                if idx <= start_idx: current_spk = spk
                else: break
                
            label = 1.0 if current_spk == 'L' else 0.0
            
            safe_start = start_idx + EXCLUSION_SAMPLES
            safe_end = end_idx
            
            if safe_end - safe_start >= SEQ_SAMPLES:
                for seq_start in range(safe_start, safe_end - SEQ_SAMPLES + 1, seq_hop):
                    e_seq = eeg[:, seq_start:seq_start + SEQ_SAMPLES]
                    al_seq = env_l[:, seq_start:seq_start + SEQ_SAMPLES]
                    ar_seq = env_r[:, seq_start:seq_start + SEQ_SAMPLES]
                    
                    e = torch.from_numpy(e_seq.copy()).unfold(-1, WIN_SAMPLES, HOP_SAMPLES).permute(1, 0, 2)
                    al = torch.from_numpy(al_seq.copy()).unfold(-1, WIN_SAMPLES, HOP_SAMPLES).permute(1, 0, 2)
                    ar = torch.from_numpy(ar_seq.copy()).unfold(-1, WIN_SAMPLES, HOP_SAMPLES).permute(1, 0, 2)
                    
                    num_windows = e.shape[0]
                    y = torch.full((num_windows,), label, dtype=torch.float32)
                    
                    sequences.append((e, al, ar, y))
    return sequences

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
            
    cache_files = sorted(list(cache_dir.glob('*_multiband.pt')))
    
    if len(cache_files) == 0:
        print("Missing cache files!")
        return
        
    print("\n=======================================================")
    print(" PHASE 96: SPECTRAL DECOMPOSITION CALIBRATION")
    print(" Hypothesis: S16 relies on High Modulations (>8Hz)")
    print("             S01 relies on Low Modulations (<4Hz)")
    print(" Protocol: Run the exact same TCN baseline across 5 filter bands.")
    print("=======================================================\n", flush=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on Device: {device}\n", flush=True)
    
    for subj_idx in EVAL_SUBJECTS:
        if subj_idx >= len(cache_files): continue
        print(f"\n==================== SUBJECT {subj_idx:02d} ====================", flush=True)
        
        cached = torch.load(cache_files[subj_idx], map_location='cpu', weights_only=False)['raw']
        
        for band_name, lowcut, highcut in MODULATION_BANDS:
            print(f"\n  --- Testing Band: {band_name} ---", flush=True)
            
            subj_trials = []
            for i in range(len(cached)):
                tr = cached[i]
                eeg = tr['eeg'].numpy()[EAR_CHANNEL_INDICES, :] 
                env_l = tr['env_l'].numpy()
                env_r = tr['env_r'].numpy()
                
                # IMPORTANT: Normalize BEFORE filtering to preserve the relative physical energy of the filtered bands!
                eeg = (eeg - np.mean(eeg, axis=1, keepdims=True)) / (np.std(eeg, axis=1, keepdims=True) + 1e-8)
                env_l = (env_l - np.mean(env_l, axis=1, keepdims=True)) / (np.std(env_l, axis=1, keepdims=True) + 1e-8)
                env_r = (env_r - np.mean(env_r, axis=1, keepdims=True)) / (np.std(env_r, axis=1, keepdims=True) + 1e-8)

                # Apply Temporal Modulation Filter
                env_l = apply_modulation_filter(env_l, lowcut, highcut, SR)
                env_r = apply_modulation_filter(env_r, lowcut, highcut, SR)
                
                min_len = min(eeg.shape[1], env_l.shape[1])
                
                subj_trials.append({
                    'eeg': eeg[:, :min_len], 
                    'env_l': env_l[:, :min_len], 
                    'env_r': env_r[:, :min_len], 
                    'meta': tr['meta']
                })
                
            seqs = extract_sequences(subj_trials)
            split_idx = int(len(seqs) * 0.8)
            calib_pool = seqs[:split_idx]
            eval_set = seqs[split_idx:]
            
            if len(eval_set) == 0 or len(calib_pool) == 0: continue
                
            train_loader = DataLoader(SpectralDataset(calib_pool), batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, num_workers=2)
            eval_loader = DataLoader(SpectralDataset(eval_set), batch_size=BATCH_SIZE, shuffle=False, pin_memory=True, num_workers=2)
            
            model = TCNAADModel(encoder_type='baseline', audio_channels=16, use_wavlm=False, attention_type='none').to(device)
            optimizer = optim.Adam(model.parameters(), lr=TRAIN_LR, weight_decay=1e-4)
            criterion = nn.BCEWithLogitsLoss()
            scaler = torch.amp.GradScaler('cuda')
            
            best_auc = 0
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
                
                model.eval()
                all_preds, all_labels = [], []
                with torch.no_grad():
                    for b_e, b_a, b_b, b_y in eval_loader:
                        b_e = b_e.to(device, non_blocking=True).float()
                        b_a = b_a.to(device, non_blocking=True).float()
                        b_b = b_b.to(device, non_blocking=True).float()
                        
                        with torch.autocast(device_type='cuda', dtype=torch.float16):
                            logits, _ = model(b_e, b_a, b_b)
                            
                        all_preds.extend(torch.sigmoid(logits).cpu().numpy().flatten())
                        all_labels.extend(b_y.numpy().flatten())
                
                if len(np.unique(all_labels)) > 1:
                    auc = roc_auc_score(all_labels, all_preds)
                    best_auc = max(best_auc, auc)
                    
            print(f"  -> AUROC for {band_name:25s}: {best_auc:.4f}", flush=True)

if __name__ == "__main__":
    main()
