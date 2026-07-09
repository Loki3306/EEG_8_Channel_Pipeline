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
# CONSTANTS (Copied exactly from Phase 96)
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

SUBJECT = 13
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
        env_l = tr['env_l'] 
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
            
    cache_file = cache_dir / f"S{SUBJECT}_multiband.pt"
    if not cache_file.exists():
        print(f"Missing cache file! {cache_file}")
        return
        
    print(f"\n=======================================================")
    print(f" PHASE 97: ABLATION FALSIFICATION SUITE")
    print(f" Testing S{SUBJECT} on HPF > 16.0 Hz")
    print(f"=======================================================\n", flush=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on Device: {device}\n", flush=True)
    
    cached = torch.load(cache_file, map_location='cpu', weights_only=False)['raw']
    
    subj_trials = []
    for i in range(len(cached)):
        tr = cached[i]
        eeg = tr['eeg'].numpy()[EAR_CHANNEL_INDICES, :] 
        env_l = tr['env_l'].numpy()
        env_r = tr['env_r'].numpy()
        
        # Normalize BEFORE filtering to preserve the relative physical energy
        eeg = (eeg - np.mean(eeg, axis=1, keepdims=True)) / (np.std(eeg, axis=1, keepdims=True) + 1e-8)
        env_l = (env_l - np.mean(env_l, axis=1, keepdims=True)) / (np.std(env_l, axis=1, keepdims=True) + 1e-8)
        env_r = (env_r - np.mean(env_r, axis=1, keepdims=True)) / (np.std(env_r, axis=1, keepdims=True) + 1e-8)

        # Apply Temporal Modulation Filter
        env_l = apply_modulation_filter(env_l, LOWCUT, HIGHCUT, SR)
        env_r = apply_modulation_filter(env_r, LOWCUT, HIGHCUT, SR)
        
        min_len = min(eeg.shape[1], env_l.shape[1])
        
        subj_trials.append({
            'eeg': eeg[:, :min_len], 
            'env_l': env_l[:, :min_len], 
            'env_r': env_r[:, :min_len], 
            'meta': tr['meta']
        })
        
    print("Extracting Sequences...")
    seqs = extract_sequences(subj_trials)
    split_idx = int(len(seqs) * 0.8)
    calib_pool = seqs[:split_idx]
    eval_set = seqs[split_idx:]
    
    train_loader = DataLoader(SpectralDataset(calib_pool), batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, num_workers=2)
    # Important: eval_loader should NOT shuffle initially, but we might want to pre-load all batches for ablation tests
    eval_loader = DataLoader(SpectralDataset(eval_set), batch_size=BATCH_SIZE, shuffle=False, pin_memory=True, num_workers=2)
    
    model = TCNAADModel(encoder_type='baseline', audio_channels=16, use_wavlm=False, attention_type='none').to(device)
    optimizer = optim.Adam(model.parameters(), lr=TRAIN_LR, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()
    scaler = torch.amp.GradScaler('cuda')
    
    print(f"\nTraining Baseline Model ({TRAIN_EPOCHS} Epochs)...")
    best_auc = 0
    best_model_state = None
    
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
            if auc > best_auc:
                best_auc = auc
                best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            print(f"  Epoch {epoch+1:02d}/{TRAIN_EPOCHS} - Val AUROC: {auc:.4f} (Best: {best_auc:.4f})")
            
    print("\n--- Training Complete. Restoring Best Model & Running Ablations ---\n")
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        
    model.eval()
    
    ablations = [
        "Baseline",
        "Zero EEG",
        "Shuffled EEG",
        "Zero Audio",
        "Shuffled Audio",
        "Swapped Audio",
        "Gaussian EEG"
    ]
    
    # We load everything into RAM for perfect global shuffling (as GPT requested)
    eval_e, eval_a, eval_b, eval_y = [], [], [], []
    for b_e, b_a, b_b, b_y in eval_loader:
        eval_e.append(b_e)
        eval_a.append(b_a)
        eval_b.append(b_b)
        eval_y.append(b_y)
        
    eval_e = torch.cat(eval_e, dim=0)
    eval_a = torch.cat(eval_a, dim=0)
    eval_b = torch.cat(eval_b, dim=0)
    eval_y = torch.cat(eval_y, dim=0)
    
    N = eval_e.size(0)
    
    for ablation in ablations:
        b_e = eval_e.clone()
        b_a = eval_a.clone()
        b_b = eval_b.clone()
        
        # Apply Ablation Logic across the ENTIRE dataset (as GPT requested)
        if ablation == "Zero EEG":
            b_e = torch.zeros_like(b_e)
        elif ablation == "Shuffled EEG":
            idx = torch.randperm(N)
            b_e = b_e[idx]
        elif ablation == "Zero Audio":
            b_a = torch.zeros_like(b_a)
            b_b = torch.zeros_like(b_b)
        elif ablation == "Shuffled Audio":
            idx = torch.randperm(N)
            b_a = b_a[idx]
            b_b = b_b[idx]
        elif ablation == "Swapped Audio":
            tmp = b_a.clone()
            b_a = b_b
            b_b = tmp
        elif ablation == "Gaussian EEG":
            mean = b_e.mean()
            std = b_e.std()
            b_e = torch.randn_like(b_e) * std + mean
            
        # Evaluate in batches
        all_preds, all_labels = [], []
        with torch.no_grad():
            for i in range(0, N, BATCH_SIZE):
                batch_e = b_e[i:i+BATCH_SIZE].to(device, non_blocking=True).float()
                batch_a = b_a[i:i+BATCH_SIZE].to(device, non_blocking=True).float()
                batch_b = b_b[i:i+BATCH_SIZE].to(device, non_blocking=True).float()
                batch_y = eval_y[i:i+BATCH_SIZE]
                
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    logits, _ = model(batch_e, batch_a, batch_b)
                    
                all_preds.extend(torch.sigmoid(logits).cpu().numpy().flatten())
                all_labels.extend(batch_y.numpy().flatten())
        
        auc = roc_auc_score(all_labels, all_preds)
        print(f"[{ablation:15s}] AUROC : {auc:.4f}")

if __name__ == "__main__":
    main()
