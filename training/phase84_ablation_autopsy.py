import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score
import numpy as np
from pathlib import Path
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.aad_tcn import TCNAADModel

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
TRAIN_EPOCHS = 15
TRAIN_LR = 1e-3

# We evaluate S11 (The Acoustic Failure) and S16 (The Acoustic Master)
EVAL_SUBJECTS = [11, 16]

class StableAASDDataset(Dataset):
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
        env_l = tr['env_l'].numpy()
        env_r = tr['env_r'].numpy()
        
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
                    
                    e = e_seq.unfold(-1, WIN_SAMPLES, HOP_SAMPLES).permute(1, 0, 2)
                    al = torch.from_numpy(al_seq).unfold(-1, WIN_SAMPLES, HOP_SAMPLES).permute(1, 0, 2)
                    ar = torch.from_numpy(ar_seq).unfold(-1, WIN_SAMPLES, HOP_SAMPLES).permute(1, 0, 2)
                    
                    num_windows = e.shape[0]
                    y = torch.full((num_windows,), label, dtype=torch.float32)
                    
                    sequences.append((e, al, ar, y))
    return sequences

def evaluate_ablation(model, eval_loader, device, mask_eeg_channel=None, mask_audio_band=None):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for b_e, b_a, b_b, b_y in eval_loader:
            if mask_eeg_channel is not None:
                b_e = b_e.clone()
                b_e[:, :, mask_eeg_channel, :] = 0.0 # Zero out the specific EEG channel
            
            if mask_audio_band is not None:
                b_a = b_a.clone()
                b_b = b_b.clone()
                b_a[:, :, mask_audio_band, :] = 0.0 # Zero out the specific Audio frequency band
                b_b[:, :, mask_audio_band, :] = 0.0
                
            b_e = b_e.to(device, non_blocking=True)
            b_a = b_a.to(device, non_blocking=True)
            b_b = b_b.to(device, non_blocking=True)
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                logits, _ = model(b_e, b_a, b_b)
                
            all_preds.extend(torch.sigmoid(logits).cpu().numpy().flatten())
            all_labels.extend(b_y.numpy().flatten())
    
    if len(np.unique(all_labels)) > 1:
        return roc_auc_score(all_labels, all_preds)
    return 0.5

def main():
    cache_dir = Path('/kaggle/working/multiband_cache')
    if not cache_dir.exists():
        cache_dir = Path('/kaggle/input/multiband-cache/multiband_cache') # Fallback if they mounted it
    if not cache_dir.exists():
        cache_dir = Path('/kaggle/input/datasets/lowkieee/aasd-universal-cache-v1/kaggle/working/multiband_cache')
        
    cache_files = sorted(list(cache_dir.glob('*_multiband.pt')))
    
    if len(cache_files) == 0:
        print(f"Missing cache files in {cache_dir}!")
        return
        
    print("\n=======================================================")
    print(" PHASE 84: ABLATION AUTOPSY (S11 vs S16)")
    print(" Revealing the spatial & spectral causes of heterogeneity.")
    print("=======================================================\n", flush=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n", flush=True)
    
    for subj_idx in EVAL_SUBJECTS:
        if subj_idx >= len(cache_files): continue
        print(f"\n==================== SUBJECT {subj_idx:02d} ====================", flush=True)
        
        cached_data = torch.load(cache_files[subj_idx], map_location='cpu', weights_only=False)['raw']
        
        subj_trials = []
        for tr in cached_data:
            eeg = tr['eeg'].numpy()[EAR_CHANNEL_INDICES, :]
            eeg = (eeg - np.mean(eeg, axis=1, keepdims=True)) / (np.std(eeg, axis=1, keepdims=True) + 1e-8)
            
            subj_trials.append({
                'eeg': torch.from_numpy(eeg).float(), 
                'env_l': tr['env_l'].float(), 
                'env_r': tr['env_r'].float(), 
                'meta': tr['meta']
            })
            
        seqs = extract_sequences(subj_trials)
        split_idx = int(len(seqs) * 0.8)
        calib_pool = seqs[:split_idx]
        eval_set = seqs[split_idx:]
        
        if len(eval_set) == 0 or len(calib_pool) == 0: continue
            
        train_loader = DataLoader(StableAASDDataset(calib_pool), batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, num_workers=2)
        eval_loader = DataLoader(StableAASDDataset(eval_set), batch_size=BATCH_SIZE, shuffle=False, pin_memory=True, num_workers=2)
        
        model = TCNAADModel(eeg_channels=8, latent_dim=64, tcn_channels=[64, 64, 64], kernel_size=2, dropout=0.3, use_wavlm=False, audio_channels=16).to(device)
        optimizer = optim.Adam(model.parameters(), lr=TRAIN_LR, weight_decay=1e-4)
        criterion = nn.BCEWithLogitsLoss()
        scaler = torch.amp.GradScaler('cuda')
        
        print("  [1] Training Multiband Baseline...")
        best_state = None
        best_auc = 0
        for epoch in range(TRAIN_EPOCHS):
            model.train()
            for b_e, b_a, b_b, b_y in train_loader:
                b_e, b_a, b_b, b_y = b_e.to(device), b_a.to(device), b_b.to(device), b_y.to(device)
                if b_e.size(0) == 1: continue 
                optimizer.zero_grad()
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    logits, _ = model(b_e, b_a, b_b)
                    loss = criterion(logits, b_y)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            
            auc = evaluate_ablation(model, eval_loader, device)
            if auc > best_auc:
                best_auc = auc
                best_state = model.state_dict()
                
        model.load_state_dict(best_state)
        baseline_auc = evaluate_ablation(model, eval_loader, device)
        print(f"  -> Baseline AUROC: {baseline_auc:.4f}\n")
        
        print("  [2] EEG Channel Ablation (Spatial Heterogeneity)")
        for c in range(8):
            auc = evaluate_ablation(model, eval_loader, device, mask_eeg_channel=c)
            drop = baseline_auc - auc
            print(f"      Mask Ch {c:02d} -> AUROC: {auc:.4f}  (Drop: {drop:+.4f})")
            
        print("\n  [3] Audio Band Ablation (Spectral Heterogeneity)")
        for b in range(16):
            auc = evaluate_ablation(model, eval_loader, device, mask_audio_band=b)
            drop = baseline_auc - auc
            print(f"      Mask Band {b:02d} -> AUROC: {auc:.4f}  (Drop: {drop:+.4f})")

if __name__ == "__main__":
    main()
