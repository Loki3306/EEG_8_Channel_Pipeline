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
BATCH_SIZE = 64
TRAIN_EPOCHS = 15
TRAIN_LR = 1e-3

# Masking 200ms of audio = 25.6 samples (round to 26)
MASK_SAMPLES = int(0.200 * SR)

# Only evaluating the two archetypes to prove the point
EVAL_TARGETS = [
    (1, 'baseline'), # S01 (Slow/Semantic)
    (16, 'fast'),    # S16 (Fast/Transient)
    (3, 'fast'),     # S03 (Fast/Transient)
    (8, 'fast')      # S08 (Fast/Transient)
]

class StableAASDDataset(Dataset):
    def __init__(self, sequences):
        self.seqs = sequences
    def __len__(self): return len(self.seqs)
    def __getitem__(self, idx):
        e, a_a, a_b, y = self.seqs[idx]
        return e, a_a, a_b, y[-1]

def extract_stable_sequences(trials):
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
                    
                    e = e_seq.unfold(-1, WIN_SAMPLES, HOP_SAMPLES).permute(1, 0, 2)
                    a_a = al_seq.unfold(-1, WIN_SAMPLES, HOP_SAMPLES).permute(1, 0, 2)
                    a_b = ar_seq.unfold(-1, WIN_SAMPLES, HOP_SAMPLES).permute(1, 0, 2)
                    
                    num_windows = e.shape[0]
                    y = torch.full((num_windows,), label, dtype=torch.float32)
                    sequences.append((e, a_a, a_b, y))
    return sequences

def train_and_eval_saliency(subj_idx, encoder_type, train_loader, eval_loader, device):
    model = TCNAADModel(audio_channels=16, encoder_type=encoder_type).to(device)
    optimizer = optim.Adam(model.parameters(), lr=TRAIN_LR, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()
    scaler = torch.amp.GradScaler('cuda')
    
    print(f"\n  [Training S{subj_idx:02d} with {encoder_type.upper()} Encoder]")
    for epoch in range(TRAIN_EPOCHS):
        model.train()
        for b_e, b_a_a, b_a_b, b_y in train_loader:
            b_e = b_e.to(device, non_blocking=True)
            b_a_a = b_a_a.to(device, non_blocking=True)
            b_a_b = b_a_b.to(device, non_blocking=True)
            b_y = b_y.to(device, non_blocking=True)
            if b_e.size(0) == 1: continue 
            optimizer.zero_grad()
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                logits, _ = model(b_e, b_a_a, b_a_b)
                loss = criterion(logits, b_y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
    # Clean Evaluation
    model.eval()
    all_preds_clean, all_labels = [], []
    with torch.no_grad():
        for b_e, b_a_a, b_a_b, b_y in eval_loader:
            b_e = b_e.to(device, non_blocking=True)
            b_a_a = b_a_a.to(device, non_blocking=True)
            b_a_b = b_a_b.to(device, non_blocking=True)
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                logits, _ = model(b_e, b_a_a, b_a_b)
            all_preds_clean.extend(torch.sigmoid(logits).cpu().numpy().flatten())
            all_labels.extend(b_y.numpy().flatten())
            
    auc_clean = roc_auc_score(all_labels, all_preds_clean) if len(np.unique(all_labels)) > 1 else 0.5
    
    # Masked Evaluation (Zero out first 200ms of audio windows)
    all_preds_masked = []
    with torch.no_grad():
        for b_e, b_a_a, b_a_b, b_y in eval_loader:
            b_e = b_e.to(device, non_blocking=True)
            b_a_a = b_a_a.to(device, non_blocking=True)
            b_a_b = b_a_b.to(device, non_blocking=True)
            
            # MASKING OPERATION:
            # b_a_a shape is [Batch, SeqLen, Channels, Time]
            # We zero out the first MASK_SAMPLES in the Time dimension
            b_a_a[:, :, :, :MASK_SAMPLES] = 0.0
            b_a_b[:, :, :, :MASK_SAMPLES] = 0.0
            
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                logits, _ = model(b_e, b_a_a, b_a_b)
            all_preds_masked.extend(torch.sigmoid(logits).cpu().numpy().flatten())
            
    auc_masked = roc_auc_score(all_labels, all_preds_masked) if len(np.unique(all_labels)) > 1 else 0.5
    
    return auc_clean, auc_masked

def main():
    cache_dir = Path('/kaggle/input/datasets/lowkieee/aasd-universal-cache-v1/kaggle/working/multiband_cache')
    if not cache_dir.exists(): cache_dir = Path('/kaggle/working/multiband_cache')
        
    cache_files = sorted(list(cache_dir.glob('*_multiband.pt')))
    
    if len(cache_files) == 0:
        print("No Multiband cache found!")
        return
        
    print("\n=======================================================")
    print(" PHASE 94: TEMPORAL SALIENCY FALSIFICATION (TRACK A)")
    print(" Hypotheses:")
    print(" S16 (Fast) is doing Spatial Decoding (relies on onset transients).")
    print(" S01 (Slow) is doing True AAD (relies on sustained semantics).")
    print(" Protocol:")
    print(" We zero out the first 200ms of every audio window during evaluation.")
    print(" If S16 collapses, we prove S16 relies on the early spatial onset.")
    print("=======================================================\n", flush=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on Device: {device}\n", flush=True)
    
    for subj_idx, optimal_encoder in EVAL_TARGETS:
        if subj_idx >= len(cache_files): continue
        print(f"\n==================== SUBJECT {subj_idx:02d} ====================", flush=True)
        
        cached_data = torch.load(cache_files[subj_idx], map_location='cpu', weights_only=False)
        trials = cached_data['raw']
        
        subj_trials = []
        for tr in trials:
            eeg = tr['eeg'].numpy()
            eeg = eeg[EAR_CHANNEL_INDICES, :] 
            eeg = (eeg - np.mean(eeg, axis=1, keepdims=True)) / (np.std(eeg, axis=1, keepdims=True) + 1e-8)
            subj_trials.append({
                'eeg': torch.from_numpy(eeg).float(), 
                'env_l': tr['env_l'].float(), 
                'env_r': tr['env_r'].float(), 
                'meta': tr['meta']
            })
            
        seqs = extract_stable_sequences(subj_trials)
        split_idx = int(len(seqs) * 0.8)
        calib_pool = seqs[:split_idx]
        eval_set = seqs[split_idx:]
        
        if len(eval_set) == 0 or len(calib_pool) == 0: continue
            
        train_loader = DataLoader(StableAASDDataset(calib_pool), batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, num_workers=0)
        eval_loader = DataLoader(StableAASDDataset(eval_set), batch_size=BATCH_SIZE, shuffle=False, pin_memory=True, num_workers=0)
        
        auc_clean, auc_masked = train_and_eval_saliency(subj_idx, optimal_encoder, train_loader, eval_loader, device)
        
        print(f"  Clean AUROC      : {auc_clean:.4f}")
        print(f"  Masked AUROC     : {auc_masked:.4f}")
        
        drop = auc_clean - auc_masked
        if drop > 0.10:
            print(f"  -> {drop*100:.1f}% COLLAPSE! Subject relied on spatial onsets.", flush=True)
        elif drop < 0.05:
            print(f"  -> SURVIVED (-{drop*100:.1f}%). Subject relied on sustained semantics.", flush=True)
        else:
            print(f"  -> MODERATE DROP (-{drop*100:.1f}%). Subject uses mixed strategies.", flush=True)

if __name__ == "__main__":
    main()
