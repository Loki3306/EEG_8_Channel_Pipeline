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

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.aad_tcn import DeepMatchMismatchTCN

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
                
            safe_start = start_idx + EXCLUSION_SAMPLES
            safe_end = end_idx
            
            if safe_end - safe_start >= SEQ_SAMPLES:
                for seq_start in range(safe_start, safe_end - SEQ_SAMPLES + 1, seq_hop):
                    e_seq = eeg[:, seq_start:seq_start + SEQ_SAMPLES]
                    wl_seq = wavlm_l[seq_start:seq_start + SEQ_SAMPLES, :]
                    wr_seq = wavlm_r[seq_start:seq_start + SEQ_SAMPLES, :]
                    
                    # Unfold EEG: [TimeWindows, Channels, WindowSamples]
                    e = torch.from_numpy(e_seq.copy()).unfold(-1, WIN_SAMPLES, HOP_SAMPLES).permute(1, 0, 2)
                    
                    # Unfold WavLM: [TimeWindows, WindowSamples, 768]
                    # Original shape: [SEQ_SAMPLES, 768]. Transpose -> [768, SEQ_SAMPLES] -> unfold -> transpose back
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
    print(f" PHASE 105: WAVLM SELF-SUPERVISED RESCUE SWEEP")
    print(f" DeepMatchMismatchTCN with WavLM Encoder on ALL Subjects")
    print(f"=======================================================\n", flush=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on Device: {device}\n", flush=True)
    
    cache_files = sorted(list(cache_dir.glob('*_wavlm.pt')))
    if len(cache_files) == 0:
        print(f"No wavlm cache files found in {cache_dir}. Please verify the Kaggle dataset path.")
        return
        
    print(f"Found {len(cache_files)} subjects in {cache_dir}.")
    
    for cache_file in cache_files:
        subj_name = cache_file.stem.split('_')[0]
        print(f"\n=======================================================")
        print(f" SUBJECT {subj_name}")
        print(f"=======================================================", flush=True)
        
        cached = torch.load(cache_file, map_location='cpu', weights_only=False)['raw']
        
        subj_trials = []
        for i in range(len(cached)):
            tr = cached[i]
            eeg = tr['eeg'].numpy()[EAR_CHANNEL_INDICES, :] 
            wavlm_l = tr['wavlm_l'].float() # [Time_WavLM, 768]
            wavlm_r = tr['wavlm_r'].float()
            
            # Interpolate WavLM to EEG length precisely (128Hz)
            target_length = eeg.shape[1]
            wavlm_l = torch.nn.functional.interpolate(wavlm_l.unsqueeze(0).transpose(1, 2), size=target_length, mode='linear', align_corners=False).transpose(1, 2).squeeze(0)
            wavlm_r = torch.nn.functional.interpolate(wavlm_r.unsqueeze(0).transpose(1, 2), size=target_length, mode='linear', align_corners=False).transpose(1, 2).squeeze(0)
            
            wavlm_l = (wavlm_l - wavlm_l.mean(dim=0, keepdim=True)) / (wavlm_l.std(dim=0, keepdim=True) + 1e-8)
            wavlm_r = (wavlm_r - wavlm_r.mean(dim=0, keepdim=True)) / (wavlm_r.std(dim=0, keepdim=True) + 1e-8)
            
            eeg = (eeg - np.mean(eeg, axis=1, keepdims=True)) / (np.std(eeg, axis=1, keepdims=True) + 1e-8)
            
            wavlm_l = wavlm_l.numpy()
            wavlm_r = wavlm_r.numpy()
            
            min_len = min(eeg.shape[1], wavlm_l.shape[0])
            
            subj_trials.append({
                'eeg': eeg[:, :min_len], 
                'wavlm_l': wavlm_l[:min_len, :], 
                'wavlm_r': wavlm_r[:min_len, :], 
                'meta': tr['meta']
            })
            
        # TRIAL-LEVEL SPLIT
        random.seed(42)
        random.shuffle(subj_trials)
        
        split_idx = int(len(subj_trials) * 0.8)
        train_trials = subj_trials[:split_idx]
        eval_trials = subj_trials[split_idx:]
        
        calib_pool = extract_match_mismatch_sequences_wavlm(train_trials)
        eval_set = extract_match_mismatch_sequences_wavlm(eval_trials)
        random.shuffle(calib_pool)
        
        if len(eval_set) == 0 or len(calib_pool) == 0:
            continue
            
        train_loader = DataLoader(MatchMismatchWavLMDataset(calib_pool), batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, num_workers=2)
        eval_loader = DataLoader(MatchMismatchWavLMDataset(eval_set), batch_size=BATCH_SIZE, shuffle=False, pin_memory=True, num_workers=2)
        
        # Phase 105 Model: DeepMatchMismatchTCN with WavLM Encoder
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
        scaler = torch.amp.GradScaler('cuda')
        
        print("Training DeepMatchMismatchTCN with WavLM Encoder...", flush=True)
        best_auc = 0
        
        for epoch in range(TRAIN_EPOCHS):
            model.train()
            for b_e, b_a, b_y in train_loader:
                b_e = b_e.to(device, non_blocking=True).float()
                b_a = b_a.to(device, non_blocking=True).float()
                b_y = b_y.to(device, non_blocking=True).float()
                if b_e.size(0) == 1: continue 
                optimizer.zero_grad()
                
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    logits, _ = model(b_e, b_a)
                    loss = criterion(logits, b_y)
                    
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                
            model.eval()
            all_preds, all_labels, all_logits = [], [], []
            with torch.no_grad():
                for b_e, b_a, b_y in eval_loader:
                    b_e = b_e.to(device, non_blocking=True).float()
                    b_a = b_a.to(device, non_blocking=True).float()
                    
                    with torch.autocast(device_type='cuda', dtype=torch.float16):
                        logits, _ = model(b_e, b_a)
                        
                    all_preds.extend(torch.sigmoid(logits).cpu().numpy().flatten())
                    all_labels.extend(b_y.numpy().flatten())
                    all_logits.extend(logits.cpu().numpy().flatten())
                    
            if len(np.unique(all_labels)) > 1:
                auc = roc_auc_score(all_labels, all_preds)
                best_auc = max(best_auc, auc)
                
                if auc < 0.5 and epoch == TRAIN_EPOCHS - 1:
                    inverted_auc = roc_auc_score(all_labels, -np.array(all_logits))
                    print(f"    -> [Diagnostic] INVERTED AUROC: {inverted_auc:.4f}")
            print(f"  Epoch {epoch+1:02d}/{TRAIN_EPOCHS} - Val AUROC: {auc:.4f} (Best: {best_auc:.4f})")
                
        print(f"Final Best AUROC: {best_auc:.4f}\n")

if __name__ == "__main__":
    main()
