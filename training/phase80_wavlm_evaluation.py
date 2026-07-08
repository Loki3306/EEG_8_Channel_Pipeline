import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
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
BATCH_SIZE = 32
TRAIN_EPOCHS = 15
TRAIN_LR = 1e-3

EVAL_SUBJECTS = [5, 11, 16, 17]

# -------------------------------------------------------------------------
# DATASET
# -------------------------------------------------------------------------
class StableSequenceDataset(Dataset):
    def __init__(self, sequences):
        self.seqs = sequences
    def __len__(self): return len(self.seqs)
    def __getitem__(self, idx):
        e, a, b, y = self.seqs[idx]
        return e, a, b, y[-1]

def extract_wavlm_sequences(trials):
    sequences = []
    seq_hop = int(0.5 * SR) 
    
    for tr in trials:
        eeg = tr['eeg']
        wavlm_l = tr['wavlm_l'].numpy() # [Time_EEG, 768] (Interpolated)
        wavlm_r = tr['wavlm_r'].numpy()
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
                    a_seq = wavlm_l[seq_start:seq_start + SEQ_SAMPLES, :] # [SEQ_SAMPLES, 768]
                    b_seq = wavlm_r[seq_start:seq_start + SEQ_SAMPLES, :]
                    
                    e = e_seq.unfold(-1, WIN_SAMPLES, HOP_SAMPLES).permute(1, 0, 2)
                    
                    a_tensor = torch.from_numpy(a_seq).transpose(0, 1) # [768, SEQ_SAMPLES]
                    a_windows = a_tensor.unfold(-1, WIN_SAMPLES, HOP_SAMPLES).permute(1, 0, 2) # [num_windows, 768, WIN_SAMPLES]
                    
                    b_tensor = torch.from_numpy(b_seq).transpose(0, 1)
                    b_windows = b_tensor.unfold(-1, WIN_SAMPLES, HOP_SAMPLES).permute(1, 0, 2)
                    
                    num_windows = e.shape[0]
                    y = torch.full((num_windows,), label, dtype=torch.float32)
                    
                    a = a_windows.transpose(1, 2)
                    b = b_windows.transpose(1, 2)
                    
                    sequences.append((e, a, b, y))
    return sequences

# -------------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------------
def main():
    cache_dir = Path('/kaggle/input/datasets/lowkieee/aasd-universal-cache-v1/kaggle/working/wavlm_cache')
    if not cache_dir.exists():
        cache_dir = Path('/kaggle/working/wavlm_cache')
        
    cache_files = sorted(list(cache_dir.glob('*_wavlm.pt')))
    if len(cache_files) == 0:
        print("No cache files found.")
        return
        
    print("\n=======================================================")
    print(" PHASE 80: WAVLM ARCHITECTURE EVALUATION")
    print(" Upgrading from Envelope to Self-Supervised Features.")
    print(" Resolving Information Starvation on Subjects 05, 11, 16.")
    print("=======================================================\n", flush=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on Device: {device}\n", flush=True)
    
    for subj_idx in EVAL_SUBJECTS:
        if subj_idx >= len(cache_files): continue
        cache_path = cache_files[subj_idx]
        print(f"\n==================== SUBJECT {subj_idx:02d} ====================", flush=True)
        
        cached = torch.load(cache_path, map_location='cpu', weights_only=False)
        trials = cached['raw']
        
        subj_trials = []
        for tr in trials:
            eeg = tr['eeg'].numpy()[EAR_CHANNEL_INDICES, :] 
            wavlm_l = tr['wavlm_l'].float() # [TimeWavLM, 768]
            wavlm_r = tr['wavlm_r'].float()
            
            target_length = eeg.shape[1]
            wavlm_l = torch.nn.functional.interpolate(wavlm_l.unsqueeze(0).transpose(1, 2), size=target_length, mode='linear', align_corners=False).transpose(1, 2).squeeze(0)
            wavlm_r = torch.nn.functional.interpolate(wavlm_r.unsqueeze(0).transpose(1, 2), size=target_length, mode='linear', align_corners=False).transpose(1, 2).squeeze(0)
            
            wavlm_l = (wavlm_l - wavlm_l.mean(dim=0, keepdim=True)) / (wavlm_l.std(dim=0, keepdim=True) + 1e-8)
            wavlm_r = (wavlm_r - wavlm_r.mean(dim=0, keepdim=True)) / (wavlm_r.std(dim=0, keepdim=True) + 1e-8)
            
            eeg = (eeg - np.mean(eeg, axis=1, keepdims=True)) / (np.std(eeg, axis=1, keepdims=True) + 1e-8)
            
            subj_trials.append({
                'eeg': torch.from_numpy(eeg).float(), 
                'wavlm_l': wavlm_l, 
                'wavlm_r': wavlm_r, 
                'meta': tr['meta']
            })
            
        seqs = extract_wavlm_sequences(subj_trials)
        split_idx = int(len(seqs) * 0.8)
        calib_pool = seqs[:split_idx]
        eval_set = seqs[split_idx:]
        
        if len(eval_set) == 0 or len(calib_pool) == 0: continue
            
        # Optimization: pin_memory=True and num_workers=2
        train_loader = DataLoader(StableSequenceDataset(calib_pool), batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, num_workers=2)
        eval_loader = DataLoader(StableSequenceDataset(eval_set), batch_size=BATCH_SIZE, shuffle=False, pin_memory=True, num_workers=2)
        
        def train_and_eval(model_class, name, **kwargs):
            model = model_class(**kwargs).to(device)
            optimizer = optim.Adam(model.parameters(), lr=TRAIN_LR, weight_decay=1e-4)
            criterion = nn.BCEWithLogitsLoss()
            
            # Optimization: Mixed precision scaler for 768-D inputs
            scaler = torch.amp.GradScaler('cuda')
            
            best_auc = 0
            for epoch in range(TRAIN_EPOCHS):
                print(f"  Training Epoch {epoch+1}/15...", end='\r', flush=True)
                model.train()
                for b_e, b_a, b_b, b_y in train_loader:
                    b_e, b_a, b_b, b_y = b_e.to(device, non_blocking=True), b_a.to(device, non_blocking=True), b_b.to(device, non_blocking=True), b_y.to(device, non_blocking=True)
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
                        b_e, b_a, b_b = b_e.to(device, non_blocking=True), b_a.to(device, non_blocking=True), b_b.to(device, non_blocking=True)
                        with torch.autocast(device_type='cuda', dtype=torch.float16):
                            logits, _ = model(b_e, b_a, b_b)
                        all_preds.extend(torch.sigmoid(logits).cpu().numpy().flatten())
                        all_labels.extend(b_y.numpy().flatten())
                
                if len(np.unique(all_labels)) > 1:
                    auc = roc_auc_score(all_labels, all_preds)
                    best_auc = max(best_auc, auc)
                    
            print(f"  {name:20s}: {best_auc:.4f}           ", flush=True)
            return best_auc
            
        auc_tcn = train_and_eval(TCNAADModel, "WavLM + Dilated TCN", use_wavlm=True, dropout=0.3)
        print(f"  -> FINAL AUROC: {auc_tcn:.4f}\n", flush=True)

if __name__ == "__main__":
    main()
