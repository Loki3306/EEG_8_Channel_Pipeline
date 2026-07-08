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
from models.aad_mamba import ContinuousMambaAADModel

# -------------------------------------------------------------------------
# CONSTANTS
# -------------------------------------------------------------------------
SR = 128
EXCLUSION_SEC = 1.5   
SEQ_SEC = 3.5         

EXCLUSION_SAMPLES = int(EXCLUSION_SEC * SR)
SEQ_SAMPLES = int(SEQ_SEC * SR)

EAR_CHANNEL_INDICES = [23, 31, 32, 40, 14, 22, 41, 49]
BATCH_SIZE = 64
TRAIN_EPOCHS = 15
TRAIN_LR = 1e-3

EVAL_TARGETS = [1, 16, 3, 8] # Benchmark subjects

class MambaAASDDataset(Dataset):
    def __init__(self, sequences):
        self.seqs = sequences
    def __len__(self): return len(self.seqs)
    def __getitem__(self, idx):
        e, a_a, a_b, y = self.seqs[idx]
        return e, a_a, a_b, y

def extract_continuous_sequences(trials):
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
                    # Extract 3.5s continuous sequences
                    e_seq = eeg[:, seq_start:seq_start + SEQ_SAMPLES]
                    al_seq = env_l[:, seq_start:seq_start + SEQ_SAMPLES]
                    ar_seq = env_r[:, seq_start:seq_start + SEQ_SAMPLES]
                    
                    # Convert to [SeqLen, Channels]
                    e_seq = e_seq.transpose(0, 1)
                    al_seq = al_seq.transpose(0, 1)
                    ar_seq = ar_seq.transpose(0, 1)
                    
                    sequences.append((e_seq, al_seq, ar_seq, label))
    return sequences

def train_and_eval_mamba(subj_idx, train_loader, eval_loader, device):
    model = ContinuousMambaAADModel().to(device)
    optimizer = optim.Adam(model.parameters(), lr=TRAIN_LR, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()
    scaler = torch.amp.GradScaler('cuda')
    
    print(f"\n  [Training S{subj_idx:02d} with MAMBA SSM]")
    for epoch in range(TRAIN_EPOCHS):
        model.train()
        for b_e, b_a_a, b_a_b, b_y in train_loader:
            b_e = b_e.to(device, non_blocking=True)
            b_a_a = b_a_a.to(device, non_blocking=True)
            b_a_b = b_a_b.to(device, non_blocking=True)
            b_y = b_y.to(device, non_blocking=True).float()
            
            optimizer.zero_grad()
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                logits = model(b_e, b_a_a, b_a_b)
                loss = criterion(logits, b_y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
    # Evaluation
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for b_e, b_a_a, b_a_b, b_y in eval_loader:
            b_e = b_e.to(device, non_blocking=True)
            b_a_a = b_a_a.to(device, non_blocking=True)
            b_a_b = b_a_b.to(device, non_blocking=True)
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                logits = model(b_e, b_a_a, b_a_b)
            all_preds.extend(torch.sigmoid(logits).cpu().numpy().flatten())
            all_labels.extend(b_y.numpy().flatten())
            
    auc = roc_auc_score(all_labels, all_preds) if len(np.unique(all_labels)) > 1 else 0.5
    return auc

def main():
    cache_dir = Path('/kaggle/input/datasets/lowkieee/multiband-cache/kaggle/working/multiband_cache')
    if not cache_dir.exists(): cache_dir = Path('/kaggle/working/multiband_cache')
        
    cache_files = sorted(list(cache_dir.glob('*_multiband.pt')))
    
    if len(cache_files) == 0:
        print("No Multiband cache found!")
        return
        
    print("\n=======================================================")
    print(" PHASE 95: STATE SPACE MODELS (MAMBA)")
    print(" Architecture: Continuous Sequence Model (No rigid CNN windows)")
    print(" Objective: Match or beat the Oracle upper bound for both clusters.")
    print("=======================================================\n", flush=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on Device: {device}\n", flush=True)
    
    for subj_idx in EVAL_TARGETS:
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
            
        seqs = extract_continuous_sequences(subj_trials)
        split_idx = int(len(seqs) * 0.8)
        calib_pool = seqs[:split_idx]
        eval_set = seqs[split_idx:]
        
        if len(eval_set) == 0 or len(calib_pool) == 0: continue
            
        train_loader = DataLoader(MambaAASDDataset(calib_pool), batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, num_workers=0)
        eval_loader = DataLoader(MambaAASDDataset(eval_set), batch_size=BATCH_SIZE, shuffle=False, pin_memory=True, num_workers=0)
        
        auc = train_and_eval_mamba(subj_idx, train_loader, eval_loader, device)
        
        print(f"  Mamba AUROC      : {auc:.4f}")

if __name__ == "__main__":
    main()
