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
from models.aad_tcn import HybridMoEAADModel

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
BATCH_SIZE = 16 # Smaller batch size because Hybrid model takes more memory
TRAIN_EPOCHS = 15
TRAIN_LR = 1e-3

# The Information-Starved subjects
EVAL_SUBJECTS = [5, 11, 16, 17]

# -------------------------------------------------------------------------
# DATASET
# -------------------------------------------------------------------------
class StableHybridDataset(Dataset):
    def __init__(self, sequences):
        self.seqs = sequences
    def __len__(self): return len(self.seqs)
    def __getitem__(self, idx):
        e, w_a, w_b, m_a, m_b, y = self.seqs[idx]
        return e, w_a, w_b, m_a, m_b, y[-1]

def extract_hybrid_sequences(trials):
    sequences = []
    seq_hop = int(0.5 * SR) 
    
    for tr in trials:
        eeg = tr['eeg']
        wavlm_l = tr['wavlm_l'].numpy() # [T, 768]
        wavlm_r = tr['wavlm_r'].numpy()
        multi_l = tr['multi_l'].numpy() # [16, T]
        multi_r = tr['multi_r'].numpy()
        
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
                    
                    wa_seq = wavlm_l[seq_start:seq_start + SEQ_SAMPLES, :]
                    wb_seq = wavlm_r[seq_start:seq_start + SEQ_SAMPLES, :]
                    
                    ma_seq = multi_l[:, seq_start:seq_start + SEQ_SAMPLES]
                    mb_seq = multi_r[:, seq_start:seq_start + SEQ_SAMPLES]
                    
                    e = e_seq.unfold(-1, WIN_SAMPLES, HOP_SAMPLES).permute(1, 0, 2)
                    
                    # WavLM format: [Channels, Time]
                    wa_tensor = torch.from_numpy(wa_seq).transpose(0, 1)
                    wa_windows = wa_tensor.unfold(-1, WIN_SAMPLES, HOP_SAMPLES).permute(1, 0, 2).transpose(1, 2)
                    
                    wb_tensor = torch.from_numpy(wb_seq).transpose(0, 1)
                    wb_windows = wb_tensor.unfold(-1, WIN_SAMPLES, HOP_SAMPLES).permute(1, 0, 2).transpose(1, 2)
                    
                    # Multiband format
                    ma_tensor = torch.from_numpy(ma_seq)
                    ma_windows = ma_tensor.unfold(-1, WIN_SAMPLES, HOP_SAMPLES).permute(1, 0, 2)
                    
                    mb_tensor = torch.from_numpy(mb_seq)
                    mb_windows = mb_tensor.unfold(-1, WIN_SAMPLES, HOP_SAMPLES).permute(1, 0, 2)
                    
                    num_windows = e.shape[0]
                    y = torch.full((num_windows,), label, dtype=torch.float32)
                    
                    sequences.append((e, wa_windows, wb_windows, ma_windows, mb_windows, y))
    return sequences

# -------------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------------
def main():
    multi_cache_dir = Path('/kaggle/input/datasets/lowkieee/aasd-universal-cache-v1/kaggle/working/multiband_cache')
    wavlm_cache_dir = Path('/kaggle/input/datasets/lowkieee/aasd-universal-cache-v1/kaggle/working/wavlm_cache')
    
    if not multi_cache_dir.exists(): multi_cache_dir = Path('/kaggle/working/multiband_cache')
    if not wavlm_cache_dir.exists(): wavlm_cache_dir = Path('/kaggle/working/wavlm_cache')
        
    multi_files = sorted(list(multi_cache_dir.glob('*_multiband.pt')))
    wavlm_files = sorted(list(wavlm_cache_dir.glob('*_wavlm.pt')))
    
    if len(multi_files) == 0 or len(wavlm_files) == 0:
        print("Missing cache files! Ensure BOTH WavLM and Multiband caches are present.")
        return
        
    print("\n=======================================================")
    print(" PHASE 82: HYBRID MIXTURE-OF-EXPERTS (MoE) EVALUATION")
    print(" Dynamically fusing Semantic (WavLM) and Cochlear (16-Band).")
    print(" Resolving extreme subject variance across 11, 16, and 17.")
    print("=======================================================\n", flush=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on Device: {device}\n", flush=True)
    
    for subj_idx in EVAL_SUBJECTS:
        if subj_idx >= len(multi_files) or subj_idx >= len(wavlm_files): continue
        print(f"\n==================== SUBJECT {subj_idx:02d} ====================", flush=True)
        
        multi_cached = torch.load(multi_files[subj_idx], map_location='cpu', weights_only=False)['raw']
        wavlm_cached = torch.load(wavlm_files[subj_idx], map_location='cpu', weights_only=False)['raw']
        
        subj_trials = []
        for i in range(min(len(multi_cached), len(wavlm_cached))):
            tr_m = multi_cached[i]
            tr_w = wavlm_cached[i]
            
            eeg = tr_m['eeg'].numpy()[EAR_CHANNEL_INDICES, :] 
            multi_l = tr_m['env_l'].float() # [16, Time]
            multi_r = tr_m['env_r'].float()
            
            wavlm_l = tr_w['wavlm_l'].float() # [Time_WavLM, 768]
            wavlm_r = tr_w['wavlm_r'].float()
            
            # Interpolate WavLM to EEG length precisely
            target_length = eeg.shape[1]
            wavlm_l = torch.nn.functional.interpolate(wavlm_l.unsqueeze(0).transpose(1, 2), size=target_length, mode='linear', align_corners=False).transpose(1, 2).squeeze(0)
            wavlm_r = torch.nn.functional.interpolate(wavlm_r.unsqueeze(0).transpose(1, 2), size=target_length, mode='linear', align_corners=False).transpose(1, 2).squeeze(0)
            
            wavlm_l = (wavlm_l - wavlm_l.mean(dim=0, keepdim=True)) / (wavlm_l.std(dim=0, keepdim=True) + 1e-8)
            wavlm_r = (wavlm_r - wavlm_r.mean(dim=0, keepdim=True)) / (wavlm_r.std(dim=0, keepdim=True) + 1e-8)
            eeg = (eeg - np.mean(eeg, axis=1, keepdims=True)) / (np.std(eeg, axis=1, keepdims=True) + 1e-8)
            
            min_len = min(eeg.shape[1], multi_l.shape[1], wavlm_l.shape[0])
            
            subj_trials.append({
                'eeg': torch.from_numpy(eeg[:, :min_len]).float(), 
                'multi_l': multi_l[:, :min_len], 
                'multi_r': multi_r[:, :min_len], 
                'wavlm_l': wavlm_l[:min_len, :],
                'wavlm_r': wavlm_r[:min_len, :],
                'meta': tr_m['meta']
            })
            
        seqs = extract_hybrid_sequences(subj_trials)
        split_idx = int(len(seqs) * 0.8)
        calib_pool = seqs[:split_idx]
        eval_set = seqs[split_idx:]
        
        if len(eval_set) == 0 or len(calib_pool) == 0: continue
            
        train_loader = DataLoader(StableHybridDataset(calib_pool), batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, num_workers=2)
        eval_loader = DataLoader(StableHybridDataset(eval_set), batch_size=BATCH_SIZE, shuffle=False, pin_memory=True, num_workers=2)
        
        def train_and_eval(model_class, name, **kwargs):
            model = model_class(**kwargs).to(device)
            optimizer = optim.Adam(model.parameters(), lr=TRAIN_LR, weight_decay=1e-4)
            criterion = nn.BCEWithLogitsLoss()
            scaler = torch.amp.GradScaler('cuda')
            
            best_auc = 0
            for epoch in range(TRAIN_EPOCHS):
                print(f"  Training Epoch {epoch+1}/15...", end='\r', flush=True)
                model.train()
                for b_e, b_wa, b_wb, b_ma, b_mb, b_y in train_loader:
                    b_e = b_e.to(device, non_blocking=True)
                    b_wa = b_wa.to(device, non_blocking=True)
                    b_wb = b_wb.to(device, non_blocking=True)
                    b_ma = b_ma.to(device, non_blocking=True)
                    b_mb = b_mb.to(device, non_blocking=True)
                    b_y = b_y.to(device, non_blocking=True)
                    if b_e.size(0) == 1: continue 
                    optimizer.zero_grad()
                    
                    with torch.autocast(device_type='cuda', dtype=torch.float16):
                        logits, alpha = model(b_e, b_wa, b_wb, b_ma, b_mb)
                        bce_loss = criterion(logits, b_y)
                        
                        # Entropy regularization to prevent gate collapse
                        # We want to keep alpha near 0.5 early on, so we maximize entropy (minimize -entropy)
                        entropy = - (alpha * torch.log(alpha + 1e-8) + (1 - alpha) * torch.log(1 - alpha + 1e-8)).mean()
                        
                        # The entropy weight decays each epoch so the gate can eventually specialize
                        entropy_weight = 0.5 * (0.8 ** epoch) 
                        
                        loss = bce_loss - entropy_weight * entropy
                        
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                
                model.eval()
                all_preds, all_labels = [], []
                with torch.no_grad():
                    for b_e, b_wa, b_wb, b_ma, b_mb, b_y in eval_loader:
                        b_e = b_e.to(device, non_blocking=True)
                        b_wa = b_wa.to(device, non_blocking=True)
                        b_wb = b_wb.to(device, non_blocking=True)
                        b_ma = b_ma.to(device, non_blocking=True)
                        b_mb = b_mb.to(device, non_blocking=True)
                        with torch.autocast(device_type='cuda', dtype=torch.float16):
                            logits, _ = model(b_e, b_wa, b_wb, b_ma, b_mb)
                        all_preds.extend(torch.sigmoid(logits).cpu().numpy().flatten())
                        all_labels.extend(b_y.numpy().flatten())
                
                if len(np.unique(all_labels)) > 1:
                    auc = roc_auc_score(all_labels, all_preds)
                    best_auc = max(best_auc, auc)
                    
            print(f"  {name:20s}: {best_auc:.4f}           ", flush=True)
            return best_auc
            
        auc_tcn = train_and_eval(HybridMoEAADModel, "Hybrid MoE TCN", dropout=0.3)
        print(f"  -> FINAL AUROC: {auc_tcn:.4f}\n", flush=True)

if __name__ == "__main__":
    main()
