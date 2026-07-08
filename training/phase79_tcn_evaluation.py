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

# The 7 Data/Model Limited subjects (Excluding pure noise subjects 02 and 09)
WEAK_SUBJECTS = [1, 5, 6, 11, 15, 16, 17]

# -------------------------------------------------------------------------
# MODELS LADDER
# -------------------------------------------------------------------------
class LinearAADModel(nn.Module):
    def __init__(self, eeg_channels=8, latent_dim=64):
        super().__init__()
        self.eeg_proj = nn.Linear(eeg_channels * WIN_SAMPLES, latent_dim)
        self.aud_proj = nn.Linear(WIN_SAMPLES, latent_dim)
        self.classifier = nn.Linear(latent_dim * 3 + 3, 1)
        
    def forward(self, eeg_seq, aud_a_seq, aud_b_seq, hidden=None):
        B, SeqLen, C, T = eeg_seq.shape
        eeg_flat = eeg_seq.reshape(B * SeqLen, C * T)
        aud_a_flat = aud_a_seq.reshape(B * SeqLen, T)
        aud_b_flat = aud_b_seq.reshape(B * SeqLen, T)
        
        p_eeg = F.normalize(self.eeg_proj(eeg_flat), dim=-1)
        p_a = F.normalize(self.aud_proj(aud_a_flat), dim=-1)
        p_b = F.normalize(self.aud_proj(aud_b_flat), dim=-1)
        
        score_a = F.cosine_similarity(p_eeg, p_a, dim=-1)
        score_b = F.cosine_similarity(p_eeg, p_b, dim=-1)
        score_diff = score_a - score_b
        
        feat = torch.cat([p_eeg, p_a, p_b, score_a.unsqueeze(-1), score_b.unsqueeze(-1), score_diff.unsqueeze(-1)], dim=-1)
        feat = feat.reshape(B, SeqLen, -1)
        
        # Mean across sequence
        feat_pool = feat.mean(dim=1)
        logits = self.classifier(feat_pool).squeeze(-1)
        return logits, None

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

def get_attended_speaker_at_time(start_idx, switch_points):
    current_spk = 'L'
    for spk, idx in switch_points:
        if idx <= start_idx: current_spk = spk
        else: break
    return current_spk

def extract_subject_sequences(trials):
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
            spk = get_attended_speaker_at_time(start_idx, sp)
            label = 1.0 if spk == 'L' else 0.0
            
            safe_start = start_idx + EXCLUSION_SAMPLES
            safe_end = end_idx
            
            if safe_end - safe_start >= SEQ_SAMPLES:
                for seq_start in range(safe_start, safe_end - SEQ_SAMPLES + 1, seq_hop):
                    e_seq = eeg[:, seq_start:seq_start + SEQ_SAMPLES]
                    a_seq = env_l[seq_start:seq_start + SEQ_SAMPLES]
                    b_seq = env_r[seq_start:seq_start + SEQ_SAMPLES]
                    
                    e = e_seq.unfold(-1, WIN_SAMPLES, HOP_SAMPLES).permute(1, 0, 2)
                    a = a_seq.unfold(-1, WIN_SAMPLES, HOP_SAMPLES)
                    b = b_seq.unfold(-1, WIN_SAMPLES, HOP_SAMPLES)
                    num_windows = e.shape[0]
                    y = torch.full((num_windows,), label, dtype=torch.float32)
                    sequences.append((e, a, b, y))
    return sequences

# -------------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------------
def main():
    cache_dir = Path('/kaggle/input/datasets/lowkieee/aasd-universal-cache-v1/kaggle/working/eeg_cache')
    if not cache_dir.exists():
        cache_dir = Path('/kaggle/working/eeg_cache')
        
    cache_files = sorted(list(cache_dir.glob('*_processed.pt')))
    if len(cache_files) == 0:
        print("No cache files found. (Run on Kaggle)")
        return
        
    print("\n=======================================================")
    print(" PHASE 79: TCN ARCHITECTURE EVALUATION")
    print(" Solving the Underfitting/Overfitting gap on weak subjects.")
    print(" Architecture: Dilated Residual Temporal Convolutional Net")
    print("=======================================================\n")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on Device: {device}\n")
    
    for subj_idx in WEAK_SUBJECTS:
        if subj_idx >= len(cache_files): continue
        cache_path = cache_files[subj_idx]
        print(f"\n==================== SUBJECT {subj_idx:02d} ====================")
        
        cached = torch.load(cache_path, map_location='cpu', weights_only=False)
        trials = cached['raw']
        
        subj_trials = []
        for tr in trials:
            eeg = tr['eeg'].numpy()[EAR_CHANNEL_INDICES, :] 
            env_l = tr['env_l'].numpy()
            env_r = tr['env_r'].numpy()
            
            eeg = (eeg - np.mean(eeg, axis=1, keepdims=True)) / (np.std(eeg, axis=1, keepdims=True) + 1e-8)
            env_l = (env_l - np.mean(env_l)) / (np.std(env_l) + 1e-8)
            env_r = (env_r - np.mean(env_r)) / (np.std(env_r) + 1e-8)
            
            subj_trials.append({
                'eeg': torch.from_numpy(eeg).float(), 
                'env_l': torch.from_numpy(env_l).float(), 
                'env_r': torch.from_numpy(env_r).float(), 
                'meta': tr['meta']
            })
            
        seqs = extract_subject_sequences(subj_trials)
        split_idx = int(len(seqs) * 0.8)
        calib_pool = seqs[:split_idx]
        eval_set = seqs[split_idx:]
        
        if len(eval_set) == 0 or len(calib_pool) == 0: continue
            
        train_loader = DataLoader(StableSequenceDataset(calib_pool), batch_size=BATCH_SIZE, shuffle=True)
        eval_loader = DataLoader(StableSequenceDataset(eval_set), batch_size=BATCH_SIZE, shuffle=False)
        
        def train_and_eval(model_class, name, **kwargs):
            model = model_class(**kwargs).to(device)
            # Incorporate Weight Decay for TCN as recommended by GPT IPC review
            optimizer = optim.Adam(model.parameters(), lr=TRAIN_LR, weight_decay=1e-4)
            criterion = nn.BCEWithLogitsLoss()
            
            best_auc = 0
            for epoch in range(TRAIN_EPOCHS):
                model.train()
                for b_e, b_a, b_b, b_y in train_loader:
                    b_e, b_a, b_b, b_y = b_e.to(device), b_a.to(device), b_b.to(device), b_y.to(device)
                    if b_e.size(0) == 1: continue 
                    optimizer.zero_grad()
                    logits, _ = model(b_e, b_a, b_b)
                    loss = criterion(logits, b_y)
                    loss.backward()
                    optimizer.step()
                
                model.eval()
                all_preds, all_labels = [], []
                with torch.no_grad():
                    for b_e, b_a, b_b, b_y in eval_loader:
                        b_e, b_a, b_b = b_e.to(device), b_a.to(device), b_b.to(device)
                        logits, _ = model(b_e, b_a, b_b)
                        all_preds.extend(torch.sigmoid(logits).cpu().numpy().flatten())
                        all_labels.extend(b_y.cpu().numpy().flatten())
                
                if len(np.unique(all_labels)) > 1:
                    auc = roc_auc_score(all_labels, all_preds)
                    best_auc = max(best_auc, auc)
                    
            print(f"  {name:20s}: {best_auc:.4f}")
            return best_auc
            
        auc_lin = train_and_eval(LinearAADModel, "Linear (Ridge)")
        auc_tcn = train_and_eval(TCNAADModel, "Dilated TCN", dropout=0.3)
        
        if auc_tcn > auc_lin:
            print(f"  -> SUCCESS: TCN outperforms Linear by {auc_tcn - auc_lin:.4f}")
        else:
            print(f"  -> FAILURE: TCN underperforms Linear by {auc_lin - auc_tcn:.4f} (Overfitting unresolved)")

if __name__ == "__main__":
    main()
