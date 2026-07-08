import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
import numpy as np
from pathlib import Path
import random
import copy

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
FT_EPOCHS = 10
FT_LR = 1e-4

# Fractions of the 80% Calibration Pool to test
# e.g., 0.05 = 5% of 80% = 4% of total data = ~20-30 sequences
DATA_FRACTIONS = [0.05, 0.10, 0.25, 0.50, 1.0]

# -------------------------------------------------------------------------
# ARCHITECTURE (Identical to Phase 71A)
# -------------------------------------------------------------------------
class LocalEncoder(nn.Module):
    def __init__(self, in_channels, out_dim=64):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, 16, kernel_size=33, padding=16)
        self.ln1 = nn.GroupNorm(1, 16) 
        self.conv2 = nn.Conv1d(16, 32, kernel_size=17, padding=8)
        self.ln2 = nn.GroupNorm(1, 32)
        self.pool = nn.AdaptiveAvgPool1d(4)
        self.fc = nn.Linear(32 * 4, out_dim)
        
    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x = F.relu(self.ln1(self.conv1(x)))
        x = F.max_pool1d(x, 2)
        x = F.relu(self.ln2(self.conv2(x)))
        x = self.pool(x)
        x = torch.flatten(x, 1) 
        return self.fc(x)

class SequenceAADModel(nn.Module):
    def __init__(self, eeg_channels=8, latent_dim=64, lstm_hidden=64):
        super().__init__()
        self.eeg_encoder = LocalEncoder(in_channels=eeg_channels, out_dim=latent_dim)
        self.aud_encoder = LocalEncoder(in_channels=1, out_dim=latent_dim)
        
        self.lstm = nn.LSTM(input_size=latent_dim * 3 + 3, hidden_size=lstm_hidden, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(lstm_hidden, 1)
        
    def forward(self, eeg_seq, aud_a_seq, aud_b_seq, hidden=None):
        B, SeqLen, C, T = eeg_seq.shape
        eeg_flat = eeg_seq.reshape(B * SeqLen, C, T)
        aud_a_flat = aud_a_seq.reshape(B * SeqLen, 1, T)
        aud_b_flat = aud_b_seq.reshape(B * SeqLen, 1, T)
        
        p_eeg = F.normalize(self.eeg_encoder(eeg_flat), dim=-1)
        p_a = F.normalize(self.aud_encoder(aud_a_flat), dim=-1)
        p_b = F.normalize(self.aud_encoder(aud_b_flat), dim=-1)
        
        score_a = F.cosine_similarity(p_eeg, p_a, dim=-1)
        score_b = F.cosine_similarity(p_eeg, p_b, dim=-1)
        score_diff = score_a - score_b
        
        lstm_feat = torch.cat([
            p_eeg, p_a, p_b, 
            score_a.unsqueeze(-1), score_b.unsqueeze(-1), score_diff.unsqueeze(-1)
        ], dim=-1)
        lstm_feat = lstm_feat.reshape(B, SeqLen, -1)
        
        self.lstm.flatten_parameters()
        lstm_out, hidden = self.lstm(lstm_feat, hidden)
        logits = self.classifier(lstm_out).squeeze(-1)
        return logits, hidden

# -------------------------------------------------------------------------
# DATASET
# -------------------------------------------------------------------------
class StableSequenceDataset(Dataset):
    def __init__(self, sequences):
        self.seqs = sequences
        
    def __len__(self):
        return len(self.seqs)
        
    def __getitem__(self, idx):
        e, a, b, y = self.seqs[idx]
        return e, a, b, y

def get_attended_speaker_at_time(start_idx, switch_points):
    current_spk = 'L'
    for spk, idx in switch_points:
        if idx <= start_idx:
            current_spk = spk
        else:
            break
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
        if boundaries[-1] != T:
            boundaries.append(T)
            
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
        
    cache_files = list(cache_dir.glob('*_processed.pt'))
    
    if len(cache_files) == 0:
        print("No cache files found.")
        return
        
    print("\n=======================================================")
    print(" PHASE 72: CALIBRATION SCALING LAWS")
    print(" Fine-tuning Universal Backbone with varying data fractions")
    print("=======================================================\n")
    
    all_subj_seqs = []
    
    print(f"Loading {len(cache_files)} subjects into RAM...")
    for subj_idx, cache_path in enumerate(sorted(cache_files)):
        cached = torch.load(cache_path, map_location='cpu', weights_only=False)
        trials = cached['raw']
        
        subj_trials = []
        for tr in trials:
            eeg = tr['eeg'].numpy()
            eeg = eeg[EAR_CHANNEL_INDICES, :] 
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
        all_subj_seqs.append(seqs)
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nTraining on Device: {device}\n")
    
    subject_indices = list(range(len(all_subj_seqs)))
    kf = KFold(n_splits=4, shuffle=True, random_state=42)
    
    results = {frac: [] for frac in DATA_FRACTIONS}
    results['zero_shot'] = []
    
    for fold, (train_subjs, test_subjs) in enumerate(kf.split(subject_indices)):
        print(f"\n==================== FOLD {fold+1}/4 ====================")
        
        ckpt_path = Path(f"maml_backbone_fold_{fold}.pt")
        if not ckpt_path.exists():
            print(f"[FATAL] Missing {ckpt_path}. Did you run phase71a_pretrain_maml_backbone.py?")
            return
            
        print(f"Loading isolated pre-trained backbone: {ckpt_path.name}")
        base_model = SequenceAADModel(eeg_channels=8).to(device)
        base_model.load_state_dict(torch.load(ckpt_path, map_location=device))
        
        criterion = nn.BCEWithLogitsLoss()
        
        for subj in test_subjs:
            seqs = all_subj_seqs[subj]
            
            # Time-split 80% Calibration Pool / 20% Evaluation Set
            split_idx = int(len(seqs) * 0.8)
            calib_pool = seqs[:split_idx]
            eval_set = seqs[split_idx:]
            
            if len(eval_set) == 0 or len(calib_pool) == 0:
                continue
                
            eval_dataset = StableSequenceDataset(eval_set)
            eval_loader = DataLoader(eval_dataset, batch_size=BATCH_SIZE, shuffle=False)
            
            # Helper to evaluate a model
            def evaluate(m):
                m.eval()
                all_preds, all_labels = [], []
                with torch.no_grad():
                    for b_e, b_a, b_b, b_y in eval_loader:
                        b_e, b_a, b_b = b_e.to(device), b_a.to(device), b_b.to(device)
                        logits, _ = m(b_e, b_a, b_b)
                        all_preds.extend(torch.sigmoid(logits).cpu().numpy().flatten())
                        all_labels.extend(b_y.cpu().numpy().flatten())
                return roc_auc_score(all_labels, all_preds) if len(np.unique(all_labels)) > 1 else 0.5

            # --- EVAL 1: ZERO-SHOT ---
            zs_auc = evaluate(base_model)
            results['zero_shot'].append(zs_auc)
            print(f"  Subject {subj:02d} | Zero-Shot AUROC: {zs_auc:.4f}")
            
            # --- EVAL 2: SCALING FRACTIONS ---
            for frac in DATA_FRACTIONS:
                num_calib = max(1, int(len(calib_pool) * frac))
                calib_subset = calib_pool[:num_calib] # Chronological sampling
                
                calib_dataset = StableSequenceDataset(calib_subset)
                calib_loader = DataLoader(calib_dataset, batch_size=BATCH_SIZE, shuffle=True)
                
                # Clone the base model for this fraction
                ft_model = copy.deepcopy(base_model)
                optimizer = optim.Adam(ft_model.parameters(), lr=FT_LR)
                
                # Fine-tune
                ft_model.train()
                for epoch in range(FT_EPOCHS):
                    for b_e, b_a, b_b, b_y in calib_loader:
                        b_e, b_a, b_b, b_y = b_e.to(device), b_a.to(device), b_b.to(device), b_y.to(device)
                        optimizer.zero_grad()
                        logits, _ = ft_model(b_e, b_a, b_b)
                        loss = criterion(logits, b_y)
                        loss.backward()
                        optimizer.step()
                        
                ft_auc = evaluate(ft_model)
                results[frac].append(ft_auc)
                
                # Calculate approx calibration time (3.5s per sequence, overlapping by 0.5s)
                # If N sequences, time is ~ 3.5 + (N-1)*0.5 seconds
                calib_time_sec = 3.5 + (num_calib - 1) * 0.5 if num_calib > 0 else 0
                calib_time_min = calib_time_sec / 60.0
                
                print(f"    - FT {frac*100:5.1f}% ({num_calib:4d} seqs | ~{calib_time_min:4.1f} min): {ft_auc:.4f}")

    print("\n=======================================================")
    print(f" FINAL SCALING LAWS (Mean across 18 Subjects):")
    print(f" Zero-Shot:    {np.mean(results['zero_shot']):.4f}")
    for frac in DATA_FRACTIONS:
        print(f" FT {frac*100:5.1f}%:   {np.mean(results[frac]):.4f}")
    print("=======================================================\n")

if __name__ == "__main__":
    main()
