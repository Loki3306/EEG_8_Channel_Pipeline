import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score
import numpy as np
from pathlib import Path

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

BATCH_SIZE = 128 
EPOCHS = 15      
LR = 1e-3

EAR_CHANNEL_INDICES = [23, 31, 32, 40, 14, 22, 41, 49]

# -------------------------------------------------------------------------
# ARCHITECTURE (GroupNorm version)
# -------------------------------------------------------------------------
class LocalEncoder(nn.Module):
    def __init__(self, in_channels, out_dim=64):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, 16, kernel_size=33, padding=16)
        # SWAPPED: GroupNorm(1, C) instead of BatchNorm1d
        self.ln1 = nn.GroupNorm(1, 16) 
        self.conv2 = nn.Conv1d(16, 32, kernel_size=17, padding=8)
        # SWAPPED: GroupNorm(1, C) instead of BatchNorm1d
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
# BLOCK DATASET LOADER
# -------------------------------------------------------------------------
def get_attended_speaker_at_time(start_idx, switch_points):
    current_spk = 'L'
    for spk, idx in switch_points:
        if idx <= start_idx:
            current_spk = spk
        else:
            break
    return current_spk

class StableSequenceDataset(Dataset):
    def __init__(self, trials):
        self.sequences = []
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
                        
                        self.sequences.append((e, a, b, y))
                        
    def __len__(self):
        return len(self.sequences)
        
    def __getitem__(self, idx):
        return self.sequences[idx]

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
    print(" PHASE 73: NORMALIZATION ABLATION (GroupNorm Test)")
    print(" Falsification Test: Does GroupNorm destroy EEG features?")
    print(" Protocol: 80% Train / 20% Test per Subject (From Scratch)")
    print("=======================================================\n")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on Device: {device}\n")
    
    subject_aurocs = []
    
    for subj_idx, cache_path in enumerate(sorted(cache_files)):
        cached = torch.load(cache_path, map_location='cpu', weights_only=False)
        trials = cached['raw']
        
        subj_trials = []
        for tr in trials:
            eeg = tr['eeg'].numpy()
            eeg = eeg[EAR_CHANNEL_INDICES, :] # 8 channels
            
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
            
        # Time-chronological split to prevent data leakage from adjacent overlapping sequences
        # We split the trials 80/20.
        split_idx = int(len(subj_trials) * 0.8)
        train_trials = subj_trials[:split_idx]
        test_trials = subj_trials[split_idx:]
        
        train_dataset = StableSequenceDataset(train_trials)
        test_dataset = StableSequenceDataset(test_trials)
        
        # If a subject has too little data to form test sequences, skip
        if len(test_dataset) == 0:
            print(f"Subject {subj_idx:02d} | Skipped (Not enough test data)")
            continue
            
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
        
        model = SequenceAADModel(eeg_channels=8).to(device)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=LR)
        
        best_auc = 0
        for epoch in range(EPOCHS):
            model.train()
            for b_e, b_a, b_b, b_y in train_loader:
                b_e, b_a, b_b, b_y = b_e.to(device), b_a.to(device), b_b.to(device), b_y.to(device)
                optimizer.zero_grad()
                logits, _ = model(b_e, b_a, b_b)
                loss = criterion(logits, b_y) 
                loss.backward()
                optimizer.step()
                
            model.eval()
            all_preds, all_labels = [], []
            with torch.no_grad():
                for b_e, b_a, b_b, b_y in test_loader:
                    b_e, b_a, b_b = b_e.to(device), b_a.to(device), b_b.to(device)
                    logits, _ = model(b_e, b_a, b_b)
                    all_preds.extend(torch.sigmoid(logits).cpu().numpy().flatten())
                    all_labels.extend(b_y.cpu().numpy().flatten())
            
            val_auc = roc_auc_score(all_labels, all_preds) if len(np.unique(all_labels)) > 1 else 0.5
            best_auc = max(best_auc, val_auc)
            
        subject_aurocs.append(best_auc)
        print(f"Subject {subj_idx:02d} | Train Seq: {len(train_dataset):4d} | Test Seq: {len(test_dataset):4d} | Best GroupNorm AUROC: {best_auc:.4f}")

    print("\n=======================================================")
    print(f" FINAL ABLATION RESULTS:")
    print(f" Average GroupNorm AUROC: {np.mean(subject_aurocs):.4f}")
    print(" (Reference: Phase 70 BatchNorm AUROC was 0.70+)")
    print("=======================================================\n")

if __name__ == "__main__":
    main()
