import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
import numpy as np
from pathlib import Path

# -------------------------------------------------------------------------
# CONSTANTS
# -------------------------------------------------------------------------
SR = 128
WIN_SEC = 2.0
HOP_SEC = 0.5
SEQ_SEC = 10.0

WIN_SAMPLES = int(WIN_SEC * SR)
HOP_SAMPLES = int(HOP_SEC * SR)
SEQ_SAMPLES = int(SEQ_SEC * SR)

NUM_WINDOWS = int((SEQ_SEC - WIN_SEC) / HOP_SEC) + 1 # 17

BATCH_SIZE = 256
EPOCHS = 10
LR = 1e-3

EAR_CHANNEL_INDICES = [23, 31, 32, 40, 14, 22, 41, 49]

# -------------------------------------------------------------------------
# ARCHITECTURE (Phase 61A: Spatial Mixer ONLY)
# -------------------------------------------------------------------------
class LocalEncoder(nn.Module):
    def __init__(self, in_channels, out_dim=64, is_eeg=False):
        super().__init__()
        self.is_eeg = is_eeg
        
        if is_eeg:
            # 1. Spatial Unmixing (Data-Driven CSP) ONLY. 
            # No dilations. No GroupNorm. No Contrastive Loss.
            self.spatial_mixer = nn.Conv1d(in_channels, in_channels, kernel_size=1)
            
        self.conv1 = nn.Conv1d(in_channels, 16, kernel_size=33, padding=16)
        self.bn1 = nn.BatchNorm1d(16)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=17, padding=8)
        self.bn2 = nn.BatchNorm1d(32)
        self.pool = nn.AdaptiveAvgPool1d(4)
        self.fc = nn.Linear(32 * 4, out_dim)
        
    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
            
        if self.is_eeg:
            x = self.spatial_mixer(x)
            
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool1d(x, 2)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = x.view(x.size(0), -1) 
        return self.fc(x)

class SequenceAADModel(nn.Module):
    def __init__(self, eeg_channels=8, latent_dim=64, lstm_hidden=64):
        super().__init__()
        self.eeg_encoder = LocalEncoder(in_channels=eeg_channels, out_dim=latent_dim, is_eeg=True)
        self.aud_encoder = LocalEncoder(in_channels=1, out_dim=latent_dim, is_eeg=False)
        
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
        
        lstm_out, hidden = self.lstm(lstm_feat, hidden)
        logits = self.classifier(lstm_out).squeeze(-1)
        return logits, hidden

# -------------------------------------------------------------------------
# DYNAMIC DATASET
# -------------------------------------------------------------------------
def get_attended_speaker_at_time(start_idx, switch_points):
    current_spk = 'L'
    for spk, idx in switch_points:
        if idx <= start_idx:
            current_spk = spk
        else:
            break
    return current_spk

class AASDSequenceDataset(Dataset):
    def __init__(self, trials):
        self.trials = trials
        self.sequences = []
        
        for t_idx, tr in enumerate(trials):
            T = tr['eeg'].shape[1]
            for seq_start in range(0, T - SEQ_SAMPLES + 1, int(SR * 2.0)):
                self.sequences.append((t_idx, seq_start))
                
    def __len__(self):
        return len(self.sequences)
        
    def __getitem__(self, idx):
        t_idx, seq_start = self.sequences[idx]
        tr = self.trials[t_idx]
        
        eeg_seq = tr['eeg'][:, seq_start:seq_start + SEQ_SAMPLES]
        a_seq = tr['env_l'][seq_start:seq_start + SEQ_SAMPLES]
        b_seq = tr['env_r'][seq_start:seq_start + SEQ_SAMPLES]
        att_seq = tr['att'][seq_start:seq_start + SEQ_SAMPLES]
        
        e = eeg_seq.unfold(-1, WIN_SAMPLES, HOP_SAMPLES).permute(1, 0, 2)
        a = a_seq.unfold(-1, WIN_SAMPLES, HOP_SAMPLES)
        b = b_seq.unfold(-1, WIN_SAMPLES, HOP_SAMPLES)
        y = att_seq.unfold(-1, WIN_SAMPLES, HOP_SAMPLES)[:, WIN_SAMPLES//2]
        
        return e, a, b, y

# -------------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------------
def main():
    cache_dir = Path('/kaggle/working/eeg_cache')
    cache_files = list(cache_dir.glob('*_processed.pt'))
    
    if len(cache_files) == 0:
        print("No cache files found.")
        return
        
    print("\n=======================================================")
    print(" PHASE 61A: ABLATION (SPATIAL MIXER ONLY)")
    print("=======================================================\n")
    
    all_trials = []
    
    print(f"Loading {len(cache_files)} subjects into RAM...")
    for subj_idx, cache_path in enumerate(sorted(cache_files)):
        cached = torch.load(cache_path, map_location='cpu', weights_only=False)
        trials = cached['raw']
        
        for tr in trials:
            eeg = tr['eeg'].numpy()
            
            # CRITICAL: Drop from 60 channels to 8 Ear-EEG channels
            eeg = eeg[EAR_CHANNEL_INDICES, :]
            
            env_l = tr['env_l'].numpy()
            env_r = tr['env_r'].numpy()
            sp = tr['meta']['switch_points']
            
            eeg = (eeg - np.mean(eeg, axis=1, keepdims=True)) / (np.std(eeg, axis=1, keepdims=True) + 1e-8)
            env_l = (env_l - np.mean(env_l)) / (np.std(env_l) + 1e-8)
            env_r = (env_r - np.mean(env_r)) / (np.std(env_r) + 1e-8)
            
            if eeg.shape[1] >= SEQ_SAMPLES:
                att_array = np.zeros(eeg.shape[1], dtype=np.float32)
                for i in range(eeg.shape[1]):
                    att_array[i] = 1.0 if get_attended_speaker_at_time(i, sp) == 'L' else 0.0
                
                all_trials.append({
                    'eeg': torch.from_numpy(eeg).float(), 
                    'env_l': torch.from_numpy(env_l).float(), 
                    'env_r': torch.from_numpy(env_r).float(), 
                    'att': torch.from_numpy(att_array)
                })
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    fold_aurocs = []
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on Device: {device}\n")
    
    for fold, (train_idx, test_idx) in enumerate(kf.split(all_trials)):
        train_trials = [all_trials[i] for i in train_idx]
        test_trials = [all_trials[i] for i in test_idx]
        
        train_ds = AASDSequenceDataset(train_trials)
        test_ds = AASDSequenceDataset(test_trials)
        
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
        test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
        
        model = SequenceAADModel(eeg_channels=8).to(device)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=LR)
        
        best_val = 0.5
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
                    probs = torch.sigmoid(logits).cpu().numpy().flatten()
                    all_preds.extend(probs)
                    all_labels.extend(b_y.cpu().numpy().flatten())
                    
            try:
                val_auc = roc_auc_score(all_labels, all_preds)
            except:
                val_auc = 0.5
            best_val = max(best_val, val_auc)
            
            print(f"    Epoch {epoch+1}/{EPOCHS} - Val AUROC: {val_auc:.4f} (Best: {best_val:.4f})")
            
        print(f"  Fold {fold+1} Best AUROC: {best_val:.4f}")
        fold_aurocs.append(best_val)
        
    print(f"\nAVERAGE 5-FOLD PHASE 61A AUROC: {np.mean(fold_aurocs):.4f}")

if __name__ == "__main__":
    main()
