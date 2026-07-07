import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
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

BATCH_SIZE = 16
EPOCHS = 20
LR = 1e-3

# -------------------------------------------------------------------------
# ARCHITECTURE
# -------------------------------------------------------------------------
class LocalEncoder(nn.Module):
    def __init__(self, in_channels, out_dim=64):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, 16, kernel_size=33, padding=16)
        self.bn1 = nn.BatchNorm1d(16)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=17, padding=8)
        self.bn2 = nn.BatchNorm1d(32)
        self.pool = nn.AdaptiveAvgPool1d(4)
        self.fc = nn.Linear(32 * 4, out_dim)
        
    def forward(self, x):
        # x: (B, C, T)
        if x.dim() == 2:
            x = x.unsqueeze(1)
            
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool1d(x, 2)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = x.view(x.size(0), -1) # flatten
        return self.fc(x)

class SequenceAADModel(nn.Module):
    def __init__(self, eeg_channels=60, latent_dim=64, lstm_hidden=64):
        super().__init__()
        self.eeg_encoder = LocalEncoder(in_channels=eeg_channels, out_dim=latent_dim)
        self.aud_encoder = LocalEncoder(in_channels=1, out_dim=latent_dim)
        
        # LSTM takes [score_A, score_B, score_A - score_B] => 3 features
        self.lstm = nn.LSTM(input_size=latent_dim * 3 + 3, hidden_size=lstm_hidden, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(lstm_hidden, 1)
        
    def forward(self, eeg_seq, aud_a_seq, aud_b_seq, hidden=None):
        # eeg_seq: (B, SeqLen, 60, WIN_SAMPLES)
        B, SeqLen, C, T = eeg_seq.shape
        
        # Flatten sequences to batch dimension for efficient parallel encoding
        eeg_flat = eeg_seq.reshape(B * SeqLen, C, T)
        aud_a_flat = aud_a_seq.reshape(B * SeqLen, 1, T)
        aud_b_flat = aud_b_seq.reshape(B * SeqLen, 1, T)
        
        p_eeg = F.normalize(self.eeg_encoder(eeg_flat), dim=-1)
        p_a = F.normalize(self.aud_encoder(aud_a_flat), dim=-1)
        p_b = F.normalize(self.aud_encoder(aud_b_flat), dim=-1)
        
        # Cosine Similarity Matching
        score_a = F.cosine_similarity(p_eeg, p_a, dim=-1) # (B*SeqLen)
        score_b = F.cosine_similarity(p_eeg, p_b, dim=-1) # (B*SeqLen)
        score_diff = score_a - score_b
        
        # Build sequence feature vector
        lstm_feat = torch.cat([
            p_eeg, p_a, p_b, 
            score_a.unsqueeze(-1), score_b.unsqueeze(-1), score_diff.unsqueeze(-1)
        ], dim=-1)
        lstm_feat = lstm_feat.reshape(B, SeqLen, -1)
        
        # Temporal Smoothing
        lstm_out, hidden = self.lstm(lstm_feat, hidden) # (B, SeqLen, hidden)
        
        logits = self.classifier(lstm_out).squeeze(-1) # (B, SeqLen)
        return logits, hidden

# -------------------------------------------------------------------------
# DATA EXTRACTION
# -------------------------------------------------------------------------
def get_attended_speaker_at_time(start_idx, switch_points):
    current_spk = 'L'
    for spk, idx in switch_points:
        if idx <= start_idx:
            current_spk = spk
        else:
            break
    return current_spk

def extract_sequence_windows(trial):
    eeg = trial['eeg'].numpy()
    env_l = trial['env_l'].numpy()
    env_r = trial['env_r'].numpy()
    switch_points = trial['meta']['switch_points']
    
    eeg = (eeg - np.mean(eeg, axis=1, keepdims=True)) / (np.std(eeg, axis=1, keepdims=True) + 1e-8)
    env_l = (env_l - np.mean(env_l)) / (np.std(env_l) + 1e-8)
    env_r = (env_r - np.mean(env_r)) / (np.std(env_r) + 1e-8)
    
    T = eeg.shape[1]
    
    seq_eeg, seq_aud_a, seq_aud_b, seq_labels = [], [], [], []
    
    # Slide 10-second sequences across the trial
    for seq_start in range(0, T - SEQ_SAMPLES + 1, int(SR * 2.0)): # 2s stride for sequences
        
        win_eeg, win_a, win_b, win_labels = [], [], [], []
        
        for w in range(NUM_WINDOWS):
            win_start = seq_start + w * HOP_SAMPLES
            win_end = win_start + WIN_SAMPLES
            
            att_spk = get_attended_speaker_at_time(win_start + WIN_SAMPLES//2, switch_points)
            
            # Real product behavior: Stream A is Left Speaker, Stream B is Right Speaker
            aud_a = env_l[win_start:win_end]
            aud_b = env_r[win_start:win_end]
            
            win_eeg.append(eeg[:, win_start:win_end])
            win_a.append(aud_a)
            win_b.append(aud_b)
            win_labels.append(1.0 if att_spk == 'L' else 0.0)
            
        seq_eeg.append(np.stack(win_eeg))
        seq_aud_a.append(np.stack(win_a))
        seq_aud_b.append(np.stack(win_b))
        seq_labels.append(np.array(win_labels))
        
    if len(seq_eeg) == 0:
        return None, None, None, None
        
    return np.stack(seq_eeg), np.stack(seq_aud_a), np.stack(seq_aud_b), np.stack(seq_labels)

# -------------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------------
def main():
    cache_path = Path('/kaggle/working/eeg_cache/S1_processed.pt')
    if not cache_path.exists():
        print("Cache not found.")
        return
        
    print("\n=======================================================")
    print(" PHASE 56: TRUE AAD LSTM SEQUENCE MODEL")
    print("=======================================================\n")
    
    cached = torch.load(cache_path, map_location='cpu', weights_only=False)
    trials = cached['raw']
    
    trial_eeg, trial_a, trial_b, trial_lbl = [], [], [], []
    for tr in trials:
        e, a, b, l = extract_sequence_windows(tr)
        if e is not None:
            trial_eeg.append(e)
            trial_a.append(a)
            trial_b.append(b)
            trial_lbl.append(l)
            
    kf = KFold(n_splits=5, shuffle=False)
    fold_aurocs = []
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on Device: {device}\n")
    
    for fold, (train_idx, test_idx) in enumerate(kf.split(trial_eeg)):
        X_tr_e = np.concatenate([trial_eeg[i] for i in train_idx], axis=0)
        X_tr_a = np.concatenate([trial_a[i] for i in train_idx], axis=0)
        X_tr_b = np.concatenate([trial_b[i] for i in train_idx], axis=0)
        Y_tr = np.concatenate([trial_lbl[i] for i in train_idx], axis=0)
        
        X_te_e = np.concatenate([trial_eeg[i] for i in test_idx], axis=0)
        X_te_a = np.concatenate([trial_a[i] for i in test_idx], axis=0)
        X_te_b = np.concatenate([trial_b[i] for i in test_idx], axis=0)
        Y_te = np.concatenate([trial_lbl[i] for i in test_idx], axis=0)
        
        train_ds = TensorDataset(torch.from_numpy(X_tr_e).float(), torch.from_numpy(X_tr_a).float(), torch.from_numpy(X_tr_b).float(), torch.from_numpy(Y_tr).float())
        test_ds = TensorDataset(torch.from_numpy(X_te_e).float(), torch.from_numpy(X_te_a).float(), torch.from_numpy(X_te_b).float(), torch.from_numpy(Y_te).float())
        
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
        
        model = SequenceAADModel().to(device)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=LR)
        
        best_val = 0.5
        for epoch in range(EPOCHS):
            model.train()
            for b_e, b_a, b_b, b_y in train_loader:
                b_e, b_a, b_b, b_y = b_e.to(device), b_a.to(device), b_b.to(device), b_y.to(device)
                optimizer.zero_grad()
                logits, _ = model(b_e, b_a, b_b) # (B, SeqLen)
                loss = criterion(logits, b_y) # BCE loss over all sequence steps
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
                    all_labels.extend(b_y.numpy().flatten())
                    
            try:
                val_auc = roc_auc_score(all_labels, all_preds)
            except:
                val_auc = 0.5
            best_val = max(best_val, val_auc)
            
        print(f"  Fold {fold+1} Best AUROC: {best_val:.4f}")
        fold_aurocs.append(best_val)
        
    print(f"\nAVERAGE 5-FOLD AUROC: {np.mean(fold_aurocs):.4f}")

if __name__ == "__main__":
    main()
