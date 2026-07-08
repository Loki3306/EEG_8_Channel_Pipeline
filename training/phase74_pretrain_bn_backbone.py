import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.model_selection import KFold
import numpy as np
from pathlib import Path

# -------------------------------------------------------------------------
# CONSTANTS & HYPERPARAMETERS
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

EPOCHS = 15
BATCH_SIZE = 64
LR = 1e-3

# -------------------------------------------------------------------------
# ARCHITECTURE (BatchNorm Version)
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
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool1d(x, 2)
        x = F.relu(self.bn2(self.conv2(x)))
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
# DATA PROCESSING
# -------------------------------------------------------------------------
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
    print(" PHASE 74: BATCHNORM UNIVERSAL PRE-TRAINING")
    print(" Pre-training isolated backbones using BatchNorm1d")
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
    # Exact same fold split as Phase 71
    kf = KFold(n_splits=4, shuffle=True, random_state=42)
    
    for fold, (train_subjs, test_subjs) in enumerate(kf.split(subject_indices)):
        print(f"\n==================== FOLD {fold+1}/4 ====================")
        print(f"Train Subjects: {train_subjs}")
        print(f"Held-out Subjects (Ignored): {test_subjs}")
        
        # Aggregate training sequences for this fold
        fold_train_seqs = []
        for subj in train_subjs:
            fold_train_seqs.extend(all_subj_seqs[subj])
            
        # Shuffle sequences
        np.random.seed(fold)
        np.random.shuffle(fold_train_seqs)
        
        model = SequenceAADModel(eeg_channels=8).to(device)
        optimizer = optim.Adam(model.parameters(), lr=LR)
        criterion = nn.BCEWithLogitsLoss()
        
        print(f"  Training BatchNorm Backbone on {len(fold_train_seqs)} sequences...")
        
        for epoch in range(EPOCHS):
            model.train()
            total_loss = 0.0
            
            # Simple manual batching
            for i in range(0, len(fold_train_seqs), BATCH_SIZE):
                batch = fold_train_seqs[i:i+BATCH_SIZE]
                if len(batch) == 0: continue
                
                b_e = torch.stack([s[0] for s in batch]).to(device)
                b_a = torch.stack([s[1] for s in batch]).to(device)
                b_b = torch.stack([s[2] for s in batch]).to(device)
                b_y = torch.stack([s[3] for s in batch]).to(device)
                
                # IMPORTANT: BatchNorm needs batch size > 1
                if b_e.size(0) == 1: continue
                
                optimizer.zero_grad()
                logits, _ = model(b_e, b_a, b_b)
                loss = criterion(logits, b_y)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item() * len(batch)
                
            avg_loss = total_loss / len(fold_train_seqs)
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"    Epoch {epoch+1:02d}/{EPOCHS} | Train Loss: {avg_loss:.4f}")
                
        # Save the isolated backbone for this fold
        save_path = f"bn_backbone_fold_{fold}.pt"
        torch.save(model.state_dict(), save_path)
        print(f"  [+] Saved {save_path}")

    print("\nPhase 74 Complete. BatchNorm fold-specific backbones are ready for Phase 75 Calibration.")

if __name__ == "__main__":
    main()
