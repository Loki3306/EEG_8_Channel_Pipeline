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

# MAML Hyperparameters
META_EPOCHS = 100
META_BATCH_SIZE = 4        # Number of subjects per meta-update
FAST_LR = 0.01             # Inner loop adaptation rate
META_LR = 1e-3             # Outer loop meta-learning rate
ADAPT_STEPS = 5            # Number of SGD steps in inner loop
SUPPORT_SIZE = 50          # Number of sequences for calibration/support
QUERY_SIZE = 150           # Number of sequences for meta-update

# -------------------------------------------------------------------------
# ARCHITECTURE (Pure CNN + LSTM)
# -------------------------------------------------------------------------
class LocalEncoder(nn.Module):
    def __init__(self, in_channels, out_dim=64):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, 16, kernel_size=33, padding=16)
        # Using GroupNorm(1, C) which is equivalent to LayerNorm
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
# SEQUENCE EXTRACTOR
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

def batchify(seqs, device):
    if len(seqs) == 0:
        return None, None, None, None
    b_e = torch.stack([s[0] for s in seqs]).to(device)
    b_a = torch.stack([s[1] for s in seqs]).to(device)
    b_b = torch.stack([s[2] for s in seqs]).to(device)
    b_y = torch.stack([s[3] for s in seqs]).to(device)
    return b_e, b_a, b_b, b_y

# -------------------------------------------------------------------------
# MAIN FOMAML LOOP (PURE PYTORCH)
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
    print(" PHASE 71: PURE PYTORCH FOMAML (No learn2learn)")
    print(" Pivot to Rapid Personalization using First-Order MAML")
    print("=======================================================\n")
    
    all_subj_seqs = []
    
    print(f"Loading {len(cache_files)} subjects into RAM...")
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
        seqs = extract_subject_sequences(subj_trials)
        all_subj_seqs.append(seqs)
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nTraining on Device: {device}\n")
    
    subject_indices = list(range(len(all_subj_seqs)))
    kf = KFold(n_splits=4, shuffle=True, random_state=42)
    
    global_zeroshot = []
    global_calibrated = []
    
    for fold, (train_subjs, test_subjs) in enumerate(kf.split(subject_indices)):
        print(f"\n==================== FOLD {fold+1}/4 ====================")
        
        model = SequenceAADModel(eeg_channels=8).to(device)
        meta_opt = optim.Adam(model.parameters(), lr=META_LR)
        criterion = nn.BCEWithLogitsLoss()
        
        print(f"--- STEP 1: META-TRAINING ON {len(train_subjs)} SUBJECTS ---")
        for epoch in range(META_EPOCHS):
            meta_opt.zero_grad()
            meta_loss_sum = 0.0
            
            meta_batch = random.sample(list(train_subjs), META_BATCH_SIZE)
            
            for subj in meta_batch:
                # 1. Clone the model for this specific task
                learner = copy.deepcopy(model)
                learner.train()
                # Fast Adaptation uses standard SGD
                inner_opt = optim.SGD(learner.parameters(), lr=FAST_LR)
                
                seqs = all_subj_seqs[subj].copy()
                random.shuffle(seqs)
                
                support = seqs[:SUPPORT_SIZE]
                query = seqs[SUPPORT_SIZE:SUPPORT_SIZE+QUERY_SIZE]
                if len(support) == 0 or len(query) == 0: continue
                
                b_e_s, b_a_s, b_b_s, b_y_s = batchify(support, device)
                b_e_q, b_a_q, b_b_q, b_y_q = batchify(query, device)
                
                # 2. Fast Adaptation (Inner Loop)
                for _ in range(ADAPT_STEPS):
                    inner_opt.zero_grad()
                    logits, _ = learner(b_e_s, b_a_s, b_b_s)
                    loss = criterion(logits, b_y_s)
                    loss.backward()
                    inner_opt.step()
                    
                # 3. Outer Loop: Evaluate on Query Set
                logits_q, _ = learner(b_e_q, b_a_q, b_b_q)
                q_loss = criterion(logits_q, b_y_q)
                meta_loss_sum += q_loss.item()
                
                # 4. FOMAML Trick: Compute gradients on the adapted model...
                learner.zero_grad()
                q_loss.backward()
                
                # ...and copy them directly into the original meta-model!
                for p_meta, p_learner in zip(model.parameters(), learner.parameters()):
                    if p_learner.grad is not None:
                        if p_meta.grad is None:
                            p_meta.grad = p_learner.grad.clone() / META_BATCH_SIZE
                        else:
                            p_meta.grad += p_learner.grad.clone() / META_BATCH_SIZE
            
            meta_opt.step()
            
            if (epoch + 1) % 10 == 0:
                print(f"  Meta-Epoch {epoch+1:03d}/{META_EPOCHS} | Avg Query Loss: {meta_loss_sum/META_BATCH_SIZE:.4f}")
                
        print("\n--- STEP 2: META-TESTING (DEPLOYMENT CALIBRATION) ---")
        for subj in test_subjs:
            seqs = all_subj_seqs[subj]
            if len(seqs) < SUPPORT_SIZE + 50:
                continue
                
            # Deployment is Chronological! First sequences are calibration.
            support = seqs[:SUPPORT_SIZE]
            query = seqs[SUPPORT_SIZE:]
            
            b_e_s, b_a_s, b_b_s, b_y_s = batchify(support, device)
            b_e_q, b_a_q, b_b_q, b_y_q = batchify(query, device)
            
            # 1. Zero-Shot Baseline (using unmodified Meta-Weights)
            model.eval()
            with torch.no_grad():
                zs_logits, _ = model(b_e_q, b_a_q, b_b_q)
                zs_preds = torch.sigmoid(zs_logits).cpu().numpy().flatten()
                labels = b_y_q.cpu().numpy().flatten()
            zs_auc = roc_auc_score(labels, zs_preds) if len(np.unique(labels)) > 1 else 0.5
            
            # 2. Fast Adaptation (MAML Deployment)
            learner = copy.deepcopy(model)
            learner.train()
            inner_opt = optim.SGD(learner.parameters(), lr=FAST_LR)
            
            for _ in range(ADAPT_STEPS):
                inner_opt.zero_grad()
                logits, _ = learner(b_e_s, b_a_s, b_b_s)
                loss = criterion(logits, b_y_s)
                loss.backward()
                inner_opt.step()
                
            learner.eval()
            with torch.no_grad():
                calib_logits, _ = learner(b_e_q, b_a_q, b_b_q)
                calib_preds = torch.sigmoid(calib_logits).cpu().numpy().flatten()
            calib_auc = roc_auc_score(labels, calib_preds) if len(np.unique(labels)) > 1 else 0.5
            
            global_zeroshot.append(zs_auc)
            global_calibrated.append(calib_auc)
            
            print(f"  Subject {subj:02d} | Zero-Shot: {zs_auc:.4f} | MAML Adapted: {calib_auc:.4f}")

    print("\n=======================================================")
    print(f" FINAL RESULTS ACROSS ALL 18 SUBJECTS:")
    print(f" Average Zero-Shot AUROC:  {np.mean(global_zeroshot):.4f}")
    print(f" Average MAML Adapted AUROC: {np.mean(global_calibrated):.4f}")
    print("=======================================================\n")

if __name__ == "__main__":
    main()
