import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split
import numpy as np
from pathlib import Path
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

BATCH_SIZE = 256
AAD_EPOCHS = 10
PROBE_EPOCHS = 10
LR = 1e-3

EAR_CHANNEL_INDICES = [23, 31, 32, 40, 14, 22, 41, 49]

# -------------------------------------------------------------------------
# ARCHITECTURE
# -------------------------------------------------------------------------
class SubjectAdapter(nn.Module):
    def __init__(self, channels=8):
        super().__init__()
        self.mixer = nn.Conv1d(channels, channels, kernel_size=1, bias=False)
        nn.init.eye_(self.mixer.weight[:, :, 0])
        
    def forward(self, x):
        return self.mixer(x)

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
        x = x.view(x.size(0), -1) 
        return self.fc(x)

class SequenceAADModel(nn.Module):
    def __init__(self, eeg_channels=8, latent_dim=64, lstm_hidden=64):
        super().__init__()
        self.adapter = SubjectAdapter(eeg_channels)
        self.eeg_encoder = LocalEncoder(in_channels=eeg_channels, out_dim=latent_dim)
        self.aud_encoder = LocalEncoder(in_channels=1, out_dim=latent_dim)
        
        self.lstm_hidden_dim = lstm_hidden
        self.lstm = nn.LSTM(input_size=latent_dim * 3 + 3, hidden_size=lstm_hidden, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(lstm_hidden, 1)
        
    def forward(self, eeg_seq, aud_a_seq, aud_b_seq, hidden=None):
        B, SeqLen, C, T = eeg_seq.shape
        eeg_flat = eeg_seq.reshape(B * SeqLen, C, T)
        eeg_flat = self.adapter(eeg_flat)
        
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
        
        # Return lstm_out for the Subject Probe
        return logits, hidden, lstm_out

class SubjectProbe(nn.Module):
    def __init__(self, embed_dim, num_subjects):
        super().__init__()
        self.fc = nn.Linear(embed_dim, num_subjects)
        
    def forward(self, embeddings):
        # embeddings shape: (B, SeqLen, EmbedDim)
        # We classify each window independently
        B, S, E = embeddings.shape
        x = embeddings.reshape(B * S, E)
        return self.fc(x)

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
            subj_id = tr['subj_id']
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
                        s = torch.full((num_windows,), subj_id, dtype=torch.long)
                        
                        self.sequences.append((e, a, b, y, s))
                        
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
    print(" PHASE 68: REPRESENTATION FORENSICS (SUBJECT PROBE)")
    print(" Task 1: Train AAD on all 18 Subjects (LOTO)")
    print(" Task 2: Freeze Backbone, Train Subject-ID Probe")
    print("=======================================================\n")
    
    all_trials = []
    
    print(f"Loading {len(cache_files)} subjects into RAM...")
    for subj_idx, cache_path in enumerate(sorted(cache_files)):
        cached = torch.load(cache_path, map_location='cpu', weights_only=False)
        trials = cached['raw']
        
        for tr in trials:
            eeg = tr['eeg'].numpy()
            eeg = eeg[EAR_CHANNEL_INDICES, :] # 8 channels
            
            env_l = tr['env_l'].numpy()
            env_r = tr['env_r'].numpy()
            
            eeg = (eeg - np.mean(eeg, axis=1, keepdims=True)) / (np.std(eeg, axis=1, keepdims=True) + 1e-8)
            env_l = (env_l - np.mean(env_l)) / (np.std(env_l) + 1e-8)
            env_r = (env_r - np.mean(env_r)) / (np.std(env_r) + 1e-8)
            
            all_trials.append({
                'eeg': torch.from_numpy(eeg).float(), 
                'env_l': torch.from_numpy(env_l).float(), 
                'env_r': torch.from_numpy(env_r).float(), 
                'meta': tr['meta'],
                'subj_id': subj_idx
            })
            
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nTraining on Device: {device}\n")
    
    # Random Split (LOTO)
    train_trials, test_trials = train_test_split(all_trials, test_size=0.2, random_state=42)
    
    train_dataset = StableSequenceDataset(train_trials)
    test_dataset = StableSequenceDataset(test_trials)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    print(f"Train Sequences: {len(train_dataset)}, Test Sequences: {len(test_dataset)}")
    
    model = SequenceAADModel(eeg_channels=8).to(device)
    criterion_aad = nn.BCEWithLogitsLoss()
    optimizer_aad = optim.Adam(model.parameters(), lr=LR)
    
    print("\n--- STEP 1: TRAINING UNIVERSAL AAD BACKBONE ---")
    best_aad_auc = 0
    for epoch in range(AAD_EPOCHS):
        model.train()
        for b_e, b_a, b_b, b_y, _ in train_loader:
            b_e, b_a, b_b, b_y = b_e.to(device), b_a.to(device), b_b.to(device), b_y.to(device)
            optimizer_aad.zero_grad()
            logits, _, _ = model(b_e, b_a, b_b)
            loss = criterion_aad(logits, b_y) 
            
            identity = torch.eye(8, device=device)
            adapter_reg = 0.01 * ((model.adapter.mixer.weight[:, :, 0] - identity)**2).sum()
            total_loss = loss + adapter_reg
            
            total_loss.backward()
            optimizer_aad.step()
            
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for b_e, b_a, b_b, b_y, _ in test_loader:
                b_e, b_a, b_b = b_e.to(device), b_a.to(device), b_b.to(device)
                logits, _, _ = model(b_e, b_a, b_b)
                all_preds.extend(torch.sigmoid(logits).cpu().numpy().flatten())
                all_labels.extend(b_y.cpu().numpy().flatten())
        val_auc = roc_auc_score(all_labels, all_preds)
        best_aad_auc = max(best_aad_auc, val_auc)
        print(f"  AAD Epoch {epoch+1}/{AAD_EPOCHS} - Val AUROC: {val_auc:.4f} (Best: {best_aad_auc:.4f})")
        
    print("\n--- STEP 2: REPRESENTATION FORENSICS (SUBJECT-ID PROBE) ---")
    # Freeze the entire AAD backbone
    for param in model.parameters():
        param.requires_grad = False
    model.eval()
    
    # Attach a Linear Probe to the LSTM hidden state
    num_subjects = len(cache_files)
    probe = SubjectProbe(embed_dim=model.lstm_hidden_dim, num_subjects=num_subjects).to(device)
    criterion_probe = nn.CrossEntropyLoss()
    optimizer_probe = optim.Adam(probe.parameters(), lr=1e-3)
    
    best_probe_acc = 0
    for epoch in range(PROBE_EPOCHS):
        probe.train()
        total_loss = 0
        for b_e, b_a, b_b, _, b_subj in train_loader:
            b_e, b_a, b_b, b_subj = b_e.to(device), b_a.to(device), b_b.to(device), b_subj.to(device)
            
            # Extract frozen embeddings
            with torch.no_grad():
                _, _, lstm_out = model(b_e, b_a, b_b)
                
            optimizer_probe.zero_grad()
            # Predict Subject ID
            subj_logits = probe(lstm_out)
            
            # Flatten predictions (B*SeqLen, NumSubj) and targets (B*SeqLen)
            subj_logits_flat = subj_logits.reshape(-1, num_subjects)
            b_subj_flat = b_subj.reshape(-1)
            
            loss = criterion_probe(subj_logits_flat, b_subj_flat)
            loss.backward()
            optimizer_probe.step()
            total_loss += loss.item()
            
        probe.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for b_e, b_a, b_b, _, b_subj in test_loader:
                b_e, b_a, b_b = b_e.to(device), b_a.to(device), b_b.to(device)
                _, _, lstm_out = model(b_e, b_a, b_b)
                subj_logits = probe(lstm_out)
                
                preds = torch.argmax(subj_logits, dim=-1).cpu().numpy().flatten()
                labels = b_subj.cpu().numpy().flatten()
                
                all_preds.extend(preds)
                all_labels.extend(labels)
                
        val_acc = accuracy_score(all_labels, all_preds)
        best_probe_acc = max(best_probe_acc, val_acc)
        chance_level = 1.0 / num_subjects
        
        print(f"  Probe Epoch {epoch+1}/{PROBE_EPOCHS} - Loss: {total_loss/len(train_loader):.4f} | Subj Val Acc: {val_acc:.4f} (Chance: {chance_level:.4f})")

    print("\n=======================================================")
    print(" FORENSIC VERDICT:")
    print(f" Final AAD AUROC: {best_aad_auc:.4f}")
    print(f" Subject-ID Accuracy: {best_probe_acc:.4f} (Chance: {1.0/num_subjects:.4f})")
    
    if best_probe_acc > 0.50:
        print(" [CRITICAL FAILURE] The latent space is massively entangled with Subject Identity.")
        print(" The backbone learned physical skull signatures, not universal auditory attention.")
    else:
        print(" [PASS] The latent space is relatively invariant to Subject Identity.")
    print("=======================================================\n")

if __name__ == "__main__":
    main()
