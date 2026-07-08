import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
import numpy as np
from pathlib import Path
import copy

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

BATCH_SIZE = 256
PRETRAIN_EPOCHS = 10
CALIB_EPOCHS = 50
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
    print(" PHASE 64: FULL ROTATED SUBJECT CALIBRATION (LOSO FOLDS)")
    print("=======================================================\n")
    
    all_trials_by_subj = []
    
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
            sp = tr['meta']['switch_points']
            
            eeg = (eeg - np.mean(eeg, axis=1, keepdims=True)) / (np.std(eeg, axis=1, keepdims=True) + 1e-8)
            env_l = (env_l - np.mean(env_l)) / (np.std(env_l) + 1e-8)
            env_r = (env_r - np.mean(env_r)) / (np.std(env_r) + 1e-8)
            
            if eeg.shape[1] >= SEQ_SAMPLES:
                att_array = np.zeros(eeg.shape[1], dtype=np.float32)
                for i in range(eeg.shape[1]):
                    att_array[i] = 1.0 if get_attended_speaker_at_time(i, sp) == 'L' else 0.0
                
                subj_trials.append({
                    'eeg': torch.from_numpy(eeg).float(), 
                    'env_l': torch.from_numpy(env_l).float(), 
                    'env_r': torch.from_numpy(env_r).float(), 
                    'att': torch.from_numpy(att_array)
                })
        all_trials_by_subj.append(subj_trials)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nTraining on Device: {device}\n")
    
    # 4-Fold Cross Validation on Subjects (18 subjects total)
    subject_indices = np.arange(len(all_trials_by_subj))
    kf = KFold(n_splits=4, shuffle=True, random_state=42)
    
    global_zeroshot = []
    global_calibrated = []
    
    for fold, (train_subjs, test_subjs) in enumerate(kf.split(subject_indices)):
        print(f"\n==================== FOLD {fold+1}/4 ====================")
        print(f"Train Subjects: {train_subjs}")
        print(f"Test Subjects:  {test_subjs}")
        
        universal_trials = []
        for i in train_subjs:
            universal_trials.extend(all_trials_by_subj[i])
            
        train_loader = DataLoader(AASDSequenceDataset(universal_trials), batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
        
        model = SequenceAADModel(eeg_channels=8).to(device)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=LR)
        
        print("\n--- STEP 1: PRE-TRAINING UNIVERSAL BACKBONE ---")
        for epoch in range(PRETRAIN_EPOCHS):
            model.train()
            for b_e, b_a, b_b, b_y in train_loader:
                b_e, b_a, b_b, b_y = b_e.to(device), b_a.to(device), b_b.to(device), b_y.to(device)
                optimizer.zero_grad()
                logits, _ = model(b_e, b_a, b_b)
                loss = criterion(logits, b_y) 
                
                identity = torch.eye(8, device=device)
                adapter_reg = 0.01 * ((model.adapter.mixer.weight[:, :, 0] - identity)**2).sum()
                total_loss = loss + adapter_reg
                
                total_loss.backward()
                optimizer.step()
            print(f"  Pre-train Epoch {epoch+1}/{PRETRAIN_EPOCHS} complete.")
            
        universal_weights = copy.deepcopy(model.state_dict())
        
        print("\n--- STEP 2: TEST-TIME CALIBRATION ---")
        for subj in test_subjs:
            subj_trials = all_trials_by_subj[subj]
            num_calib_trials = 10
            calib_trials = subj_trials[:num_calib_trials]
            test_trials = subj_trials[num_calib_trials:]
            
            if len(test_trials) == 0: continue
            
            calib_loader = DataLoader(AASDSequenceDataset(calib_trials), batch_size=BATCH_SIZE, shuffle=True)
            test_loader = DataLoader(AASDSequenceDataset(test_trials), batch_size=BATCH_SIZE, shuffle=False)
            
            model.load_state_dict(universal_weights)
            
            # Zero-Shot evaluation
            model.eval()
            all_preds, all_labels = [], []
            with torch.no_grad():
                for b_e, b_a, b_b, b_y in test_loader:
                    b_e, b_a, b_b = b_e.to(device), b_a.to(device), b_b.to(device)
                    logits, _ = model(b_e, b_a, b_b)
                    all_preds.extend(torch.sigmoid(logits).cpu().numpy().flatten())
                    all_labels.extend(b_y.cpu().numpy().flatten())
            zs_auc = roc_auc_score(all_labels, all_preds) if len(np.unique(all_labels)) > 1 else 0.5
            
            # Calibration
            for param in model.parameters(): param.requires_grad = False
            calib_params = []
            for param in model.adapter.parameters():
                param.requires_grad = True
                calib_params.append(param)
            for m in model.eeg_encoder.modules():
                if isinstance(m, nn.BatchNorm1d):
                    for param in m.parameters():
                        param.requires_grad = True
                        calib_params.append(param)
                        
            calib_optimizer = optim.Adam(calib_params, lr=1e-2) 
            
            for epoch in range(CALIB_EPOCHS):
                model.train() 
                for b_e, b_a, b_b, b_y in calib_loader:
                    b_e, b_a, b_b, b_y = b_e.to(device), b_a.to(device), b_b.to(device), b_y.to(device)
                    calib_optimizer.zero_grad()
                    logits, _ = model(b_e, b_a, b_b)
                    bce_loss = criterion(logits, b_y)
                    identity = torch.eye(8, device=device)
                    adapter_reg = 0.001 * ((model.adapter.mixer.weight[:, :, 0] - identity)**2).sum()
                    loss = bce_loss + adapter_reg
                    loss.backward()
                    calib_optimizer.step()
                    
            # Calibrated evaluation
            model.eval()
            all_preds, all_labels = [], []
            with torch.no_grad():
                for b_e, b_a, b_b, b_y in test_loader:
                    b_e, b_a, b_b = b_e.to(device), b_a.to(device), b_b.to(device)
                    logits, _ = model(b_e, b_a, b_b)
                    all_preds.extend(torch.sigmoid(logits).cpu().numpy().flatten())
                    all_labels.extend(b_y.cpu().numpy().flatten())
            calib_auc = roc_auc_score(all_labels, all_preds) if len(np.unique(all_labels)) > 1 else 0.5
            
            global_zeroshot.append(zs_auc)
            global_calibrated.append(calib_auc)
            
            print(f"  Subject {subj:02d} | Zero-Shot: {zs_auc:.4f} | Calibrated: {calib_auc:.4f}")

    print("\n=======================================================")
    print(f" FINAL RESULTS ACROSS ALL 18 SUBJECTS:")
    print(f" Average Zero-Shot AUROC:  {np.mean(global_zeroshot):.4f}")
    print(f" Average Calibrated AUROC: {np.mean(global_calibrated):.4f}")
    print("=======================================================\n")

if __name__ == "__main__":
    main()
