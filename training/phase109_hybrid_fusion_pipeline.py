import numpy as np
from scipy.linalg import solve
from scipy.stats import pearsonr
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from sklearn.metrics import roc_auc_score
import random
from scipy import signal
import copy

# -------------------------------------------------------------------------
# CONSTANTS & CONFIGURATION
# -------------------------------------------------------------------------
SR = 128
EAR_CHANNEL_INDICES = [23, 31, 32, 40, 14, 22, 41, 49]
TARGET_SUBJECTS = ['S05', 'S08', 'S10', 'S11', 'S13', 'S16']
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Phase 108 mTRF Constants
MAX_LAG_SEC = 0.400 
MAX_LAG_SAMPLES = int(MAX_LAG_SEC * SR)
MTRF_LAMBDA = 100.0 # Fixed based on Phase 108 results to speed up

# Phase 109 Deep Learning Constants
BATCH_SIZE = 128
EPOCHS = 15
LEARNING_RATE = 1e-4
SEQ_SAMPLES = int(3.5 * SR)

# -------------------------------------------------------------------------
# SIGNAL PROCESSING
# -------------------------------------------------------------------------
def apply_modulation_filter(env, lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    if lowcut is None and highcut is not None:
        b, a = signal.butter(order, highcut / nyq, btype='low')
    elif highcut is None and lowcut is not None:
        b, a = signal.butter(order, lowcut / nyq, btype='high')
    else:
        low = lowcut / nyq
        high = highcut / nyq
        b, a = signal.butter(order, [low, high], btype='band')
        
    filtered = signal.filtfilt(b, a, env, axis=1)
    return filtered

def create_toeplitz_features(eeg, max_lag_samples):
    C, T = eeg.shape
    T_eff = T - max_lag_samples
    X = np.zeros((T_eff, C * max_lag_samples))
    for tau in range(max_lag_samples):
        X[:, tau*C : (tau+1)*C] = eeg[:, tau : tau+T_eff].T
    return X

# -------------------------------------------------------------------------
# STRATIFIED SPLITTING (LEAK FREE)
# -------------------------------------------------------------------------
def get_trial_dominant_speaker(tr):
    sp = tr['meta']['switch_points']
    T = tr['eeg'].shape[1]
    
    boundaries = [0]
    boundaries.extend([idx for spk, idx in sp])
    boundaries = sorted(set(boundaries))
    if boundaries[-1] != T: boundaries.append(T)
        
    l_duration = 0
    r_duration = 0
    
    for i in range(len(boundaries) - 1):
        start_idx = boundaries[i]
        end_idx = boundaries[i+1]
        current_spk = 'L'
        for spk, idx in sp:
            if idx <= start_idx: current_spk = spk
            else: break
            
        if current_spk == 'L': l_duration += (end_idx - start_idx)
        else: r_duration += (end_idx - start_idx)
        
    return 'L' if l_duration >= r_duration else 'R'

def stratified_trial_split(trials, train_ratio=0.8):
    l_trials = []
    r_trials = []
    
    for i, tr in enumerate(trials):
        if get_trial_dominant_speaker(tr) == 'L':
            l_trials.append(i)
        else:
            r_trials.append(i)
            
    random.seed(42)
    random.shuffle(l_trials)
    random.shuffle(r_trials)
    
    l_split = int(len(l_trials) * train_ratio)
    r_split = int(len(r_trials) * train_ratio)
    
    train_indices = l_trials[:l_split] + r_trials[:r_split]
    eval_indices = l_trials[l_split:] + r_trials[r_split:]
    
    random.shuffle(train_indices)
    random.shuffle(eval_indices)
    return train_indices, eval_indices

# -------------------------------------------------------------------------
# STAGE 1: mTRF PIPELINE
# -------------------------------------------------------------------------
def extract_mtrf_matrices(trials):
    X_list = []
    Y_attended_list = []
    
    for tr in trials:
        eeg = tr['eeg']
        env_l = tr['env_l'][0]
        env_r = tr['env_r'][0]
        T = eeg.shape[1]
        
        X_trial = create_toeplitz_features(eeg, MAX_LAG_SAMPLES)
        T_eff = X_trial.shape[0]
        
        sp = tr['meta']['switch_points']
        boundaries = [0]
        boundaries.extend([idx for spk, idx in sp])
        boundaries = sorted(set(boundaries))
        if boundaries[-1] != T: boundaries.append(T)
            
        Y_att = np.zeros(T)
        for i in range(len(boundaries) - 1):
            start_idx = boundaries[i]
            end_idx = boundaries[i+1]
            current_spk = 'L'
            for spk, idx in sp:
                if idx <= start_idx: current_spk = spk
                else: break
            if current_spk == 'L':
                Y_att[start_idx:end_idx] = env_l[start_idx:end_idx]
            else:
                Y_att[start_idx:end_idx] = env_r[start_idx:end_idx]
                
        Y_trial = Y_att[:T_eff]
        X_list.append(X_trial)
        Y_attended_list.append(Y_trial)
        
    return np.vstack(X_list), np.concatenate(Y_attended_list)

def fit_ridge(X, y, lam):
    XTX = X.T @ X
    XTy = X.T @ y
    I = np.eye(XTX.shape[0])
    W = solve(XTX + lam * I, XTy, assume_a='pos')
    return W

def apply_mtrf_reconstruction(trials, W):
    augmented_trials = []
    for tr in trials:
        eeg = tr['eeg']
        T = eeg.shape[1]
        
        X_trial = create_toeplitz_features(eeg, MAX_LAG_SAMPLES)
        Y_hat_eff = X_trial @ W
        
        # Pad the missing MAX_LAG_SAMPLES with zeros to maintain temporal alignment
        Y_hat = np.zeros(T)
        Y_hat[:Y_hat_eff.shape[0]] = Y_hat_eff
        
        new_tr = {
            'eeg': eeg,
            'env_l': tr['env_l'],
            'env_r': tr['env_r'],
            'y_hat': np.expand_dims(Y_hat, axis=0),
            'meta': tr['meta']
        }
        augmented_trials.append(new_tr)
    return augmented_trials

# -------------------------------------------------------------------------
# STAGE 2: HYBRID TCN PIPELINE
# -------------------------------------------------------------------------
class HybridDataset(Dataset):
    def __init__(self, augmented_trials, seq_len):
        self.samples = []
        hop = int(0.5 * SR)
        
        for tr in augmented_trials:
            eeg = tr['eeg']
            env_l = tr['env_l'][0]
            env_r = tr['env_r'][0]
            y_hat = tr['y_hat'][0]
            T = eeg.shape[1]
            
            sp = tr['meta']['switch_points']
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
                    
                safe_start = start_idx + int(1.5 * SR)
                safe_end = end_idx
                
                if safe_end - safe_start >= seq_len:
                    for seq_start in range(safe_start, safe_end - seq_len + 1, hop):
                        eeg_seq = eeg[:, seq_start:seq_start + seq_len]
                        env_l_seq = env_l[seq_start:seq_start + seq_len]
                        env_r_seq = env_r[seq_start:seq_start + seq_len]
                        y_hat_seq = y_hat[seq_start:seq_start + seq_len]
                        
                        self.samples.append({
                            'eeg': torch.FloatTensor(eeg_seq),
                            'env_l': torch.FloatTensor(env_l_seq).unsqueeze(0),
                            'env_r': torch.FloatTensor(env_r_seq).unsqueeze(0),
                            'y_hat': torch.FloatTensor(y_hat_seq).unsqueeze(0),
                            'label': 1.0 if current_spk == 'L' else 0.0
                        })
                        
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        return self.samples[idx]

class HybridFusionTCN(nn.Module):
    def __init__(self, eeg_channels=8):
        super().__init__()
        
        # Input: 8 EEG + 1 Env + 1 Y_hat (mTRF) = 10 channels
        self.conv_blocks = nn.Sequential(
            nn.Conv1d(eeg_channels + 2, 32, kernel_size=15, padding=7),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            nn.Conv1d(32, 64, kernel_size=11, padding=5, dilation=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            nn.Conv1d(64, 64, kernel_size=7, padding=3, dilation=4),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 1)
        )
        
    def forward_match(self, eeg, env, y_hat):
        x = torch.cat([eeg, env, y_hat], dim=1)
        feat = self.conv_blocks(x).squeeze(-1)
        return self.classifier(feat)
        
    def forward(self, eeg, env_l, env_r, y_hat):
        score_l = self.forward_match(eeg, env_l, y_hat)
        score_r = self.forward_match(eeg, env_r, y_hat)
        return torch.cat([score_l, score_r], dim=1)

def evaluate_hybrid(model, dataloader):
    model.eval()
    preds = []
    labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            eeg = batch['eeg'].to(DEVICE)
            env_l = batch['env_l'].to(DEVICE)
            env_r = batch['env_r'].to(DEVICE)
            y_hat = batch['y_hat'].to(DEVICE)
            label = batch['label'].numpy()
            
            logits = model(eeg, env_l, env_r, y_hat)
            score_l = logits[:, 0].cpu().numpy()
            score_r = logits[:, 1].cpu().numpy()
            
            # Predict Left if score_l > score_r
            diff = score_l - score_r
            preds.extend(diff)
            labels.extend(label)
            
    if len(np.unique(labels)) > 1:
        return roc_auc_score(labels, preds)
    return 0.5

# -------------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------------
def main():
    possible_paths = [
        Path('/kaggle/input/datasets/lowkieee/multiband-cache/kaggle/working/multiband_cache'),
        Path('/kaggle/input/multiband-cache/kaggle/working/multiband_cache'),
        Path('/kaggle/input/aasd-universal-cache-v1/kaggle/working/multiband_cache'),
        Path('/kaggle/working/multiband_cache')
    ]
    
    cache_dir = Path('/kaggle/working/multiband_cache')
    for p in possible_paths:
        if p.exists() and len(list(p.glob('*_multiband.pt'))) > 0:
            cache_dir = p
            break
            
    print(f"\n=======================================================")
    print(f" PHASE 109: PRODUCTION HYBRID FUSION PIPELINE")
    print(f" Integrating classical mTRF reconstruction directly into TCN")
    print(f"=======================================================\n", flush=True)
    
    cache_files = sorted(list(cache_dir.glob('*_multiband.pt')))
    filtered_files = [f for f in cache_files if f.stem.split('_')[0] in TARGET_SUBJECTS]
    
    final_results = {}
    
    for cache_file in filtered_files:
        subj_name = cache_file.stem.split('_')[0]
        print(f"\n=======================================================")
        print(f" SUBJECT {subj_name}")
        print(f"=======================================================", flush=True)
        
        cached = torch.load(cache_file, map_location='cpu', weights_only=False)['raw']
        
        raw_trials = []
        for i in range(len(cached)):
            tr = cached[i]
            eeg = tr['eeg'].numpy()[EAR_CHANNEL_INDICES, :] 
            env_l = tr['env_l'].numpy()
            env_r = tr['env_r'].numpy()
            
            # Broadband / Phonemic filter (8Hz lowpass) to ensure mTRF and TCN use biological bands
            eeg = apply_modulation_filter(eeg, None, 8.0, SR)
            env_l = apply_modulation_filter(env_l, None, 8.0, SR)
            env_r = apply_modulation_filter(env_r, None, 8.0, SR)
            
            eeg = (eeg - np.mean(eeg, axis=1, keepdims=True)) / (np.std(eeg, axis=1, keepdims=True) + 1e-8)
            env_l = (env_l - np.mean(env_l, axis=1, keepdims=True)) / (np.std(env_l, axis=1, keepdims=True) + 1e-8)
            env_r = (env_r - np.mean(env_r, axis=1, keepdims=True)) / (np.std(env_r, axis=1, keepdims=True) + 1e-8)
            
            min_len = min(eeg.shape[1], env_l.shape[1])
            raw_trials.append({
                'eeg': eeg[:, :min_len], 
                'env_l': env_l[:, :min_len], 
                'env_r': env_r[:, :min_len], 
                'meta': tr['meta']
            })
            
        train_indices, eval_indices = stratified_trial_split(raw_trials, train_ratio=0.8)
        
        raw_train_trials = [raw_trials[i] for i in train_indices]
        raw_eval_trials = [raw_trials[i] for i in eval_indices]
        
        # 1. Classical Stage: Fit mTRF on Train
        print("  [Stage 1] Fitting Classical mTRF...", flush=True)
        X_train, Y_train = extract_mtrf_matrices(raw_train_trials)
        W_mtrf = fit_ridge(X_train, Y_train, MTRF_LAMBDA)
        
        # 2. Hybrid Augmentation: Generate Reconstructed Envelopes
        print("  [Stage 2] Generating mTRF Features...", flush=True)
        aug_train_trials = apply_mtrf_reconstruction(raw_train_trials, W_mtrf)
        aug_eval_trials = apply_mtrf_reconstruction(raw_eval_trials, W_mtrf)
        
        # Split Train into Train/Val for Early Stopping
        aug_train_idx, aug_val_idx = stratified_trial_split(aug_train_trials, train_ratio=0.8)
        final_train_trials = [aug_train_trials[i] for i in aug_train_idx]
        final_val_trials = [aug_train_trials[i] for i in aug_val_idx]
        
        # 3. Deep Learning Stage
        print("  [Stage 3] Training Hybrid TCN Fusion...", flush=True)
        train_ds = HybridDataset(final_train_trials, SEQ_SAMPLES)
        val_ds = HybridDataset(final_val_trials, SEQ_SAMPLES)
        eval_ds = HybridDataset(aug_eval_trials, SEQ_SAMPLES)
        
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
        eval_loader = DataLoader(eval_ds, batch_size=BATCH_SIZE, shuffle=False)
        
        model = HybridFusionTCN(eeg_channels=len(EAR_CHANNEL_INDICES)).to(DEVICE)
        optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-3)
        criterion = nn.BCEWithLogitsLoss()
        
        best_val_auc = 0
        best_state = None
        
        for ep in range(EPOCHS):
            model.train()
            for batch in train_loader:
                eeg = batch['eeg'].to(DEVICE)
                env_l = batch['env_l'].to(DEVICE)
                env_r = batch['env_r'].to(DEVICE)
                y_hat = batch['y_hat'].to(DEVICE)
                label = batch['label'].float().unsqueeze(1).to(DEVICE)
                
                optimizer.zero_grad()
                logits = model(eeg, env_l, env_r, y_hat)
                
                # Create target tensor for [L, R] logits
                # If label is 1 (Left), target is [1, 0]
                # If label is 0 (Right), target is [0, 1]
                target = torch.cat([label, 1.0 - label], dim=1)
                
                loss = criterion(logits, target)
                loss.backward()
                optimizer.step()
                
            val_auc = evaluate_hybrid(model, val_loader)
            # Correcting for true polarity (we don't blindly invert based on noise)
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_state = copy.deepcopy(model.state_dict())
                
        # 4. Deployment Eval
        model.load_state_dict(best_state)
        eval_auc = evaluate_hybrid(model, eval_loader)
        
        print(f"  [Deployment] Final Hybrid AUROC: {eval_auc:.4f}")
        final_results[subj_name] = eval_auc

    print("\n\n=======================================================")
    print(" PHASE 109 HYBRID FUSION PIPELINE RESULTS")
    print("=======================================================")
    print(f"{'Subject':<10} {'Deployment AUROC':<10}")
    for subj, auroc in final_results.items():
        print(f"{subj:<10} {auroc:.4f}")

if __name__ == '__main__':
    main()
