import os
import sys
import numpy as np
import scipy.io
import scipy.io.wavfile
import scipy.signal
from sklearn.metrics import roc_auc_score
import time
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.phase32_5_spatial_fix import load_aasd_subject_trials

class PyTorchMTRF(nn.Module):
    def __init__(self, in_channels=62, lags=16):
        super().__init__()
        # A single causal convolutional layer
        # Output is 1 channel (the predicted envelope)
        self.lags = lags
        self.conv = nn.Conv1d(in_channels, 1, kernel_size=lags+1, bias=True)
        
    def forward(self, x):
        # Causal padding: pad the left side by 'lags' samples
        x_pad = nn.functional.pad(x, (self.lags, 0))
        return self.conv(x_pad).squeeze(1) # [B, T]

def pearson_loss(pred, target):
    """
    Minimizes negative Pearson Correlation.
    pred: [B, T]
    target: [B, T]
    """
    pred_mean = pred.mean(dim=1, keepdim=True)
    target_mean = target.mean(dim=1, keepdim=True)
    
    pred_centered = pred - pred_mean
    target_centered = target - target_mean
    
    cov = (pred_centered * target_centered).sum(dim=1)
    std_pred = torch.sqrt((pred_centered ** 2).sum(dim=1) + 1e-8)
    std_target = torch.sqrt((target_centered ** 2).sum(dim=1) + 1e-8)
    
    corr = cov / (std_pred * std_target)
    return 1.0 - corr.mean() # 1 - Pearson, so minimizing maximizes correlation

def train_pytorch_mtrf():
    print("--- 1. Loading AASD Dataset ---")
    data_root = '/kaggle/input/datasets/lokeshgile/aasd-processed-eeg/Processed EEG'
    mat_files = []
    for root, dirs, files in os.walk(data_root):
        for file in files:
            if file.endswith('.mat') and not file.startswith('._'):
                mat_files.append(os.path.join(root, file))
    
    if not mat_files:
        print("ERROR: No .mat files found. Please run on Kaggle.")
        return
        
    sub_path = next((p for p in mat_files if 'S18' in p), mat_files[0])
    
    b, a = scipy.signal.butter(4, [1.0/64.0, 8.0/64.0], btype='band')
    audio_dir = '/kaggle/input/datasets/lokeshgile/aasd-audio-gammatones'
    wav_dir = '/kaggle/input/datasets/lokeshgile/aasd-audio/Stimuli Audio'
    
    trials = load_aasd_subject_trials(sub_path, b, a, audio_dir, wav_dir)
    print(f"Loaded {len(trials)} trials from {os.path.basename(sub_path)}")
    
    train_trials = trials[:40]
    test_trials = trials[40:]
    
    # 5-second windows
    window_len = 64 * 5
    hop_len = 64 * 1
    
    print("\n--- 2. Building Training Windows ---")
    X_train, Y_train = [], []
    for trial in train_trials:
        eeg = trial['eeg']
        env_l = trial['env_l']
        env_r = trial['env_r']
        switch_points = trial['meta']['switch_points']
        
        # Build attended envelope
        att = torch.zeros_like(env_l)
        if len(switch_points) == 0:
            switch_points = [('R', 0)]
            
        if switch_points[0][1] > 0:
            initial_state = 'R' if switch_points[0][0] == 'L' else 'L'
        else:
            initial_state = switch_points[0][0]
            
        current_state = initial_state
        prev_idx = 0
        for state, idx_64 in switch_points:
            if idx_64 > prev_idx:
                if current_state == 'L':
                    att[prev_idx:idx_64] = env_l[prev_idx:idx_64]
                else:
                    att[prev_idx:idx_64] = env_r[prev_idx:idx_64]
            prev_idx, current_state = idx_64, state
            
        if current_state == 'L':
            att[prev_idx:] = env_l[prev_idx:]
        else:
            att[prev_idx:] = env_r[prev_idx:]
            
        for start in range(0, eeg.shape[1] - window_len + 1, hop_len):
            end = start + window_len
            
            # Skip transition windows
            is_transition = False
            for _, sw_idx in switch_points:
                if start < sw_idx < end:
                    is_transition = True
                    break
            
            if not is_transition:
                # Normalize EEG window
                eeg_w = eeg[:, start:end].clone()
                eeg_w = (eeg_w - eeg_w.mean(dim=1, keepdim=True)) / (eeg_w.std(dim=1, keepdim=True) + 1e-8)
                
                # Normalize Audio window
                att_w = att[start:end].clone()
                att_w = (att_w - att_w.mean()) / (att_w.std() + 1e-8)
                
                X_train.append(eeg_w)
                Y_train.append(att_w)

    X_train = torch.stack(X_train)
    Y_train = torch.stack(Y_train)
    
    print(f"Generated {len(X_train)} training windows.")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_eeg_channels = X_train.size(1)
    print(f"Device: {device} | Channels: {num_eeg_channels}")
    
    dataset = TensorDataset(X_train, Y_train)
    dataloader = DataLoader(dataset, batch_size=256, shuffle=True)
    
    model = PyTorchMTRF(in_channels=num_eeg_channels, lags=16).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    
    print("\n--- 3. Training PyTorch mTRF ---")
    start_time = time.time()
    model.train()
    for epoch in range(1, 101):
        total_loss = 0.0
        for batch_eeg, batch_att in dataloader:
            batch_eeg = batch_eeg.to(device)
            batch_att = batch_att.to(device)
            
            optimizer.zero_grad()
            pred = model(batch_eeg)
            loss = pearson_loss(pred, batch_att)
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d} | Train Pearson Loss: {total_loss / len(dataloader):.4f}")
            
    print(f"Training completed in {time.time() - start_time:.1f}s")
    
    print("\n--- 4. Testing PyTorch mTRF ---")
    model.eval()
    sim_att = []
    sim_unatt = []
    
    with torch.no_grad():
        for trial in test_trials:
            eeg = trial['eeg']
            env_l = trial['env_l']
            env_r = trial['env_r']
            switch_points = trial['meta']['switch_points']
            
            att = torch.zeros_like(env_l)
            unatt = torch.zeros_like(env_l)
            
            if len(switch_points) == 0:
                switch_points = [('R', 0)]
                
            if switch_points[0][1] > 0:
                initial_state = 'R' if switch_points[0][0] == 'L' else 'L'
            else:
                initial_state = switch_points[0][0]
                
            current_state = initial_state
            prev_idx = 0
            for state, idx_64 in switch_points:
                if idx_64 > prev_idx:
                    if current_state == 'L':
                        att[prev_idx:idx_64] = env_l[prev_idx:idx_64]
                        unatt[prev_idx:idx_64] = env_r[prev_idx:idx_64]
                    else:
                        att[prev_idx:idx_64] = env_r[prev_idx:idx_64]
                        unatt[prev_idx:idx_64] = env_l[prev_idx:idx_64]
                prev_idx, current_state = idx_64, state
                
            if current_state == 'L':
                att[prev_idx:] = env_l[prev_idx:]
                unatt[prev_idx:] = env_r[prev_idx:]
            else:
                att[prev_idx:] = env_r[prev_idx:]
                unatt[prev_idx:] = env_l[prev_idx:]
                
            for start in range(0, eeg.shape[1] - window_len + 1, hop_len):
                end = start + window_len
                
                eeg_w = eeg[:, start:end].clone()
                eeg_w = (eeg_w - eeg_w.mean(dim=1, keepdim=True)) / (eeg_w.std(dim=1, keepdim=True) + 1e-8)
                
                att_w = att[start:end].clone()
                unatt_w = unatt[start:end].clone()
                
                # Predict
                pred_w = model(eeg_w.unsqueeze(0).to(device)).squeeze(0).cpu()
                
                # Pearson corr
                corr_att = np.corrcoef(pred_w.numpy(), att_w.numpy())[0, 1]
                corr_unatt = np.corrcoef(pred_w.numpy(), unatt_w.numpy())[0, 1]
                
                if not np.isnan(corr_att) and not np.isnan(corr_unatt):
                    sim_att.append(corr_att)
                    sim_unatt.append(corr_unatt)
                    
    sim_att = np.array(sim_att)
    sim_unatt = np.array(sim_unatt)
    
    margin = sim_att - sim_unatt
    acc = np.mean(margin > 0)
    
    y_true = np.concatenate([np.ones(len(sim_att)), np.zeros(len(sim_unatt))])
    y_scores = np.concatenate([sim_att, sim_unatt])
    auroc = roc_auc_score(y_true, y_scores)
    
    print(f"Test P(Att): {sim_att.mean():.4f}")
    print(f"Test P(Unatt): {sim_unatt.mean():.4f}")
    print(f"Margin Mean: {margin.mean():.4f}")
    print(f"Margin Std: {margin.std():.4f}")
    print(f"Test Accuracy: {acc*100:.1f}%")
    print(f"Test AUROC: {auroc:.4f}")

if __name__ == "__main__":
    train_pytorch_mtrf()
