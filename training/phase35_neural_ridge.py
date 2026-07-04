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

class ResidualSpatialAdapter(nn.Module):
    def __init__(self, in_channels=16, hidden=32):
        super().__init__()
        # 1x1 convolutions act as spatial mixing matrices across the 16 channels
        self.conv1 = nn.Conv1d(in_channels, hidden, kernel_size=1)
        self.gelu = nn.GELU()
        self.norm = nn.BatchNorm1d(hidden)
        self.conv2 = nn.Conv1d(hidden, in_channels, kernel_size=1)
        
        # Initialize second convolution to exactly zero
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)
        
        # Alpha controls the magnitude of the residual
        self.alpha = nn.Parameter(torch.tensor(0.0))
        
    def forward(self, x):
        res = self.conv1(x)
        res = self.norm(res)
        res = self.gelu(res)
        res = self.conv2(res)
        return x + self.alpha * res

class NeuralRidgeDecoder(nn.Module):
    def __init__(self, in_channels=16, lags=24):
        super().__init__()
        self.lags = lags
        self.adapter = ResidualSpatialAdapter(in_channels=in_channels, hidden=32)
        
        # Classical Ridge Decoder (Frozen)
        self.ridge_decoder = nn.Conv1d(in_channels, 1, kernel_size=lags+1, bias=False)
        for p in self.ridge_decoder.parameters():
            p.requires_grad = False
            
    def load_analytical_weights(self, W):
        """
        W: numpy array of shape [16 * 25] ordered as c0_lag0..24, c1_lag0..24...
        PyTorch Conv1d weight: [out_channels, in_channels, kernel_size] = [1, 16, 25]
        PyTorch Causal Padding maps kernel index K=24 to Lag=0, and K=0 to Lag=24.
        """
        W_reshaped = W.reshape(16, 25) # [Channels, Lags]
        # Reverse the lag dimension for PyTorch causal convolution
        W_pytorch = np.flip(W_reshaped, axis=1).copy()
        
        with torch.no_grad():
            self.ridge_decoder.weight.copy_(torch.from_numpy(W_pytorch).unsqueeze(0))
            
    def forward(self, x):
        # x: [B, C, T]
        x_adapted = self.adapter(x)
        x_pad = nn.functional.pad(x_adapted, (self.lags, 0))
        return self.ridge_decoder(x_pad).squeeze(1)

def safe_corr_torch(x, y, eps=1e-8):
    x_mean = x.mean(dim=-1, keepdim=True)
    y_mean = y.mean(dim=-1, keepdim=True)
    x_centered = x - x_mean
    y_centered = y - y_mean
    
    cov = (x_centered * y_centered).sum(dim=-1)
    x_var = (x_centered ** 2).sum(dim=-1)
    y_var = (y_centered ** 2).sum(dim=-1)
    
    corr = cov / (torch.sqrt(x_var * y_var) + eps)
    return corr

def pearson_loss(pred, target):
    corr = safe_corr_torch(pred, target)
    return 1.0 - corr.mean()

def build_lagged_matrix(eeg, max_lag):
    lags = np.arange(0, max_lag + 1)
    C, T = eeg.shape
    num_lags = len(lags)
    
    out_T = T - max_lag
    X = np.zeros((out_T, C * num_lags), dtype=np.float32)
    
    for c in range(C):
        for i, lag in enumerate(lags):
            start_idx = max_lag - lag
            end_idx = T - lag
            idx = c * num_lags + i
            X[:, idx] = eeg[c, start_idx:end_idx]
            
    return X

def run_neural_ridge():
    print("--- 1. Loading AASD Dataset ---")
    data_root = '/kaggle/input/datasets/lokeshgile/aasd-processed-eeg/Processed EEG'
    mat_files = []
    for root, dirs, files in os.walk(data_root):
        for file in files:
            if file.endswith('.mat') and not file.startswith('._'):
                mat_files.append(os.path.join(root, file))
    
    sub_path = next((p for p in mat_files if 'S18' in p), mat_files[0])
    b, a = scipy.signal.butter(4, [1.0/64.0, 8.0/64.0], btype='band')
    audio_dir = '/kaggle/input/datasets/lokeshgile/aasd-audio-gammatones'
    wav_dir = '/kaggle/input/datasets/lokeshgile/aasd-audio/Stimuli Audio'
    
    trials = load_aasd_subject_trials(sub_path, b, a, audio_dir, wav_dir)
    print(f"Loaded {len(trials)} trials from {os.path.basename(sub_path)}")
    
    train_trials = trials[:40]
    test_trials = trials[40:]
    
    # Optimal parameters from Phase 34
    max_lag = 24
    alpha_ridge = 10000.0
    selected_channels = [29, 26, 31, 5, 13, 21, 22, 44, 55, 41, 45, 15, 17, 56, 61, 60]
    num_lags = max_lag + 1
    C = len(selected_channels)
    
    window_len = 64 * 5
    hop_len = 64 * 1
    
    print("\n--- 2. Computing Analytical Ridge Solution ---")
    X_train_list, Y_train_list = [], []
    X_train_raw, Y_train_raw = [], [] # For neural network dataloader
    
    for trial in train_trials:
        eeg_full = trial['eeg'].numpy()
        env_l = trial['env_l'].numpy()
        env_r = trial['env_r'].numpy()
        switch_points = trial['meta']['switch_points']
        
        eeg_full = (eeg_full - eeg_full.mean(axis=1, keepdims=True)) / (eeg_full.std(axis=1, keepdims=True) + 1e-8)
        env_l = (env_l - env_l.mean()) / (env_l.std() + 1e-8)
        env_r = (env_r - env_r.mean()) / (env_r.std() + 1e-8)
        
        # Subset channels
        eeg = eeg_full[selected_channels, :]
        
        X = build_lagged_matrix(eeg, max_lag)
        att = np.zeros(eeg.shape[1], dtype=np.float32)
        if len(switch_points) == 0:
            switch_points = [('R', 0)]
            
        initial_state = 'R' if (switch_points[0][1] > 0 and switch_points[0][0] == 'L') else switch_points[0][0]
        if switch_points[0][1] > 0:
            initial_state = 'R' if switch_points[0][0] == 'L' else 'L'
            
        current_state = initial_state
        prev_idx = 0
        for state, idx_64 in switch_points:
            if idx_64 > prev_idx:
                if current_state == 'L': att[prev_idx:idx_64] = env_l[prev_idx:idx_64]
                else: att[prev_idx:idx_64] = env_r[prev_idx:idx_64]
            prev_idx, current_state = idx_64, state
            
        if current_state == 'L': att[prev_idx:] = env_l[prev_idx:]
        else: att[prev_idx:] = env_r[prev_idx:]
            
        Y = att[max_lag:]
        X_train_list.append(X)
        Y_train_list.append(Y)
        
        # Prepare NN windows
        for start in range(0, eeg.shape[1] - window_len + 1, hop_len):
            end = start + window_len
            is_transition = False
            for _, sw_idx in switch_points:
                if start < sw_idx < end:
                    is_transition = True
                    break
            
            if not is_transition:
                X_train_raw.append(torch.from_numpy(eeg[:, start:end]))
                Y_train_raw.append(torch.from_numpy(att[start:end]))
        
    X_train = np.vstack(X_train_list)
    Y_train = np.concatenate(Y_train_list)
    
    cov_X = X_train.T @ X_train
    cov_XY = X_train.T @ Y_train
    
    ridge_matrix = cov_X + alpha_ridge * np.eye(cov_X.shape[0])
    W_analytical = np.linalg.solve(ridge_matrix, cov_XY)
    
    print("Analytical Ridge Weights Computed.")
    
    print("\n--- 3. Initializing Neural Ridge Hybrid ---")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = NeuralRidgeDecoder(in_channels=C, lags=max_lag).to(device)
    model.load_analytical_weights(W_analytical)
    
    # Prove that it perfectly matches the analytical solution at initialization
    model.eval()
    X_nn = torch.stack(X_train_raw)
    Y_nn = torch.stack(Y_train_raw)
    dataset = TensorDataset(X_nn, Y_nn)
    dataloader = DataLoader(dataset, batch_size=256, shuffle=True)
    
    init_loss = 0.0
    with torch.no_grad():
        for bx, by in dataloader:
            pred = model(bx.to(device)).cpu()
            init_loss += pearson_loss(pred, by).item()
    print(f"Pre-Training Neural Ridge Loss (Identical to Classical Ridge): {init_loss/len(dataloader):.4f}")
    
    print("\n--- 4. Training Residual Spatial Adapter ---")
    # Optimize only the spatial adapter
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3, weight_decay=1e-2)
    
    model.train()
    start_time = time.time()
    for epoch in range(1, 101):
        total_loss = 0.0
        for bx, by in dataloader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            pred = model(bx)
            loss = pearson_loss(pred, by)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d} | Train Pearson Loss: {total_loss / len(dataloader):.4f}")
            
    print(f"Neural Hybrid Training completed in {time.time() - start_time:.1f}s")
    
    print("\n--- 5. Testing Neural Ridge Hybrid ---")
    model.eval()
    sim_att, sim_unatt = [], []
    
    with torch.no_grad():
        for trial in test_trials:
            eeg_full = trial['eeg'].numpy()
            env_l = trial['env_l'].numpy()
            env_r = trial['env_r'].numpy()
            switch_points = trial['meta']['switch_points']
            
            eeg_full = (eeg_full - eeg_full.mean(axis=1, keepdims=True)) / (eeg_full.std(axis=1, keepdims=True) + 1e-8)
            env_l = (env_l - env_l.mean()) / (env_l.std() + 1e-8)
            env_r = (env_r - env_r.mean()) / (env_r.std() + 1e-8)
            
            eeg = eeg_full[selected_channels, :]
            
            att = np.zeros(eeg.shape[1], dtype=np.float32)
            unatt = np.zeros(eeg.shape[1], dtype=np.float32)
            if len(switch_points) == 0: switch_points = [('R', 0)]
                
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
                
                eeg_w = torch.from_numpy(eeg[:, start:end]).unsqueeze(0).to(device)
                att_w = att[start:end]
                unatt_w = unatt[start:end]
                
                pred_w = model(eeg_w).squeeze().cpu().numpy()
                
                c_a = np.corrcoef(pred_w, att_w)[0, 1]
                c_u = np.corrcoef(pred_w, unatt_w)[0, 1]
                
                if not np.isnan(c_a) and not np.isnan(c_u):
                    sim_att.append(c_a)
                    sim_unatt.append(c_u)
                    
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
    run_neural_ridge()
