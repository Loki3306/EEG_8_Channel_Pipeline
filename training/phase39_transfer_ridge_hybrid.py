import os
import sys
import numpy as np
import scipy.io
import scipy.signal
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
from sklearn.metrics import roc_auc_score
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.aad_conformer import AADConformer
from training.phase32_5_spatial_fix import load_aasd_subject_trials
from training.phase35_neural_ridge import pearson_loss, build_lagged_matrix

def safe_pearson(x, y):
    if np.var(x) < 1e-8 or np.var(y) < 1e-8:
        return np.nan
    return np.corrcoef(x, y)[0, 1]

class HybridTransferDecoder(nn.Module):
    def __init__(self, pretrained_conformer, selected_channels, embed_dim=64, max_lag=24):
        super().__init__()
        
        # HARDWARE CONSTRAINT: We can only use 8 physical channels
        self.selected_channels = selected_channels
        
        # 1. Frozen Conformer Backbone
        self.backbone = pretrained_conformer
        for param in self.backbone.parameters():
            param.requires_grad = False
            
        # 2. Dynamic Switch Decoder (Neural Ridge Adapter)
        self.max_lag = max_lag
        
        # We need a temporal Conv1d (Ridge)
        self.ridge_decoder = nn.Conv1d(embed_dim, 1, kernel_size=max_lag + 1, bias=False)
        
        # A tiny residual adapter for fast fine-tuning on the latent space
        self.residual_adapter = nn.Sequential(
            nn.Conv1d(embed_dim, embed_dim, kernel_size=1),
            nn.SiLU(),
            nn.Conv1d(embed_dim, embed_dim, kernel_size=1)
        )
        nn.init.zeros_(self.residual_adapter[-1].weight)
        nn.init.zeros_(self.residual_adapter[-1].bias)

    def extract_features(self, x):
        # x is [B, 8, T]
        T = x.size(-1)
        x = x.unsqueeze(1) # [B, 1, 8, T]
        x = self.backbone.temporal_conv(x)
        x = self.backbone.temporal_norm(x)
        x = self.backbone.spatial_conv(x)
        x = self.backbone.spatial_norm(x)
        x = self.backbone.stem_act(x)
        x = self.backbone.stem_dropout(x)
        x = x.squeeze(2) # [B, F, T]
        
        x = self.backbone.tokenization(x)
        x = self.backbone.pos_encoder(x)
        
        for block in self.backbone.conformer_blocks:
            x = block(x)
            
        x = self.backbone.upsample(x)
        if x.size(-1) != T:
            x = F.interpolate(x, size=T, mode='linear', align_corners=False)
        x = self.backbone.upsample_act(x)
        return x # [B, 64, T]

    def _build_lagged_tensor(self, x):
        return F.pad(x, (self.max_lag, 0))

    def load_analytical_weights(self, W):
        W_reshaped = W.reshape(64, self.max_lag + 1) # [Channels, Lags]
        # Reverse the lag dimension for PyTorch causal convolution
        W_pytorch = np.flip(W_reshaped, axis=1).copy()
        with torch.no_grad():
            self.ridge_decoder.weight.copy_(torch.from_numpy(W_pytorch).float().unsqueeze(0))

    def forward(self, x):
        # x: [B, 62, T] -> Downselect to 8 physical channels
        x_8ch = x[:, self.selected_channels, :]
        
        with torch.no_grad():
            z = self.extract_features(x_8ch)
            
        z_residual = self.residual_adapter(z)
        z_combined = z + z_residual
        
        z_lagged = self._build_lagged_tensor(z_combined)
        out = self.ridge_decoder(z_lagged)
        return out.squeeze(1)


def run_test_suite(model, test_trials, device, max_lag, hop_len, window_len, suite_mode, offset_sec=0.0):
    sim_att, sim_unatt = [], []
    for trial_idx, trial in enumerate(test_trials):
        eeg_full = trial['eeg'].numpy()
        env_l = trial['env_l'].numpy()
        env_r = trial['env_r'].numpy()
        switch_points = trial['meta']['switch_points']
        
        if suite_mode == "SHUFFLE_AUDIO":
            alt_trial = test_trials[(trial_idx + 1) % len(test_trials)]
            env_l = alt_trial['env_l'].numpy()
            env_r = alt_trial['env_r'].numpy()
            switch_points = alt_trial['meta']['switch_points']
            
            min_len = min(eeg_full.shape[1], len(env_l))
            eeg_full = eeg_full[:, :min_len]
            env_l = env_l[:min_len]
            env_r = env_r[:min_len]
            
        eeg = (eeg_full - eeg_full.mean(axis=1, keepdims=True)) / (eeg_full.std(axis=1, keepdims=True) + 1e-8)
        env_l = (env_l - env_l.mean()) / (env_l.std() + 1e-8)
        env_r = (env_r - env_r.mean()) / (env_r.std() + 1e-8)
        
        if suite_mode == "REVERSE_EEG":
            eeg = np.flip(eeg, axis=1).copy()
        elif suite_mode == "NOISE_EEG":
            eeg = np.random.randn(*eeg.shape).astype(np.float32)
            
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
            idx_64 = min(idx_64, eeg.shape[1])
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
            
        if suite_mode == "TEMPORAL_OFFSET":
            offset_samples = int(offset_sec * 64)
            if offset_samples > 0:
                att = np.concatenate([np.zeros(offset_samples), att[:-offset_samples]])
                unatt = np.concatenate([np.zeros(offset_samples), unatt[:-offset_samples]])
            elif offset_samples < 0:
                att = np.concatenate([att[-offset_samples:], np.zeros(-offset_samples)])
                unatt = np.concatenate([unatt[-offset_samples:], np.zeros(-offset_samples)])
                
        eeg_tensor = torch.from_numpy(eeg).float().unsqueeze(0).to(device)
        with torch.no_grad():
            pred = model(eeg_tensor).squeeze(0).cpu().numpy()
            
        for start in range(0, eeg.shape[1] - window_len + 1, hop_len):
            end = start + window_len
            
            pred_w = pred[start:end]
            att_w = att[start:end]
            unatt_w = unatt[start:end]
            
            var_pred = np.var(pred_w)
            var_att = np.var(att_w)
            var_unatt = np.var(unatt_w)
            
            if var_pred > 1e-8 and var_att > 1e-8:
                c_a = np.corrcoef(pred_w, att_w)[0, 1]
            else:
                c_a = np.nan
                
            if var_pred > 1e-8 and var_unatt > 1e-8:
                c_u = np.corrcoef(pred_w, unatt_w)[0, 1]
            else:
                c_u = np.nan
            
            if not np.isnan(c_a) and not np.isnan(c_u):
                sim_att.append(c_a)
                sim_unatt.append(c_u)
                
    if len(sim_att) == 0:
        return 0.5, 0.5, 0.0, 0.0
        
    sim_att = np.array(sim_att)
    sim_unatt = np.array(sim_unatt)
    
    margin = sim_att - sim_unatt
    acc = np.mean(margin > 0)
    y_true = np.concatenate([np.ones(len(sim_att)), np.zeros(len(sim_unatt))])
    y_scores = np.concatenate([sim_att, sim_unatt])
    
    try:
        auroc = roc_auc_score(y_true, y_scores)
    except ValueError:
        auroc = 0.5
    
    return auroc, acc, sim_att.mean(), sim_unatt.mean()

def run_transfer_learning():
    print("=======================================================")
    print(" PHASE 39: KUL -> AASD TRANSFER LEARNING (HYBRID)      ")
    print("=======================================================")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # 1. Load Pretrained KUL Conformer
    ckpt_path = REPO_ROOT / 'conformer_loso_results' / 'checkpoints' / 'seed_123' / 'model_S1.pt'
    if not ckpt_path.exists():
        print(f"ERROR: Could not find checkpoint {ckpt_path}")
        return
        
    pretrained_conformer = AADConformer(in_channels=8).to(device)
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=True)
    
    # Handle DataParallel prefix if necessary
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k
        new_state_dict[name] = v
        
    pretrained_conformer.load_state_dict(new_state_dict, strict=False)
    pretrained_conformer.eval()
    print(f"Loaded Pretrained KUL Conformer (8-Channel, 64-Dim Latent)")
    
    data_root = '/kaggle/input/datasets/lokeshgile/aasd-processed-eeg/Processed EEG'
    mat_files = []
    for root, dirs, files in os.walk(data_root):
        for file in files:
            if file.endswith('.mat') and not file.startswith('._'):
                mat_files.append(os.path.join(root, file))
                
    if not mat_files:
        print("ERROR: No .mat files found. Please run on Kaggle.")
        return
        
    b, a = scipy.signal.butter(4, [1.0/64.0, 8.0/64.0], btype='band')
    audio_dir = '/kaggle/input/datasets/lokeshgile/aasd-audio-gammatones'
    wav_dir = '/kaggle/input/datasets/lokeshgile/aasd-audio/Stimuli Audio'
    
    max_lag = 24
    window_len = 64 * 5
    hop_len = 64 * 1
    
    # We will just test Transfer Learning Within-Subject (Subject 18) for speed and verification
    # Once verified, we can do LOSO.
    target_subject = 'S18'
    target_path = next(p for p in mat_files if target_subject in os.path.basename(p))
    
    print(f"\n--- 1. Loading AASD Dataset ({target_subject}) ---")
    trials = load_aasd_subject_trials(target_path, b, a, audio_dir, wav_dir)
    print(f"Loaded {len(trials)} trials from {os.path.basename(target_path)}")
    
    train_trials = trials[:40]
    test_trials = trials[40:]
    
    # ---------------------------------------------------------
    # HARDWARE CONSTRAINT: 8 PHYSICAL CHANNELS
    # Using specific neuroscan indices mapped to KUL spatial montage
    # Fp1(0), Fp2(2), F7(5), F8(13), T7(23), T8(31), P7(41), P8(49)
    # ---------------------------------------------------------
    PHYSICAL_8_CHANNELS = [0, 2, 5, 13, 23, 31, 41, 49]
    
    # 2. Extract Latents and Build Covariance Matrix
    model = HybridTransferDecoder(pretrained_conformer, selected_channels=PHYSICAL_8_CHANNELS, embed_dim=64, max_lag=max_lag).to(device)
    
    print("\n--- 2. Initializing Ridge Decoder (Analytical) ---")
    
    # To initialize the Ridge decoder analytically, we must extract the latents via the randomly initialized spatial adapter
    model.eval()
    Z_train_list = []
    Y_train_list = []
    
    with torch.no_grad():
        for trial in train_trials:
            eeg = trial['eeg'].numpy()
            env_l = trial['env_l'].numpy()
            env_r = trial['env_r'].numpy()
            switch_points = trial['meta']['switch_points']
            
            eeg = (eeg - eeg.mean(axis=1, keepdims=True)) / (eeg.std(axis=1, keepdims=True) + 1e-8)
            env_l = (env_l - env_l.mean()) / (env_l.std() + 1e-8)
            env_r = (env_r - env_r.mean()) / (env_r.std() + 1e-8)
            
            eeg_tensor = torch.from_numpy(eeg).float().unsqueeze(0).to(device)
            # Downselect to 8 channels and extract features
            eeg_8ch = eeg_tensor[:, PHYSICAL_8_CHANNELS, :]
            z = model.extract_features(eeg_8ch).squeeze(0).cpu().numpy()
            
            Z = build_lagged_matrix(z, max_lag) # [T, 64 * max_lag]
            
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
            Z_train_list.append(Z)
            Y_train_list.append(Y)
            
    Z_mat = np.vstack(Z_train_list)
    Y_mat = np.concatenate(Y_train_list)
    
    Z_tensor = torch.from_numpy(Z_mat).float().to(device)
    Y_tensor = torch.from_numpy(Y_mat).float().to(device)
    
    cov_Z = (Z_tensor.T @ Z_tensor).cpu().numpy()
    cov_ZY = (Z_tensor.T @ Y_tensor).cpu().numpy()
    
    alpha = 10000.0
    ridge_matrix = cov_Z + alpha * np.eye(cov_Z.shape[0])
    W_analytical = np.linalg.solve(ridge_matrix, cov_ZY)
    
    model.load_analytical_weights(W_analytical)
    
    # PRE-TRAIN (Pure Ridge on Frozen Latents)
    auroc_base, _, _, _ = run_test_suite(model, test_trials, device, max_lag, hop_len, window_len, "NORMAL")
    print(f"  -> PRE-TRAIN AUROC: {auroc_base:.4f} (Zero-Shot Latent Feature Extraction)")
    
    # 3. Personalized Fine-Tuning
    # We fine-tune ONLY the Residual Adapter (to slightly adjust the latents to the new subject's skull)
    train_X_nn = torch.stack([trial['eeg'] for trial in train_trials])
    train_Y_nn = []
    
    for trial in train_trials:
        env_l = trial['env_l'].numpy()
        env_r = trial['env_r'].numpy()
        switch_points = trial['meta']['switch_points']
        
        env_l = (env_l - env_l.mean()) / (env_l.std() + 1e-8)
        env_r = (env_r - env_r.mean()) / (env_r.std() + 1e-8)
        
        att = np.zeros(len(env_l), dtype=np.float32)
        if len(switch_points) == 0: switch_points = [('R', 0)]
            
        initial_state = 'R' if (switch_points[0][1] > 0 and switch_points[0][0] == 'L') else switch_points[0][0]
        if switch_points[0][1] > 0:
            initial_state = 'R' if switch_points[0][0] == 'L' else 'L'
            
        current_state = initial_state
        prev_idx = 0
        for state, idx_64 in switch_points:
            idx_64 = min(idx_64, len(env_l))
            if idx_64 > prev_idx:
                if current_state == 'L': att[prev_idx:idx_64] = env_l[prev_idx:idx_64]
                else: att[prev_idx:idx_64] = env_r[prev_idx:idx_64]
            prev_idx, current_state = idx_64, state
            
        if current_state == 'L': att[prev_idx:] = env_l[prev_idx:]
        else: att[prev_idx:] = env_r[prev_idx:]
            
        train_Y_nn.append(torch.from_numpy(att).float())
        
    train_Y_nn = torch.stack(train_Y_nn)
    
    dataset = TensorDataset(train_X_nn, train_Y_nn)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    # Only optimizing the Residual Adapter!
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3, weight_decay=1e-2)
    model.train()
    
    print("\n--- 3. Training Residual Latent Adapter (AASD Transfer) ---")
    for epoch in range(1, 26): 
        epoch_loss = 0.0
        for bx, by in dataloader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            pred = model(bx)
            loss = pearson_loss(pred, by)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
        if epoch % 5 == 0:
            print(f"Epoch {epoch:2d} | Train Pearson Loss: {epoch_loss / len(dataloader):.4f}")
            
    print("\n--- 4. Falsification Testing ---")
    model.eval()
    suites = [
        ("NORMAL", "NORMAL", 0.0),
        ("SHUFFLE_AUDIO", "SHUFFLE_AUDIO", 0.0),
        ("REVERSE_EEG", "REVERSE_EEG", 0.0),
        ("NOISE_EEG", "NOISE_EEG", 0.0)
    ]
    
    for suite_name, mode, offset in suites:
        auroc, acc, _, _ = run_test_suite(model, test_trials, device, max_lag, hop_len, window_len, mode, offset)
        print(f"{suite_name:<15} | AUROC: {auroc:.4f} | Accuracy: {acc*100:.1f}%")

if __name__ == "__main__":
    run_transfer_learning()
