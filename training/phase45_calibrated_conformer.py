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
import gc
import copy

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.aad_conformer import AADConformer
from training.phase35_neural_ridge import pearson_loss, build_lagged_matrix

def safe_pearson(x, y):
    if np.var(x) < 1e-8 or np.var(y) < 1e-8:
        return 0.0
    return np.corrcoef(x, y)[0, 1]

class LayerwiseAdaptationModel(nn.Module):
    def __init__(self, pretrained_conformer, selected_channels, embed_dim=64, max_lag=24):
        super().__init__()
        self.selected_channels = selected_channels
        self.max_lag = max_lag
        
        self.backbone = pretrained_conformer
        
        self.ridge_decoder = nn.Conv1d(embed_dim, 1, kernel_size=max_lag + 1, bias=False)
        # Initialize randomly, but it will be trained
        
        # We don't need residual adapter for PROJECTION_ONLY, but keeping it for compatibility
        self.residual_adapter = nn.Sequential(
            nn.Conv1d(embed_dim, embed_dim, kernel_size=1),
            nn.SiLU(),
            nn.Conv1d(embed_dim, embed_dim, kernel_size=1)
        )
        self.reset_residual_adapter()
        for param in self.residual_adapter.parameters():
            param.requires_grad = False

    def reset_residual_adapter(self):
        nn.init.zeros_(self.residual_adapter[-1].weight)
        nn.init.zeros_(self.residual_adapter[-1].bias)

    def extract_features(self, x):
        T = x.size(-1)
        x = x.unsqueeze(1)
        x = self.backbone.temporal_conv(x)
        x = self.backbone.temporal_norm(x)
        x = self.backbone.spatial_conv(x)
        x = self.backbone.spatial_norm(x)
        x = self.backbone.stem_act(x)
        x = self.backbone.stem_dropout(x)
        x = x.squeeze(2)
        
        x = self.backbone.tokenization(x)
        x = self.backbone.pos_encoder(x)
        
        for block in self.backbone.conformer_blocks:
            x = block(x)
            
        x = self.backbone.upsample(x)
        if x.size(-1) != T:
            x = F.interpolate(x, size=T, mode='linear', align_corners=False)
        x = self.backbone.upsample_act(x)
        return x

    def _build_lagged_tensor(self, x):
        return F.pad(x, (self.max_lag, 0))

    def load_analytical_weights(self, W):
        W_reshaped = W.reshape(64, self.max_lag + 1)
        W_pytorch = np.flip(W_reshaped, axis=1).copy()
        with torch.no_grad():
            self.ridge_decoder.weight.copy_(torch.from_numpy(W_pytorch).float().unsqueeze(0))

    def forward(self, x):
        x_8ch = x[:, self.selected_channels, :]
        z = self.extract_features(x_8ch)
        
        z_residual = self.residual_adapter(z)
        z_combined = z + z_residual
            
        z_lagged = self._build_lagged_tensor(z_combined)
        out = self.ridge_decoder(z_lagged)
        return out.squeeze(1)


def run_test_suite(model, test_trials, device, max_lag, hop_len, window_len):
    sim_att, sim_unatt = [], []
    for trial in test_trials:
        eeg_full = trial['eeg'].numpy()
        env_l = trial['env_l'].numpy()
        env_r = trial['env_r'].numpy()
        switch_points = trial['meta']['switch_points']
        
        eeg = (eeg_full - eeg_full.mean(axis=1, keepdims=True)) / (eeg_full.std(axis=1, keepdims=True) + 1e-8)
        env_l = (env_l - env_l.mean()) / (env_l.std() + 1e-8)
        env_r = (env_r - env_r.mean()) / (env_r.std() + 1e-8)
        
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
            
        eeg_tensor = torch.from_numpy(eeg).float().unsqueeze(0).to(device)
        with torch.no_grad():
            pred = model(eeg_tensor).squeeze(0).cpu().numpy()
            
        num_windows = (eeg.shape[1] - window_len) // hop_len + 1
        if num_windows <= 0: continue
        
        for i in range(num_windows):
            start = i * hop_len
            end = start + window_len
            
            pred_w = pred[start:end]
            att_w = att[start:end]
            unatt_w = unatt[start:end]
            
            var_pred = np.var(pred_w)
            if var_pred > 1e-8 and np.var(att_w) > 1e-8:
                sim_att.append(safe_pearson(pred_w, att_w))
            else:
                sim_att.append(0.0)
                
            if var_pred > 1e-8 and np.var(unatt_w) > 1e-8:
                sim_unatt.append(safe_pearson(pred_w, unatt_w))
            else:
                sim_unatt.append(0.0)
                
    if not sim_att: return 0.5
    y_true = np.concatenate([np.ones(len(sim_att)), np.zeros(len(sim_unatt))])
    y_scores = np.concatenate([sim_att, sim_unatt])
    return roc_auc_score(y_true, y_scores)


def collate_fn(batch):
    max_len = max(x.size(1) for x, y in batch)
    x_padded = torch.zeros(len(batch), batch[0][0].size(0), max_len)
    y_padded = torch.zeros(len(batch), max_len)
    for i, (x, y) in enumerate(batch):
        x_padded[i, :, :x.size(1)] = x
        y_padded[i, :y.size(0)] = y
    return x_padded, y_padded

def load_clean_backbone():
    pretrained_conformer = AADConformer(channels=64, embed_dim=64).to('cpu')
    ckpt_path = REPO_ROOT / 'conformer_checkpoints_seed1.zip'
    import zipfile
    import io
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing {ckpt_path}")
        
    with zipfile.ZipFile(ckpt_path, 'r') as z:
        pt_files = [f for f in z.namelist() if f.endswith('.pt')]
        pt_files.sort()
        best_ckpt = pt_files[-1]
        with z.open(best_ckpt) as f:
            buffer = io.BytesIO(f.read())
            state_dict = torch.load(buffer, map_location='cpu', weights_only=False)
            
    new_state_dict = {k[7:] if k.startswith('module.') else k: v for k, v in state_dict.items()}
    pretrained_conformer.load_state_dict(new_state_dict, strict=False)
    return pretrained_conformer

def get_weights_numpy(model):
    weight_tensor = model.ridge_decoder.weight.data.squeeze(0).cpu().numpy()
    W_reshaped = np.flip(weight_tensor, axis=1)
    W_flat = W_reshaped.flatten()
    return W_flat

def main():
    print("=======================================================")
    print(" PHASE 45: 2+3 CALIBRATED PROJECTION HEAD              ")
    print("=======================================================")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    cache_dir = Path('/kaggle/working/eeg_cache')
    if not cache_dir.exists():
        print("ERROR: Cache directory not found! Run Phase 41 first to generate cache.")
        return
        
    subject_ids = []
    for pt_file in cache_dir.glob("*_processed.pt"):
        subject_ids.append(pt_file.name.split('_')[0])
    subject_ids.sort()
    
    PHYSICAL_8_CHANNELS = [0, 2, 5, 13, 23, 31, 41, 49]
    max_lag = 24
    window_len = 64 * 5
    hop_len = 64 * 1
    calibration_trials = 5
    lambda_val = 1000.0 # Prior Regularization Strength
    
    print(f"Found {len(subject_ids)} subjects. Beginning Cross-Subject LOSO Evaluation...\n")
    
    all_auroc_stage1 = []
    all_auroc_stage2 = []
    
    for test_subject in subject_ids:
        print(f"==========================================")
        print(f" LOSO FOLD: Test Subject {test_subject}")
        print(f"==========================================")
        
        # Build Cross-Subject Train Data
        train_data = []
        for sid in subject_ids:
            if sid != test_subject:
                cached = torch.load(cache_dir / f"{sid}_processed.pt", weights_only=False)
                train_data.extend(list(zip(cached['X'], cached['Y'])))
                
        # Build Within-Subject Test Data
        cached = torch.load(cache_dir / f"{test_subject}_processed.pt", weights_only=False)
        all_test_X = cached['X']
        all_test_Y = cached['Y']
        all_test_raw = cached['raw']
        
        calib_X = all_test_X[:calibration_trials]
        calib_Y = all_test_Y[:calibration_trials]
        eval_raw = all_test_raw[calibration_trials:]
        
        train_loader = DataLoader(train_data, batch_size=4, shuffle=True, collate_fn=collate_fn)
        
        # Load KUL Pretrained Backbone
        model = LayerwiseAdaptationModel(load_clean_backbone(), PHYSICAL_8_CHANNELS, embed_dim=64, max_lag=max_lag).to(device)
        
        # Initialize Ridge Decoder Randomly (because we are moving from 64 to 8 channels, we need to learn a new projection)
        nn.init.xavier_uniform_(model.ridge_decoder.weight)
        
        # Freeze all layers except ridge_decoder
        for param in model.parameters():
            param.requires_grad = False
        for param in model.ridge_decoder.parameters():
            param.requires_grad = True
            
        print(f"  -> [Stage 1] Pre-training Projection Head on 17 subjects for 10 epochs...")
        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3, weight_decay=1e-2)
        
        model.eval() # Keep BatchNorms frozen
        model.ridge_decoder.train() # Only train the decoder
        
        pretrain_epochs = 10
        for epoch in range(1, pretrain_epochs + 1):
            epoch_loss = 0.0
            for bx, by in train_loader:
                bx, by = bx.to(device), by.to(device)
                optimizer.zero_grad()
                pred = model(bx)
                loss = pearson_loss(pred, by)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.ridge_decoder.parameters(), max_norm=1.0)
                optimizer.step()
                epoch_loss += loss.item()
            if epoch == pretrain_epochs or epoch % 5 == 0:
                print(f"     Epoch {epoch:02d} | Loss: {epoch_loss/len(train_loader):.4f}")
                
        model.eval()
        auroc_s1 = run_test_suite(model, eval_raw, device, max_lag, hop_len, window_len)
        print(f"  -> [Stage 1] Pre-Trained (Zero-Shot) AUROC: {auroc_s1:.4f}")
        all_auroc_stage1.append(auroc_s1)
        
        # -------------------------------------------------------------
        # STAGE 2: Prior-Regularized Analytical Ridge Calibration
        # -------------------------------------------------------------
        print(f"  -> [Stage 2] Prior-Regularized Calibration on {calibration_trials} trials of {test_subject}...")
        
        # Extract Prior Weights (W_0) from Stage 1
        W_0 = get_weights_numpy(model)
        num_features = 64 * (max_lag + 1)
        
        ZtZ = np.zeros((num_features, num_features), dtype=np.float32)
        ZtY = np.zeros(num_features, dtype=np.float32)
        
        with torch.no_grad():
            for x, y in zip(calib_X, calib_Y):
                eeg_tensor = x.unsqueeze(0).to(device)
                eeg_8ch = eeg_tensor[:, PHYSICAL_8_CHANNELS, :]
                z = model.extract_features(eeg_8ch).squeeze(0).cpu().numpy()
                Z = build_lagged_matrix(z, max_lag).astype(np.float32)
                y_aligned = y.numpy()[max_lag:].astype(np.float32)
                
                ZtZ += Z.T @ Z
                ZtY += Z.T @ y_aligned
                
        # W = (Z^T Z + \lambda I)^{-1} (Z^T Y + \lambda W_0)
        regularization_matrix = ZtZ + lambda_val * np.eye(num_features, dtype=np.float32)
        target_vector = ZtY + lambda_val * W_0
        
        try:
            W_calibrated = np.linalg.solve(regularization_matrix, target_vector)
            model.load_analytical_weights(W_calibrated)
        except np.linalg.LinAlgError:
            print("     [WARNING] Singular matrix encountered. Falling back to Stage 1 weights.")
            pass # Keep W_0
            
        auroc_s2 = run_test_suite(model, eval_raw, device, max_lag, hop_len, window_len)
        print(f"  -> [Stage 2] Calibrated AUROC: {auroc_s2:.4f}")
        print(f"  -> Improvement: {auroc_s2 - auroc_s1:+.4f}\n")
        all_auroc_stage2.append(auroc_s2)
        
        del cached, all_test_X, all_test_Y, all_test_raw, train_data, train_loader, model, optimizer
        gc.collect()
        
    print("=======================================================")
    print(" SUMMARY: PHASE 45 CALIBRATED PROJECTION HEAD          ")
    print("=======================================================")
    print(f"{'Subject':<10} | {'Zero-Shot':<12} | {'Calibrated':<12} | {'Diff':<12}")
    print("-" * 55)
    for sid, a1, a2 in zip(subject_ids, all_auroc_stage1, all_auroc_stage2):
        print(f"{sid:<10} | {a1:.4f}       | {a2:.4f}       | {a2-a1:+.4f}")
    print("-" * 55)
    print(f"{'MEAN':<10} | {np.mean(all_auroc_stage1):.4f}       | {np.mean(all_auroc_stage2):.4f}       | {np.mean(all_auroc_stage2)-np.mean(all_auroc_stage1):+.4f}")

if __name__ == '__main__':
    main()
