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
from scipy.stats import wilcoxon
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

class LayerwiseAdaptationModel(nn.Module):
    def __init__(self, pretrained_conformer, selected_channels, embed_dim=64, max_lag=24):
        super().__init__()
        self.selected_channels = selected_channels
        self.max_lag = max_lag
        
        self.backbone = pretrained_conformer
        
        self.ridge_decoder = nn.Conv1d(embed_dim, 1, kernel_size=max_lag + 1, bias=False)
        for param in self.ridge_decoder.parameters():
            param.requires_grad = False
            
        self.residual_adapter = nn.Sequential(
            nn.Conv1d(embed_dim, embed_dim, kernel_size=1),
            nn.SiLU(),
            nn.Conv1d(embed_dim, embed_dim, kernel_size=1)
        )
        self.reset_residual_adapter()

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
            
        for start in range(0, eeg.shape[1] - window_len + 1, hop_len):
            end = start + window_len
            pred_w = pred[start:end]
            att_w = att[start:end]
            unatt_w = unatt[start:end]
            
            var_pred = np.var(pred_w)
            if var_pred > 1e-8 and np.var(att_w) > 1e-8:
                c_a = np.corrcoef(pred_w, att_w)[0, 1]
            else:
                c_a = np.nan
                
            if var_pred > 1e-8 and np.var(unatt_w) > 1e-8:
                c_u = np.corrcoef(pred_w, unatt_w)[0, 1]
            else:
                c_u = np.nan
            
            if not np.isnan(c_a) and not np.isnan(c_u):
                sim_att.append(c_a)
                sim_unatt.append(c_u)
                
    if len(sim_att) == 0:
        return 0.5
        
    sim_att = np.array(sim_att)
    sim_unatt = np.array(sim_unatt)
    margin = sim_att - sim_unatt
    y_true = np.concatenate([np.ones(len(sim_att)), np.zeros(len(sim_unatt))])
    y_scores = np.concatenate([sim_att, sim_unatt])
    return roc_auc_score(y_true, y_scores)


def run_loso_validation():
    print("=======================================================")
    print(" PHASE 41: AASD LEAVE-ONE-SUBJECT-OUT VALIDATION       ")
    print("=======================================================")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    ckpt_path = REPO_ROOT / 'conformer_loso_results' / 'checkpoints' / 'seed_123' / 'model_S1.pt'
    if not ckpt_path.exists():
        print(f"ERROR: Could not find checkpoint {ckpt_path}")
        return
        
    def load_clean_backbone():
        pretrained_conformer = AADConformer(in_channels=8).to(device)
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=True)
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        new_state_dict = {k[7:] if k.startswith('module.') else k: v for k, v in state_dict.items()}
        pretrained_conformer.load_state_dict(new_state_dict, strict=False)
        return pretrained_conformer
        
    data_root = '/kaggle/input/datasets/lokeshgile/aasd-processed-eeg/Processed EEG'
    audio_dir = '/kaggle/input/datasets/lokeshgile/aasd-audio-gammatones'
    wav_dir = '/kaggle/input/datasets/lokeshgile/aasd-audio/Stimuli Audio'
    
    mat_files = []
    if os.path.exists(data_root):
        for root, dirs, files in os.walk(data_root):
            for file in files:
                if file.endswith('.mat') and not file.startswith('._'):
                    mat_files.append(os.path.join(root, file))
    mat_files.sort()
    
    if not mat_files:
        print("ERROR: No .mat files found. Please run on Kaggle.")
        return
        
    b, a = scipy.signal.butter(4, [1.0/64.0, 8.0/64.0], btype='band')
    
    # Pre-process and cache all subjects to disk to avoid Out-Of-Memory (OOM)
    cache_dir = Path('/kaggle/working/eeg_cache')
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    print("Pre-processing and caching subjects to disk...")
    subject_ids = []
    
    import gc
    for mat_file in mat_files:
        subj_id = os.path.basename(mat_file).split('.')[0]
        subject_ids.append(subj_id)
        
        cache_path = cache_dir / f"{subj_id}_processed.pt"
        if not cache_path.exists():
            print(f"  Processing {subj_id} from .mat...")
            trials = load_aasd_subject_trials(mat_file, b, a, audio_dir, wav_dir)
            
            X_tensors, Y_tensors, raw_trials = [], [], []
            for trial in trials:
                eeg = trial['eeg'].numpy()
                env_l = trial['env_l'].numpy()
                env_r = trial['env_r'].numpy()
                switch_points = trial['meta']['switch_points']
                
                # Normalize
                eeg = (eeg - eeg.mean(axis=1, keepdims=True)) / (eeg.std(axis=1, keepdims=True) + 1e-8)
                env_l = (env_l - env_l.mean()) / (env_l.std() + 1e-8)
                env_r = (env_r - env_r.mean()) / (env_r.std() + 1e-8)
                
                att = np.zeros(eeg.shape[1], dtype=np.float32)
                if len(switch_points) == 0: switch_points = [('R', 0)]
                if switch_points[0][1] > 0:
                    current_state = 'R' if switch_points[0][0] == 'L' else 'L'
                else:
                    current_state = switch_points[0][0]
                
                prev_idx = 0
                for state, idx_64 in switch_points:
                    idx_64 = min(idx_64, eeg.shape[1])
                    if idx_64 > prev_idx:
                        if current_state == 'L': att[prev_idx:idx_64] = env_l[prev_idx:idx_64]
                        else: att[prev_idx:idx_64] = env_r[prev_idx:idx_64]
                    prev_idx, current_state = idx_64, state
                if current_state == 'L': att[prev_idx:] = env_l[prev_idx:]
                else: att[prev_idx:] = env_r[prev_idx:]
                
                X_tensors.append(torch.from_numpy(eeg).float())
                Y_tensors.append(torch.from_numpy(att).float())
                
                # We need raw trial data for evaluation testing
                raw_trials.append({
                    'eeg': torch.from_numpy(eeg_full := trial['eeg'].numpy()), 
                    'env_l': torch.from_numpy(trial['env_l'].numpy()),
                    'env_r': torch.from_numpy(trial['env_r'].numpy()),
                    'meta': trial['meta']
                })
                
            torch.save({'X': X_tensors, 'Y': Y_tensors, 'raw': raw_trials}, cache_path)
            del trials, X_tensors, Y_tensors, raw_trials
            gc.collect()
        else:
            print(f"  Found cached {subj_id}")
            
    PHYSICAL_8_CHANNELS = [0, 2, 5, 13, 23, 31, 41, 49]
    max_lag = 24
    window_len = 64 * 5
    hop_len = 64 * 1
    
    configs = [
        ("ZERO_SHOT", []),
        ("LATENT_ONLY", ["residual_adapter"]),
        ("PROJECTION_ONLY", ["backbone.upsample"])
    ]
    
    results = {cfg[0]: [] for cfg in configs}
    
    print("\n--- 3. Running Within-Subject Validation ---")
    
    for test_idx, subject_id in enumerate(subject_ids):
        print(f"\n==========================================")
        print(f" SUBJECT {test_idx+1}/{len(subject_ids)}: {subject_id}")
        print(f"==========================================")
        
        cached = torch.load(cache_dir / f"{subject_id}_processed.pt", weights_only=False)
        all_trials_X = cached['X']
        all_trials_Y = cached['Y']
        raw_trials = cached['raw']
        
        # 40 Train, 20 Test
        train_X, test_X = all_trials_X[:40], all_trials_X[40:]
        train_Y, test_Y = all_trials_Y[:40], all_trials_Y[40:]
        test_raw = raw_trials[40:]
        
        # Calculate Analytical Ridge on Train Trials (Within-Subject)
        print("  -> Computing Within-Subject Analytical Ridge (40 trials)...")
        temp_model = LayerwiseAdaptationModel(load_clean_backbone(), PHYSICAL_8_CHANNELS, embed_dim=64, max_lag=max_lag).to(device)
        temp_model.eval()
        
        num_features = 64 * (max_lag + 1)
        ZtZ = np.zeros((num_features, num_features), dtype=np.float32)
        ZtY = np.zeros(num_features, dtype=np.float32)
        
        with torch.no_grad():
            for x, y in zip(train_X, train_Y):
                eeg_tensor = x.unsqueeze(0).to(device)
                eeg_8ch = eeg_tensor[:, PHYSICAL_8_CHANNELS, :]
                z = temp_model.extract_features(eeg_8ch).squeeze(0).cpu().numpy()
                Z = build_lagged_matrix(z, max_lag).astype(np.float32)
                y_aligned = y.numpy()[max_lag:].astype(np.float32)
                
                ZtZ += Z.T @ Z
                ZtY += Z.T @ y_aligned
                
        lambda_reg = 1e4
        I = np.eye(num_features, dtype=np.float32)
        W_analytical = np.linalg.inv(ZtZ + lambda_reg * I) @ ZtY
        del ZtZ, ZtY, temp_model
        
        # Build Train Loader
        def collate_fn(batch):
            max_len = max(x.size(1) for x, y in batch)
            x_padded = torch.zeros(len(batch), batch[0][0].size(0), max_len)
            y_padded = torch.zeros(len(batch), max_len)
            for i, (x, y) in enumerate(batch):
                x_padded[i, :, :x.size(1)] = x
                y_padded[i, :y.size(0)] = y
            return x_padded, y_padded
            
        train_data = list(zip(train_X, train_Y))
        train_loader = DataLoader(train_data, batch_size=4, shuffle=True, collate_fn=collate_fn)
        
        for config_name, layers_to_unfreeze in configs:
            model = LayerwiseAdaptationModel(load_clean_backbone(), PHYSICAL_8_CHANNELS, embed_dim=64, max_lag=max_lag).to(device)
            model.load_analytical_weights(W_analytical)
            
            # Freeze everything first
            for param in model.parameters():
                param.requires_grad = False
                
            # Unfreeze target layers
            for name, param in model.named_parameters():
                for target_layer in layers_to_unfreeze:
                    if name.startswith(target_layer):
                        param.requires_grad = True
                            
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            if trainable_params > 0:
                optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4, weight_decay=1e-2)
                
                for epoch in range(1, 11): 
                    model.eval()
                    for target_layer in layers_to_unfreeze:
                        parts = target_layer.split('.')
                        mod = model
                        for p in parts:
                            if p.isdigit(): mod = mod[int(p)]
                            else: mod = getattr(mod, p)
                        mod.train()
                        
                    epoch_loss = 0.0
                    for bx, by in train_loader:
                        bx, by = bx.to(device), by.to(device)
                        optimizer.zero_grad()
                        pred = model(bx)
                        loss = pearson_loss(pred, by)
                        loss.backward()
                        optimizer.step()
                        epoch_loss += loss.item()
                
            model.eval()
            auroc = run_test_suite(model, test_raw, device, max_lag, hop_len, window_len)
            results[config_name].append(auroc)
            print(f"   [{config_name:<15}] AUROC: {auroc:.4f}")
            
        del cached, train_data, train_loader, all_trials_X, all_trials_Y, raw_trials
        gc.collect()
            
    print("\n=======================================================")
    print(" SUMMARY: WITHIN-SUBJECT ADAPTATION (18 SUBJECTS) ")
    print("=======================================================")
    
    print(f"{'Subject':<10} | {'ZERO_SHOT':<12} | {'LATENT':<12} | {'PROJECTION':<12}")
    print("-" * 55)
    for i, subj in enumerate(subject_ids):
        zs = results["ZERO_SHOT"][i]
        lat = results["LATENT_ONLY"][i]
        proj = results["PROJECTION_ONLY"][i]
        print(f"{subj:<10} | {zs:<12.4f} | {lat:<12.4f} | {proj:<12.4f}")
        
    print("-" * 55)
    
    zs_arr = np.array(results["ZERO_SHOT"])
    lat_arr = np.array(results["LATENT_ONLY"])
    proj_arr = np.array(results["PROJECTION_ONLY"])
    
    print(f"MEAN (STD)")
    print(f"ZERO_SHOT      : {np.mean(zs_arr):.4f} ({np.std(zs_arr):.4f})")
    print(f"LATENT_ONLY    : {np.mean(lat_arr):.4f} ({np.std(lat_arr):.4f})")
    print(f"PROJECTION_ONLY: {np.mean(proj_arr):.4f} ({np.std(proj_arr):.4f})")
    
    # Statistical tests
    try:
        _, p_proj_vs_zs = wilcoxon(proj_arr, zs_arr)
        _, p_proj_vs_lat = wilcoxon(proj_arr, lat_arr)
        _, p_lat_vs_zs = wilcoxon(lat_arr, zs_arr)
        
        print("\nStatistical Significance (Wilcoxon Signed-Rank Test):")
        print(f"PROJECTION vs ZERO_SHOT: p = {p_proj_vs_zs:.4f}")
        print(f"LATENT vs ZERO_SHOT    : p = {p_lat_vs_zs:.4f}")
        print(f"PROJECTION vs LATENT   : p = {p_proj_vs_lat:.4f}")
    except Exception as e:
        print(f"Could not compute wilcoxon stats: {e}")

if __name__ == '__main__':
    run_loso_validation()
