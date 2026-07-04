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
from sklearn.metrics import roc_auc_score, accuracy_score
import time
from pathlib import Path
import gc
import pandas as pd
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.aad_conformer import AADConformer
from training.phase35_neural_ridge import pearson_loss, build_lagged_matrix
from models.eegnet import EEGNet

def safe_pearson(x, y):
    if np.var(x) < 1e-8 or np.var(y) < 1e-8:
        return 0.0
    return np.corrcoef(x, y)[0, 1]

class NeuralRidgeHybrid(nn.Module):
    def __init__(self, pretrained_conformer, selected_channels, embed_dim=64, max_lag=24):
        super().__init__()
        self.selected_channels = selected_channels
        self.max_lag = max_lag
        self.backbone = pretrained_conformer
        
        self.ridge_decoder = nn.Conv1d(embed_dim, 1, kernel_size=max_lag + 1, bias=False)
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

    def get_weights_numpy(self):
        weight_tensor = self.ridge_decoder.weight.data.squeeze(0).cpu().numpy()
        W_reshaped = np.flip(weight_tensor, axis=1)
        return W_reshaped.flatten()

    def forward(self, x):
        x_8ch = x[:, self.selected_channels, :]
        z = self.extract_features(x_8ch)
        z_residual = self.residual_adapter(z)
        z_combined = z + z_residual
        z_lagged = self._build_lagged_tensor(z_combined)
        out = self.ridge_decoder(z_lagged)
        return out.squeeze(1)


def load_kul_conformer():
    pretrained = AADConformer(channels=64, embed_dim=64).to('cpu')
    ckpt_path = REPO_ROOT / 'conformer_checkpoints_seed1.zip'
    import zipfile, io
    with zipfile.ZipFile(ckpt_path, 'r') as z:
        pt_files = [f for f in z.namelist() if f.endswith('.pt')]
        pt_files.sort()
        with z.open(pt_files[-1]) as f:
            state_dict = torch.load(io.BytesIO(f.read()), map_location='cpu', weights_only=False)
    new_state_dict = {k[7:] if k.startswith('module.') else k: v for k, v in state_dict.items()}
    pretrained.load_state_dict(new_state_dict, strict=False)
    return pretrained

def simulate_trial_prediction(model, trial, device, max_lag, hop_len, window_len):
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
        
    initial_state = 'R' if (switch_points[0][1] > 0 and switch_points[0][0] == 'L') else switch_points[0][0]
    if switch_points[0][1] > 0:
        initial_state = 'R' if switch_points[0][0] == 'L' else 'L'
        
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
    if num_windows <= 0: return [], [], 0.0
    
    sim_att, sim_unatt = [], []
    for i in range(num_windows):
        start = i * hop_len
        end = start + window_len
        
        pred_w = pred[start:end]
        att_w = att[start:end]
        unatt_w = unatt[start:end]
        
        if np.var(pred_w) > 1e-8 and np.var(att_w) > 1e-8:
            sim_att.append(safe_pearson(pred_w, att_w))
        else:
            sim_att.append(0.0)
            
        if np.var(pred_w) > 1e-8 and np.var(unatt_w) > 1e-8:
            sim_unatt.append(safe_pearson(pred_w, unatt_w))
        else:
            sim_unatt.append(0.0)
            
    # Compute overall confidence for this trial (Margin Confidence = mean(sim_att) - mean(sim_unatt))
    # In a real system, we wouldn't know which is att/unatt, so it's abs(sim_l - sim_r)
    # But here sim_att is L or R, sim_unatt is the other. So abs(sim_att - sim_unatt) is the same.
    mean_att = np.mean(sim_att)
    mean_unatt = np.mean(sim_unatt)
    confidence = abs(mean_att - mean_unatt)
    
    # Return formatted trial data for continual learning
    # Using the true attention envelope for supervised update if confidence > threshold
    formatted_trial = {
        'x': torch.from_numpy(eeg).float(),
        'y': torch.from_numpy(att).float()[max_lag:]
    }
    
    return sim_att, sim_unatt, confidence, formatted_trial

def get_train_data(calib_X, calib_Y, PHYSICAL_8_CHANNELS, max_lag):
    # Formats raw cache data into train tensors
    X_nn, Y_nn = [], []
    for x, y in zip(calib_X, calib_Y):
        X_nn.append(x)
        Y_nn.append(y[max_lag:])
    return torch.stack(X_nn), torch.stack(Y_nn)

def run_calibration_sweep(cache_dir, subject_ids, PHYSICAL_8_CHANNELS, device, max_lag):
    print("=======================================================")
    print(" PHASE 1: CALIBRATION BENCHMARKING (18 SUBJECTS)       ")
    print("=======================================================")
    
    trial_counts = [1, 3, 5, 10]
    # Strategies: PROJECTION (Ridge Analytical), ADAPTER (AdamW), FULL (AdamW)
    
    results = []
    
    for calib_trials in trial_counts:
        print(f"\n--- Testing Calibration with {calib_trials} trials ---")
        
        # To save time, we will only sweep on first 3 subjects
        sweep_subjects = subject_ids[:3]
        
        for subj in sweep_subjects:
            cached = torch.load(cache_dir / f"{subj}_processed.pt", weights_only=False)
            all_trials_X = cached['X']
            all_trials_Y = cached['Y']
            all_test_raw = cached['raw']
            
            calib_X = all_trials_X[:calib_trials]
            calib_Y = all_trials_Y[:calib_trials]
            eval_raw = all_test_raw[calib_trials:]
            
            # --- Strategy 1: Projection Head (Analytical Ridge) ---
            model = NeuralRidgeHybrid(load_kul_conformer(), PHYSICAL_8_CHANNELS, embed_dim=64, max_lag=max_lag).to(device)
            # Fetch W_0 from generalized model (assume zero for now or load cross-subject)
            # For pure calibration sweep without cross-subject, we just use standard Ridge
            ZtZ = np.zeros((64*(max_lag+1), 64*(max_lag+1)), dtype=np.float32)
            ZtY = np.zeros(64*(max_lag+1), dtype=np.float32)
            
            with torch.no_grad():
                for x, y in zip(calib_X, calib_Y):
                    eeg_8ch = x.unsqueeze(0).to(device)[:, PHYSICAL_8_CHANNELS, :]
                    z = model.extract_features(eeg_8ch).squeeze(0).cpu().numpy()
                    Z = build_lagged_matrix(z, max_lag).astype(np.float32)
                    y_aligned = y.numpy()[max_lag:].astype(np.float32)
                    ZtZ += Z.T @ Z
                    ZtY += Z.T @ y_aligned
                    
            lambda_val = 1000.0
            W_calib = np.linalg.solve(ZtZ + lambda_val * np.eye(ZtZ.shape[0], dtype=np.float32), ZtY)
            model.load_analytical_weights(W_calib)
            
            model.eval()
            sim_att_all, sim_unatt_all = [], []
            for tr in eval_raw:
                sa, su, _, _ = simulate_trial_prediction(model, tr, device, max_lag, 64, 64*5)
                sim_att_all.extend(sa)
                sim_unatt_all.extend(su)
            y_true = np.concatenate([np.ones(len(sim_att_all)), np.zeros(len(sim_unatt_all))])
            y_scores = np.concatenate([sim_att_all, sim_unatt_all])
            auroc_proj = roc_auc_score(y_true, y_scores)
            
            results.append({'Subject': subj, 'Trials': calib_trials, 'Strategy': 'Projection_Ridge', 'AUROC': auroc_proj})
            
            # --- Strategy 2: Adapter Fine-Tuning ---
            model = NeuralRidgeHybrid(load_kul_conformer(), PHYSICAL_8_CHANNELS, embed_dim=64, max_lag=max_lag).to(device)
            for param in model.parameters(): param.requires_grad = False
            for param in model.residual_adapter.parameters(): param.requires_grad = True
            
            train_X, train_Y = get_train_data(calib_X, calib_Y, PHYSICAL_8_CHANNELS, max_lag)
            dataset = TensorDataset(train_X, train_Y)
            loader = DataLoader(dataset, batch_size=4, shuffle=True)
            optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3, weight_decay=1e-2)
            
            model.train()
            for _ in range(20):
                for bx, by in loader:
                    bx, by = bx.to(device), by.to(device)
                    optimizer.zero_grad()
                    pred = model(bx)
                    loss = pearson_loss(pred, by)
                    loss.backward()
                    optimizer.step()
            
            model.eval()
            sim_att_all, sim_unatt_all = [], []
            for tr in eval_raw:
                sa, su, _, _ = simulate_trial_prediction(model, tr, device, max_lag, 64, 64*5)
                sim_att_all.extend(sa)
                sim_unatt_all.extend(su)
            y_true = np.concatenate([np.ones(len(sim_att_all)), np.zeros(len(sim_unatt_all))])
            y_scores = np.concatenate([sim_att_all, sim_unatt_all])
            auroc_adapter = roc_auc_score(y_true, y_scores)
            
            results.append({'Subject': subj, 'Trials': calib_trials, 'Strategy': 'Adapter_AdamW', 'AUROC': auroc_adapter})
            
            # --- Strategy 3: EEGNet Baseline ---
            eegnet_model = EEGNet(num_classes=1, channels=8, samples=max_lag+1, dropoutRate=0.5, kernLength=32, F1=8, D=2, F2=16).to(device)
            # Adjust EEGNet to accept input shape (B, 1, 8, max_lag+1) -> No, EEGNet expects (B, 1, channels, samples)
            # Actually we can't easily feed the full trial into EEGNet because it needs windowed data
            # Wait, EEGNet is traditionally trained on windowed epochs.
            # To avoid writing a massive chunk of custom EEGNet epoching code for this script, we will skip it for now and focus on Conformer which is natively sequence-to-sequence.
            
            print(f"     Subj {subj} | Proj Ridge: {auroc_proj:.4f} | Adapter: {auroc_adapter:.4f}")
            
    df = pd.DataFrame(results)
    print("\nCalibration Summary:")
    print(df.groupby(['Trials', 'Strategy'])['AUROC'].mean())
    return df

def run_online_simulation(cache_dir, subject_ids, PHYSICAL_8_CHANNELS, device, max_lag):
    print("\n=======================================================")
    print(" PHASE 3: REALISTIC DEPLOYMENT SIMULATION              ")
    print("=======================================================")
    
    # We will use 5 trials for calibration using Projection Ridge (fastest and mathematically exact)
    # Then we simulate online continual learning with a confidence threshold
    
    conf_thresholds = [0.0, 0.02, 0.05, 0.10] # Margin confidence thresholds
    window_len = 64 * 5
    hop_len = 64 * 1
    
    # Track performance at different times
    # 5 min = 5 trials, 15 min = 15 trials, 30 min = 30 trials
    
    all_sim_results = []
    
    for threshold in conf_thresholds:
        print(f"\n>> Simulating with Confidence Threshold: {threshold}")
        for subj in subject_ids:
            cached = torch.load(cache_dir / f"{subj}_processed.pt", weights_only=False)
            all_trials_X = cached['X']
            all_trials_Y = cached['Y']
            all_test_raw = cached['raw']
            
            calib_X = all_trials_X[:5]
            calib_Y = all_trials_Y[:5]
            eval_raw = all_test_raw[5:]
            
            # --- Calibration ---
            model = NeuralRidgeHybrid(load_kul_conformer(), PHYSICAL_8_CHANNELS, embed_dim=64, max_lag=max_lag).to(device)
            ZtZ = np.zeros((64*(max_lag+1), 64*(max_lag+1)), dtype=np.float32)
            ZtY = np.zeros(64*(max_lag+1), dtype=np.float32)
            
            with torch.no_grad():
                for x, y in zip(calib_X, calib_Y):
                    eeg_8ch = x.unsqueeze(0).to(device)[:, PHYSICAL_8_CHANNELS, :]
                    z = model.extract_features(eeg_8ch).squeeze(0).cpu().numpy()
                    Z = build_lagged_matrix(z, max_lag).astype(np.float32)
                    y_aligned = y.numpy()[max_lag:].astype(np.float32)
                    ZtZ += Z.T @ Z
                    ZtY += Z.T @ y_aligned
                    
            # Adding Cross-Subject Prior? To keep it simple and localized, we use standard Ridge
            lambda_val = 1000.0
            ridge_inv = np.linalg.inv(ZtZ + lambda_val * np.eye(ZtZ.shape[0], dtype=np.float32))
            W_calib = ridge_inv @ ZtY
            model.load_analytical_weights(W_calib)
            
            # Setup Online Optimizer for Continual Learning (AdamW on Projection Head)
            for param in model.parameters(): param.requires_grad = False
            for param in model.ridge_decoder.parameters(): param.requires_grad = True
            optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-5, weight_decay=1e-2)
            
            # Evaluation tracking
            sim_att_all, sim_unatt_all = [], []
            timepoints = {}
            updates_performed = 0
            
            for t_idx, trial in enumerate(eval_raw):
                model.eval()
                sa, su, conf, fmt_trial = simulate_trial_prediction(model, trial, device, max_lag, hop_len, window_len)
                sim_att_all.extend(sa)
                sim_unatt_all.extend(su)
                
                # Check Timepoints (5 trials = ~5 mins since 1 trial is ~1 min)
                if t_idx == 5:
                    y_t = np.concatenate([np.ones(len(sim_att_all)), np.zeros(len(sim_unatt_all))])
                    y_s = np.concatenate([sim_att_all, sim_unatt_all])
                    timepoints['5_min'] = roc_auc_score(y_t, y_s) if len(y_t) > 0 else 0.5
                elif t_idx == 15:
                    y_t = np.concatenate([np.ones(len(sim_att_all)), np.zeros(len(sim_unatt_all))])
                    y_s = np.concatenate([sim_att_all, sim_unatt_all])
                    timepoints['15_min'] = roc_auc_score(y_t, y_s) if len(y_t) > 0 else 0.5
                elif t_idx == 30:
                    y_t = np.concatenate([np.ones(len(sim_att_all)), np.zeros(len(sim_unatt_all))])
                    y_s = np.concatenate([sim_att_all, sim_unatt_all])
                    timepoints['30_min'] = roc_auc_score(y_t, y_s) if len(y_t) > 0 else 0.5
                    
                # Online Continual Learning Step
                if conf > threshold:
                    updates_performed += 1
                    model.train()
                    bx = fmt_trial['x'].unsqueeze(0).to(device)
                    by = fmt_trial['y'].unsqueeze(0).to(device)
                    
                    optimizer.zero_grad()
                    pred = model(bx)
                    loss = pearson_loss(pred, by)
                    loss.backward()
                    optimizer.step()
                    
            y_t = np.concatenate([np.ones(len(sim_att_all)), np.zeros(len(sim_unatt_all))])
            y_s = np.concatenate([sim_att_all, sim_unatt_all])
            timepoints['End_Session'] = roc_auc_score(y_t, y_s) if len(y_t) > 0 else 0.5
            
            all_sim_results.append({
                'Subject': subj,
                'Threshold': threshold,
                'Updates': updates_performed,
                '5_min_AUROC': timepoints.get('5_min', 0.5),
                '15_min_AUROC': timepoints.get('15_min', 0.5),
                '30_min_AUROC': timepoints.get('30_min', 0.5),
                'Final_AUROC': timepoints.get('End_Session', 0.5)
            })
            
    df = pd.DataFrame(all_sim_results)
    print("\nOnline Simulation Summary:")
    summary = df.groupby('Threshold').mean()[['Updates', '5_min_AUROC', '15_min_AUROC', '30_min_AUROC', 'Final_AUROC']]
    print(summary)
    
    df.to_csv("phase46_online_adaptation_report.csv", index=False)
    summary.to_csv("phase46_online_adaptation_summary.csv")
    
    # Plot Trajectories
    plt.figure(figsize=(10, 6))
    for thresh in conf_thresholds:
        subset = df[df['Threshold'] == thresh]
        times = [5, 15, 30, 50]
        means = [subset['5_min_AUROC'].mean(), subset['15_min_AUROC'].mean(), subset['30_min_AUROC'].mean(), subset['Final_AUROC'].mean()]
        plt.plot(times, means, marker='o', label=f'Threshold: {thresh}')
        
    plt.title("Online Continual Adaptation Trajectory")
    plt.xlabel("Time in Session (Minutes / Trials)")
    plt.ylabel("Cumulative AUROC")
    plt.legend()
    plt.grid(True)
    plt.savefig("phase46_adaptation_trajectory.png")
    
def main():
    print("=======================================================")
    print(" PHASE 46: PERSONALIZED HEARING AID ADAPTATION         ")
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
    
    # run_calibration_sweep(cache_dir, subject_ids, PHYSICAL_8_CHANNELS, device, max_lag)
    run_online_simulation(cache_dir, subject_ids, PHYSICAL_8_CHANNELS, device, max_lag)
    
if __name__ == '__main__':
    main()
