import os
import sys
import numpy as np
import scipy.io
import scipy.signal
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score, accuracy_score
import time
from pathlib import Path
import gc
import pandas as pd
import matplotlib.pyplot as plt
import random

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.aad_conformer import AADConformer
from training.phase35_neural_ridge import build_lagged_matrix

# --- CONSTANTS & CONFIG ---
PHYSICAL_8_CHANNELS = [0, 2, 5, 13, 23, 31, 41, 49]
MAX_LAG = 24
WINDOW_LEN = 64 * 5
HOP_LEN = 64 * 1
CALIBRATION_TRIALS = 5
LAMBDA_RIDGE = 1000.0
UPDATE_INTERVAL_TRIALS = 5 # Replay buffer flushes every 5 trials

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

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
        z_lagged = self._build_lagged_tensor(z)
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

# Singleton pattern to prevent reloading from disk
_global_kul_conformer = None
def get_cached_kul_conformer():
    global _global_kul_conformer
    if _global_kul_conformer is None:
        _global_kul_conformer = load_kul_conformer()
    return _global_kul_conformer

def simulate_trial_unsupervised(model, trial, device, max_lag, hop_len, window_len):
    """
    Computes predictions and unsupervised confidence metrics without ground truth leakage.
    Returns:
    - true_att_corr: array of correlation with the true attended envelope (for evaluation only)
    - true_unatt_corr: array of correlation with the true unattended envelope (for evaluation only)
    - trial_confidence: float, absolute difference between L and R mean correlations
    - pseudo_target_env: numpy array of the envelope chosen as the pseudo-target
    """
    eeg_full = trial['eeg'].numpy()
    env_l_raw = trial['env_l'].numpy()
    env_r_raw = trial['env_r'].numpy()
    switch_points = trial['meta']['switch_points']
    
    eeg = (eeg_full - eeg_full.mean(axis=1, keepdims=True)) / (eeg_full.std(axis=1, keepdims=True) + 1e-8)
    env_l = (env_l_raw - env_l_raw.mean()) / (env_l_raw.std() + 1e-8)
    env_r = (env_r_raw - env_r_raw.mean()) / (env_r_raw.std() + 1e-8)
    
    # Ground Truth construction (strictly for evaluation metrics, NEVER for updates)
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
    if num_windows <= 0: return [], [], 0.0, None, None
    
    # Evaluate windowed correlation
    corr_L, corr_R = [], []
    true_att_corr, true_unatt_corr = [], []
    
    for i in range(num_windows):
        start = i * hop_len
        end = start + window_len
        
        pred_w = pred[start:end]
        att_w = att[start:end]
        unatt_w = unatt[start:end]
        env_l_w = env_l[start:end]
        env_r_w = env_r[start:end]
        
        true_att_corr.append(safe_pearson(pred_w, att_w))
        true_unatt_corr.append(safe_pearson(pred_w, unatt_w))
        corr_L.append(safe_pearson(pred_w, env_l_w))
        corr_R.append(safe_pearson(pred_w, env_r_w))
            
    # Unsupervised Confidence Calculation
    mean_corr_L = np.mean(corr_L)
    mean_corr_R = np.mean(corr_R)
    confidence = abs(mean_corr_L - mean_corr_R)
    
    # Pseudo-Label Selection
    pseudo_target = env_l if mean_corr_L > mean_corr_R else env_r
    
    return true_att_corr, true_unatt_corr, confidence, eeg, pseudo_target

def build_ridge_covariance(model, x_eeg, y_env, device, max_lag):
    """Computes ZtZ and ZtY for a single trial."""
    eeg_8ch = torch.from_numpy(x_eeg).float().unsqueeze(0).to(device)[:, PHYSICAL_8_CHANNELS, :]
    with torch.no_grad():
        z = model.extract_features(eeg_8ch).squeeze(0).cpu().numpy()
    Z = build_lagged_matrix(z, max_lag).astype(np.float32)
    y_aligned = y_env[max_lag:].astype(np.float32)
    ZtZ = Z.T @ Z
    ZtY = Z.T @ y_aligned
    return ZtZ, ZtY

def solve_ridge(ZtZ, ZtY, lambda_val):
    return np.linalg.solve(ZtZ + lambda_val * np.eye(ZtZ.shape[0], dtype=np.float32), ZtY)

def evaluate_model(model, eval_trials, device):
    sim_att_all, sim_unatt_all = [], []
    for tr in eval_trials:
        model.eval()
        sa, su, _, _, _ = simulate_trial_unsupervised(model, tr, device, MAX_LAG, HOP_LEN, WINDOW_LEN)
        sim_att_all.extend(sa)
        sim_unatt_all.extend(su)
    y_t = np.concatenate([np.ones(len(sim_att_all)), np.zeros(len(sim_unatt_all))])
    y_s = np.concatenate([sim_att_all, sim_unatt_all])
    return roc_auc_score(y_t, y_s) if len(y_t) > 0 else 0.5

def run_unsupervised_continual_learning(cache_dir, subject_ids, device):
    print("\n=======================================================")
    print(" PHASE 46 V2: UNSUPERVISED CONTINUAL LEARNING          ")
    print("=======================================================")
    
    all_sim_results = []
    
    for subj in subject_ids:
        print(f"\n>> Processing Subject: {subj}")
        cached = torch.load(cache_dir / f"{subj}_processed.pt", weights_only=False)
        all_test_raw = cached['raw']
        
        # 1. Split into Calibration (5 trials) and Continuous (Remaining ~45 trials)
        calib_raw = all_test_raw[:CALIBRATION_TRIALS]
        eval_raw = all_test_raw[CALIBRATION_TRIALS:]
        
        # 2. Establish Zero-Shot Baseline
        model = NeuralRidgeHybrid(get_cached_kul_conformer(), PHYSICAL_8_CHANNELS, embed_dim=64, max_lag=MAX_LAG).to(device)
        model.load_analytical_weights(np.zeros(64*(MAX_LAG+1), dtype=np.float32)) # Zero-shot starts flat
        zs_auroc = evaluate_model(model, eval_raw, device)
        print(f"   [Zero-Shot] Final AUROC: {zs_auroc:.4f}")
        
        # 3. Supervised Calibration (Initial Fitting Session)
        ZtZ_total = np.zeros((64*(MAX_LAG+1), 64*(MAX_LAG+1)), dtype=np.float32)
        ZtY_total = np.zeros(64*(MAX_LAG+1), dtype=np.float32)
        calib_confidences = []
        
        for tr in calib_raw:
            # During calibration, we theoretically have the ground truth, or we use a short structured task.
            # Here we extract true env just for the ZtZ and ZtY calculation to simulate the calibration session.
            # Note: We also calculate pseudo-confidence to build the threshold distribution.
            sa, su, conf, eeg, pseudo_target = simulate_trial_unsupervised(model, tr, device, MAX_LAG, HOP_LEN, WINDOW_LEN)
            calib_confidences.append(conf)
            
            # Use true envelope for initial supervised calibration
            eeg_full = tr['eeg'].numpy()
            env_l = tr['env_l'].numpy()
            env_r = tr['env_r'].numpy()
            switch_points = tr['meta']['switch_points']
            eeg = (eeg_full - eeg_full.mean(axis=1, keepdims=True)) / (eeg_full.std(axis=1, keepdims=True) + 1e-8)
            env_l = (env_l - env_l.mean()) / (env_l.std() + 1e-8)
            env_r = (env_r - env_r.mean()) / (env_r.std() + 1e-8)
            
            true_att = np.zeros(eeg.shape[1], dtype=np.float32)
            current_state = 'R' if (len(switch_points) > 0 and switch_points[0][1] > 0 and switch_points[0][0] == 'L') else (switch_points[0][0] if len(switch_points)>0 else 'R')
            if len(switch_points)>0 and switch_points[0][1] > 0: current_state = 'R' if switch_points[0][0] == 'L' else 'L'
            prev_idx = 0
            for state, idx_64 in switch_points:
                idx_64 = min(idx_64, eeg.shape[1])
                if idx_64 > prev_idx:
                    true_att[prev_idx:idx_64] = env_l[prev_idx:idx_64] if current_state == 'L' else env_r[prev_idx:idx_64]
                prev_idx, current_state = idx_64, state
            true_att[prev_idx:] = env_l[prev_idx:] if current_state == 'L' else env_r[prev_idx:]
            
            ztz, zty = build_ridge_covariance(model, eeg, true_att, device, MAX_LAG)
            ZtZ_total += ztz
            ZtY_total += zty
            
        # Solve Analytical Ridge for Calibration
        W_calib = solve_ridge(ZtZ_total, ZtY_total, LAMBDA_RIDGE)
        model.load_analytical_weights(W_calib)
        
        calib_auroc = evaluate_model(model, eval_raw, device)
        print(f"   [Calibration Only] Final AUROC: {calib_auroc:.4f}")
        
        # 4. Determine Data-Driven Thresholds
        p50 = np.percentile(calib_confidences, 50)
        p75 = np.percentile(calib_confidences, 75)
        p90 = np.percentile(calib_confidences, 90)
        p95 = np.percentile(calib_confidences, 95)
        thresholds = {'P50': p50, 'P75': p75, 'P90': p90, 'P95': p95}
        
        # 5. Online Continual Learning (Unsupervised)
        for t_name, threshold in thresholds.items():
            # Reset model to post-calibration state
            ZtZ_online = ZtZ_total.copy()
            ZtY_online = ZtY_total.copy()
            model.load_analytical_weights(W_calib)
            
            timepoints = {}
            updates_performed = 0
            replay_buffer_ztz = np.zeros_like(ZtZ_total)
            replay_buffer_zty = np.zeros_like(ZtY_total)
            trials_since_update = 0
            
            # Evaluation Tracking
            sim_att_all, sim_unatt_all = [], []
            
            for t_idx, trial in enumerate(eval_raw):
                model.eval()
                sa, su, conf, eeg, pseudo_target = simulate_trial_unsupervised(model, trial, device, MAX_LAG, HOP_LEN, WINDOW_LEN)
                sim_att_all.extend(sa)
                sim_unatt_all.extend(su)
                
                # Unsupervised Confidence Acceptance
                if conf > threshold:
                    ztz, zty = build_ridge_covariance(model, eeg, pseudo_target, device, MAX_LAG)
                    replay_buffer_ztz += ztz
                    replay_buffer_zty += zty
                    updates_performed += 1
                
                trials_since_update += 1
                
                # Periodic Replay Buffer Flush (Every N minutes)
                if trials_since_update >= UPDATE_INTERVAL_TRIALS:
                    ZtZ_online += replay_buffer_ztz
                    ZtY_online += replay_buffer_zty
                    W_online = solve_ridge(ZtZ_online, ZtY_online, LAMBDA_RIDGE)
                    model.load_analytical_weights(W_online)
                    
                    replay_buffer_ztz.fill(0)
                    replay_buffer_zty.fill(0)
                    trials_since_update = 0
                
                # Record timepoints (t_idx is 0-indexed, so t_idx == 4 means after 5 trials)
                if t_idx == 4:
                    y_t = np.concatenate([np.ones(len(sim_att_all)), np.zeros(len(sim_unatt_all))])
                    y_s = np.concatenate([sim_att_all, sim_unatt_all])
                    timepoints['5_min'] = roc_auc_score(y_t, y_s) if len(y_t) > 0 else 0.5
                elif t_idx == 14:
                    y_t = np.concatenate([np.ones(len(sim_att_all)), np.zeros(len(sim_unatt_all))])
                    y_s = np.concatenate([sim_att_all, sim_unatt_all])
                    timepoints['15_min'] = roc_auc_score(y_t, y_s) if len(y_t) > 0 else 0.5
                elif t_idx == 29:
                    y_t = np.concatenate([np.ones(len(sim_att_all)), np.zeros(len(sim_unatt_all))])
                    y_s = np.concatenate([sim_att_all, sim_unatt_all])
                    timepoints['30_min'] = roc_auc_score(y_t, y_s) if len(y_t) > 0 else 0.5
            
            # Ensure final flush if any
            if trials_since_update > 0:
                ZtZ_online += replay_buffer_ztz
                ZtY_online += replay_buffer_zty
                W_online = solve_ridge(ZtZ_online, ZtY_online, LAMBDA_RIDGE)
                model.load_analytical_weights(W_online)
                
            y_t = np.concatenate([np.ones(len(sim_att_all)), np.zeros(len(sim_unatt_all))])
            y_s = np.concatenate([sim_att_all, sim_unatt_all])
            final_auroc = roc_auc_score(y_t, y_s) if len(y_t) > 0 else 0.5
            
            all_sim_results.append({
                'Subject': subj,
                'Condition': f'Online_{t_name}',
                'Updates_Accepted': updates_performed,
                '5_min_AUROC': timepoints.get('5_min', 0.5),
                '15_min_AUROC': timepoints.get('15_min', 0.5),
                '30_min_AUROC': timepoints.get('30_min', 0.5),
                'Final_AUROC': final_auroc
            })
            print(f"   [Online {t_name}] Final AUROC: {final_auroc:.4f} | Updates: {updates_performed}/{len(eval_raw)}")
            
        # Add static baselines for dataframe completeness
        all_sim_results.append({'Subject': subj, 'Condition': 'Zero_Shot', 'Updates_Accepted': 0, '5_min_AUROC': zs_auroc, '15_min_AUROC': zs_auroc, '30_min_AUROC': zs_auroc, 'Final_AUROC': zs_auroc})
        all_sim_results.append({'Subject': subj, 'Condition': 'Calibration_Only', 'Updates_Accepted': 0, '5_min_AUROC': calib_auroc, '15_min_AUROC': calib_auroc, '30_min_AUROC': calib_auroc, 'Final_AUROC': calib_auroc})
            
    df = pd.DataFrame(all_sim_results)
    print("\n=======================================================")
    print(" SUMMARY: PHASE 46 V2 RESULTS                          ")
    print("=======================================================")
    summary = df.groupby('Condition').mean()[['Updates_Accepted', '5_min_AUROC', '15_min_AUROC', '30_min_AUROC', 'Final_AUROC']]
    print(summary)
    
    df.to_csv("phase46_v2_unsupervised_online_report.csv", index=False)
    summary.to_csv("phase46_v2_unsupervised_online_summary.csv")
    
    # Plot Trajectories
    plt.figure(figsize=(10, 6))
    conditions = df['Condition'].unique()
    for cond in conditions:
        subset = df[df['Condition'] == cond]
        times = [5, 15, 30, 50] # Assuming average end of session is ~50 trials
        means = [subset['5_min_AUROC'].mean(), subset['15_min_AUROC'].mean(), subset['30_min_AUROC'].mean(), subset['Final_AUROC'].mean()]
        plt.plot(times, means, marker='o', label=f'{cond}')
        
    plt.title("Unsupervised Continual Adaptation Trajectory")
    plt.xlabel("Time in Session (Minutes / Trials)")
    plt.ylabel("Cumulative AUROC")
    plt.legend()
    plt.grid(True)
    plt.savefig("phase46_v2_adaptation_trajectory.png")

def main():
    set_seed(42)
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
    
    run_unsupervised_continual_learning(cache_dir, subject_ids, device)
    
if __name__ == '__main__':
    main()
