import os
import sys
import numpy as np
import scipy.io
import scipy.signal
import torch
import torch.nn as nn
import torch.nn.functional as F
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
UPDATE_INTERVAL_TRIALS = 5 
FORGETTING_FACTOR = 0.995

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

_global_kul_conformer = None
def get_cached_kul_conformer():
    global _global_kul_conformer
    if _global_kul_conformer is None:
        pretrained = AADConformer(in_channels=8).to('cpu')
        
        kaggle_ckpt = Path('/kaggle/input/datasets/lowkieee/eeg-aad-conformer-seed1-checkpoints/checkpoints/seed_1/model_S1.pt')
        local_ckpt = REPO_ROOT / 'conformer_loso_results' / 'checkpoints' / 'seed_123' / 'model_S1.pt'
        
        if kaggle_ckpt.exists():
            checkpoint = torch.load(kaggle_ckpt, map_location='cpu', weights_only=False)
            state_dict = checkpoint.get('model_state_dict', checkpoint)
        elif local_ckpt.exists():
            checkpoint = torch.load(local_ckpt, map_location='cpu', weights_only=False)
            state_dict = checkpoint.get('model_state_dict', checkpoint)
        else:
            ckpt_path = REPO_ROOT / 'conformer_checkpoints_seed1.zip'
            import zipfile, io
            with zipfile.ZipFile(ckpt_path, 'r') as z:
                pt_files = [f for f in z.namelist() if f.endswith('.pt')]
                pt_files.sort()
                with z.open(pt_files[-1]) as f:
                    checkpoint = torch.load(io.BytesIO(f.read()), map_location='cpu', weights_only=False)
                    state_dict = checkpoint.get('model_state_dict', checkpoint)
                    
        new_state_dict = {k[7:] if k.startswith('module.') else k: v for k, v in state_dict.items()}
        pretrained.load_state_dict(new_state_dict, strict=False)
        _global_kul_conformer = pretrained
    return _global_kul_conformer

def build_ground_truth_envelope(trial):
    """Builds true attended and unattended envelopes strictly for metric evaluation."""
    eeg_full = trial['eeg'].numpy()
    env_l_raw = trial['env_l'].numpy()
    env_r_raw = trial['env_r'].numpy()
    switch_points = trial['meta']['switch_points']
    
    env_l = (env_l_raw - env_l_raw.mean()) / (env_l_raw.std() + 1e-8)
    env_r = (env_r_raw - env_r_raw.mean()) / (env_r_raw.std() + 1e-8)
    
    att = np.zeros(eeg_full.shape[1], dtype=np.float32)
    unatt = np.zeros(eeg_full.shape[1], dtype=np.float32)
    if len(switch_points) == 0: switch_points = [('R', 0)]
    initial_state = 'R' if (switch_points[0][1] > 0 and switch_points[0][0] == 'L') else switch_points[0][0]
    if switch_points[0][1] > 0:
        initial_state = 'R' if switch_points[0][0] == 'L' else 'L'
        
    current_state = initial_state
    prev_idx = 0
    for state, idx_64 in switch_points:
        idx_64 = min(idx_64, eeg_full.shape[1])
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
        
    return att, unatt

def simulate_trial_unsupervised(model, trial, device, z_cached=None):
    """
    Computes predictions, median-based confidence, and pseudo-targets.
    Optionally accepts a pre-cached latent feature matrix `z_cached` to avoid forward pass.
    """
    eeg_full = trial['eeg'].numpy()
    env_l_raw = trial['env_l'].numpy()
    env_r_raw = trial['env_r'].numpy()
    
    eeg = (eeg_full - eeg_full.mean(axis=1, keepdims=True)) / (eeg_full.std(axis=1, keepdims=True) + 1e-8)
    env_l = (env_l_raw - env_l_raw.mean()) / (env_l_raw.std() + 1e-8)
    env_r = (env_r_raw - env_r_raw.mean()) / (env_r_raw.std() + 1e-8)
    
    att, unatt = build_ground_truth_envelope(trial)
    
    if z_cached is None:
        eeg_tensor = torch.from_numpy(eeg).float().unsqueeze(0).to(device)
        with torch.no_grad():
            pred = model(eeg_tensor).squeeze(0).cpu().numpy()
    else:
        # If z is cached, apply the linear projection directly
        W_tensor = model.ridge_decoder.weight.data
        z_lagged = model._build_lagged_tensor(torch.from_numpy(z_cached).unsqueeze(0).to(device))
        with torch.no_grad():
            pred = F.conv1d(z_lagged, W_tensor).squeeze(0).squeeze(0).cpu().numpy()
        
    num_windows = (eeg.shape[1] - WINDOW_LEN) // HOP_LEN + 1
    if num_windows <= 0: return [], [], 0.0, None, None
    
    corr_L, corr_R = [], []
    true_att_corr, true_unatt_corr = [], []
    
    for i in range(num_windows):
        start = i * HOP_LEN
        end = start + WINDOW_LEN
        
        pred_w = pred[start:end]
        att_w = att[start:end]
        unatt_w = unatt[start:end]
        env_l_w = env_l[start:end]
        env_r_w = env_r[start:end]
        
        true_att_corr.append(safe_pearson(pred_w, att_w))
        true_unatt_corr.append(safe_pearson(pred_w, unatt_w))
        corr_L.append(safe_pearson(pred_w, env_l_w))
        corr_R.append(safe_pearson(pred_w, env_r_w))
            
    # Robust Median Confidence
    med_corr_L = np.median(corr_L)
    med_corr_R = np.median(corr_R)
    confidence = abs(med_corr_L - med_corr_R)
    
    pseudo_target = env_l if med_corr_L > med_corr_R else env_r
    
    return true_att_corr, true_unatt_corr, confidence, eeg, pseudo_target

def build_ridge_covariance(z_numpy, y_env):
    """Builds Ridge ZtZ and ZtY directly from pre-extracted latent Z numpy array."""
    Z = build_lagged_matrix(z_numpy, MAX_LAG).astype(np.float32)
    y_aligned = y_env[MAX_LAG:].astype(np.float32)
    return Z.T @ Z, Z.T @ y_aligned

def solve_ridge(ZtZ, ZtY, lambda_val):
    return np.linalg.solve(ZtZ + lambda_val * np.eye(ZtZ.shape[0], dtype=np.float32), ZtY)

def get_trial_z(model, eeg, device):
    eeg_8ch = torch.from_numpy(eeg).float().unsqueeze(0).to(device)[:, PHYSICAL_8_CHANNELS, :]
    with torch.no_grad():
        z = model.extract_features(eeg_8ch).squeeze(0).cpu().numpy()
    return z

def precompute_subject_covariances(cache_dir, subject_ids, device):
    print("--- Precomputing Global Calibration Covariances ---")
    model = NeuralRidgeHybrid(get_cached_kul_conformer(), PHYSICAL_8_CHANNELS, embed_dim=64, max_lag=MAX_LAG).to(device)
    model.eval()
    
    subj_covs = {}
    for subj in subject_ids:
        cached = torch.load(cache_dir / f"{subj}_processed.pt", weights_only=False)
        calib_raw = cached['raw'][:CALIBRATION_TRIALS]
        
        ZtZ_subj = np.zeros((64*(MAX_LAG+1), 64*(MAX_LAG+1)), dtype=np.float32)
        ZtY_subj = np.zeros(64*(MAX_LAG+1), dtype=np.float32)
        
        for tr in calib_raw:
            eeg_full = tr['eeg'].numpy()
            eeg = (eeg_full - eeg_full.mean(axis=1, keepdims=True)) / (eeg_full.std(axis=1, keepdims=True) + 1e-8)
            z_numpy = get_trial_z(model, eeg, device)
            att, _ = build_ground_truth_envelope(tr)
            
            ztz, zty = build_ridge_covariance(z_numpy, att)
            ZtZ_subj += ztz
            ZtY_subj += zty
            
        subj_covs[subj] = (ZtZ_subj, ZtY_subj)
        
    return subj_covs, model

def compute_trial_auroc(true_att_corr_list, true_unatt_corr_list):
    y_t = np.concatenate([np.ones(len(true_att_corr_list)), np.zeros(len(true_unatt_corr_list))])
    y_s = np.concatenate([true_att_corr_list, true_unatt_corr_list])
    return roc_auc_score(y_t, y_s) if len(y_t) > 0 else 0.5

def run_unsupervised_continual_learning(cache_dir, subject_ids, device):
    print("\n=======================================================")
    print(" PHASE 46 V3: ADVANCED UNSUPERVISED CONTINUAL LEARNING ")
    print("=======================================================")
    
    subj_covs, model = precompute_subject_covariances(cache_dir, subject_ids, device)
    all_sim_results = []
    
    for subj in subject_ids:
        print(f"\n>> Processing Subject: {subj}")
        cached = torch.load(cache_dir / f"{subj}_processed.pt", weights_only=False)
        all_test_raw = cached['raw']
        
        calib_raw = all_test_raw[:CALIBRATION_TRIALS]
        eval_raw = all_test_raw[CALIBRATION_TRIALS:]
        
        # 1. Establish Generalized Prior (Leave-One-Subject-Out)
        ZtZ_prior = np.zeros((64*(MAX_LAG+1), 64*(MAX_LAG+1)), dtype=np.float32)
        ZtY_prior = np.zeros(64*(MAX_LAG+1), dtype=np.float32)
        for other_subj, (ztz, zty) in subj_covs.items():
            if other_subj != subj:
                ZtZ_prior += ztz
                ZtY_prior += zty
                
        W_general = solve_ridge(ZtZ_prior, ZtY_prior, LAMBDA_RIDGE)
        
        # 2. Evaluate Zero-Shot (using W_general)
        model.load_analytical_weights(W_general)
        
        # Cache latent Z for all evaluation trials to massively speed up online simulation
        eval_cached_z = []
        zs_sa, zs_su = [], []
        for tr in eval_raw:
            eeg_full = tr['eeg'].numpy()
            eeg = (eeg_full - eeg_full.mean(axis=1, keepdims=True)) / (eeg_full.std(axis=1, keepdims=True) + 1e-8)
            z_numpy = get_trial_z(model, eeg, device)
            eval_cached_z.append((tr, eeg, z_numpy))
            
            sa, su, _, _, _ = simulate_trial_unsupervised(model, tr, device, z_cached=z_numpy)
            zs_sa.extend(sa)
            zs_su.extend(su)
        zs_auroc = compute_trial_auroc(zs_sa, zs_su)
        print(f"   [Zero-Shot (Generalized)] AUROC: {zs_auroc:.4f}")
        
        # 3. Supervised Calibration (Initial Fitting Session)
        # Adds the subject's 5 trials to the Generalized Prior
        ZtZ_calib_only, ZtY_calib_only = subj_covs[subj]
        ZtZ_fitted = ZtZ_prior + ZtZ_calib_only
        ZtY_fitted = ZtY_prior + ZtY_calib_only
        
        W_fitted = solve_ridge(ZtZ_fitted, ZtY_fitted, LAMBDA_RIDGE)
        model.load_analytical_weights(W_fitted)
        
        calib_sa, calib_su = [], []
        for (tr, eeg, z_numpy) in eval_cached_z:
            sa, su, _, _, _ = simulate_trial_unsupervised(model, tr, device, z_cached=z_numpy)
            calib_sa.extend(sa)
            calib_su.extend(su)
        calib_auroc = compute_trial_auroc(calib_sa, calib_su)
        print(f"   [Calibration + Prior]     AUROC: {calib_auroc:.4f}")
        
        # Determine Data-Driven Confidence Thresholds from Calibration Trials
        calib_confidences = []
        for tr in calib_raw:
            # We must use the model state *after* calibration to get representative confidence
            _, _, conf, _, _ = simulate_trial_unsupervised(model, tr, device, z_cached=None)
            calib_confidences.append(conf)
            
        p50 = np.percentile(calib_confidences, 50)
        p75 = np.percentile(calib_confidences, 75)
        p90 = np.percentile(calib_confidences, 90)
        p95 = np.percentile(calib_confidences, 95)
        thresholds = {'P50': p50, 'P75': p75, 'P90': p90, 'P95': p95}
        
        # 4. Online Continual Learning (Unsupervised RLS)
        for t_name, threshold in thresholds.items():
            ZtZ_online = ZtZ_fitted.copy()
            ZtY_online = ZtY_fitted.copy()
            model.load_analytical_weights(W_fitted)
            
            timepoints = {}
            updates_accepted = 0
            replay_ztz = np.zeros_like(ZtZ_fitted)
            replay_zty = np.zeros_like(ZtY_fitted)
            trials_since_update = 0
            
            sim_att_all, sim_unatt_all = [], []
            
            for t_idx, (trial, eeg, z_numpy) in enumerate(eval_cached_z):
                sa, su, conf, _, pseudo_target = simulate_trial_unsupervised(model, trial, device, z_cached=z_numpy)
                sim_att_all.extend(sa)
                sim_unatt_all.extend(su)
                
                # Confidence-Weighted Replay Buffer Accumulation
                if conf > threshold:
                    ztz, zty = build_ridge_covariance(z_numpy, pseudo_target)
                    replay_ztz += conf * ztz
                    replay_zty += conf * zty
                    updates_accepted += 1
                
                trials_since_update += 1
                
                # Periodic Exponentially-Forgetting RLS Update
                if trials_since_update >= UPDATE_INTERVAL_TRIALS:
                    ZtZ_online = FORGETTING_FACTOR * ZtZ_online + replay_ztz
                    ZtY_online = FORGETTING_FACTOR * ZtY_online + replay_zty
                    W_online = solve_ridge(ZtZ_online, ZtY_online, LAMBDA_RIDGE)
                    model.load_analytical_weights(W_online)
                    
                    replay_ztz.fill(0)
                    replay_zty.fill(0)
                    trials_since_update = 0
                
                if t_idx == 4:
                    timepoints['5_min'] = compute_trial_auroc(sim_att_all, sim_unatt_all)
                elif t_idx == 14:
                    timepoints['15_min'] = compute_trial_auroc(sim_att_all, sim_unatt_all)
                elif t_idx == 29:
                    timepoints['30_min'] = compute_trial_auroc(sim_att_all, sim_unatt_all)
            
            # Final flush if leftover trials
            if trials_since_update > 0:
                ZtZ_online = FORGETTING_FACTOR * ZtZ_online + replay_ztz
                ZtY_online = FORGETTING_FACTOR * ZtY_online + replay_zty
                W_online = solve_ridge(ZtZ_online, ZtY_online, LAMBDA_RIDGE)
                model.load_analytical_weights(W_online)
                
            final_auroc = compute_trial_auroc(sim_att_all, sim_unatt_all)
            
            all_sim_results.append({
                'Subject': subj,
                'Condition': f'Online_{t_name}',
                'Updates_Accepted': updates_accepted,
                '5_min_AUROC': timepoints.get('5_min', 0.5),
                '15_min_AUROC': timepoints.get('15_min', 0.5),
                '30_min_AUROC': timepoints.get('30_min', 0.5),
                'Final_AUROC': final_auroc
            })
            print(f"   [Online {t_name}] AUROC: {final_auroc:.4f} | Updates: {updates_accepted}/{len(eval_raw)}")
            
        all_sim_results.append({'Subject': subj, 'Condition': 'Zero_Shot', 'Updates_Accepted': 0, '5_min_AUROC': zs_auroc, '15_min_AUROC': zs_auroc, '30_min_AUROC': zs_auroc, 'Final_AUROC': zs_auroc})
        all_sim_results.append({'Subject': subj, 'Condition': 'Calibration_Only', 'Updates_Accepted': 0, '5_min_AUROC': calib_auroc, '15_min_AUROC': calib_auroc, '30_min_AUROC': calib_auroc, 'Final_AUROC': calib_auroc})
            
    df = pd.DataFrame(all_sim_results)
    print("\n=======================================================")
    print(" SUMMARY: PHASE 46 V3 RESULTS                          ")
    print("=======================================================")
    summary = df.groupby('Condition').mean()[['Updates_Accepted', '5_min_AUROC', '15_min_AUROC', '30_min_AUROC', 'Final_AUROC']]
    print(summary)
    
    df.to_csv("phase46_v3_unsupervised_online_report.csv", index=False)
    summary.to_csv("phase46_v3_unsupervised_online_summary.csv")
    
    # Plot Trajectories
    plt.figure(figsize=(10, 6))
    conditions = df['Condition'].unique()
    for cond in conditions:
        subset = df[df['Condition'] == cond]
        times = [5, 15, 30, 50]
        means = [subset['5_min_AUROC'].mean(), subset['15_min_AUROC'].mean(), subset['30_min_AUROC'].mean(), subset['Final_AUROC'].mean()]
        plt.plot(times, means, marker='o', label=f'{cond}')
        
    plt.title("V3 Unsupervised Continual Adaptation Trajectory")
    plt.xlabel("Time in Session (Minutes / Trials)")
    plt.ylabel("Cumulative AUROC")
    plt.legend()
    plt.grid(True)
    plt.savefig("phase46_v3_adaptation_trajectory.png")

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
