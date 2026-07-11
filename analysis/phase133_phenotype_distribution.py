import os
import numpy as np
import torch
import pandas as pd
from pathlib import Path
from sklearn.metrics import roc_auc_score
from scipy import signal
import time
import concurrent.futures
import multiprocessing as mp

# -------------------------------------------------------------------------
# CONSTANTS & CONFIGURATION
# -------------------------------------------------------------------------
SR = 128
EAR_CHANNEL_INDICES = [23, 31, 32, 40, 14, 22, 41, 49]
MAX_LAG_SEC = 0.400 
MAX_LAG_SAMPLES = int(MAX_LAG_SEC * SR)

BROADBAND = (0.5, 8.0)
RIDGE_LAMBDA_GLOBAL = 100.0
LAMBDA_PRIOR = 5000.0  
PCA_COMPONENTS = 60
N_FOLDS = 4
N_NULL_SHIFTS = 200

def apply_modulation_filter(env, lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype='band')
    return signal.filtfilt(b, a, env, axis=1)

def create_toeplitz_features_pt(eeg, max_lag_samples):
    C, T = eeg.shape
    T_eff = T - max_lag_samples
    X = torch.zeros((T_eff, C * max_lag_samples), dtype=eeg.dtype, device=eeg.device)
    for tau in range(max_lag_samples):
        X[:, tau*C : (tau+1)*C] = eeg[:, tau : tau+T_eff].T
    return X

def batch_pearsonr_pt(x, y):
    x_mean = x - x.mean(dim=1, keepdim=True)
    y_mean = y - y.mean(dim=1, keepdim=True)
    num = (x_mean * y_mean).sum(dim=1)
    den = torch.sqrt((x_mean**2).sum(dim=1) * (y_mean**2).sum(dim=1))
    return num / (den + 1e-8)

def solve_ridge_pt(XTX, XTy, lam=10.0):
    F = XTX.shape[0]
    I = torch.eye(F, device=XTX.device, dtype=XTX.dtype)
    jitter = 1e-6 * torch.randn(F, F, device=XTX.device, dtype=XTX.dtype) * I
    return torch.linalg.solve(XTX + lam * I + jitter, XTy)

def process_subject(cache_file, device_id):
    device = torch.device(f'cuda:{device_id}' if torch.cuda.is_available() else 'cpu')
    subj_name = cache_file.stem.split('_')[0]
    
    cached = torch.load(cache_file, map_location='cpu', weights_only=False)['raw']
    N_TRIALS = len(cached)
    
    all_X = []
    all_YL = []
    all_YR = []
    all_Ymeta = []
    
    for tr in cached:
        eeg_raw = tr['eeg'].numpy()[EAR_CHANNEL_INDICES, :] 
        env_l_raw = tr['env_l'].numpy()
        env_r_raw = tr['env_r'].numpy()
        
        min_len = min(eeg_raw.shape[1], env_l_raw.shape[1])
        eeg_raw = eeg_raw[:, :min_len]
        env_l_raw = env_l_raw[:, :min_len]
        env_r_raw = env_r_raw[:, :min_len]
        
        eeg_f = apply_modulation_filter(eeg_raw, BROADBAND[0], BROADBAND[1], SR)
        env_l_f = apply_modulation_filter(env_l_raw, BROADBAND[0], BROADBAND[1], SR)
        env_r_f = apply_modulation_filter(env_r_raw, BROADBAND[0], BROADBAND[1], SR)
        
        eeg_f = (eeg_f - np.mean(eeg_f, axis=1, keepdims=True)) / (np.std(eeg_f, axis=1, keepdims=True) + 1e-8)
        env_l_f = (env_l_f - np.mean(env_l_f, axis=1, keepdims=True)) / (np.std(env_l_f, axis=1, keepdims=True) + 1e-8)
        env_r_f = (env_r_f - np.mean(env_r_f, axis=1, keepdims=True)) / (np.std(env_r_f, axis=1, keepdims=True) + 1e-8)
        
        eeg = torch.tensor(eeg_f, dtype=torch.float32, device=device)
        env_l = torch.tensor(env_l_f[0], dtype=torch.float32, device=device)
        env_r = torch.tensor(env_r_f[0], dtype=torch.float32, device=device)
        
        X_trial = create_toeplitz_features_pt(eeg, MAX_LAG_SAMPLES)
        T_eff = X_trial.shape[0]
        
        all_X.append(X_trial)
        all_YL.append(env_l[:T_eff])
        all_YR.append(env_r[:T_eff])
        
        sp = tr['meta']['switch_points']
        current_spk = sp[0][0] if len(sp) > 0 else 'L'
        all_Ymeta.append(1.0 if current_spk == 'L' else 0.0)

    # Output Trackers
    eval_labels = []
    
    diffs_static = []
    diffs_abs_cca = []
    diffs_map_cca = []
    
    for test_idx in range(N_TRIALS):
        # 1. Build Static Prior (Cross-Trial)
        train_X_list = []
        train_Y_match_list = []
        
        for i in range(N_TRIALS):
            if i != test_idx:
                train_X_list.append(all_X[i])
                train_Y_match_list.append(all_YL[i] if all_Ymeta[i] == 1.0 else all_YR[i])
                    
        X_train_cat = torch.cat(train_X_list, dim=0)
        Y_match_cat = torch.cat(train_Y_match_list, dim=0)
        
        U, S, V = torch.pca_lowrank(X_train_cat, q=PCA_COMPONENTS)
        X_train_pca = torch.matmul(X_train_cat, V)
        
        XTX_train = X_train_pca.T @ X_train_pca
        XTy_train = (X_train_pca.T @ Y_match_cat).unsqueeze(-1)
        W_prior = solve_ridge_pt(XTX_train, XTy_train, lam=RIDGE_LAMBDA_GLOBAL)
        
        # Test Data
        X_test_toeplitz = all_X[test_idx]
        X_test_pca = torch.matmul(X_test_toeplitz, V)
        YL_test = all_YL[test_idx]
        YR_test = all_YR[test_idx]
        T_eff = X_test_pca.shape[0]
        block_size = T_eff // N_FOLDS
        
        # Strategy 1: STATIC PREDICTION (Entire trial)
        # We simply evaluate W_prior on the whole test trial
        Y_hat_static_L = (X_test_pca @ W_prior).squeeze(-1)
        Y_hat_static_R = (X_test_pca @ W_prior).squeeze(-1) # Wait, W_prior is evaluated against both!
        # Actually, r_L is corr(X*W_prior, Y_L) and r_R is corr(X*W_prior, Y_R)
        r_L_stat = batch_pearsonr_pt(Y_hat_static_L.unsqueeze(0), YL_test.unsqueeze(0)).item()
        r_R_stat = batch_pearsonr_pt(Y_hat_static_R.unsqueeze(0), YR_test.unsqueeze(0)).item()
        diffs_static.append(r_L_stat - r_R_stat)
        
        # Strategy 2 & 3: Within-Trial Adaptive
        z_scores_L_abs = []
        z_scores_R_abs = []
        z_scores_L_map = []
        z_scores_R_map = []
        
        for fold in range(N_FOLDS):
            test_start = fold * block_size
            test_end = (fold + 1) * block_size if fold < N_FOLDS - 1 else T_eff
            
            train_mask = torch.ones(T_eff, dtype=torch.bool, device=device)
            train_mask[test_start:test_end] = False
            
            X_cv_train = X_test_pca[train_mask]
            YL_cv_train = YL_test[train_mask]
            YR_cv_train = YR_test[train_mask]
            
            X_cv_test = X_test_pca[~train_mask]
            YL_cv_test = YL_test[~train_mask]
            YR_cv_test = YR_test[~train_mask]
            
            XTX_cv = X_cv_train.T @ X_cv_train
            XTy_L = (X_cv_train.T @ YL_cv_train).unsqueeze(-1)
            XTy_R = (X_cv_train.T @ YR_cv_train).unsqueeze(-1)
            
            # --- Unconstrained CCA (for ABS) ---
            W_L_unconst = solve_ridge_pt(XTX_cv, XTy_L, lam=RIDGE_LAMBDA_GLOBAL)
            W_R_unconst = solve_ridge_pt(XTX_cv, XTy_R, lam=RIDGE_LAMBDA_GLOBAL)
            
            Y_hat_L_unconst = (X_cv_test @ W_L_unconst).squeeze(-1)
            Y_hat_R_unconst = (X_cv_test @ W_R_unconst).squeeze(-1)
            
            r_L_true_unc = batch_pearsonr_pt(Y_hat_L_unconst.unsqueeze(0), YL_cv_test.unsqueeze(0)).item()
            r_R_true_unc = batch_pearsonr_pt(Y_hat_R_unconst.unsqueeze(0), YR_cv_test.unsqueeze(0)).item()
            
            # --- Bayesian MAP CCA ---
            W_L_map = solve_ridge_pt(XTX_cv, XTy_L + LAMBDA_PRIOR * W_prior, lam=LAMBDA_PRIOR)
            W_R_map = solve_ridge_pt(XTX_cv, XTy_R + LAMBDA_PRIOR * W_prior, lam=LAMBDA_PRIOR)
            
            Y_hat_L_map = (X_cv_test @ W_L_map).squeeze(-1)
            Y_hat_R_map = (X_cv_test @ W_R_map).squeeze(-1)
            
            r_L_true_map = batch_pearsonr_pt(Y_hat_L_map.unsqueeze(0), YL_cv_test.unsqueeze(0)).item()
            r_R_true_map = batch_pearsonr_pt(Y_hat_R_map.unsqueeze(0), YR_cv_test.unsqueeze(0)).item()
            
            # --- Null Distributions ---
            T_test_block = YL_cv_test.shape[0]
            shifts = torch.randint(SR * 1, T_test_block - SR * 1, (N_NULL_SHIFTS,), device=device)
            YL_null = torch.stack([torch.roll(YL_cv_test, shifts[i].item()) for i in range(N_NULL_SHIFTS)])
            YR_null = torch.stack([torch.roll(YR_cv_test, shifts[i].item()) for i in range(N_NULL_SHIFTS)])
            
            # Nulls for Unconstrained (ABS)
            r_L_null_unc = batch_pearsonr_pt(Y_hat_L_unconst.unsqueeze(0).expand(N_NULL_SHIFTS, -1), YL_null)
            r_R_null_unc = batch_pearsonr_pt(Y_hat_R_unconst.unsqueeze(0).expand(N_NULL_SHIFTS, -1), YR_null)
            z_L_unc = (r_L_true_unc - r_L_null_unc.mean().item()) / (r_L_null_unc.std().item() + 1e-8)
            z_R_unc = (r_R_true_unc - r_R_null_unc.mean().item()) / (r_R_null_unc.std().item() + 1e-8)
            z_scores_L_abs.append(z_L_unc)
            z_scores_R_abs.append(z_R_unc)
            
            # Nulls for MAP
            r_L_null_map = batch_pearsonr_pt(Y_hat_L_map.unsqueeze(0).expand(N_NULL_SHIFTS, -1), YL_null)
            r_R_null_map = batch_pearsonr_pt(Y_hat_R_map.unsqueeze(0).expand(N_NULL_SHIFTS, -1), YR_null)
            z_L_map = (r_L_true_map - r_L_null_map.mean().item()) / (r_L_null_map.std().item() + 1e-8)
            z_R_map = (r_R_true_map - r_R_null_map.mean().item()) / (r_R_null_map.std().item() + 1e-8)
            z_scores_L_map.append(z_L_map)
            z_scores_R_map.append(z_R_map)
            
        diffs_abs_cca.append(abs(np.mean(z_scores_L_abs)) - abs(np.mean(z_scores_R_abs)))
        diffs_map_cca.append(np.mean(z_scores_L_map) - np.mean(z_scores_R_map))
        
        eval_labels.append(all_Ymeta[test_idx])
        
    eval_labels = np.array(eval_labels)
    
    def get_auc(diffs):
        diffs = np.array(diffs)
        if len(diffs) > 0 and len(np.unique(eval_labels)) > 1:
            probs = (diffs - np.min(diffs)) / (np.max(diffs) - np.min(diffs) + 1e-8)
            return roc_auc_score(eval_labels, probs)
        return 0.5

    auc_stat = get_auc(diffs_static)
    auc_abs = get_auc(diffs_abs_cca)
    auc_map = get_auc(diffs_map_cca)
    
    print(f"  [{subj_name}] Static: {auc_stat:.3f} | ABS CCA: {auc_abs:.3f} | MAP CCA: {auc_map:.3f}")
    
    return subj_name, auc_stat, auc_abs, auc_map

def main():
    mp.set_start_method('spawn', force=True)
    
    cache_dir = Path('/kaggle/working/multiband_cache')
    possible_paths = [
        Path('/kaggle/input/datasets/lowkieee/multiband-cache/kaggle/working/multiband_cache'),
        Path('/kaggle/input/multiband-cache/kaggle/working/multiband_cache'),
        Path('/kaggle/input/aasd-universal-cache-v1/kaggle/working/multiband_cache'),
        cache_dir
    ]
    for p in possible_paths:
        if p.exists() and len(list(p.glob('*_multiband.pt'))) > 0:
            cache_dir = p
            break
            
    num_gpus = torch.cuda.device_count()
    num_workers = min(mp.cpu_count(), num_gpus if num_gpus > 0 else mp.cpu_count())
    
    print(f"\n=======================================================")
    print(f" PHASE 133: PHENOTYPE DISTRIBUTION ANALYSIS")
    print(f" CPUs detected: {mp.cpu_count()} | GPUs detected: {num_gpus}")
    print(f"=======================================================\n", flush=True)
    
    cache_files = sorted(list(cache_dir.glob('*_multiband.pt')))
    
    results = []
    futures = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        for idx, cache_file in enumerate(cache_files):
            device_id = idx % num_gpus if num_gpus > 0 else 0
            futures.append(executor.submit(process_subject, cache_file, device_id))
            
        for future in concurrent.futures.as_completed(futures):
            subj, auc_stat, auc_abs, auc_map = future.result()
            results.append({
                'Subject': subj,
                'Static': auc_stat,
                'ABS': auc_abs,
                'MAP': auc_map,
                'Delta_ABS': auc_abs - auc_stat,
                'Delta_MAP': auc_map - auc_stat
            })

    df = pd.DataFrame(results).sort_values('Subject')
    
    print("\n=======================================================")
    print(" PHASE 133 PHENOTYPE ADAPTATION BENEFIT")
    print("=======================================================")
    print(df.to_string(index=False, float_format='%.3f'))
    
    # Statistical analysis of Delta_MAP
    print("\n=== Delta_MAP Distribution (Is it bimodal?) ===")
    delta_map = df['Delta_MAP'].values
    mean_d = np.mean(delta_map)
    std_d = np.std(delta_map)
    print(f"Mean Delta_MAP: {mean_d:.3f}, Std: {std_d:.3f}")
    
    # Basic histogram
    bins = np.linspace(-0.2, 0.2, 10)
    hist, _ = np.histogram(delta_map, bins=bins)
    print("\nHistogram of Delta_MAP (bin_edges, counts):")
    for i in range(len(hist)):
        print(f"  [{bins[i]:.2f} to {bins[i+1]:.2f}]: {'*' * hist[i]} ({hist[i]})")

if __name__ == '__main__':
    main()
