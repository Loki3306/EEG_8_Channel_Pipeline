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
LAMBDA_MAP = 5000.0  
LAMBDA_ABS = 100.0
PCA_COMPONENTS = 60

WIN_SEC = 5.0
PAST_SEC = 15.0
WIN_SAMPLES = int(WIN_SEC * SR)
PAST_SAMPLES = int(PAST_SEC * SR)

PRE_SWITCH_SAMPLES = int(1.0 * SR)
POST_SWITCH_SAMPLES = int(3.0 * SR)

SOFTMAX_BETA = 10.0  # Higher temp because evidence is in [0, 1]

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

def get_masks(sp, length):
    # mask_true: 1.0 for L, 0.0 for R
    # mask_valid: True for clean tracking, False for blackout zones
    mask_true = np.zeros(length, dtype=np.float32)
    mask_valid = np.ones(length, dtype=bool)
    
    if len(sp) == 0:
        mask_true[:] = 1.0
        return mask_true, mask_valid
        
    current_state = 1.0 if sp[0][0] == 'R' else 0.0 
    last_idx = 0
    for spk, idx in sp:
        end_idx = min(idx, length)
        mask_true[last_idx:end_idx] = current_state
        current_state = 1.0 if spk == 'L' else 0.0
        last_idx = end_idx
        
        # Apply Blackout Zone around switch
        b_start = max(0, idx - PRE_SWITCH_SAMPLES)
        b_end = min(length, idx + POST_SWITCH_SAMPLES)
        mask_valid[b_start:b_end] = False
        
        if last_idx >= length:
            break
            
    if last_idx < length:
        mask_true[last_idx:] = current_state
        
    return mask_true, mask_valid

def batch_pearsonr_pt(x, y):
    x_mean = x - x.mean(dim=0, keepdim=True)
    y_mean = y - y.mean(dim=0, keepdim=True)
    num = (x_mean * y_mean).sum(dim=0)
    den = torch.sqrt((x_mean**2).sum(dim=0) * (y_mean**2).sum(dim=0))
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
    all_Ytrue = []
    all_mask_true = []
    all_mask_valid = []
    
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
        
        mask_t, mask_v = get_masks(tr['meta'].get('switch_points', []), min_len)
        mask_t = torch.tensor(mask_t[MAX_LAG_SAMPLES:min_len], dtype=torch.float32, device=device)
        mask_v = torch.tensor(mask_v[MAX_LAG_SAMPLES:min_len], dtype=torch.bool, device=device)
        
        all_X.append(X_trial)
        all_YL.append(env_l[:T_eff])
        all_YR.append(env_r[:T_eff])
        all_mask_true.append(mask_t)
        all_mask_valid.append(mask_v)
        
        Y_true = env_l[:T_eff] * mask_t + env_r[:T_eff] * (1.0 - mask_t)
        all_Ytrue.append(Y_true)

    trial_diffs = []
    trial_labels = []
    
    expert_track = {'Static': 0, 'MAP': 0, 'ABS': 0}
    
    for test_idx in range(N_TRIALS):
        # 1. Train Universal Static Prior (excluding blackout zones!)
        train_X_list = []
        train_Y_true_list = []
        for i in range(N_TRIALS):
            if i != test_idx:
                X_clean = all_X[i][all_mask_valid[i]]
                Y_clean = all_Ytrue[i][all_mask_valid[i]]
                if X_clean.shape[0] > 0:
                    train_X_list.append(X_clean)
                    train_Y_true_list.append(Y_clean)
                    
        X_train_cat = torch.cat(train_X_list, dim=0)
        Y_true_cat = torch.cat(train_Y_true_list, dim=0)
        
        U, S, V = torch.pca_lowrank(X_train_cat, q=PCA_COMPONENTS)
        X_train_pca = torch.matmul(X_train_cat, V)
        
        XTX_train = X_train_pca.T @ X_train_pca
        XTy_train = (X_train_pca.T @ Y_true_cat).unsqueeze(-1)
        W_prior = solve_ridge_pt(XTX_train, XTy_train, lam=RIDGE_LAMBDA_GLOBAL)
        
        X_test_pca = torch.matmul(all_X[test_idx], V)
        YL_test = all_YL[test_idx]
        YR_test = all_YR[test_idx]
        mask_t_test = all_mask_true[test_idx]
        mask_v_test = all_mask_valid[test_idx]
        
        T_eff = X_test_pca.shape[0]
        
        window_diffs = []
        
        # 2. Continuous 5-second decoding
        for start in range(0, T_eff - WIN_SAMPLES + 1, WIN_SAMPLES):
            end = start + WIN_SAMPLES
            
            # Skip if any part of the window is in a blackout zone
            if not mask_v_test[start:end].all().item():
                continue
                
            mask_win = mask_t_test[start:end]
            mean_mask = mask_win.mean().item()
            
            if mean_mask >= 0.8:
                label = 1.0
            elif mean_mask <= 0.2:
                label = 0.0
            else:
                continue 
                
            X_test_win = X_test_pca[start:end]
            YL_test_win = YL_test[start:end]
            YR_test_win = YR_test[start:end]
            
            # Past 15 seconds for adaptation (excluding blackout zones)
            train_start = max(0, start - PAST_SAMPLES)
            
            # Extract valid past samples
            past_valid = mask_v_test[train_start:start]
            if not past_valid.any().item():
                W_MAP_L = W_prior
                W_MAP_R = W_prior
                W_ABS_L = W_prior
                W_ABS_R = W_prior
            else:
                X_train_win = X_test_pca[train_start:start][past_valid]
                YL_train_win = YL_test[train_start:start][past_valid]
                YR_train_win = YR_test[train_start:start][past_valid]
                
                XTX_tr = X_train_win.T @ X_train_win
                XTy_L = (X_train_win.T @ YL_train_win).unsqueeze(-1)
                XTy_R = (X_train_win.T @ YR_train_win).unsqueeze(-1)
                
                W_MAP_L = solve_ridge_pt(XTX_tr, XTy_L + LAMBDA_MAP * W_prior, lam=LAMBDA_MAP)
                W_MAP_R = solve_ridge_pt(XTX_tr, XTy_R + LAMBDA_MAP * W_prior, lam=LAMBDA_MAP)
                
                W_ABS_L = solve_ridge_pt(XTX_tr, XTy_L, lam=LAMBDA_ABS)
                W_ABS_R = solve_ridge_pt(XTX_tr, XTy_R, lam=LAMBDA_ABS)
                
            def get_E(W, Y):
                Y_hat = (X_test_win @ W).squeeze(-1)
                r = batch_pearsonr_pt(Y_hat, Y).item()
                return abs(r)
                
            E_S_L = get_E(W_prior, YL_test_win)
            E_M_L = get_E(W_MAP_L, YL_test_win)
            E_A_L = get_E(W_ABS_L, YL_test_win)
            
            E_S_R = get_E(W_prior, YR_test_win)
            E_M_R = get_E(W_MAP_R, YR_test_win)
            E_A_R = get_E(W_ABS_R, YR_test_win)
            
            # MoE Blending
            ev_L = np.array([E_S_L, E_M_L, E_A_L])
            exp_L = np.exp(SOFTMAX_BETA * ev_L)
            w_L = exp_L / np.sum(exp_L)
            final_L = np.sum(w_L * ev_L)
            
            ev_R = np.array([E_S_R, E_M_R, E_A_R])
            exp_R = np.exp(SOFTMAX_BETA * ev_R)
            w_R = exp_R / np.sum(exp_R)
            final_R = np.sum(w_R * ev_R)
            
            window_diffs.append(final_L - final_R)
            
            if label == 1.0:
                expert_track[['Static', 'MAP', 'ABS'][np.argmax(ev_L)]] += 1
            else:
                expert_track[['Static', 'MAP', 'ABS'][np.argmax(ev_R)]] += 1

        # 3. 60-Second Reintegration
        if len(window_diffs) > 0:
            trial_mean_diff = np.mean(window_diffs)
            
            # Compute Dominant Ground Truth for the valid parts of the trial
            valid_mask_t = mask_t_test[mask_v_test]
            dominant_label = 1.0 if valid_mask_t.mean().item() >= 0.5 else 0.0
            
            trial_diffs.append(trial_mean_diff)
            trial_labels.append(dominant_label)

    trial_diffs = np.array(trial_diffs)
    trial_labels = np.array(trial_labels)
    
    if len(trial_diffs) > 0 and len(np.unique(trial_labels)) > 1:
        probs = (trial_diffs - np.min(trial_diffs)) / (np.max(trial_diffs) - np.min(trial_diffs) + 1e-8)
        global_auc = roc_auc_score(trial_labels, probs)
    else:
        global_auc = 0.5
        
    print(f"  [{subj_name}] P-MoE Integrated AUROC: {global_auc:.4f} | Favored: S({expert_track['Static']}) M({expert_track['MAP']}) A({expert_track['ABS']})")
    
    return subj_name, global_auc

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
    print(f" PHASE 138: BLACKOUT ZONES & 60-SECOND REINTEGRATION")
    print(f" CPUs detected: {mp.cpu_count()} | GPUs detected: {num_gpus}")
    print(f"=======================================================\n", flush=True)
    
    cache_files = sorted(list(cache_dir.glob('*_multiband.pt')))
    
    start_time = time.time()
    final_results = {}
    
    futures = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        for idx, cache_file in enumerate(cache_files):
            device_id = idx % num_gpus if num_gpus > 0 else 0
            futures.append(executor.submit(process_subject, cache_file, device_id))
            
        for future in concurrent.futures.as_completed(futures):
            subj_name, auc = future.result()
            final_results[subj_name] = auc

    print(f"\nTotal Pipeline Execution Time: {time.time() - start_time:.2f}s")
    print("\n=======================================================")
    print(" PHASE 138 REINTEGRATED MoE RESULTS")
    print("=======================================================")
    print(f"{'Subject':<10} {'Integrated AUROC':<20}")
    
    sorted_results = sorted(final_results.items(), key=lambda x: int(x[0][1:]))
    for subj, auc in sorted_results:
        print(f"{subj:<10} {auc:.4f}")

if __name__ == '__main__':
    main()
