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
N_FOLDS = 4
N_NULL_SHIFTS = 200

SOFTMAX_BETA = 2.0  # Temperature for Bayesian Model Averaging

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

def evaluate_expert_for_stream(X_test_pca, Y_test, W_prior, expert_type, device):
    T_eff = X_test_pca.shape[0]
    block_size = T_eff // N_FOLDS
    
    z_folds = []
    
    for fold in range(N_FOLDS):
        test_start = fold * block_size
        test_end = (fold + 1) * block_size if fold < N_FOLDS - 1 else T_eff
        
        train_mask = torch.ones(T_eff, dtype=torch.bool, device=device)
        train_mask[test_start:test_end] = False
        
        X_cv_train = X_test_pca[train_mask]
        Y_cv_train = Y_test[train_mask]
        X_cv_test = X_test_pca[~train_mask]
        Y_cv_test = Y_test[~train_mask]
        
        XTX_cv = X_cv_train.T @ X_cv_train
        XTy_cv = (X_cv_train.T @ Y_cv_train).unsqueeze(-1)
        
        if expert_type == 'Static':
            W = W_prior
        elif expert_type == 'MAP':
            W = solve_ridge_pt(XTX_cv, XTy_cv + LAMBDA_MAP * W_prior, lam=LAMBDA_MAP)
        elif expert_type == 'ABS':
            W = solve_ridge_pt(XTX_cv, XTy_cv, lam=LAMBDA_ABS)
            
        Y_hat = (X_cv_test @ W).squeeze(-1)
        r_true = batch_pearsonr_pt(Y_hat.unsqueeze(0), Y_cv_test.unsqueeze(0)).item()
        
        T_test_block = Y_cv_test.shape[0]
        shifts = torch.randint(SR * 1, T_test_block - SR * 1, (N_NULL_SHIFTS,), device=device)
        Y_null = torch.stack([torch.roll(Y_cv_test, shifts[i].item()) for i in range(N_NULL_SHIFTS)])
        r_null = batch_pearsonr_pt(Y_hat.unsqueeze(0).expand(N_NULL_SHIFTS, -1), Y_null)
        
        z = (r_true - r_null.mean().item()) / (r_null.std().item() + 1e-8)
        z_folds.append(z)
        
    mean_z = np.mean(z_folds)
    return mean_z

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

    eval_diffs = []
    eval_labels = []
    
    # Store per-expert scores for diagnostic tracking
    expert_track = {'Static': 0, 'MAP': 0, 'ABS': 0}
    
    for test_idx in range(N_TRIALS):
        # Build Static Prior
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
        
        X_test_pca = torch.matmul(all_X[test_idx], V)
        YL_test = all_YL[test_idx]
        YR_test = all_YR[test_idx]
        
        # --- PROBABILISTIC MIXTURE OF EXPERTS ---
        final_zs = []
        for Y_cand in [YL_test, YR_test]:
            z_S = evaluate_expert_for_stream(X_test_pca, Y_cand, W_prior, 'Static', device)
            z_M = evaluate_expert_for_stream(X_test_pca, Y_cand, W_prior, 'MAP', device)
            z_A = evaluate_expert_for_stream(X_test_pca, Y_cand, W_prior, 'ABS', device)
            
            # Evidence computation: The magnitude of the held-out Z-score
            E_S = abs(z_S)
            E_M = abs(z_M)
            E_A = abs(z_A)
            
            # Softmax Model Averaging
            evidences = np.array([E_S, E_M, E_A])
            exp_E = np.exp(SOFTMAX_BETA * evidences)
            weights = exp_E / np.sum(exp_E)
            
            # Final blended positive likelihood score
            z_final = weights[0]*E_S + weights[1]*E_M + weights[2]*E_A
            final_zs.append(z_final)
            
            # Track which expert dominated for Match stream
            if len(final_zs) == 1 and all_Ymeta[test_idx] == 1.0:
                best_idx = np.argmax(weights)
                expert_track[['Static', 'MAP', 'ABS'][best_idx]] += 1
            elif len(final_zs) == 2 and all_Ymeta[test_idx] == 0.0:
                best_idx = np.argmax(weights)
                expert_track[['Static', 'MAP', 'ABS'][best_idx]] += 1
                
        # Unsigned Decision Rule (z_L and z_R are already positive magnitudes)
        z_L, z_R = final_zs
        diff = z_L - z_R
        
        eval_diffs.append(diff)
        eval_labels.append(all_Ymeta[test_idx])
        
    eval_diffs = np.array(eval_diffs)
    eval_labels = np.array(eval_labels)
    
    if len(eval_diffs) > 0 and len(np.unique(eval_labels)) > 1:
        probs = (eval_diffs - np.min(eval_diffs)) / (np.max(eval_diffs) - np.min(eval_diffs) + 1e-8)
        global_auc = roc_auc_score(eval_labels, probs)
    else:
        global_auc = 0.5
        
    print(f"  [{subj_name}] Probabilistic MoE AUROC: {global_auc:.4f} | Favored: Static({expert_track['Static']}) MAP({expert_track['MAP']}) ABS({expert_track['ABS']})")
    
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
    print(f" PHASE 135: PROBABILISTIC MIXTURE OF EXPERTS (P-MoE)")
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
    print(" PHASE 135 PROBABILISTIC MoE RESULTS")
    print("=======================================================")
    print(f"{'Subject':<10} {'AUROC':<10}")
    
    sorted_results = sorted(final_results.items(), key=lambda x: int(x[0][1:]))
    for subj, auc in sorted_results:
        print(f"{subj:<10} {auc:.4f}")

if __name__ == '__main__':
    main()
