import os
import numpy as np
import torch
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
LAMBDA_PRIOR = 5000.0  # Acts as 'virtual samples' pulling towards W_prior
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
    
    # Pre-process all trials into memory
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

    # Leave-One-Trial-Out Framework
    all_eval_diffs = []
    all_eval_labels = []
    
    for test_idx in range(N_TRIALS):
        # 1. Build Cross-Trial Prior from other 59 trials
        train_X_list = []
        train_Y_match_list = []
        
        for i in range(N_TRIALS):
            if i != test_idx:
                train_X_list.append(all_X[i])
                if all_Ymeta[i] == 1.0:
                    train_Y_match_list.append(all_YL[i])
                else:
                    train_Y_match_list.append(all_YR[i])
                    
        X_train_cat = torch.cat(train_X_list, dim=0)
        Y_match_cat = torch.cat(train_Y_match_list, dim=0)
        
        # Fit PCA on the 59 training trials
        U, S, V = torch.pca_lowrank(X_train_cat, q=PCA_COMPONENTS)
        X_train_pca = torch.matmul(X_train_cat, V)
        
        # Compute W_prior
        XTX_train = X_train_pca.T @ X_train_pca
        XTy_train = (X_train_pca.T @ Y_match_cat).unsqueeze(-1)
        W_prior = solve_ridge_pt(XTX_train, XTy_train, lam=RIDGE_LAMBDA_GLOBAL)
        
        # 2. Project Test Trial using the SAME PCA V matrix
        X_test_toeplitz = all_X[test_idx]
        X_test_pca = torch.matmul(X_test_toeplitz, V)
        YL_test = all_YL[test_idx]
        YR_test = all_YR[test_idx]
        
        # 3. Within-Trial MAP Estimation
        T_eff = X_test_pca.shape[0]
        block_size = T_eff // N_FOLDS
        z_scores_L = []
        z_scores_R = []
        
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
            
            # MAP Ridge Estimation: (XTX + lambda*I)^-1 (XTy + lambda * W_prior)
            XTX_cv = X_cv_train.T @ X_cv_train
            XTy_L = (X_cv_train.T @ YL_cv_train).unsqueeze(-1)
            XTy_R = (X_cv_train.T @ YR_cv_train).unsqueeze(-1)
            
            W_L = solve_ridge_pt(XTX_cv, XTy_L + LAMBDA_PRIOR * W_prior, lam=LAMBDA_PRIOR)
            W_R = solve_ridge_pt(XTX_cv, XTy_R + LAMBDA_PRIOR * W_prior, lam=LAMBDA_PRIOR)
            
            Y_hat_L = (X_cv_test @ W_L).squeeze(-1)
            Y_hat_R = (X_cv_test @ W_R).squeeze(-1)
            
            r_L_true = batch_pearsonr_pt(Y_hat_L.unsqueeze(0), YL_cv_test.unsqueeze(0)).item()
            r_R_true = batch_pearsonr_pt(Y_hat_R.unsqueeze(0), YR_cv_test.unsqueeze(0)).item()
            
            # Null Distribution Generation
            T_test_block = YL_cv_test.shape[0]
            shift_min = SR * 1 
            shift_max = T_test_block - SR * 1
            shifts = torch.randint(shift_min, shift_max, (N_NULL_SHIFTS,), device=device)
            
            YL_null = torch.stack([torch.roll(YL_cv_test, shifts[i].item()) for i in range(N_NULL_SHIFTS)])
            YR_null = torch.stack([torch.roll(YR_cv_test, shifts[i].item()) for i in range(N_NULL_SHIFTS)])
            
            r_L_null = batch_pearsonr_pt(Y_hat_L.unsqueeze(0).expand(N_NULL_SHIFTS, -1), YL_null)
            r_R_null = batch_pearsonr_pt(Y_hat_R.unsqueeze(0).expand(N_NULL_SHIFTS, -1), YR_null)
            
            z_L = (r_L_true - r_L_null.mean().item()) / (r_L_null.std().item() + 1e-8)
            z_R = (r_R_true - r_R_null.mean().item()) / (r_R_null.std().item() + 1e-8)
            
            z_scores_L.append(z_L)
            z_scores_R.append(z_R)
            
        final_z_L = np.mean(z_scores_L)
        final_z_R = np.mean(z_scores_R)
        
        # ORIGINAL RULE: z_L > z_R
        # We do NOT use abs() because W_prior handles phase inversions automatically!
        diff = final_z_L - final_z_R
        all_eval_diffs.append(diff)
        all_eval_labels.append(all_Ymeta[test_idx])
        
    all_eval_diffs = np.array(all_eval_diffs)
    all_eval_labels = np.array(all_eval_labels)
    
    if len(all_eval_diffs) > 0 and len(np.unique(all_eval_labels)) > 1:
        probs = (all_eval_diffs - np.min(all_eval_diffs)) / (np.max(all_eval_diffs) - np.min(all_eval_diffs) + 1e-8)
        global_auc = roc_auc_score(all_eval_labels, probs)
    else:
        global_auc = 0.5
        
    print(f"  [{subj_name}] Finished on {device}. MAP Adaptive AUROC: {global_auc:.4f}")
    
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
    print(f" PHASE 132: BAYESIAN MAP ADAPTIVE CCA")
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
    print(" PHASE 132 MAP ADAPTIVE RESULTS")
    print("=======================================================")
    print(f"{'Subject':<10} {'MAP AUROC':<10}")
    
    sorted_results = sorted(final_results.items(), key=lambda x: int(x[0][1:]))
    for subj, auc in sorted_results:
        print(f"{subj:<10} {auc:.4f}")

if __name__ == '__main__':
    main()
