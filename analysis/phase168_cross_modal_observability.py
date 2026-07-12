import os
# MUST set before importing torch or numpy to prevent thread thrashing in multiprocessing!
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import time
import numpy as np
import torch
from pathlib import Path
from scipy import signal
import concurrent.futures
import multiprocessing as mp
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score

# -------------------------------------------------------------------------
# CONSTANTS & CONFIGURATION
# -------------------------------------------------------------------------
SR = 128
EAR_CHANNEL_INDICES = [23, 31, 32, 40, 14, 22, 41, 49]
MAX_LAG_SEC = 0.400 
MAX_LAG_SAMPLES = int(MAX_LAG_SEC * SR)
SEQ_SAMPLES = int(3.0 * SR)
SEQ_HOP = int(0.5 * SR)

BROADBAND = (0.5, 8.0)
RIDGE_LAMBDA = 2.0

def apply_modulation_filter(env, lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    b, a = signal.butter(order, [lowcut / nyq, highcut / nyq], btype='band')
    return signal.filtfilt(b, a, env, axis=1)

def create_toeplitz_features_pt(eeg, max_lag_samples):
    C, T = eeg.shape
    T_eff = T - max_lag_samples
    X = torch.zeros((T_eff, C * max_lag_samples), dtype=eeg.dtype, device=eeg.device)
    for tau in range(max_lag_samples):
        X[:, tau*C : (tau+1)*C] = eeg[:, tau : tau+T_eff].T
    return X

def prepare_subject_windows(cache_file, device):
    cached = torch.load(cache_file, map_location='cpu', weights_only=False)['raw']
    windows = []
    
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
        
        env_l_f = (env_l_f - np.mean(env_l_f, axis=1, keepdims=True)) / (np.std(env_l_f, axis=1, keepdims=True) + 1e-8)
        env_r_f = (env_r_f - np.mean(env_r_f, axis=1, keepdims=True)) / (np.std(env_r_f, axis=1, keepdims=True) + 1e-8)
        
        eeg = torch.tensor(eeg_f.copy(), dtype=torch.float32, device=device)
        env_l = torch.tensor(env_l_f[0].copy(), dtype=torch.float32, device=device)
        env_r = torch.tensor(env_r_f[0].copy(), dtype=torch.float32, device=device)
        
        T = eeg.shape[1]
        X_trial = create_toeplitz_features_pt(eeg, MAX_LAG_SAMPLES)
        T_eff = X_trial.shape[0]
        
        Y_l_eff = env_l[:T_eff]
        Y_r_eff = env_r[:T_eff]
        
        sp = tr['meta']['switch_points']
        boundaries = [0] + [idx for spk, idx in sp]
        boundaries = sorted(set(boundaries))
        if boundaries[-1] != T: boundaries.append(T)
            
        for i in range(len(boundaries) - 1):
            start_idx = boundaries[i]
            end_idx = boundaries[i+1]
            
            current_spk = 'L'
            for spk, idx in sp:
                if idx <= start_idx: current_spk = spk
                else: break
                
            safe_start = start_idx + int(1.5 * SR)
            safe_end = end_idx
            
            if safe_end - safe_start >= SEQ_SAMPLES:
                for seq_start in range(safe_start, safe_end - SEQ_SAMPLES + 1, SEQ_HOP):
                    if seq_start + SEQ_SAMPLES <= T_eff:
                        X_win = X_trial[seq_start:seq_start + SEQ_SAMPLES]
                        Y_L_win = Y_l_eff[seq_start:seq_start + SEQ_SAMPLES]
                        Y_R_win = Y_r_eff[seq_start:seq_start + SEQ_SAMPLES]
                        label = 1 if current_spk == 'L' else 0
                        
                        windows.append({
                            'X': X_win,
                            'Y_L': Y_L_win,
                            'Y_R': Y_R_win,
                            'label': label
                        })
    return windows

def process_subject(cache_file):
    torch.set_num_threads(1)
    device = torch.device('cpu')
    subj_name = cache_file.stem.split('_')[0]
    
    windows = prepare_subject_windows(cache_file, device)
    if len(windows) == 0:
        return subj_name, None
        
    F = windows[0]['X'].shape[1]
    I = torch.eye(F, device=device)
    
    # 1. Compute Oracle Trajectory W_t and Unlabeled Cross-Modal Stats
    W_oracles = []
    cross_modal_features = []
    
    for w in windows:
        X = w['X']
        Y_true = w['Y_L'] if w['label'] == 1 else w['Y_R']
        
        # Oracle W_t
        C_xx = X.T @ X
        C_xy = X.T @ Y_true
        W_t = torch.linalg.solve(C_xx + RIDGE_LAMBDA * I, C_xy)
        W_oracles.append(W_t.cpu().numpy())
        
        # Unlabeled Cross-Modal Stats (C_xL, C_xR)
        C_xL = X.T @ w['Y_L']
        C_xR = X.T @ w['Y_R']
        
        # Combine them (816 dimensions)
        feats = torch.cat([C_xL, C_xR]).cpu().numpy()
        cross_modal_features.append(feats)
        
    W_oracles = np.stack(W_oracles)
    cross_modal_features = np.stack(cross_modal_features)
    
    # 2. PCA on Oracle Trajectory to extract latent state a_t
    # In Phase 160, 18 components explained ~85% of variance
    pca = PCA(n_components=18)
    a_t = pca.fit_transform(W_oracles)
    var_explained = np.sum(pca.explained_variance_ratio_)
    
    # 3. Predict a_t from [C_xL, C_xR] using RidgeCV (K-Fold CV)
    # We evaluate if the unlabeled features can predict the oracle state
    kf = KFold(n_splits=5, shuffle=False) # Temporal blocks are better, but standard KFold for simplicity
    
    y_preds = np.zeros_like(a_t)
    
    for train_idx, test_idx in kf.split(cross_modal_features):
        X_train, X_test = cross_modal_features[train_idx], cross_modal_features[test_idx]
        y_train, y_test = a_t[train_idx], a_t[test_idx]
        
        # Multi-output Ridge CV
        reg = RidgeCV(alphas=np.logspace(-2, 4, 10))
        reg.fit(X_train, y_train)
        
        y_preds[test_idx] = reg.predict(X_test)
        
    # 4. Compute R^2 score for each latent dimension
    r2_scores = []
    for d in range(a_t.shape[1]):
        r2 = r2_score(a_t[:, d], y_preds[:, d])
        r2_scores.append(r2)
        
    # We care about the Mean R^2 across the 18 latent dimensions
    mean_r2 = np.mean(r2_scores)
    max_r2 = np.max(r2_scores)
    
    print(f"[{subj_name}] PCA Var: {var_explained*100:.1f}% | Mean R2: {mean_r2:.3f} | Max R2: {max_r2:.3f}")
    
    return subj_name, {
        'mean_r2': mean_r2,
        'max_r2': max_r2,
        'var_explained': var_explained
    }

def main():
    print("=======================================================")
    print(" PHASE 168: CROSS-MODAL OBSERVABILITY ANALYSIS")
    print(" Predicting Oracle Trajectory from Unlabeled [CxL, CxR]")
    print("=======================================================\n")
    
    cache_dir = Path('/kaggle/input/datasets/lowkieee/multiband-cache/kaggle/working/multiband_cache')
    possible_paths = [
        Path('/kaggle/input/datasets/lokeshgile/aasd-universal-cache-v1'),
        cache_dir,
        Path('/kaggle/working/multiband_cache')
    ]
    for p in possible_paths:
        if p.exists() and len(list(p.glob('*_multiband.pt'))) > 0:
            cache_dir = p
            break
            
    cache_files = sorted(list(cache_dir.glob('*_multiband.pt')))
    
    start_time = time.time()
    results = {}
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=mp.cpu_count()) as executor:
        futures = {executor.submit(process_subject, cf): cf for cf in cache_files}
        for future in concurrent.futures.as_completed(futures):
            subj, metrics = future.result()
            if metrics is not None:
                results[subj] = metrics
                
    print(f"\nExtraction & Regression Time: {time.time() - start_time:.2f}s\n")
    
    global_mean_r2 = []
    global_max_r2 = []
    
    subjects_sorted = sorted(results.keys())
    for subj in subjects_sorted:
        metrics = results[subj]
        
        global_mean_r2.append(metrics['mean_r2'])
        global_max_r2.append(metrics['max_r2'])
        
        print(f"--- Subject: {subj} ---")
        print(f"  Mean R^2 (18 dims) : {metrics['mean_r2']:.3f}")
        print(f"  Max R^2 (Best dim) : {metrics['max_r2']:.3f}")
        
    print("=======================================================")
    print(" GLOBAL OBSERVABILITY AVERAGES")
    print("=======================================================")
    print(f"Global Mean R^2 : {np.mean(global_mean_r2):.3f}")
    print(f"Global Max R^2  : {np.mean(global_max_r2):.3f}")
    print("=======================================================")
    
if __name__ == '__main__':
    main()
