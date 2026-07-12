import os
import time
import numpy as np
import torch
from pathlib import Path
from scipy import signal
import concurrent.futures
import multiprocessing as mp

from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

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
RIDGE_LAMBDA = 100.0
FORGETTING_FACTOR_BASE = 0.98  
ALPHA_BASE = 1.0 - FORGETTING_FACTOR_BASE
EFFECTIVE_WINDOWS = 1.0 / ALPHA_BASE
EFFECTIVE_LAMBDA = RIDGE_LAMBDA * EFFECTIVE_WINDOWS

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

def compute_covariance(X):
    # X shape: [Samples, Channels]
    X_centered = X - torch.mean(X, dim=0, keepdim=True)
    cov = (X_centered.T @ X_centered) / (X.shape[0] - 1)
    return cov

def get_upper_triangular(cov):
    # Returns flattened upper triangular part including diagonal
    idx = torch.triu_indices(cov.shape[0], cov.shape[1])
    return cov[idx[0], idx[1]]

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
        
        # Do not Z-score per trial to preserve impedance differences for Covariance
        eeg_f = apply_modulation_filter(eeg_raw, BROADBAND[0], BROADBAND[1], SR)
        env_l_f = apply_modulation_filter(env_l_raw, BROADBAND[0], BROADBAND[1], SR)
        env_r_f = apply_modulation_filter(env_r_raw, BROADBAND[0], BROADBAND[1], SR)
        
        env_l_f = (env_l_f - np.mean(env_l_f, axis=1, keepdims=True)) / (np.std(env_l_f, axis=1, keepdims=True) + 1e-8)
        env_r_f = (env_r_f - np.mean(env_r_f, axis=1, keepdims=True)) / (np.std(env_r_f, axis=1, keepdims=True) + 1e-8)
        
        eeg = torch.tensor(eeg_f, dtype=torch.float32, device=device)
        env_l = torch.tensor(env_l_f[0], dtype=torch.float32, device=device)
        env_r = torch.tensor(env_r_f[0], dtype=torch.float32, device=device)
        
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
    device = torch.device('cpu') # Use CPU for parallel multiprocessing
    subj_name = cache_file.stem.split('_')[0]
    
    windows = prepare_subject_windows(cache_file, device)
    if len(windows) < 200:
        return subj_name, None
        
    CALIB_WINDOWS = 240
    calib_set = windows[:CALIB_WINDOWS]
    track_set = windows[CALIB_WINDOWS:]
    
    F = windows[0]['X'].shape[1]
    I = torch.eye(F, device=device)
    
    # ---------------------------------------------------------
    # 1. INITIAL CALIBRATION
    # ---------------------------------------------------------
    Rxx_calib = torch.zeros((F, F), device=device)
    Rxy_calib = torch.zeros((F,), device=device)
    C_base = torch.zeros((8, 8), device=device)
    count = 0
    for w in calib_set:
        X = w['X']
        Y_true = w['Y_L'] if w['label'] == 1 else w['Y_R']
        Rxx_calib += X.T @ X
        Rxy_calib += X.T @ Y_true
        
        # Spatial covariance (from lag 0)
        X_spatial = X[:, :8]
        C_base += compute_covariance(X_spatial)
        
        count += 1
        
    if count > 0:
        Rxx_calib = (Rxx_calib / count) * EFFECTIVE_WINDOWS
        Rxy_calib = (Rxy_calib / count) * EFFECTIVE_WINDOWS
        C_base = C_base / count
        
    W_0 = torch.linalg.solve(Rxx_calib + EFFECTIVE_LAMBDA * I, Rxy_calib) if count > 0 else torch.zeros(F, device=device)
    
    # ---------------------------------------------------------
    # 2. EXTRACT TRACKING TRAJECTORIES (Oracle W_t and Unsupervised C_t)
    # ---------------------------------------------------------
    Rxx_t, Rxy_t = Rxx_calib.clone(), Rxy_calib.clone()
    C_t = C_base.clone()
    
    features_X = [] # Unsupervised covariances
    targets_W = []  # Oracle decoders
    
    for w in track_set:
        X = w['X']
        Y_true = w['Y_L'] if w['label'] == 1 else w['Y_R']
        
        # Oracle Update for W_t
        Rxx_t = FORGETTING_FACTOR_BASE * Rxx_t + X.T @ X
        Rxy_t = FORGETTING_FACTOR_BASE * Rxy_t + X.T @ Y_true
        W_t = torch.linalg.solve(Rxx_t + EFFECTIVE_LAMBDA * I, Rxy_t)
        
        # Unsupervised Update for C_t (Using EMA to match RLS memory)
        X_spatial = X[:, :8]
        current_C = compute_covariance(X_spatial)
        C_t = FORGETTING_FACTOR_BASE * C_t + ALPHA_BASE * current_C
        
        # Extract rich unsupervised features
        cov_flat = get_upper_triangular(C_t)
        
        # Compute correlation matrix
        d = torch.diag(C_t)
        std_dev = torch.sqrt(torch.clamp(d, min=1e-8))
        corr_matrix = C_t / torch.outer(std_dev, std_dev)
        corr_flat = get_upper_triangular(corr_matrix)
        
        # Compute eigenvalues
        eigvals = torch.linalg.eigvalsh(C_t)
        
        # Concatenate features: 36 (cov) + 36 (corr) + 8 (eig) = 80 features
        rich_features = torch.cat([cov_flat, corr_flat, eigvals])
        
        features_X.append(rich_features.numpy())
        targets_W.append(W_t.numpy())
        
    features_X = np.array(features_X)
    targets_W = np.array(targets_W)
    
    # ---------------------------------------------------------
    # 3. REGRESSION: Predict Oracle W_t from Unsupervised C_t
    # ---------------------------------------------------------
    # Use TimeSeriesSplit to prevent future leakage
    tscv = TimeSeriesSplit(n_splits=5)
    
    r2_scores = []
    cos_sims = []
    func_corrs = []
    
    # We need the original X matrices to compute functional correlation
    # We'll just extract them directly from track_set
    X_track_tensors = [w['X'].cpu().numpy() for w in track_set]
    
    for train_idx, test_idx in tscv.split(features_X):
        # 1. Split Data
        X_tr, X_te = features_X[train_idx], features_X[test_idx]
        y_tr, y_te = targets_W[train_idx], targets_W[test_idx]
        
        # 2. Strict Causal Scaling (Fit only on Train)
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr)
        X_te = scaler.transform(X_te)
        
        # Train MultiOutput Ridge Regressor
        reg = MultiOutputRegressor(Ridge(alpha=1.0))
        reg.fit(X_tr, y_tr)
        
        y_pred = reg.predict(X_te)
        
        # Evaluate R^2 (variance explained)
        r2 = r2_score(y_te, y_pred)
        r2_scores.append(r2)
        
        # Evaluate Cosine Similarity & Functional Correlation
        for i, idx in enumerate(test_idx):
            pred_w = y_pred[i]
            true_w = y_te[i]
            
            # Cosine Sim
            pred_norm = pred_w / (np.linalg.norm(pred_w) + 1e-8)
            true_norm = true_w / (np.linalg.norm(true_w) + 1e-8)
            cos_sims.append(np.dot(pred_norm, true_norm))
            
            # Functional Corr
            X_win = X_track_tensors[idx]
            out_pred = X_win @ pred_w
            out_true = X_win @ true_w
            
            out_pred_centered = out_pred - np.mean(out_pred)
            out_true_centered = out_true - np.mean(out_true)
            
            num = np.dot(out_pred_centered, out_true_centered)
            den = np.linalg.norm(out_pred_centered) * np.linalg.norm(out_true_centered)
            func_corrs.append(num / (den + 1e-8))
            
    mean_r2 = np.mean(r2_scores)
    mean_cos = np.mean(cos_sims)
    mean_func = np.mean(func_corrs)
    
    return subj_name, {'r2': mean_r2, 'cos_sim': mean_cos, 'func_corr': mean_func}

def main():
    print("=======================================================")
    print(" PHASE 164: ORACLE DRIFT OBSERVABILITY (CONCEPT DRIFT)")
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
    
    print("Extracting trajectories and running regression...")
    start_time = time.time()
    
    results = {}
    
    # Process sequentially or parallel. CPU parallel is safe here.
    with concurrent.futures.ProcessPoolExecutor(max_workers=mp.cpu_count()) as executor:
        futures = {executor.submit(process_subject, cf): cf for cf in cache_files}
        for future in concurrent.futures.as_completed(futures):
            subj, metrics = future.result()
            if metrics is not None:
                results[subj] = metrics
                
    print(f"Total Time: {time.time() - start_time:.2f}s\n")
    
    global_r2 = []
    global_cos = []
    global_func = []
    
    subjects_sorted = sorted(results.keys())
    for subj in subjects_sorted:
        r2 = results[subj]['r2']
        cos = results[subj]['cos_sim']
        func = results[subj]['func_corr']
        global_r2.append(r2)
        global_cos.append(cos)
        global_func.append(func)
        print(f"--- Subject: {subj} ---")
        print(f"  Predictive R^2  : {r2:.4f}")
        print(f"  Mean Cosine Sim : {cos:.4f}")
        print(f"  Func Correlation: {func:.4f}\n")
        
    print("=======================================================")
    print(" GLOBAL OBSERVABILITY AVERAGES")
    print("=======================================================")
    print(f"Mean R^2          : {np.mean(global_r2):.4f}")
    print(f"Mean Cosine Sim   : {np.mean(global_cos):.4f}")
    print(f"Mean Func Corr    : {np.mean(global_func):.4f}")
    print("=======================================================")
    
if __name__ == '__main__':
    main()
