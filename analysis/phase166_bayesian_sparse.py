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

# Base parameters for initial calibration
CALIB_MINUTES = 2.0
CALIB_WINDOWS = int((CALIB_MINUTES * 60) / 0.5)

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

def evaluate_window(w, W):
    X_win = w['X']
    pred = X_win @ W
    
    # Calculate correlation to true envelopes
    pred_centered = pred - torch.mean(pred)
    Y_L_c = w['Y_L'] - torch.mean(w['Y_L'])
    Y_R_c = w['Y_R'] - torch.mean(w['Y_R'])
    
    corr_L = torch.sum(pred_centered * Y_L_c) / (torch.norm(pred_centered) * torch.norm(Y_L_c) + 1e-8)
    corr_R = torch.sum(pred_centered * Y_R_c) / (torch.norm(pred_centered) * torch.norm(Y_R_c) + 1e-8)
    
    pred_label = 1 if corr_L > corr_R else 0
    return 1 if pred_label == w['label'] else 0

def run_bayesian_tracking(track_set, C_xx_calib, C_xy_calib, I, anchor_interval_sec, device, Q_DRIFT, R_NOISE, P_INIT):
    """
    Runs tracking with Bayesian State Estimation (Kalman Filter for C_xy)
    anchor_interval_sec = 0 means full Oracle (update every window).
    anchor_interval_sec = None means Fixed (never update C_xy).
    """
    C_xx = C_xx_calib.clone()
    C_xy = C_xy_calib.clone()
    
    F = C_xx.shape[0]
    
    # Unsupervised EMA decay for C_xx
    ALPHA_XX = 0.02
    
    # Kalman Filter Parameters for C_xy
    P_UNCERTAINTY = P_INIT
    
    W = torch.linalg.solve(C_xx + RIDGE_LAMBDA * I, C_xy)
    
    correct = 0
    total = 0
    
    windows_per_sec = 2 # 0.5s hop
    interval_windows = int(anchor_interval_sec * windows_per_sec) if anchor_interval_sec else None
    
    for i, w in enumerate(track_set):
        # 1. Predict with current frozen W
        correct += evaluate_window(w, W)
        total += 1
        
        X = w['X']
        
        # 2. CONTINUOUS UNSUPERVISED UPDATE (C_xx)
        C_xx = (1.0 - ALPHA_XX) * C_xx + ALPHA_XX * (X.T @ X)
        
        # 3. KALMAN PREDICTION STEP (C_xy uncertainty grows)
        P_UNCERTAINTY += Q_DRIFT
        
        # 4. KALMAN OBSERVATION STEP (if anchor arrives)
        if interval_windows is not None:
            if interval_windows == 0 or (i % interval_windows == 0):
                Y_true = w['Y_L'] if w['label'] == 1 else w['Y_R']
                Z_xy = X.T @ Y_true
                
                # Compute Kalman Gain
                K = P_UNCERTAINTY / (P_UNCERTAINTY + R_NOISE)
                
                # Update C_xy and P
                C_xy = C_xy + K * (Z_xy - C_xy)
                P_UNCERTAINTY = (1.0 - K) * P_UNCERTAINTY
                
        # Re-solve for W (only necessary if C_xx or C_xy changed)
        W = torch.linalg.solve(C_xx + RIDGE_LAMBDA * I, C_xy)
                
    return correct / total

def process_subject(cache_file):
    torch.set_num_threads(1)
    device = torch.device('cpu')
    subj_name = cache_file.stem.split('_')[0]
    
    windows = prepare_subject_windows(cache_file, device)
    if len(windows) < CALIB_WINDOWS:
        return subj_name, None
        
    calib_set = windows[:CALIB_WINDOWS]
    track_set = windows[CALIB_WINDOWS:]
    
    F = windows[0]['X'].shape[1]
    I = torch.eye(F, device=device)
    
    # ---------------------------------------------------------
    # 1. INITIAL CALIBRATION (First 2 mins)
    # ---------------------------------------------------------
    Rxx_calib = torch.zeros((F, F), device=device)
    Rxy_calib = torch.zeros((F,), device=device)
    
    Z_list = []
    
    count = 0
    for w in calib_set:
        X = w['X']
        Y_true = w['Y_L'] if w['label'] == 1 else w['Y_R']
        
        Z_t = X.T @ Y_true
        Z_list.append(Z_t)
        
        Rxx_calib += X.T @ X
        Rxy_calib += Z_t
        count += 1
        
    if count > 0:
        # True mean covariance to keep scale exactly at 1.0
        C_xx_calib = Rxx_calib / count
        C_xy_calib = Rxy_calib / count
        
        # Dynamically estimate observation noise R
        Z_tensor = torch.stack(Z_list)
        var_per_feature = torch.var(Z_tensor, dim=0)
        R_NOISE = torch.mean(var_per_feature).item()
    else:
        R_NOISE = 1.0
        
    # To match Oracle EMA (K=0.02), Q/R = K^2 / (1-K) = 4.08e-4
    Q_DRIFT = R_NOISE * 4.08e-4
    # Initialize P to steady state so the first anchor doesn't get ignored
    P_INIT = 0.02 * R_NOISE
        
    # ---------------------------------------------------------
    # 2. TRACKING SCENARIOS
    # ---------------------------------------------------------
    
    # Scenario A: Fixed Baseline (No updates to C_xy, but C_xx tracks!)
    acc_fixed = run_bayesian_tracking(track_set, C_xx_calib, C_xy_calib, I, anchor_interval_sec=None, device=device, Q_DRIFT=Q_DRIFT, R_NOISE=R_NOISE, P_INIT=P_INIT)
    
    # Scenario B: Sparse Anchor (1 min)
    acc_sparse_1m = run_bayesian_tracking(track_set, C_xx_calib, C_xy_calib, I, anchor_interval_sec=60, device=device, Q_DRIFT=Q_DRIFT, R_NOISE=R_NOISE, P_INIT=P_INIT)
    
    # Scenario C: Sparse Anchor (5 min)
    acc_sparse_5m = run_bayesian_tracking(track_set, C_xx_calib, C_xy_calib, I, anchor_interval_sec=300, device=device, Q_DRIFT=Q_DRIFT, R_NOISE=R_NOISE, P_INIT=P_INIT)
    
    # Scenario D: Oracle Upper Bound (Every window - 0.5s)
    acc_oracle = run_bayesian_tracking(track_set, C_xx_calib, C_xy_calib, I, anchor_interval_sec=0, device=device, Q_DRIFT=Q_DRIFT, R_NOISE=R_NOISE, P_INIT=P_INIT)
    
    print(f"[{subj_name}] Finished! (Oracle: {acc_oracle*100:.2f}%)")
    
    return subj_name, {
        'fixed': acc_fixed,
        'sparse_1m': acc_sparse_1m,
        'sparse_5m': acc_sparse_5m,
        'oracle': acc_oracle
    }

def main():
    print("=======================================================")
    print(" PHASE 166: BAYESIAN SPARSE ANCHORING (KALMAN FILTER)")
    print(" Decoupling unsupervised C_xx from sparsely supervised C_xy")
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
                
    print(f"Extraction & Tracking Time: {time.time() - start_time:.2f}s\n")
    
    global_fixed = []
    global_1m = []
    global_5m = []
    global_oracle = []
    
    subjects_sorted = sorted(results.keys())
    for subj in subjects_sorted:
        fixed = results[subj]['fixed'] * 100
        sp_1m = results[subj]['sparse_1m'] * 100
        sp_5m = results[subj]['sparse_5m'] * 100
        oracle = results[subj]['oracle'] * 100
        
        global_fixed.append(fixed)
        global_1m.append(sp_1m)
        global_5m.append(sp_5m)
        global_oracle.append(oracle)
        
        print(f"--- Subject: {subj} ---")
        print(f"  Unsupervised C_xx : {fixed:.2f}%")
        print(f"  Sparse KF (5 min) : {sp_5m:.2f}%")
        print(f"  Sparse KF (1 min) : {sp_1m:.2f}%")
        print(f"  Oracle KF (0.5s)  : {oracle:.2f}%\n")
        
    print("=======================================================")
    print(" GLOBAL OBSERVABILITY AVERAGES")
    print("=======================================================")
    print(f"Mean Unsupervised C_xx : {np.mean(global_fixed):.2f}%")
    print(f"Mean Sparse KF (5 min) : {np.mean(global_5m):.2f}%")
    print(f"Mean Sparse KF (1 min) : {np.mean(global_1m):.2f}%")
    print(f"Mean Oracle KF (Upper) : {np.mean(global_oracle):.2f}%")
    print("=======================================================")
    
if __name__ == '__main__':
    main()
