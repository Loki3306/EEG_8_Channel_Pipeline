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
RIDGE_LAMBDA = 2.0

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

def run_bayesian_tracking(track_set, C_xx_calib, C_xy_calib, I, N_windows, gamma, device, Q_DRIFT, R_NOISE, P_INIT, is_oracle=False):
    """
    Runs tracking with Quality-Controlled Pseudo-Observations
    N_windows: Number of windows to aggregate before updating C_xy
    gamma: Temperature for sigmoid probability
    is_oracle: If True, uses ground truth instead of prediction for p_t
    """
    C_xx = C_xx_calib.clone()
    C_xy = C_xy_calib.clone()
    
    F = C_xx.shape[0]
    ALPHA_XX = 0.02
    
    P_UNCERTAINTY = P_INIT
    W = torch.linalg.solve(C_xx + RIDGE_LAMBDA * I, C_xy)
    
    correct = 0
    total = 0
    
    Z_buffer = []
    
    for i, w in enumerate(track_set):
        # 1. Evaluate current window
        X = w['X']
        pred = X @ W
        
        pred_centered = pred - torch.mean(pred)
        Y_L_c = w['Y_L'] - torch.mean(w['Y_L'])
        Y_R_c = w['Y_R'] - torch.mean(w['Y_R'])
        
        corr_L = torch.sum(pred_centered * Y_L_c) / (torch.norm(pred_centered) * torch.norm(Y_L_c) + 1e-8)
        corr_R = torch.sum(pred_centered * Y_R_c) / (torch.norm(pred_centered) * torch.norm(Y_R_c) + 1e-8)
        
        pred_label = 1 if corr_L > corr_R else 0
        if pred_label == w['label']: correct += 1
        total += 1
        
        # 2. CONTINUOUS UNSUPERVISED UPDATE (C_xx)
        C_xx = (1.0 - ALPHA_XX) * C_xx + ALPHA_XX * (X.T @ X)
        
        # 3. COMPUTE EXPECTED OBSERVATION E[Z_t]
        if is_oracle:
            p_t = 1.0 if w['label'] == 1 else 0.0
        else:
            e_t = (corr_L - corr_R).item()
            # Clip to prevent overflow
            e_t = max(min(e_t, 1.0), -1.0)
            p_t = 1.0 / (1.0 + np.exp(-gamma * e_t))
            
        E_Z_t = p_t * (X.T @ w['Y_L']) + (1.0 - p_t) * (X.T @ w['Y_R'])
        Z_buffer.append(E_Z_t)
        
        # 4. TEMPORAL AGGREGATION & KALMAN UPDATE
        if N_windows is not None:
            if len(Z_buffer) >= N_windows:
                # Aggregate soft observations
                Z_agg = torch.stack(Z_buffer).mean(dim=0)
                Z_buffer.clear()
                
                # Kalman Prediction Step (accumulate N windows of drift)
                P_UNCERTAINTY += N_windows * Q_DRIFT
                
                # Kalman Observation Step (variance reduced by N)
                R_agg = R_NOISE / N_windows
                
                # Compute Kalman Gain
                K = P_UNCERTAINTY / (P_UNCERTAINTY + R_agg)
                
                # Update C_xy and P
                C_xy = C_xy + K * (Z_agg - C_xy)
                P_UNCERTAINTY = (1.0 - K) * P_UNCERTAINTY
                
        # Re-solve for W (done continuously because C_xx updates every window)
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
        C_xx_calib = Rxx_calib / count
        C_xy_calib = Rxy_calib / count
        
        Z_tensor = torch.stack(Z_list)
        var_per_feature = torch.var(Z_tensor, dim=0)
        R_NOISE = torch.mean(var_per_feature).item()
    else:
        R_NOISE = 1.0
        
    # To match Oracle EMA (K=0.02) exactly, Q/R = K^2 / (1-K) = 4.08e-4
    Q_DRIFT = R_NOISE * 4.08e-4
    P_INIT = 0.02 * R_NOISE
        
    # ---------------------------------------------------------
    # 2. TRACKING SCENARIOS
    # ---------------------------------------------------------
    
    # Scenario A: Fixed Baseline
    acc_fixed = run_bayesian_tracking(track_set, C_xx_calib, C_xy_calib, I, N_windows=None, gamma=1.0, device=device, Q_DRIFT=Q_DRIFT, R_NOISE=R_NOISE, P_INIT=P_INIT)
    
    # Scenario B: Hard Pseudo-Labels (gamma=1000, N=20 -> 10s)
    acc_hard = run_bayesian_tracking(track_set, C_xx_calib, C_xy_calib, I, N_windows=20, gamma=1000.0, device=device, Q_DRIFT=Q_DRIFT, R_NOISE=R_NOISE, P_INIT=P_INIT)
    
    # Scenario C: Soft Pseudo-Labels (gamma=40, N=20 -> 10s)
    acc_soft_20 = run_bayesian_tracking(track_set, C_xx_calib, C_xy_calib, I, N_windows=20, gamma=40.0, device=device, Q_DRIFT=Q_DRIFT, R_NOISE=R_NOISE, P_INIT=P_INIT)
    
    # Scenario D: Soft Pseudo-Labels (gamma=40, N=10 -> 5s)
    acc_soft_10 = run_bayesian_tracking(track_set, C_xx_calib, C_xy_calib, I, N_windows=10, gamma=40.0, device=device, Q_DRIFT=Q_DRIFT, R_NOISE=R_NOISE, P_INIT=P_INIT)
    
    # Scenario E: Oracle (N=1)
    acc_oracle = run_bayesian_tracking(track_set, C_xx_calib, C_xy_calib, I, N_windows=1, gamma=1.0, device=device, Q_DRIFT=Q_DRIFT, R_NOISE=R_NOISE, P_INIT=P_INIT, is_oracle=True)
    
    print(f"[{subj_name}] Finished! (Oracle: {acc_oracle*100:.2f}%)")
    
    return subj_name, {
        'fixed': acc_fixed,
        'hard_20': acc_hard,
        'soft_20': acc_soft_20,
        'soft_10': acc_soft_10,
        'oracle': acc_oracle
    }

def main():
    print("=======================================================")
    print(" PHASE 167: QUALITY-CONTROLLED PSEUDO-OBSERVATIONS")
    print(" Soft Labels + Temporal Aggregation via Kalman Filter")
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
    global_hard = []
    global_s20 = []
    global_s10 = []
    global_oracle = []
    
    subjects_sorted = sorted(results.keys())
    for subj in subjects_sorted:
        fixed = results[subj]['fixed'] * 100
        hard = results[subj]['hard_20'] * 100
        s20 = results[subj]['soft_20'] * 100
        s10 = results[subj]['soft_10'] * 100
        oracle = results[subj]['oracle'] * 100
        
        global_fixed.append(fixed)
        global_hard.append(hard)
        global_s20.append(s20)
        global_s10.append(s10)
        global_oracle.append(oracle)
        
        print(f"--- Subject: {subj} ---")
        print(f"  Fixed Baseline    : {fixed:.2f}%")
        print(f"  Hard Labels (N=20): {hard:.2f}%")
        print(f"  Soft Labels (N=20): {s20:.2f}%")
        print(f"  Soft Labels (N=10): {s10:.2f}%")
        print(f"  Oracle KF (N=1)   : {oracle:.2f}%\n")
        
    print("=======================================================")
    print(" GLOBAL TRACKING AVERAGES")
    print("=======================================================")
    print(f"Mean Fixed Baseline  : {np.mean(global_fixed):.2f}%")
    print(f"Mean Hard Lbl (N=20) : {np.mean(global_hard):.2f}%")
    print(f"Mean Soft Lbl (N=20) : {np.mean(global_s20):.2f}%")
    print(f"Mean Soft Lbl (N=10) : {np.mean(global_s10):.2f}%")
    print(f"Mean Oracle KF (N=1) : {np.mean(global_oracle):.2f}%")
    print("=======================================================")
    
if __name__ == '__main__':
    main()
