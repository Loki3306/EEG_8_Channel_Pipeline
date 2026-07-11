import os
import time
import numpy as np
import torch
from pathlib import Path
from sklearn.metrics import roc_auc_score
from scipy import signal
import multiprocessing as mp
import concurrent.futures

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
FORGETTING_FACTOR_BASE = 0.98  # Base forgetting factor (memory ~ 50 windows)
ALPHA_BASE = 1.0 - FORGETTING_FACTOR_BASE

SOFTMAX_BETA = 15.0 # Temperature for probabilistic updates

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

def batch_pearsonr_pt(x, y):
    x_mean = x - x.mean(dim=0, keepdim=True)
    y_mean = y - y.mean(dim=0, keepdim=True)
    num = (x_mean * y_mean).sum(dim=0)
    den = torch.sqrt((x_mean**2).sum(dim=0) * (y_mean**2).sum(dim=0))
    return num / (den + 1e-8)

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
        
        eeg_f = (eeg_f - np.mean(eeg_f, axis=1, keepdims=True)) / (np.std(eeg_f, axis=1, keepdims=True) + 1e-8)
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

def process_subject(cache_file, device_id):
    device = torch.device(f'cuda:{device_id}' if torch.cuda.is_available() else 'cpu')
    subj_name = cache_file.stem.split('_')[0]
    
    windows = prepare_subject_windows(cache_file, device)
    if len(windows) < 200:
        return subj_name, 0.5, 0.5, 0.5
        
    CALIB_WINDOWS = 240
    calib_set = windows[:CALIB_WINDOWS]
    track_set = windows[CALIB_WINDOWS:]
    
    F = windows[0]['X'].shape[1]
    
    # ---------------------------------------------------------
    # 1. DUAL CALIBRATION
    # ---------------------------------------------------------
    Rxx_calib_L = torch.zeros((F, F), device=device)
    Rxy_calib_L = torch.zeros((F,), device=device)
    count_L = 0
    
    Rxx_calib_R = torch.zeros((F, F), device=device)
    Rxy_calib_R = torch.zeros((F,), device=device)
    count_R = 0
    
    for w in calib_set:
        X = w['X']
        if w['label'] == 1:
            Rxx_calib_L += X.T @ X
            Rxy_calib_L += X.T @ w['Y_L']
            count_L += 1
        else:
            Rxx_calib_R += X.T @ X
            Rxy_calib_R += X.T @ w['Y_R']
            count_R += 1
            
    # Normalize to equivalent 50-window scale (1 / (1 - 0.98))
    EFFECTIVE_WINDOWS = 1.0 / ALPHA_BASE
    
    if count_L > 0:
        Rxx_calib_L = (Rxx_calib_L / count_L) * EFFECTIVE_WINDOWS
        Rxy_calib_L = (Rxy_calib_L / count_L) * EFFECTIVE_WINDOWS
    if count_R > 0:
        Rxx_calib_R = (Rxx_calib_R / count_R) * EFFECTIVE_WINDOWS
        Rxy_calib_R = (Rxy_calib_R / count_R) * EFFECTIVE_WINDOWS
        
    I = torch.eye(F, device=device)
    W_L_0 = torch.linalg.solve(Rxx_calib_L + RIDGE_LAMBDA * I, Rxy_calib_L) if count_L > 0 else torch.zeros(F, device=device)
    W_R_0 = torch.linalg.solve(Rxx_calib_R + RIDGE_LAMBDA * I, Rxy_calib_R) if count_R > 0 else torch.zeros(F, device=device)
    
    if count_L == 0: W_L_0 = W_R_0.clone()
    if count_R == 0: W_R_0 = W_L_0.clone()
    
    labels = [w['label'] for w in track_set]
    
    # ---------------------------------------------------------
    # BRANCH 1: ORACLE DUAL-TRACKER (Upper Bound)
    # ---------------------------------------------------------
    Rxx_oL, Rxy_oL, W_oL = Rxx_calib_L.clone(), Rxy_calib_L.clone(), W_L_0.clone()
    Rxx_oR, Rxy_oR, W_oR = Rxx_calib_R.clone(), Rxy_calib_R.clone(), W_R_0.clone()
    scores_o = []
    
    for w in track_set:
        c_L = batch_pearsonr_pt(w['X'] @ W_oL, w['Y_L']).item()
        c_R = batch_pearsonr_pt(w['X'] @ W_oR, w['Y_R']).item()
        scores_o.append(c_L - c_R)
        
        # Oracle knows truth. It ONLY updates the attended decoder.
        # The unattended decoder PERFECTLY FREEZES.
        if w['label'] == 1:
            Rxx_oL = FORGETTING_FACTOR_BASE * Rxx_oL + w['X'].T @ w['X']
            Rxy_oL = FORGETTING_FACTOR_BASE * Rxy_oL + w['X'].T @ w['Y_L']
            W_oL = torch.linalg.solve(Rxx_oL + RIDGE_LAMBDA * I, Rxy_oL)
        else:
            Rxx_oR = FORGETTING_FACTOR_BASE * Rxx_oR + w['X'].T @ w['X']
            Rxy_oR = FORGETTING_FACTOR_BASE * Rxy_oR + w['X'].T @ w['Y_R']
            W_oR = torch.linalg.solve(Rxx_oR + RIDGE_LAMBDA * I, Rxy_oR)
            
    auc_o = roc_auc_score(labels, scores_o)
    
    # ---------------------------------------------------------
    # BRANCH 2: HARD DECISION DUAL-TRACKER (Collapse Baseline)
    # ---------------------------------------------------------
    Rxx_hL, Rxy_hL, W_hL = Rxx_calib_L.clone(), Rxy_calib_L.clone(), W_L_0.clone()
    Rxx_hR, Rxy_hR, W_hR = Rxx_calib_R.clone(), Rxy_calib_R.clone(), W_R_0.clone()
    scores_h = []
    
    for w in track_set:
        c_L = batch_pearsonr_pt(w['X'] @ W_hL, w['Y_L']).item()
        c_R = batch_pearsonr_pt(w['X'] @ W_hR, w['Y_R']).item()
        scores_h.append(c_L - c_R)
        
        pred_label = 1 if c_L > c_R else 0
        
        if pred_label == 1:
            Rxx_hL = FORGETTING_FACTOR_BASE * Rxx_hL + w['X'].T @ w['X']
            Rxy_hL = FORGETTING_FACTOR_BASE * Rxy_hL + w['X'].T @ w['Y_L']
            W_hL = torch.linalg.solve(Rxx_hL + RIDGE_LAMBDA * I, Rxy_hL)
        else:
            Rxx_hR = FORGETTING_FACTOR_BASE * Rxx_hR + w['X'].T @ w['X']
            Rxy_hR = FORGETTING_FACTOR_BASE * Rxy_hR + w['X'].T @ w['Y_R']
            W_hR = torch.linalg.solve(Rxx_hR + RIDGE_LAMBDA * I, Rxy_hR)
            
    auc_h = roc_auc_score(labels, scores_h)
    
    # ---------------------------------------------------------
    # BRANCH 3: BAYESIAN DUAL-TRACKER (The Cure)
    # ---------------------------------------------------------
    Rxx_bL, Rxy_bL, W_bL = Rxx_calib_L.clone(), Rxy_calib_L.clone(), W_L_0.clone()
    Rxx_bR, Rxy_bR, W_bR = Rxx_calib_R.clone(), Rxy_calib_R.clone(), W_R_0.clone()
    scores_b = []
    
    for w in track_set:
        c_L = batch_pearsonr_pt(w['X'] @ W_bL, w['Y_L']).item()
        c_R = batch_pearsonr_pt(w['X'] @ W_bR, w['Y_R']).item()
        scores_b.append(c_L - c_R)
        
        # 1. Posterior Probabilities
        max_c = max(c_L, c_R)
        exp_L = np.exp(SOFTMAX_BETA * (c_L - max_c))
        exp_R = np.exp(SOFTMAX_BETA * (c_R - max_c))
        p_L = exp_L / (exp_L + exp_R + 1e-8)
        p_R = 1.0 - p_L
        
        # 2. Dynamic Freezing (Variable Forgetting Factors)
        alpha_L = p_L * ALPHA_BASE
        lambda_L = 1.0 - alpha_L
        
        alpha_R = p_R * ALPHA_BASE
        lambda_R = 1.0 - alpha_R
        
        # 3. Probabilistic Updates
        XX = w['X'].T @ w['X']
        Rxx_bL = lambda_L * Rxx_bL + p_L * XX
        Rxy_bL = lambda_L * Rxy_bL + p_L * (w['X'].T @ w['Y_L'])
        W_bL = torch.linalg.solve(Rxx_bL + RIDGE_LAMBDA * I, Rxy_bL)
        
        Rxx_bR = lambda_R * Rxx_bR + p_R * XX
        Rxy_bR = lambda_R * Rxy_bR + p_R * (w['X'].T @ w['Y_R'])
        W_bR = torch.linalg.solve(Rxx_bR + RIDGE_LAMBDA * I, Rxy_bR)
        
    auc_b = roc_auc_score(labels, scores_b)
    
    return subj_name, auc_o, auc_h, auc_b

def main():
    mp.set_start_method('spawn', force=True)
    
    print("=======================================================")
    print(" PHASE 151b: BAYESIAN DUAL-TRACKER")
    print("=======================================================\n")
    
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
            
    cache_files = sorted(list(cache_dir.glob('*_multiband.pt')))
    num_gpus = torch.cuda.device_count()
    num_workers = min(mp.cpu_count(), num_gpus if num_gpus > 0 else mp.cpu_count())
    
    print(f"Softmax Temperature (Beta): {SOFTMAX_BETA}")
    print(f"Base Forgetting Factor: {FORGETTING_FACTOR_BASE}")
    print("Architecture: Independent Left and Right Decoders with dynamic probabilistic freezing.\n")
    
    start_time = time.time()
    results = []
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(process_subject, cf, idx % max(1, num_gpus)): cf for idx, cf in enumerate(cache_files)}
        for future in concurrent.futures.as_completed(futures):
            subj, auc_o, auc_h, auc_b = future.result()
            results.append((subj, auc_o, auc_h, auc_b))
            print(f"[{subj:3s}] Oracle Dual: {auc_o:.3f} | Hard Dual: {auc_h:.3f} | Bayes Dual: {auc_b:.3f}")

    print("\n=======================================================")
    print(" FINAL RESULTS")
    print("=======================================================")
    print(f"{'Subj':<5} | {'Oracle':<8} | {'Hard':<8} | {'Bayes':<8}")
    print("-" * 37)
    
    results = sorted(results, key=lambda x: int(x[0][1:]))
    for subj, auc_o, auc_h, auc_b in results:
        print(f"{subj:<5} | {auc_o:<8.3f} | {auc_h:<8.3f} | {auc_b:<8.3f}")
        
    print("-" * 37)
    mean_o = np.mean([x[1] for x in results])
    mean_h = np.mean([x[2] for x in results])
    mean_b = np.mean([x[3] for x in results])
    print(f"{'MEAN':<5} | {mean_o:<8.3f} | {mean_h:<8.3f} | {mean_b:<8.3f}")
    
    if mean_b > mean_h + 0.05:
        print("\n[MASSIVE SUCCESS] Bayesian Dual-Tracker breaks the Positive Feedback Loop!")
        print("We have successfully implemented a mathematically sound Unsupervised Adaptive Tracker.")
    else:
        print("\n[FAILURE] The dual-hypothesis tracker is still insufficient to break the collapse.")

if __name__ == '__main__':
    main()
