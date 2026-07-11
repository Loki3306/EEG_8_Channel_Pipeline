import os
import time
import numpy as np
import torch
from pathlib import Path
from sklearn.metrics import roc_auc_score, accuracy_score
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
FORGETTING_FACTOR = 0.98  # roughly 50 windows memory (25 seconds)

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
            
        # Parse window labels carefully
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
        
    # Split: 2 minutes of data for Calibration (~240 windows)
    CALIB_WINDOWS = 240
    calib_set = windows[:CALIB_WINDOWS]
    track_set = windows[CALIB_WINDOWS:]
    
    F = windows[0]['X'].shape[1]
    
    # ---------------------------------------------------------
    # 1. CALIBRATION (Anchor the spatial geometry)
    # ---------------------------------------------------------
    Rxx_calib = torch.zeros((F, F), device=device)
    Rxy_calib = torch.zeros((F,), device=device)
    
    for w in calib_set:
        X = w['X']
        Y_true = w['Y_L'] if w['label'] == 1 else w['Y_R']
        Rxx_calib += X.T @ X
        Rxy_calib += X.T @ Y_true
        
    I = torch.eye(F, device=device)
    W_0 = torch.linalg.solve(Rxx_calib + RIDGE_LAMBDA * I, Rxy_calib)
    
    # ---------------------------------------------------------
    # 2. STATIC TRACKING (Baseline)
    # ---------------------------------------------------------
    static_scores = []
    labels = []
    
    for w in track_set:
        Y_hat = w['X'] @ W_0
        c_L = batch_pearsonr_pt(Y_hat, w['Y_L'])
        c_R = batch_pearsonr_pt(Y_hat, w['Y_R'])
        static_scores.append((c_L - c_R).item())
        labels.append(w['label'])
        
    static_auc = roc_auc_score(labels, static_scores)
    
    # ---------------------------------------------------------
    # 3. UNSUPERVISED DECISION-DIRECTED ADAPTIVE TRACKING
    # ---------------------------------------------------------
    Rxx = Rxx_calib.clone()
    Rxy = Rxy_calib.clone()
    W_t = W_0.clone()
    
    adaptive_scores = []
    
    for w in track_set:
        # Step A: Predict using current weights
        Y_hat = w['X'] @ W_t
        c_L = batch_pearsonr_pt(Y_hat, w['Y_L']).item()
        c_R = batch_pearsonr_pt(Y_hat, w['Y_R']).item()
        adaptive_scores.append(c_L - c_R)
        
        # Step B: Unsupervised Decision
        pred_label = 1 if c_L > c_R else 0
        
        # Step C: Adapt (RLS Ridge Update with Forgetting Factor)
        Y_winner = w['Y_L'] if pred_label == 1 else w['Y_R']
        Rxx = FORGETTING_FACTOR * Rxx + w['X'].T @ w['X']
        Rxy = FORGETTING_FACTOR * Rxy + w['X'].T @ Y_winner
        
        W_t = torch.linalg.solve(Rxx + RIDGE_LAMBDA * I, Rxy)
        
    adaptive_auc = roc_auc_score(labels, adaptive_scores)
    
    # ---------------------------------------------------------
    # 4. ORACLE ADAPTIVE TRACKING (Upper Bound)
    # ---------------------------------------------------------
    Rxx_o = Rxx_calib.clone()
    Rxy_o = Rxy_calib.clone()
    W_to = W_0.clone()
    
    oracle_scores = []
    
    for w in track_set:
        Y_hat = w['X'] @ W_to
        c_L = batch_pearsonr_pt(Y_hat, w['Y_L']).item()
        c_R = batch_pearsonr_pt(Y_hat, w['Y_R']).item()
        oracle_scores.append(c_L - c_R)
        
        # Oracle Update
        Y_true = w['Y_L'] if w['label'] == 1 else w['Y_R']
        Rxx_o = FORGETTING_FACTOR * Rxx_o + w['X'].T @ w['X']
        Rxy_o = FORGETTING_FACTOR * Rxy_o + w['X'].T @ Y_true
        
        W_to = torch.linalg.solve(Rxx_o + RIDGE_LAMBDA * I, Rxy_o)
        
    oracle_auc = roc_auc_score(labels, oracle_scores)
    
    return subj_name, static_auc, adaptive_auc, oracle_auc

def main():
    mp.set_start_method('spawn', force=True)
    
    print("=======================================================")
    print(" PHASE 150: DECISION-DIRECTED ADAPTIVE DECODER")
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
    
    print(f"Running Adaptive Tracking on {len(cache_files)} subjects...")
    print(f"Forgetting Factor: {FORGETTING_FACTOR} (~25s memory)")
    print(f"Calibration: First 2 minutes. Tracking: Next 58 minutes.\n")
    
    start_time = time.time()
    results = []
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(process_subject, cf, idx % max(1, num_gpus)): cf for idx, cf in enumerate(cache_files)}
        for future in concurrent.futures.as_completed(futures):
            subj, s_auc, a_auc, o_auc = future.result()
            results.append((subj, s_auc, a_auc, o_auc))
            print(f"[{subj:3s}] Static: {s_auc:.3f} | Unsup Adapt: {a_auc:.3f} | Oracle Adapt: {o_auc:.3f}")

    print("\n=======================================================")
    print(" FINAL RESULTS (TRACKING PHASE ONLY)")
    print("=======================================================")
    print(f"{'Subj':<5} | {'Static':<10} | {'Adaptive':<10} | {'Oracle':<10}")
    print("-" * 45)
    
    results = sorted(results, key=lambda x: int(x[0][1:]))
    for subj, s, a, o in results:
        print(f"{subj:<5} | {s:<10.3f} | {a:<10.3f} | {o:<10.3f}")
        
    mean_s = np.mean([x[1] for x in results])
    mean_a = np.mean([x[2] for x in results])
    mean_o = np.mean([x[3] for x in results])
    
    print("-" * 45)
    print(f"{'MEAN':<5} | {mean_s:<10.3f} | {mean_a:<10.3f} | {mean_o:<10.3f}")
    
    if mean_a > mean_s + 0.05:
        print("\n[MASSIVE SUCCESS] The Unsupervised Adaptive Decoder successfully tracked the rotating geometry!")
        print("We have fundamentally solved the Spatial Drift problem for Ear-EEG!")
    else:
        print("\n[FAILURE] The Adaptive Decoder collapsed (Error Accumulation / Positive Feedback Loop).")

if __name__ == '__main__':
    main()
