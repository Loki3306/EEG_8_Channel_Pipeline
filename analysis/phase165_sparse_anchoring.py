import os
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

def run_tracking_simulation(track_set, Rxx_calib, Rxy_calib, I, anchor_interval_sec, device):
    """
    Runs tracking with intermittent anchors.
    anchor_interval_sec = 0 means full Oracle (update every window).
    anchor_interval_sec = None means Fixed (never update).
    Otherwise, we update the decoder every `anchor_interval_sec` seconds.
    """
    Rxx = Rxx_calib.clone()
    Rxy = Rxy_calib.clone()
    
    # When jumping time, we decay memory exactly by the base factor to the power of skipped windows
    BASE_LAMBDA = 0.98
    
    F = Rxx.shape[0]
    W = torch.linalg.solve(Rxx + RIDGE_LAMBDA * I, Rxy)
    
    correct = 0
    total = 0
    
    windows_per_sec = 2 # 0.5s hop
    interval_windows = int(anchor_interval_sec * windows_per_sec) if anchor_interval_sec else None
    
    for i, w in enumerate(track_set):
        # 1. Predict with current frozen W
        correct += evaluate_window(w, W)
        total += 1
        
        # 2. Update if it's an anchor window
        if interval_windows is not None:
            if interval_windows == 0 or (i % interval_windows == 0):
                X = w['X']
                Y_true = w['Y_L'] if w['label'] == 1 else w['Y_R']
                
                # Update tracking matrices
                forget_factor = BASE_LAMBDA if interval_windows == 0 else (BASE_LAMBDA ** interval_windows)
                
                Rxx = forget_factor * Rxx + X.T @ X
                Rxy = forget_factor * Rxy + X.T @ Y_true
                
                # Re-solve for W
                W = torch.linalg.solve(Rxx + RIDGE_LAMBDA * I, Rxy)
                
    return correct / total

def process_subject(cache_file):
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
    
    count = 0
    for w in calib_set:
        X = w['X']
        Y_true = w['Y_L'] if w['label'] == 1 else w['Y_R']
        Rxx_calib += X.T @ X
        Rxy_calib += X.T @ Y_true
        count += 1
        
    if count > 0:
        # Scale to avoid exploding magnitudes, simulating effective memory of ~50 windows
        Rxx_calib = (Rxx_calib / count) * 50.0
        Rxy_calib = (Rxy_calib / count) * 50.0
        
    # ---------------------------------------------------------
    # 2. TRACKING SCENARIOS
    # ---------------------------------------------------------
    
    # Scenario A: Fixed Baseline (No updates)
    acc_fixed = run_tracking_simulation(track_set, Rxx_calib, Rxy_calib, I, anchor_interval_sec=None, device=device)
    
    # Scenario B: Sparse Anchor (1 min)
    acc_sparse_1m = run_tracking_simulation(track_set, Rxx_calib, Rxy_calib, I, anchor_interval_sec=60, device=device)
    
    # Scenario C: Sparse Anchor (5 min)
    acc_sparse_5m = run_tracking_simulation(track_set, Rxx_calib, Rxy_calib, I, anchor_interval_sec=300, device=device)
    
    # Scenario D: Oracle Upper Bound (Every window - 0.5s)
    acc_oracle = run_tracking_simulation(track_set, Rxx_calib, Rxy_calib, I, anchor_interval_sec=0, device=device)
    
    return subj_name, {
        'fixed': acc_fixed,
        'sparse_1m': acc_sparse_1m,
        'sparse_5m': acc_sparse_5m,
        'oracle': acc_oracle
    }

def main():
    print("=======================================================")
    print(" PHASE 165: SEMI-SUPERVISED SPARSE ANCHORING")
    print(" Roadmap 1: Can intermittent ground truth (e.g. button press)")
    print(" recover the Concept Drift?")
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
        print(f"  Fixed (0.00 Hz)   : {fixed:.2f}%")
        print(f"  Sparse (1/5 min)  : {sp_5m:.2f}%")
        print(f"  Sparse (1/1 min)  : {sp_1m:.2f}%")
        print(f"  Oracle (2.00 Hz)  : {oracle:.2f}%\n")
        
    print("=======================================================")
    print(" GLOBAL OBSERVABILITY AVERAGES")
    print("=======================================================")
    print(f"Mean Fixed Baseline : {np.mean(global_fixed):.2f}%")
    print(f"Mean Sparse (5 min) : {np.mean(global_5m):.2f}%")
    print(f"Mean Sparse (1 min) : {np.mean(global_1m):.2f}%")
    print(f"Mean Oracle Upper   : {np.mean(global_oracle):.2f}%")
    print("=======================================================")
    
if __name__ == '__main__':
    main()
