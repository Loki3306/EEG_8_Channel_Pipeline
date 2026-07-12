import os
import numpy as np
import torch
from pathlib import Path
from scipy import signal
import multiprocessing as mp
import concurrent.futures
import pandas as pd

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

WINDOWS_PER_MIN = 120
CALIB_MINUTES = [1, 2, 5, 10, 20]
CALIB_SIZES = [m * WINDOWS_PER_MIN for m in CALIB_MINUTES]
HOLDOUT_MINUTES = 20
HOLDOUT_WINDOWS = HOLDOUT_MINUTES * WINDOWS_PER_MIN

HALF_LIFE_TRAIN_MIN = 10
HALF_LIFE_TRAIN_WIN = HALF_LIFE_TRAIN_MIN * WINDOWS_PER_MIN
HALF_LIFE_BLOCK_MIN = 5
HALF_LIFE_BLOCK_WIN = HALF_LIFE_BLOCK_MIN * WINDOWS_PER_MIN

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

def get_pearsonr(x, y):
    x_m = x - np.mean(x)
    y_m = y - np.mean(y)
    num = np.dot(x_m, y_m)
    den = np.linalg.norm(x_m) * np.linalg.norm(y_m)
    return num / (den + 1e-8)

def prepare_subject_windows_continuous(cache_file, device):
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
        
        X_trial = create_toeplitz_features_pt(eeg, MAX_LAG_SAMPLES)
        T_eff = X_trial.shape[0]
        
        Y_l_eff = env_l[:T_eff]
        Y_r_eff = env_r[:T_eff]
        
        sp = tr['meta']['switch_points']
        current_spk = 'L'
        sp_idx = 0
        labels_eff = np.zeros(T_eff, dtype=int)
        for t in range(T_eff):
            if sp_idx < len(sp) and t >= sp[sp_idx][1]:
                current_spk = sp[sp_idx][0]
                sp_idx += 1
            labels_eff[t] = 1 if current_spk == 'L' else 0
            
        for seq_start in range(0, T_eff - SEQ_SAMPLES + 1, SEQ_HOP):
            seq_end = seq_start + SEQ_SAMPLES
            X_win = X_trial[seq_start:seq_end]
            Y_L_win = Y_l_eff[seq_start:seq_end]
            Y_R_win = Y_r_eff[seq_start:seq_end]
            
            win_labels = labels_eff[seq_start:seq_end]
            label = 1 if np.mean(win_labels) >= 0.5 else 0
            
            windows.append({
                'X': X_win,
                'Y_L': Y_L_win,
                'Y_R': Y_R_win,
                'label': label
            })
            
    return windows

def train_ridge_decoder(windows, device):
    if not windows:
        return None
    F = windows[0]['X'].shape[1]
    Rxx = torch.zeros((F, F), device=device)
    Rxy = torch.zeros((F,), device=device)
    for w in windows:
        X = w['X']
        Y_true = w['Y_L'] if w['label'] == 1 else w['Y_R']
        Rxx += X.T @ X
        Rxy += X.T @ Y_true
    I = torch.eye(F, device=device)
    W = torch.linalg.solve(Rxx + RIDGE_LAMBDA * I, Rxy).cpu().numpy()
    return W

def evaluate_decoder(W, windows):
    if not windows or W is None:
        return 0.5
    correct = 0
    for w in windows:
        X_cpu = w['X'].cpu().numpy()
        YL_cpu = w['Y_L'].cpu().numpy()
        YR_cpu = w['Y_R'].cpu().numpy()
        label = w['label']
        
        preds = X_cpu @ W
        c_L = get_pearsonr(preds, YL_cpu)
        c_R = get_pearsonr(preds, YR_cpu)
        pred_label = 1 if c_L > c_R else 0
        if pred_label == label:
            correct += 1
    return correct / len(windows)

def process_subject(cache_file, device_id):
    device = torch.device(f'cuda:{device_id}' if torch.cuda.is_available() else 'cpu')
    subj_name = cache_file.stem.split('_')[0]
    
    windows = prepare_subject_windows_continuous(cache_file, device)
    total_windows = len(windows)
    
    expA_results = []
    expB_results = []
    
    # Require at least 40 minutes of data for meaningful test
    if total_windows < 4800:
        return subj_name, expA_results, expB_results
        
    # --- EXPERIMENT A: CALIBRATION CURVE ---
    holdout_start = total_windows - HOLDOUT_WINDOWS
    holdout_set = windows[holdout_start:]
    
    for c_min, c_win in zip(CALIB_MINUTES, CALIB_SIZES):
        if c_win >= holdout_start:
            continue
        calib_set = windows[:c_win]
        W = train_ridge_decoder(calib_set, device)
        acc = evaluate_decoder(W, holdout_set)
        expA_results.append({'subject': subj_name, 'calib_min': c_min, 'accuracy': acc})
        
    # --- EXPERIMENT B: DECODER HALF-LIFE ---
    hl_calib_set = windows[:HALF_LIFE_TRAIN_WIN]
    W_hl = train_ridge_decoder(hl_calib_set, device)
    
    current_idx = HALF_LIFE_TRAIN_WIN
    block_idx = 0
    while current_idx + HALF_LIFE_BLOCK_WIN <= total_windows:
        test_block = windows[current_idx : current_idx + HALF_LIFE_BLOCK_WIN]
        acc = evaluate_decoder(W_hl, test_block)
        delta_t_min = HALF_LIFE_TRAIN_MIN + block_idx * HALF_LIFE_BLOCK_MIN
        expB_results.append({'subject': subj_name, 'delta_t_min': delta_t_min, 'accuracy': acc})
        
        current_idx += HALF_LIFE_BLOCK_WIN
        block_idx += 1
        
    return subj_name, expA_results, expB_results

def main():
    mp.set_start_method('spawn', force=True)
    print("=======================================================")
    print(" PHASE 155: DECODER LIFETIME & OBSERVABILITY AUDIT")
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
    num_workers = mp.cpu_count()
    
    all_expA = []
    all_expB = []
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(process_subject, cf, idx % max(1, num_gpus)): cf for idx, cf in enumerate(cache_files)}
        for future in concurrent.futures.as_completed(futures):
            subj, expA, expB = future.result()
            all_expA.extend(expA)
            all_expB.extend(expB)
            print(f"[{subj}] Processed {len(expA)} calib sweeps and {len(expB)} half-life blocks.")

    if all_expA and all_expB:
        dfA = pd.DataFrame(all_expA)
        dfB = pd.DataFrame(all_expB)
        
        dfA.to_csv("phase155_experimentA_calibration.csv", index=False)
        dfB.to_csv("phase155_experimentB_halflife.csv", index=False)
        
        print("\n=======================================================")
        print(" EXPERIMENT A: MEAN CALIBRATION CURVE (Holdout = Last 20 Min)")
        print("=======================================================")
        meanA = dfA.groupby('calib_min')['accuracy'].mean().reset_index()
        for _, row in meanA.iterrows():
            print(f"  Train: {int(row['calib_min']):2d} minutes  ->  Accuracy: {row['accuracy']*100:.1f}%")
            
        print("\n=======================================================")
        print(" EXPERIMENT B: MEAN DECODER HALF-LIFE (Train = First 10 Min)")
        print("=======================================================")
        meanB = dfB.groupby('delta_t_min')['accuracy'].mean().reset_index()
        for _, row in meanB.iterrows():
            print(f"  Evaluate @ Minute {int(row['delta_t_min']):3d}  ->  Accuracy: {row['accuracy']*100:.1f}%")
            
        print("\nPhase 155 Complete! Results saved to CSV.")

if __name__ == '__main__':
    main()
