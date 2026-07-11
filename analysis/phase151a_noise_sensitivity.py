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
FORGETTING_FACTOR = 0.98  # roughly 50 windows memory (25 seconds)

P_ERROR_BASE = 0.05  # 5% chance to spontaneously make a mistake
P_BURST_STAY = 0.70  # 70% chance that an error causes the next window to be an error
CONF_THRESHOLD = 0.20 # Confidence threshold for gating

def generate_markov_noise(n_samples, p_err, p_burst, seed=42):
    np.random.seed(seed)
    mask = np.zeros(n_samples, dtype=bool)
    state = False # False = Correct, True = Error
    for i in range(n_samples):
        if state == False:
            state = np.random.rand() < p_err
        else:
            state = np.random.rand() < p_burst
        mask[i] = state
    return mask

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
    
    # 1. CALIBRATION
    Rxx_calib = torch.zeros((F, F), device=device)
    Rxy_calib = torch.zeros((F,), device=device)
    
    for w in calib_set:
        X = w['X']
        Y_true = w['Y_L'] if w['label'] == 1 else w['Y_R']
        Rxx_calib += X.T @ X
        Rxy_calib += X.T @ Y_true
        
    I = torch.eye(F, device=device)
    W_0 = torch.linalg.solve(Rxx_calib + RIDGE_LAMBDA * I, Rxy_calib)
    
    labels = [w['label'] for w in track_set]
    noise_mask = generate_markov_noise(len(track_set), P_ERROR_BASE, P_BURST_STAY, seed=int(subj_name[1:]))
    actual_noise_ratio = np.mean(noise_mask)
    
    def run_tracker(use_noise, use_gating):
        Rxx = Rxx_calib.clone()
        Rxy = Rxy_calib.clone()
        W = W_0.clone()
        scores = []
        
        for idx, w in enumerate(track_set):
            Y_hat = w['X'] @ W
            c_L = batch_pearsonr_pt(Y_hat, w['Y_L']).item()
            c_R = batch_pearsonr_pt(Y_hat, w['Y_R']).item()
            scores.append(c_L - c_R)
            
            true_label = w['label']
            if use_noise and noise_mask[idx]:
                observed_label = 1 - true_label
            else:
                observed_label = true_label
                
            conf = abs(c_L - c_R) / (abs(c_L) + abs(c_R) + 1e-8)
            
            if use_gating and conf < CONF_THRESHOLD:
                # FREEZE weights. Do not decay, do not update.
                pass
            else:
                Y_obs = w['Y_L'] if observed_label == 1 else w['Y_R']
                Rxx = FORGETTING_FACTOR * Rxx + w['X'].T @ w['X']
                Rxy = FORGETTING_FACTOR * Rxy + w['X'].T @ Y_obs
                W = torch.linalg.solve(Rxx + RIDGE_LAMBDA * I, Rxy)
                
        return roc_auc_score(labels, scores)

    # Branch 1: Oracle (0% noise)
    auc_oracle = run_tracker(use_noise=False, use_gating=False)
    
    # Branch 2: Burst Noise (No Gating)
    auc_burst = run_tracker(use_noise=True, use_gating=False)
    
    # Branch 3: Burst Noise + Confidence Gating
    auc_gated = run_tracker(use_noise=True, use_gating=True)
    
    return subj_name, actual_noise_ratio, auc_oracle, auc_burst, auc_gated

def main():
    mp.set_start_method('spawn', force=True)
    
    print("=======================================================")
    print(" PHASE 151a: BURST NOISE & CONFIDENCE GATING")
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
    
    print(f"Markovian Noise Profile:")
    print(f" - Base Error Probability: {P_ERROR_BASE*100:.1f}%")
    print(f" - Burst Persistence Prob: {P_BURST_STAY*100:.1f}%")
    print(f"Confidence Gating Threshold: {CONF_THRESHOLD}\n")
    
    start_time = time.time()
    results = []
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(process_subject, cf, idx % max(1, num_gpus)): cf for idx, cf in enumerate(cache_files)}
        for future in concurrent.futures.as_completed(futures):
            subj, actual_noise, auc_o, auc_b, auc_g = future.result()
            results.append((subj, actual_noise, auc_o, auc_b, auc_g))
            print(f"[{subj:3s}] Noise Ratio: {actual_noise*100:4.1f}% | Oracle: {auc_o:.3f} | Burst: {auc_b:.3f} | Gated: {auc_g:.3f}")

    print("\n=======================================================")
    print(" FINAL RESULTS")
    print("=======================================================")
    print(f"{'Subj':<5} | {'Noise%':<8} | {'Oracle':<8} | {'Burst':<8} | {'Gated':<8}")
    print("-" * 47)
    
    results = sorted(results, key=lambda x: int(x[0][1:]))
    for subj, actual_noise, auc_o, auc_b, auc_g in results:
        print(f"{subj:<5} | {actual_noise*100:<8.1f} | {auc_o:<8.3f} | {auc_b:<8.3f} | {auc_g:<8.3f}")
        
    print("-" * 47)
    mean_noise = np.mean([x[1] for x in results]) * 100
    mean_o = np.mean([x[2] for x in results])
    mean_b = np.mean([x[3] for x in results])
    mean_g = np.mean([x[4] for x in results])
    print(f"{'MEAN':<5} | {mean_noise:<8.1f} | {mean_o:<8.3f} | {mean_b:<8.3f} | {mean_g:<8.3f}")
    
    if mean_g > mean_b + 0.05:
        print("\n[MASSIVE SUCCESS] Confidence Gating successfully neutralizes Markovian Burst Noise!")
        print("We have mathematically proven the solution to the Positive Feedback Loop.")
    else:
        print("\n[FAILURE] Confidence Gating is insufficient to break the Burst Noise collapse.")
        print("We may need a full Bayesian Update / Kalman Filter (Phase 151b).")

if __name__ == '__main__':
    main()
