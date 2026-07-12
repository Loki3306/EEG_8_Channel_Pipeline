import os
import numpy as np
import torch
from pathlib import Path
import multiprocessing as mp
import concurrent.futures
from scipy import signal

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

# EM Configuration
GAMMA = 0.98
EM_ITERATIONS = 10
BETA = 10.0

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
        return subj_name, 0.0, 0.0
        
    CALIB_WINDOWS = 240
    calib_set = windows[:CALIB_WINDOWS]
    track_set = windows[CALIB_WINDOWS:]
    
    F = windows[0]['X'].shape[1]
    
    # ---------------------------------------------------------
    # 1. INITIAL CALIBRATION (W_0)
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
        Rxx_calib = (Rxx_calib / count) * (1.0 / (1.0 - 0.98))
        Rxy_calib = (Rxy_calib / count) * (1.0 / (1.0 - 0.98))
        
    I = torch.eye(F, device=device)
    W_0 = torch.linalg.solve(Rxx_calib + (RIDGE_LAMBDA * (1.0 / (1.0 - 0.98))) * I, Rxy_calib) if count > 0 else torch.zeros(F, device=device)
    
    # ---------------------------------------------------------
    # 2. OFFLINE EM PRE-COMPUTATION
    # ---------------------------------------------------------
    N = len(track_set)
    if N == 0:
        return subj_name, 0.0, 0.0
        
    X_all = torch.stack([w['X'] for w in track_set])          # [N, 384, 408]
    Y_L_all = torch.stack([w['Y_L'] for w in track_set])      # [N, 384]
    Y_R_all = torch.stack([w['Y_R'] for w in track_set])      # [N, 384]
    labels = torch.tensor([w['label'] for w in track_set], device=device) # [N]
    
    # Rxx_tau and Rxy_tau for each window
    Rxx_tau = torch.bmm(X_all.transpose(1, 2), X_all)         # [N, 408, 408]
    Rxy_L_tau = torch.bmm(X_all.transpose(1, 2), Y_L_all.unsqueeze(2)).squeeze(2) # [N, 408]
    Rxy_R_tau = torch.bmm(X_all.transpose(1, 2), Y_R_all.unsqueeze(2)).squeeze(2) # [N, 408]
    
    # Symmetric Exponential Smoothing Kernel
    t_idx = torch.arange(N, device=device).float()
    dist_matrix = torch.abs(t_idx.unsqueeze(0) - t_idx.unsqueeze(1))
    K = (GAMMA ** dist_matrix)                                # [N, N]
    
    # Pre-smooth Rxx (since it doesn't depend on latent labels)
    Rxx_smoothed = torch.einsum('tn,nij->tij', K, Rxx_tau)    # [N, 408, 408]
    
    # Ridge Penalty
    I_F = torch.eye(F, device=device).unsqueeze(0).expand(N, F, F)
    EFFECTIVE_WINDOWS_SMOOTHED = K.sum(dim=1).view(N, 1, 1)   # [N, 1, 1]
    lambda_term = RIDGE_LAMBDA * EFFECTIVE_WINDOWS_SMOOTHED * I_F
    
    # Pre-invert the regularized Rxx for all t
    Rxx_inv = torch.linalg.inv(Rxx_smoothed + lambda_term)    # [N, 408, 408]
    del Rxx_smoothed, Rxx_tau, lambda_term, dist_matrix
    torch.cuda.empty_cache()
    
    # ---------------------------------------------------------
    # 3. EXPECTATION-MAXIMIZATION (EM) ITERATIONS
    # ---------------------------------------------------------
    # Initialize decoders to W_0
    W_t = W_0.unsqueeze(0).expand(N, F)                       # [N, 408]
    
    acc_history = []
    
    # Normalization helper for correlations
    YL_mean = Y_L_all.mean(dim=1, keepdim=True)
    YL_std = Y_L_all.std(dim=1, keepdim=True) + 1e-8
    YL_norm = (Y_L_all - YL_mean) / YL_std
    
    YR_mean = Y_R_all.mean(dim=1, keepdim=True)
    YR_std = Y_R_all.std(dim=1, keepdim=True) + 1e-8
    YR_norm = (Y_R_all - YR_mean) / YR_std

    for iteration in range(EM_ITERATIONS):
        # --- E-STEP (True Probabilistic EM based on MSE) ---
        pred_t = torch.bmm(X_all, W_t.unsqueeze(2)).squeeze(2) # [N, 384]
        
        # Compute MSE for Left and Right hypotheses
        mse_L = torch.mean((Y_L_all - pred_t)**2, dim=1) # [N]
        mse_R = torch.mean((Y_R_all - pred_t)**2, dim=1) # [N]
        
        # Convert MSE to log-likelihoods
        # log p(Y | W) = -beta * MSE
        log_L = -BETA * mse_L
        log_R = -BETA * mse_R
        
        # Log-sum-exp trick for numerical stability
        max_log = torch.max(log_L, log_R)
        exp_L = torch.exp(log_L - max_log)
        exp_R = torch.exp(log_R - max_log)
        q_t_raw = exp_L / (exp_L + exp_R + 1e-8) # [N]
        
        # Temporal smoothing of q_t (Simulating an HMM forward-backward prior)
        # We use a moving average over 5 windows (~2.5 seconds)
        kernel_size = 5
        pad = kernel_size // 2
        q_t_padded = torch.nn.functional.pad(q_t_raw.unsqueeze(0).unsqueeze(0), (pad, pad), mode='replicate')
        q_t = torch.nn.functional.avg_pool1d(q_t_padded, kernel_size, stride=1).squeeze(0).squeeze(0)
        
        # Record accuracy of current predictions (for logging)
        preds = (q_t > 0.5).float()
        acc = (preds == labels).float().mean().item()
        acc_history.append(acc)
        
        # --- M-STEP ---
        # Expected Rxy given q_t
        Rxy_expected_tau = q_t.unsqueeze(1) * Rxy_L_tau + (1.0 - q_t).unsqueeze(1) * Rxy_R_tau # [N, 408]
        
        # Smooth Rxy across time
        Rxy_smoothed = torch.einsum('tn,ni->ti', K, Rxy_expected_tau) # [N, 408]
        
        # Solve for new decoders W_t
        W_t = torch.bmm(Rxx_inv, Rxy_smoothed.unsqueeze(2)).squeeze(2) # [N, 408]
        
    initial_acc = acc_history[0]
    final_acc = acc_history[-1]
    return subj_name, initial_acc, final_acc

def main():
    mp.set_start_method('spawn', force=True)
    print("=======================================================")
    print(" PHASE 159: OFFLINE EM TRACKER")
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
    num_gpus = torch.cuda.device_count()
    # Limit max_workers to prevent OOM
    num_workers = min(mp.cpu_count(), 2)
    
    results = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(process_subject, cf, idx % max(1, num_gpus)): cf for idx, cf in enumerate(cache_files)}
        for future in concurrent.futures.as_completed(futures):
            subj, initial_acc, final_acc = future.result()
            results.append((subj, initial_acc, final_acc))
            print(f"[{subj}] Initial (Fixed W0) Acc: {initial_acc:.3f} -> Final EM Acc: {final_acc:.3f}")
            
    print("\n=======================================================")
    print(" FINAL TRACKING RESULTS (EM)")
    print("=======================================================")
    print(f"{'Subj':<5} | {'W0 Acc':<8} | {'EM Acc':<8}")
    print("-" * 30)
    for subj, i_acc, f_acc in sorted(results):
        print(f"{subj:<5} | {i_acc:<8.3f} | {f_acc:<8.3f}")
    print("-" * 30)
    
    print(f"{'MEAN':<5} | {np.mean([r[1] for r in results]):<8.3f} | {np.mean([r[2] for r in results]):<8.3f}")

if __name__ == '__main__':
    main()
