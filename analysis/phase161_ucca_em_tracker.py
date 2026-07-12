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

# EM Tracker Config
BETA = 1.0
GAMMA = 0.999
EM_ITERATIONS = 5

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
                        windows.append({'X': X_win, 'Y_L': Y_L_win, 'Y_R': Y_R_win, 'label': label})
    return windows

def process_subject(cache_file, device_id):
    device = torch.device(f'cuda:{device_id}' if torch.cuda.is_available() else 'cpu')
    subj_name = cache_file.stem.split('_')[0]
    windows = prepare_subject_windows(cache_file, device)
    
    if len(windows) < 400:
        return subj_name, 0.0, 0.0, 0.0
        
    F = windows[0]['X'].shape[1]
    
    # ---------------------------------------------------------
    # 1. COMPUTE UNSUPERVISED CCA DECODER (W_UCCA)
    # ---------------------------------------------------------
    # We use all windows to compute the global Unsupervised Difference Decoder
    X_global = torch.cat([w['X'] for w in windows], dim=0) # [Total_Samples, F]
    Y_L_global = torch.cat([w['Y_L'] for w in windows], dim=0)
    Y_R_global = torch.cat([w['Y_R'] for w in windows], dim=0)
    
    # Difference Envelope
    Y_Diff = Y_L_global - Y_R_global
    
    # Solve Ridge
    XTX = X_global.T @ X_global
    XTy = X_global.T @ Y_Diff
    I = torch.eye(F, device=device)
    W_UCCA = torch.linalg.solve(XTX + RIDGE_LAMBDA * I, XTy)
    
    # Free memory
    del X_global, Y_L_global, Y_R_global, Y_Diff, XTX, XTy
    torch.cuda.empty_cache()
    
    # ---------------------------------------------------------
    # 2. ALIGN SIGN USING CALIBRATION DATA
    # ---------------------------------------------------------
    CALIB_WINDOWS = 240 # First 2 minutes
    track_set = windows[CALIB_WINDOWS:]
    
    # Test correlation on calibration block with True Labels
    calib_correct = 0
    for w in windows[:CALIB_WINDOWS]:
        pred = (w['X'] @ W_UCCA).mean().item()
        # If true label is L, Y_L - Y_R is positive. So pred > 0 means L.
        pred_label = 1 if pred > 0 else 0
        if pred_label == w['label']:
            calib_correct += 1
            
    # If UCCA correlated negatively with the true label, flip the sign
    if calib_correct < (CALIB_WINDOWS / 2):
        W_UCCA = -W_UCCA
        
    W_0 = W_UCCA
    
    # ---------------------------------------------------------
    # 3. OFFLINE EM PRE-COMPUTATION
    # ---------------------------------------------------------
    N = len(track_set)
    if N == 0:
        return subj_name, 0.0, 0.0, 0.0
        
    X_all = torch.stack([w['X'] for w in track_set])          # [N, 384, 408]
    Y_L_all = torch.stack([w['Y_L'] for w in track_set])      # [N, 384]
    Y_R_all = torch.stack([w['Y_R'] for w in track_set])      # [N, 384]
    labels = torch.tensor([w['label'] for w in track_set], device=device) # [N]
    
    Rxx_tau = torch.bmm(X_all.transpose(1, 2), X_all)         # [N, 408, 408]
    Rxy_L_tau = torch.bmm(X_all.transpose(1, 2), Y_L_all.unsqueeze(2)).squeeze(2) # [N, 408]
    Rxy_R_tau = torch.bmm(X_all.transpose(1, 2), Y_R_all.unsqueeze(2)).squeeze(2) # [N, 408]
    
    t_idx = torch.arange(N, device=device).float()
    dist_matrix = torch.abs(t_idx.unsqueeze(0) - t_idx.unsqueeze(1))
    K = (GAMMA ** dist_matrix)                                # [N, N]
    del t_idx, dist_matrix
    
    Rxx_smoothed = torch.matmul(K, Rxx_tau.view(N, -1)).view(N, F, F)
    del Rxx_tau
    torch.cuda.empty_cache()
    
    EFFECTIVE_WINDOWS_SMOOTHED = K.sum(dim=1).view(N)         # [N]
    ridge_penalties = RIDGE_LAMBDA * EFFECTIVE_WINDOWS_SMOOTHED
    idx = torch.arange(F, device=device)
    Rxx_smoothed[:, idx, idx] += ridge_penalties.unsqueeze(1)
    
    Rxx_inv = torch.empty_like(Rxx_smoothed)
    CHUNK = 500
    for i in range(0, N, CHUNK):
        Rxx_inv[i:i+CHUNK] = torch.linalg.inv(Rxx_smoothed[i:i+CHUNK])
        
    del Rxx_smoothed
    torch.cuda.empty_cache()
    
    # ---------------------------------------------------------
    # 4. EXPECTATION-MAXIMIZATION (EM) ITERATIONS
    # ---------------------------------------------------------
    W_t = W_0.unsqueeze(0).expand(N, F)                       # [N, 408]
    acc_history = []
    
    for iteration in range(EM_ITERATIONS):
        # --- E-STEP ---
        pred_t = torch.bmm(X_all, W_t.unsqueeze(2)).squeeze(2) # [N, 384]
        
        mse_L = torch.mean((Y_L_all - pred_t)**2, dim=1) # [N]
        mse_R = torch.mean((Y_R_all - pred_t)**2, dim=1) # [N]
        
        log_L = -BETA * mse_L
        log_R = -BETA * mse_R
        
        max_log = torch.max(log_L, log_R)
        exp_L = torch.exp(log_L - max_log)
        exp_R = torch.exp(log_R - max_log)
        q_t_raw = exp_L / (exp_L + exp_R + 1e-8) # [N]
        
        kernel_size = 5
        pad = kernel_size // 2
        q_t_padded = torch.nn.functional.pad(q_t_raw.unsqueeze(0).unsqueeze(0), (pad, pad), mode='replicate')
        q_t = torch.nn.functional.avg_pool1d(q_t_padded, kernel_size, stride=1).squeeze(0).squeeze(0)
        
        preds = (q_t > 0.5).float()
        acc = (preds == labels).float().mean().item()
        acc_history.append(acc)
        
        # --- M-STEP ---
        Rxy_expected_tau = q_t.unsqueeze(1) * Rxy_L_tau + (1.0 - q_t).unsqueeze(1) * Rxy_R_tau # [N, 408]
        Rxy_smoothed = torch.einsum('tn,ni->ti', K, Rxy_expected_tau) # [N, 408]
        W_t = torch.bmm(Rxx_inv, Rxy_smoothed.unsqueeze(2)).squeeze(2) # [N, 408]
        
    initial_ucca_acc = acc_history[0]
    final_em_acc = acc_history[-1]
    
    # Evaluate STANDARD supervised W_0 (first 2 minutes) for comparison
    X_calib = torch.cat([w['X'] for w in windows[:CALIB_WINDOWS]], dim=0)
    Y_calib = torch.cat([w['Y_L'] if w['label'] == 1 else w['Y_R'] for w in windows[:CALIB_WINDOWS]], dim=0)
    XTX_calib = X_calib.T @ X_calib
    XTy_calib = X_calib.T @ Y_calib
    W_supervised = torch.linalg.solve(XTX_calib + RIDGE_LAMBDA * I, XTy_calib)
    
    W_t_sup = W_supervised.unsqueeze(0).expand(N, F)
    pred_sup = torch.bmm(X_all, W_t_sup.unsqueeze(2)).squeeze(2)
    mse_L_sup = torch.mean((Y_L_all - pred_sup)**2, dim=1)
    mse_R_sup = torch.mean((Y_R_all - pred_sup)**2, dim=1)
    preds_sup = (mse_L_sup < mse_R_sup).float()
    supervised_baseline_acc = (preds_sup == labels).float().mean().item()
    
    return subj_name, supervised_baseline_acc, initial_ucca_acc, final_em_acc

def main():
    mp.set_start_method('spawn', force=True)
    print("=======================================================")
    print(" PHASE 161: UCCA-BOOTSTRAPPED EM TRACKER")
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
    num_workers = max(1, num_gpus)
    
    results = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(process_subject, cf, idx % max(1, num_gpus)): cf for idx, cf in enumerate(cache_files)}
        for future in concurrent.futures.as_completed(futures):
            subj, sup_acc, ucca_acc, em_acc = future.result()
            if sup_acc > 0:
                results.append((subj, sup_acc, ucca_acc, em_acc))
                print(f"[{subj}] Supervised (2min) Acc: {sup_acc:.3f} | Global UCCA Acc: {ucca_acc:.3f} -> Final EM Acc: {em_acc:.3f}")
                
    results.sort(key=lambda x: x[0])
    
    print("\n=======================================================")
    print(" FINAL TRACKING RESULTS (UCCA-EM)")
    print("=======================================================")
    print(f"{'Subj':<5} | {'Sup(2m) Acc':<11} | {'UCCA Acc':<9} | {'EM Acc':<8}")
    print("-" * 45)
    for res in results:
        print(f"{res[0]:<5} | {res[1]:.3f}       | {res[2]:.3f}     | {res[3]:.3f}")
    print("-" * 45)
    
    mean_sup = np.mean([r[1] for r in results])
    mean_ucca = np.mean([r[2] for r in results])
    mean_em = np.mean([r[3] for r in results])
    print(f"{'MEAN':<5} | {mean_sup:.3f}       | {mean_ucca:.3f}     | {mean_em:.3f}")

if __name__ == '__main__':
    main()
