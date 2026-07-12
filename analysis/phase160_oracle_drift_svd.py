import os
import numpy as np
import torch
from pathlib import Path
import multiprocessing as mp
import concurrent.futures
from scipy import signal
from sklearn.decomposition import PCA

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
FORGETTING_FACTOR = 0.98

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
    
    if len(windows) < 200:
        return subj_name, None, None
        
    F = windows[0]['X'].shape[1]
    N = len(windows)
    
    # Compute Oracle W_t using symmetric smoothing
    X_all = torch.stack([w['X'] for w in windows])          # [N, 384, 408]
    labels = [w['label'] for w in windows]
    Y_all = torch.stack([w['Y_L'] if lbl == 1 else w['Y_R'] for w, lbl in zip(windows, labels)]) # [N, 384]
    
    Rxx_tau = torch.bmm(X_all.transpose(1, 2), X_all)         # [N, 408, 408]
    Rxy_tau = torch.bmm(X_all.transpose(1, 2), Y_all.unsqueeze(2)).squeeze(2) # [N, 408]
    
    t_idx = torch.arange(N, device=device).float()
    dist_matrix = torch.abs(t_idx.unsqueeze(0) - t_idx.unsqueeze(1))
    K = (FORGETTING_FACTOR ** dist_matrix)
    del t_idx, dist_matrix
    
    Rxx_smoothed = torch.matmul(K, Rxx_tau.view(N, -1)).view(N, F, F)
    del Rxx_tau
    torch.cuda.empty_cache()
    
    EFFECTIVE_WINDOWS = K.sum(dim=1).view(N)
    idx = torch.arange(F, device=device)
    Rxx_smoothed[:, idx, idx] += RIDGE_LAMBDA * EFFECTIVE_WINDOWS.unsqueeze(1)
    
    # Chunked inverse
    Rxx_inv = torch.empty_like(Rxx_smoothed)
    CHUNK = 500
    for i in range(0, N, CHUNK):
        Rxx_inv[i:i+CHUNK] = torch.linalg.inv(Rxx_smoothed[i:i+CHUNK])
    del Rxx_smoothed
    torch.cuda.empty_cache()
    
    Rxy_smoothed = torch.einsum('tn,ni->ti', K, Rxy_tau) # [N, 408]
    W_t = torch.bmm(Rxx_inv, Rxy_smoothed.unsqueeze(2)).squeeze(2) # [N, 408]
    
    W_t_np = W_t.cpu().numpy()
    
    # 1. PCA on W_t directly (Absolute Subspace)
    pca_abs = PCA()
    pca_abs.fit(W_t_np)
    var_abs = np.cumsum(pca_abs.explained_variance_ratio_)
    dim_90_abs = np.argmax(var_abs >= 0.90) + 1
    
    # 2. PCA on Delta W_t (Velocity Subspace)
    delta_W = np.diff(W_t_np, axis=0) # [N-1, 408]
    pca_vel = PCA()
    pca_vel.fit(delta_W)
    var_vel = np.cumsum(pca_vel.explained_variance_ratio_)
    dim_90_vel = np.argmax(var_vel >= 0.90) + 1
    
    return subj_name, dim_90_abs, dim_90_vel

def main():
    mp.set_start_method('spawn', force=True)
    print("=======================================================")
    print(" PHASE 160: ORACLE DRIFT SVD ANALYSIS")
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
            subj, d_abs, d_vel = future.result()
            if d_abs is not None:
                results.append((subj, d_abs, d_vel))
                print(f"[{subj}] 90% Variance Dims - Absolute: {d_abs:3d}/408 | Velocity: {d_vel:3d}/408")
                
    print("\n=======================================================")
    print(" OVERALL DRIFT DIMENSIONALITY")
    print("=======================================================")
    print(f"MEAN Absolute Dims (90% var): {np.mean([r[1] for r in results]):.1f} / 408")
    print(f"MEAN Velocity Dims (90% var): {np.mean([r[2] for r in results]):.1f} / 408")

if __name__ == '__main__':
    main()
