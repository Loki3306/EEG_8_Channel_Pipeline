import os
import numpy as np
import torch
from pathlib import Path
from scipy import signal
import time
import concurrent.futures
import multiprocessing as mp
from sklearn.metrics.pairwise import cosine_similarity

# -------------------------------------------------------------------------
# CONSTANTS & CONFIGURATION
# -------------------------------------------------------------------------
SR = 128
EAR_CHANNEL_INDICES = [23, 31, 32, 40, 14, 22, 41, 49]
MAX_LAG_SEC = 0.400 
MAX_LAG_SAMPLES = int(MAX_LAG_SEC * SR)
SEQ_SAMPLES = int(3.5 * SR)
SEQ_HOP = int(0.5 * SR)
BROADBAND = (0.5, 8.0)
RIDGE_LAMBDA = 100.0

def apply_modulation_filter(env, lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype='band')
    return signal.filtfilt(b, a, env, axis=1)

def create_toeplitz_features_pt(eeg, max_lag_samples):
    C, T = eeg.shape
    T_eff = T - max_lag_samples
    X = torch.zeros((T_eff, C * max_lag_samples), dtype=eeg.dtype, device=eeg.device)
    for tau in range(max_lag_samples):
        X[:, tau*C : (tau+1)*C] = eeg[:, tau : tau+T_eff].T
    return X

def batch_pearsonr_pt(x, y):
    x_mean = x - x.mean(dim=1, keepdim=True)
    y_mean = y - y.mean(dim=1, keepdim=True)
    num = (x_mean * y_mean).sum(dim=1)
    den = torch.sqrt((x_mean**2).sum(dim=1) * (y_mean**2).sum(dim=1))
    return num / (den + 1e-8)

def solve_ridge_pt(XTX, XTy, lam=1.0):
    F = XTX.shape[0]
    I = torch.eye(F, device=XTX.device, dtype=XTX.dtype)
    jitter = 1e-6 * torch.randn(F, F, device=XTX.device, dtype=XTX.dtype) * I
    return torch.linalg.solve(XTX + lam * I + jitter, XTy)

def prepare_trial_data(tr, device):
    eeg_raw = tr['eeg'].numpy()[EAR_CHANNEL_INDICES, :] 
    env_l_raw = tr['env_l'].numpy()
    env_l_f = apply_modulation_filter(env_l_raw, BROADBAND[0], BROADBAND[1], SR)
    env_l_f = (env_l_f - np.mean(env_l_f, axis=1, keepdims=True)) / (np.std(env_l_f, axis=1, keepdims=True) + 1e-8)
    env_l = torch.tensor(env_l_f[0], dtype=torch.float32, device=device)
    
    eeg_f = apply_modulation_filter(eeg_raw, BROADBAND[0], BROADBAND[1], SR)
    eeg_f = (eeg_f - np.mean(eeg_f, axis=1, keepdims=True)) / (np.std(eeg_f, axis=1, keepdims=True) + 1e-8)
    eeg = torch.tensor(eeg_f, dtype=torch.float32, device=device)
    
    T = eeg.shape[1]
    X_trial = create_toeplitz_features_pt(eeg, MAX_LAG_SAMPLES)
    T_eff = X_trial.shape[0]
    F = X_trial.shape[1]
    
    Y_l_eff = env_l[:T_eff]
    
    XTX = (X_trial.T @ X_trial)
    XTy = (X_trial.T @ Y_l_eff).unsqueeze(-1)
    
    # Pre-slice into 3.5s sequences for fast generalization testing
    seq_indices = []
    for seq_start in range(0, T_eff - SEQ_SAMPLES + 1, SEQ_HOP):
        seq_indices.append((seq_start, seq_start + SEQ_SAMPLES))
        
    return {
        'X': X_trial,
        'Y': Y_l_eff,
        'XTX': XTX,
        'XTy': XTy,
        'seq_indices': seq_indices
    }

def get_mean_off_diag(matrix):
    np.fill_diagonal(matrix, np.nan)
    return np.nanmean(matrix)

def get_adjacent_vs_distant(matrix):
    N = matrix.shape[0]
    adj = []
    dist = []
    for i in range(N):
        for j in range(N):
            if i == j: continue
            if abs(i - j) == 1:
                adj.append(matrix[i, j])
            elif abs(i - j) > 10:
                dist.append(matrix[i, j])
    return np.mean(adj), np.mean(dist)

def process_subject(cache_file, device_id):
    device = torch.device(f'cuda:{device_id}' if torch.cuda.is_available() else 'cpu')
    subj_name = cache_file.stem.split('_')[0]
    cached = torch.load(cache_file, map_location='cpu', weights_only=False)['raw']
    
    N_TRIALS = len(cached)
    data_cache = []
    
    W_all = []
    S_all = []
    T_all = []
    C_all = []
    
    for t_idx in range(N_TRIALS):
        tr_info = prepare_trial_data(cached[t_idx], device)
        data_cache.append(tr_info)
        
        W = solve_ridge_pt(tr_info['XTX'], tr_info['XTy'], lam=RIDGE_LAMBDA).squeeze(-1).cpu().numpy()
        W = W / (np.linalg.norm(W) + 1e-8)
        W_all.append(W)
        
        # Spatial vs Temporal
        W_reshaped = W.reshape(8, 51)
        S = np.sum(W_reshaped**2, axis=1)
        S = S / (np.linalg.norm(S) + 1e-8)
        S_all.append(S)
        
        T = np.sum(W_reshaped**2, axis=0)
        T = T / (np.linalg.norm(T) + 1e-8)
        T_all.append(T)
        
        # Covariance
        C = (tr_info['XTX'] / tr_info['X'].shape[0]).cpu().numpy().flatten()
        C = C / (np.linalg.norm(C) + 1e-8)
        C_all.append(C)
        
    W_all = np.array(W_all)
    S_all = np.array(S_all)
    T_all = np.array(T_all)
    C_all = np.array(C_all)
    
    sim_W = cosine_similarity(W_all)
    sim_S = cosine_similarity(S_all)
    sim_T = cosine_similarity(T_all)
    sim_C = cosine_similarity(C_all)
    
    mean_W = get_mean_off_diag(sim_W)
    mean_S = get_mean_off_diag(sim_S)
    mean_T = get_mean_off_diag(sim_T)
    mean_C = get_mean_off_diag(sim_C)
    
    adj_W, dist_W = get_adjacent_vs_distant(sim_W)
    
    # Cross-Decoder Generalization Matrix (Perf(i,j))
    perf_matrix = np.zeros((N_TRIALS, N_TRIALS))
    
    for i in range(N_TRIALS):
        W_i_tensor = torch.tensor(W_all[i], dtype=torch.float32, device=device).unsqueeze(-1)
        for j in range(N_TRIALS):
            tr_j = data_cache[j]
            if len(tr_j['seq_indices']) == 0: continue
                
            Y_hat = tr_j['X'] @ W_i_tensor.squeeze(-1)
            Y_hat_seqs = torch.stack([Y_hat[s:e] for s, e in tr_j['seq_indices']])
            Y_seqs = torch.stack([tr_j['Y'][s:e] for s, e in tr_j['seq_indices']])
            
            scores = batch_pearsonr_pt(Y_hat_seqs, Y_seqs)
            perf_matrix[i, j] = scores.mean().item()
            
    perf_diag = np.nanmean(np.diag(perf_matrix))
    perf_adj, perf_dist = get_adjacent_vs_distant(perf_matrix)
    
    print(f"\n[{subj_name}] Metrics:")
    print(f"  Similarity -> W: {mean_W:.3f} | Spatial: {mean_S:.3f} | Temporal: {mean_T:.3f} | Covariance: {mean_C:.3f}")
    print(f"  Drift      -> Adj W: {adj_W:.3f}, Distant W: {dist_W:.3f}")
    print(f"  Generalize -> Train/Test(Self): {perf_diag:.3f} | Adj Trial: {perf_adj:.3f} | Dist Trial: {perf_dist:.3f}")
    
    return subj_name

def main():
    mp.set_start_method('spawn', force=True)
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
            
    num_gpus = torch.cuda.device_count()
    num_workers = min(mp.cpu_count(), num_gpus if num_gpus > 0 else mp.cpu_count())
    
    print(f"\n=======================================================")
    print(f" PHASE 128: FILTER STABILITY & NON-STATIONARITY AUDIT")
    print(f"=======================================================\n", flush=True)
    
    cache_files = sorted(list(cache_dir.glob('*_multiband.pt')))
    
    start_time = time.time()
    
    futures = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        for idx, cache_file in enumerate(cache_files):
            device_id = idx % num_gpus if num_gpus > 0 else 0
            futures.append(executor.submit(process_subject, cache_file, device_id))
            
        for future in concurrent.futures.as_completed(futures):
            future.result()

    print(f"\nTotal Pipeline Execution Time: {time.time() - start_time:.2f}s")

if __name__ == '__main__':
    main()
