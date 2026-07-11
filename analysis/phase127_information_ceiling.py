import os
import numpy as np
import torch
from pathlib import Path
from sklearn.metrics import roc_auc_score
from scipy import signal
import time
import concurrent.futures
import multiprocessing as mp

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
RIDGE_LAMBDA = 1.0 # Small lambda to allow maximum fitting to the training data

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
    F = X_trial.shape[1]
    
    Y_l_eff = env_l[:T_eff]
    Y_r_eff = env_r[:T_eff]
    
    sp = tr['meta']['switch_points']
    boundaries = [0] + [idx for spk, idx in sp]
    boundaries = sorted(set(boundaries))
    if boundaries[-1] != T: boundaries.append(T)
        
    mask_L = torch.zeros(T_eff, dtype=torch.bool, device=device)
    mask_R = torch.zeros(T_eff, dtype=torch.bool, device=device)
    
    seq_indices, Y_meta = [], []
    
    for i in range(len(boundaries) - 1):
        start_idx = boundaries[i]
        end_idx = boundaries[i+1]
        
        current_spk = 'L'
        for spk, idx in sp:
            if idx <= start_idx: current_spk = spk
            else: break
            
        safe_eff_end = min(end_idx, T_eff)
        if start_idx < safe_eff_end:
            if current_spk == 'L':
                mask_L[start_idx:safe_eff_end] = True
            else:
                mask_R[start_idx:safe_eff_end] = True
            
        safe_start = start_idx + int(1.5 * SR)
        safe_end = end_idx
        
        if safe_end - safe_start >= SEQ_SAMPLES:
            for seq_start in range(safe_start, safe_end - SEQ_SAMPLES + 1, SEQ_HOP):
                if seq_start + SEQ_SAMPLES <= T_eff:
                    seq_indices.append((seq_start, seq_start + SEQ_SAMPLES))
                    Y_meta.append(1.0 if current_spk == 'L' else 0.0)
                
    X_L = X_trial[mask_L]
    Y_L = Y_l_eff[mask_L]
    
    X_R = X_trial[mask_R]
    Y_R = Y_r_eff[mask_R]
    
    XTX_L = (X_L.T @ X_L) if X_L.shape[0] > 0 else torch.zeros((F, F), device=device)
    XTy_L = (X_L.T @ Y_L).unsqueeze(-1) if X_L.shape[0] > 0 else torch.zeros((F, 1), device=device)
    
    XTX_R = (X_R.T @ X_R) if X_R.shape[0] > 0 else torch.zeros((F, F), device=device)
    XTy_R = (X_R.T @ Y_R).unsqueeze(-1) if X_R.shape[0] > 0 else torch.zeros((F, 1), device=device)
    
    return {
        'X': X_trial,
        'Y_l': Y_l_eff,
        'Y_r': Y_r_eff,
        'XTX_L': XTX_L,
        'XTy_L': XTy_L,
        'XTX_R': XTX_R,
        'XTy_R': XTy_R,
        'seq_indices': seq_indices,
        'Y_meta': torch.tensor(Y_meta, dtype=torch.float32, device=device)
    }

def evaluate_trial_scores(W_L, W_R, tr_info):
    if len(tr_info['seq_indices']) == 0:
        return None, None
        
    Y_hat_full_L = tr_info['X'] @ W_L.squeeze(-1) # (T_eff,)
    Y_hat_full_R = tr_info['X'] @ W_R.squeeze(-1) # (T_eff,)
    
    Y_hat_seqs_L = torch.stack([Y_hat_full_L[s:e] for s, e in tr_info['seq_indices']])
    Y_hat_seqs_R = torch.stack([Y_hat_full_R[s:e] for s, e in tr_info['seq_indices']])
    
    Y_l_seqs = torch.stack([tr_info['Y_l'][s:e] for s, e in tr_info['seq_indices']])
    Y_r_seqs = torch.stack([tr_info['Y_r'][s:e] for s, e in tr_info['seq_indices']])
    
    score_L = batch_pearsonr_pt(Y_hat_seqs_L, Y_l_seqs)
    score_R = batch_pearsonr_pt(Y_hat_seqs_R, Y_r_seqs)
    
    return score_L, score_R

def process_subject(cache_file, device_id):
    device = torch.device(f'cuda:{device_id}' if torch.cuda.is_available() else 'cpu')
    subj_name = cache_file.stem.split('_')[0]
    
    cached = torch.load(cache_file, map_location='cpu', weights_only=False)['raw']
    
    N_TRIALS = len(cached)
    
    all_eval_diffs = []
    all_eval_labels = []
    all_match_correlations = []
    all_mismatch_correlations = []
    
    for t_idx in range(N_TRIALS):
        # Prepare the single trial
        tr_info = prepare_trial_data(cached[t_idx], device)
        
        # CHEATING: Train on this EXACT SAME trial!
        W_final_L = solve_ridge_pt(tr_info['XTX_L'], tr_info['XTy_L'], lam=RIDGE_LAMBDA)
        W_final_R = solve_ridge_pt(tr_info['XTX_R'], tr_info['XTy_R'], lam=RIDGE_LAMBDA)
        
        # CHEATING: Evaluate on the EXACT SAME trial!
        score_L, score_R = evaluate_trial_scores(W_final_L, W_final_R, tr_info)
        
        if score_L is not None:
            score_L = score_L.cpu().numpy()
            score_R = score_R.cpu().numpy()
            y_meta = tr_info['Y_meta'].cpu().numpy()
            
            diffs = score_L - score_R
            all_eval_diffs.extend(diffs.tolist())
            all_eval_labels.extend(y_meta.tolist())
            
            # Record raw correlations for insight
            for i in range(len(score_L)):
                if y_meta[i] == 1.0: # Attended L
                    all_match_correlations.append(score_L[i])
                    all_mismatch_correlations.append(score_R[i])
                else: # Attended R
                    all_match_correlations.append(score_R[i])
                    all_mismatch_correlations.append(score_L[i])
            
    all_eval_diffs = np.array(all_eval_diffs)
    all_eval_labels = np.array(all_eval_labels)
    
    if len(all_eval_diffs) > 0 and len(np.unique(all_eval_labels)) > 1:
        probs = (all_eval_diffs - np.min(all_eval_diffs)) / (np.max(all_eval_diffs) - np.min(all_eval_diffs) + 1e-8)
        global_auc = roc_auc_score(all_eval_labels, probs)
    else:
        global_auc = 0.5
        
    mean_match_r = np.mean(all_match_correlations) if all_match_correlations else 0.0
    mean_mismatch_r = np.mean(all_mismatch_correlations) if all_mismatch_correlations else 0.0
        
    print(f"  [{subj_name}] Finished on {device}. CHEATER AUROC: {global_auc:.4f} | Mean Match R: {mean_match_r:.4f} | Mean Mismatch R: {mean_mismatch_r:.4f}")
    
    return subj_name, global_auc

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
    print(f" PHASE 127: INFORMATION CEILING (CHEATER UPPER BOUND)")
    print(f" CPUs detected: {mp.cpu_count()} | GPUs detected: {num_gpus}")
    print(f"=======================================================\n", flush=True)
    
    cache_files = sorted(list(cache_dir.glob('*_multiband.pt')))
    
    start_time = time.time()
    final_results = {}
    
    futures = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        for idx, cache_file in enumerate(cache_files):
            device_id = idx % num_gpus if num_gpus > 0 else 0
            futures.append(executor.submit(process_subject, cache_file, device_id))
            
        for future in concurrent.futures.as_completed(futures):
            subj_name, auc = future.result()
            final_results[subj_name] = auc

    print(f"\nTotal Pipeline Execution Time: {time.time() - start_time:.2f}s")
    print("\n=======================================================")
    print(" PHASE 127 CHEATER UPPER BOUND RESULTS")
    print("=======================================================")
    print(f"{'Subject':<10} {'CHEATER AUROC':<10}")
    
    sorted_results = sorted(final_results.items(), key=lambda x: int(x[0][1:]))
    for subj, auc in sorted_results:
        print(f"{subj:<10} {auc:.4f}")

if __name__ == '__main__':
    main()
