import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import time
import numpy as np
import torch
from pathlib import Path
from scipy import signal
import concurrent.futures
import multiprocessing as mp
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

# -------------------------------------------------------------------------
# CONSTANTS & CONFIGURATION
# -------------------------------------------------------------------------
SR = 128
EAR_CHANNEL_INDICES = [23, 31, 32, 40, 14, 22, 41, 49]
MAX_LAG_SEC = 0.400 
MAX_LAG_SAMPLES = int(MAX_LAG_SEC * SR)
SEQ_SAMPLES = int(15.0 * SR)  # 15s sequence
SEQ_HOP = int(2.0 * SR)
BROADBAND = (0.5, 8.0)

def apply_modulation_filter(eeg_raw, lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    b, a = signal.butter(order, [lowcut / nyq, highcut / nyq], btype='band')
    return signal.filtfilt(b, a, eeg_raw, axis=1)

def create_toeplitz_features(eeg, max_lag_samples):
    # eeg: (C, T)
    C, T = eeg.shape
    T_eff = T - max_lag_samples
    X = np.zeros((T_eff, C * max_lag_samples), dtype=np.float32)
    for tau in range(max_lag_samples):
        X[:, tau*C : (tau+1)*C] = eeg[:, tau : tau+T_eff].T
    return X

def extract_sequences(cache_file):
    cached = torch.load(cache_file, map_location='cpu', weights_only=False)['raw']
    sequences = []
    
    for tr in cached:
        eeg_raw = tr['eeg'].numpy()[EAR_CHANNEL_INDICES, :]
        env_l = tr['env_l'].numpy().flatten()
        env_r = tr['env_r'].numpy().flatten()
        
        # Bandpass filter
        eeg_f = apply_modulation_filter(eeg_raw, BROADBAND[0], BROADBAND[1], SR)
        env_l = apply_modulation_filter(np.expand_dims(env_l, 0), BROADBAND[0], BROADBAND[1], SR).flatten()
        env_r = apply_modulation_filter(np.expand_dims(env_r, 0), BROADBAND[0], BROADBAND[1], SR).flatten()
        
        # Standardize
        eeg_f = (eeg_f - np.mean(eeg_f, axis=1, keepdims=True)) / (np.std(eeg_f, axis=1, keepdims=True) + 1e-8)
        env_l = (env_l - np.mean(env_l)) / (np.std(env_l) + 1e-8)
        env_r = (env_r - np.mean(env_r)) / (np.std(env_r) + 1e-8)
        
        # Create Toeplitz features (Lag expansion)
        X_trial = create_toeplitz_features(eeg_f, MAX_LAG_SAMPLES)
        T_eff = X_trial.shape[0]
        Y_L_eff = env_l[:T_eff]
        Y_R_eff = env_r[:T_eff]
        
        sp = tr['meta']['switch_points']
        boundaries = [0] + [idx for spk, idx in sp]
        boundaries = sorted(set(boundaries))
        if boundaries[-1] != T_eff + MAX_LAG_SAMPLES: 
            boundaries.append(T_eff + MAX_LAG_SAMPLES)
            
        for i in range(len(boundaries) - 1):
            start_idx = boundaries[i]
            end_idx = boundaries[i+1]
            
            current_spk = 'L'
            for spk, idx in sp:
                if idx <= start_idx: current_spk = spk
                else: break
                
            safe_start = start_idx + int(1.5 * SR)
            safe_end = end_idx
            
            # Extract valid windows for the current attention block
            if safe_end - safe_start >= SEQ_SAMPLES:
                for seq_start in range(safe_start, safe_end - SEQ_SAMPLES + 1, SEQ_HOP):
                    if seq_start + SEQ_SAMPLES <= T_eff:
                        X_win = X_trial[seq_start:seq_start + SEQ_SAMPLES]
                        Y_L_win = Y_L_eff[seq_start:seq_start + SEQ_SAMPLES]
                        Y_R_win = Y_R_eff[seq_start:seq_start + SEQ_SAMPLES]
                        
                        sequences.append({
                            'X': X_win,
                            'Y_L': Y_L_win,
                            'Y_R': Y_R_win,
                            'label': 1 if current_spk == 'L' else 0
                        })
                    
    return sequences

def calc_corr(pred, target):
    p_c = pred - np.mean(pred)
    t_c = target - np.mean(target)
    num = np.sum(p_c * t_c)
    den = np.sqrt(np.sum(p_c**2) * np.sum(t_c**2)) + 1e-8
    return num / den

def process_subject(cache_file):
    subj_name = cache_file.stem.split('_')[0]
    sequences = extract_sequences(cache_file)
    
    if len(sequences) < 50:
        return subj_name, None, None
        
    tscv = TimeSeriesSplit(n_splits=5)
    
    acc_shared = []
    acc_residual = []
    
    # We must operate on flattened sequences for Ridge regression training
    # For evaluation we need to evaluate per 15s sequence
    
    for fold, (train_idx, test_idx) in enumerate(tscv.split(sequences)):
        overlap_margin = int(SEQ_SAMPLES / SEQ_HOP)
        if len(test_idx) > overlap_margin:
            test_idx = test_idx[overlap_margin:]
            
        train_seqs = [sequences[i] for i in train_idx]
        test_seqs = [sequences[i] for i in test_idx]
        
        # 1. Prepare Shared Data
        X_train_shared = []
        Y_train_shared = []
        
        X_train_L = []
        Y_train_L = []
        
        X_train_R = []
        Y_train_R = []
        
        for seq in train_seqs:
            X_train_shared.append(seq['X'])
            if seq['label'] == 1:
                Y_train_shared.append(seq['Y_L'])
                X_train_L.append(seq['X'])
                Y_train_L.append(seq['Y_L'])
            else:
                Y_train_shared.append(seq['Y_R'])
                X_train_R.append(seq['X'])
                Y_train_R.append(seq['Y_R'])
                
        X_train_shared = np.vstack(X_train_shared)
        Y_train_shared = np.concatenate(Y_train_shared)
        
        # Train Shared Model
        scaler = StandardScaler()
        X_train_shared_scaled = scaler.fit_transform(X_train_shared)
        
        ridge_shared = RidgeCV(alphas=np.logspace(-1, 5, 10))
        ridge_shared.fit(X_train_shared_scaled, Y_train_shared)
        
        # 2. Prepare Residuals
        if len(X_train_L) > 0 and len(X_train_R) > 0:
            X_train_L = np.vstack(X_train_L)
            Y_train_L = np.concatenate(Y_train_L)
            X_train_L_scaled = scaler.transform(X_train_L)
            
            Y_pred_L_shared = ridge_shared.predict(X_train_L_scaled)
            E_L = Y_train_L - Y_pred_L_shared
            
            ridge_delta_L = RidgeCV(alphas=np.logspace(-1, 5, 10))
            ridge_delta_L.fit(X_train_L_scaled, E_L)
            
            X_train_R = np.vstack(X_train_R)
            Y_train_R = np.concatenate(Y_train_R)
            X_train_R_scaled = scaler.transform(X_train_R)
            
            Y_pred_R_shared = ridge_shared.predict(X_train_R_scaled)
            E_R = Y_train_R - Y_pred_R_shared
            
            ridge_delta_R = RidgeCV(alphas=np.logspace(-1, 5, 10))
            ridge_delta_R.fit(X_train_R_scaled, E_R)
        else:
            # Fallback if a fold lacks L or R examples (extremely rare)
            ridge_delta_L = None
            ridge_delta_R = None
            
        # 3. Evaluation
        correct_shared = 0
        correct_residual = 0
        total = 0
        
        for seq in test_seqs:
            X_test_seq = scaler.transform(seq['X'])
            Y_L_seq = seq['Y_L']
            Y_R_seq = seq['Y_R']
            
            # Shared Evaluation
            Y_pred_shared = ridge_shared.predict(X_test_seq)
            corr_shared_L = calc_corr(Y_pred_shared, Y_L_seq)
            corr_shared_R = calc_corr(Y_pred_shared, Y_R_seq)
            if (corr_shared_L > corr_shared_R) == seq['label']:
                correct_shared += 1
                
            # Residual Evaluation
            if ridge_delta_L is not None and ridge_delta_R is not None:
                Y_pred_L_model = Y_pred_shared + ridge_delta_L.predict(X_test_seq)
                Y_pred_R_model = Y_pred_shared + ridge_delta_R.predict(X_test_seq)
                
                # Within-Decoder Discriminative Scoring
                score_L = calc_corr(Y_pred_L_model, Y_L_seq) - calc_corr(Y_pred_L_model, Y_R_seq)
                score_R = calc_corr(Y_pred_R_model, Y_R_seq) - calc_corr(Y_pred_R_model, Y_L_seq)
                
                if (score_L > score_R) == seq['label']:
                    correct_residual += 1
            else:
                if (corr_shared_L > corr_shared_R) == seq['label']:
                    correct_residual += 1
                    
            total += 1
            
        acc_shared.append(correct_shared / total)
        acc_residual.append(correct_residual / total)
        
    mean_shared = np.mean(acc_shared)
    mean_residual = np.mean(acc_residual)
    print(f"[{subj_name}] Shared Ridge: {mean_shared*100:.1f}% | Residual Decoders: {mean_residual*100:.1f}%")
    
    return subj_name, mean_shared, mean_residual

def main():
    print("=======================================================")
    print(" PHASE 171: STIMULUS-CONDITIONED RESIDUAL DECODERS")
    print(" W_L = W_shared + Delta_L  |  W_R = W_shared + Delta_R")
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
            subj, acc_s, acc_r = future.result()
            if acc_s is not None:
                results[subj] = (acc_s, acc_r)
                
    print(f"\nExtraction & Training Time: {time.time() - start_time:.2f}s\n")
    
    global_s = []
    global_r = []
    
    subjects_sorted = sorted(results.keys())
    for subj in subjects_sorted:
        s, r = results[subj]
        global_s.append(s * 100)
        global_r.append(r * 100)
        print(f"--- Subject: {subj} ---")
        print(f"  Shared Ridge    : {s*100:.1f}%")
        print(f"  Residual Ridge  : {r*100:.1f}%\n")
        
    print("=======================================================")
    print(" GLOBAL OBSERVABILITY AVERAGES")
    print("=======================================================")
    print(f"Global Shared Ridge    : {np.mean(global_s):.2f}%")
    print(f"Global Residual Decoders: {np.mean(global_r):.2f}%")
    print("=======================================================")
    
if __name__ == '__main__':
    main()
