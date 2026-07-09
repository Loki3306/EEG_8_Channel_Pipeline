import numpy as np
from scipy.linalg import solve
from scipy.stats import pearsonr
import torch
from pathlib import Path
from sklearn.metrics import roc_auc_score
import random
from scipy import signal
import time
import os

# -------------------------------------------------------------------------
# CONSTANTS
# -------------------------------------------------------------------------
SR = 128
EAR_CHANNEL_INDICES = [23, 31, 32, 40, 14, 22, 41, 49]
TARGET_SUBJECTS = ['S05', 'S08', 'S10', 'S11', 'S13', 'S16']

# mTRF Design Parameters
MAX_LAG_SEC = 0.400 
MAX_LAG_SAMPLES = int(MAX_LAG_SEC * SR)

LAMBDAS = [1e-3, 1e-2, 1e-1, 1, 10, 100, 1000]

def apply_modulation_filter(env, lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    if lowcut is None and highcut is not None:
        b, a = signal.butter(order, highcut / nyq, btype='low')
    elif highcut is None and lowcut is not None:
        b, a = signal.butter(order, lowcut / nyq, btype='high')
    else:
        low = lowcut / nyq
        high = highcut / nyq
        b, a = signal.butter(order, [low, high], btype='band')
        
    filtered = signal.filtfilt(b, a, env, axis=1)
    return filtered

def get_trial_dominant_speaker(tr):
    sp = tr['meta']['switch_points']
    T = tr['eeg'].shape[1]
    
    boundaries = [0]
    boundaries.extend([idx for spk, idx in sp])
    boundaries = sorted(set(boundaries))
    if boundaries[-1] != T: boundaries.append(T)
        
    l_duration = 0
    r_duration = 0
    
    for i in range(len(boundaries) - 1):
        start_idx = boundaries[i]
        end_idx = boundaries[i+1]
        current_spk = 'L'
        for spk, idx in sp:
            if idx <= start_idx: current_spk = spk
            else: break
            
        if current_spk == 'L': l_duration += (end_idx - start_idx)
        else: r_duration += (end_idx - start_idx)
        
    return 'L' if l_duration >= r_duration else 'R'

def stratified_trial_split(trials, train_ratio=0.8):
    l_trials = []
    r_trials = []
    
    for i, tr in enumerate(trials):
        if get_trial_dominant_speaker(tr) == 'L':
            l_trials.append(i)
        else:
            r_trials.append(i)
            
    random.seed(42)
    random.shuffle(l_trials)
    random.shuffle(r_trials)
    
    l_split = int(len(l_trials) * train_ratio)
    r_split = int(len(r_trials) * train_ratio)
    
    train_indices = l_trials[:l_split] + r_trials[:r_split]
    eval_indices = l_trials[l_split:] + r_trials[r_split:]
    
    random.shuffle(train_indices)
    random.shuffle(eval_indices)
    
    return train_indices, eval_indices

def create_toeplitz_features(eeg, max_lag_samples):
    C, T = eeg.shape
    T_eff = T - max_lag_samples
    
    X = np.zeros((T_eff, C * max_lag_samples))
    for tau in range(max_lag_samples):
        X[:, tau*C : (tau+1)*C] = eeg[:, tau : tau+T_eff].T
    return X

def extract_mtrf_matrices(trials):
    X_list = []
    Y_attended_list = []
    
    for tr in trials:
        eeg = tr['eeg'] # (C, T)
        env_l = tr['env_l'][0] # (T)
        env_r = tr['env_r'][0] # (T)
        
        T = eeg.shape[1]
        
        X_trial = create_toeplitz_features(eeg, MAX_LAG_SAMPLES)
        T_eff = X_trial.shape[0]
        
        sp = tr['meta']['switch_points']
        
        boundaries = [0]
        boundaries.extend([idx for spk, idx in sp])
        boundaries = sorted(set(boundaries))
        if boundaries[-1] != T: boundaries.append(T)
            
        Y_att = np.zeros(T)
        
        for i in range(len(boundaries) - 1):
            start_idx = boundaries[i]
            end_idx = boundaries[i+1]
            current_spk = 'L'
            for spk, idx in sp:
                if idx <= start_idx: current_spk = spk
                else: break
                
            if current_spk == 'L':
                Y_att[start_idx:end_idx] = env_l[start_idx:end_idx]
            else:
                Y_att[start_idx:end_idx] = env_r[start_idx:end_idx]
                
        Y_trial = Y_att[:T_eff]
        
        X_list.append(X_trial)
        Y_attended_list.append(Y_trial)
        
    return np.vstack(X_list), np.concatenate(Y_attended_list)

def fit_ridge(X, y, lam):
    XTX = X.T @ X
    XTy = X.T @ y
    I = np.eye(XTX.shape[0])
    W = solve(XTX + lam * I, XTy, assume_a='pos')
    return W

def evaluate_mtrf(W, trials):
    y_preds = []
    y_labels = []
    
    SEQ_SAMPLES = int(3.5 * SR)
    seq_hop = int(0.5 * SR)
    
    for tr in trials:
        eeg = tr['eeg']
        env_l = tr['env_l'][0]
        env_r = tr['env_r'][0]
        T = eeg.shape[1]
        
        sp = tr['meta']['switch_points']
        boundaries = [0]
        boundaries.extend([idx for spk, idx in sp])
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
                for seq_start in range(safe_start, safe_end - SEQ_SAMPLES + 1, seq_hop):
                    eeg_seq = eeg[:, seq_start:seq_start + SEQ_SAMPLES]
                    env_l_seq = env_l[seq_start:seq_start + SEQ_SAMPLES]
                    env_r_seq = env_r[seq_start:seq_start + SEQ_SAMPLES]
                    
                    X_seq = create_toeplitz_features(eeg_seq, MAX_LAG_SAMPLES)
                    T_eff = X_seq.shape[0]
                    
                    Y_l_eff = env_l_seq[:T_eff]
                    Y_r_eff = env_r_seq[:T_eff]
                    
                    Y_hat = X_seq @ W
                    
                    if np.std(Y_hat) < 1e-8 or np.std(Y_l_eff) < 1e-8 or np.std(Y_r_eff) < 1e-8:
                        r_L = 0
                        r_R = 0
                    else:
                        r_L, _ = pearsonr(Y_hat, Y_l_eff)
                        r_R, _ = pearsonr(Y_hat, Y_r_eff)
                    
                    score_L = r_L - r_R
                    label_L = 1.0 if current_spk == 'L' else 0.0
                    
                    y_preds.append(score_L)
                    y_labels.append(label_L)
                    
    return y_preds, y_labels

def main():
    possible_paths = [
        Path('/kaggle/input/datasets/lowkieee/multiband-cache/kaggle/working/multiband_cache'),
        Path('/kaggle/input/multiband-cache/kaggle/working/multiband_cache'),
        Path('/kaggle/input/aasd-universal-cache-v1/kaggle/working/multiband_cache'),
        Path('/kaggle/working/multiband_cache')
    ]
    
    cache_dir = Path('/kaggle/working/multiband_cache')
    for p in possible_paths:
        if p.exists() and len(list(p.glob('*_multiband.pt'))) > 0:
            cache_dir = p
            break
            
    print(f"\n=======================================================")
    print(f" PHASE 108: mTRF BACKWARD DECODER (RIDGE REGRESSION)")
    print(f" Strict Leakage-Free Baseline. Pearson Correlation.")
    print(f"=======================================================\n", flush=True)
    
    cache_files = sorted(list(cache_dir.glob('*_multiband.pt')))
    filtered_files = [f for f in cache_files if f.stem.split('_')[0] in TARGET_SUBJECTS]
    
    final_results = {}
    
    for cache_file in filtered_files:
        subj_name = cache_file.stem.split('_')[0]
        print(f"\n=======================================================")
        print(f" SUBJECT {subj_name}")
        print(f"=======================================================", flush=True)
        
        cached = torch.load(cache_file, map_location='cpu', weights_only=False)['raw']
        
        raw_trials = []
        for i in range(len(cached)):
            tr = cached[i]
            eeg = tr['eeg'].numpy()[EAR_CHANNEL_INDICES, :] 
            env_l = tr['env_l'].numpy()
            env_r = tr['env_r'].numpy()
            
            eeg = apply_modulation_filter(eeg, None, 8.0, SR)
            env_l = apply_modulation_filter(env_l, None, 8.0, SR)
            env_r = apply_modulation_filter(env_r, None, 8.0, SR)
            
            eeg = (eeg - np.mean(eeg, axis=1, keepdims=True)) / (np.std(eeg, axis=1, keepdims=True) + 1e-8)
            env_l = (env_l - np.mean(env_l, axis=1, keepdims=True)) / (np.std(env_l, axis=1, keepdims=True) + 1e-8)
            env_r = (env_r - np.mean(env_r, axis=1, keepdims=True)) / (np.std(env_r, axis=1, keepdims=True) + 1e-8)
            
            min_len = min(eeg.shape[1], env_l.shape[1])
            raw_trials.append({
                'eeg': eeg[:, :min_len], 
                'env_l': env_l[:, :min_len], 
                'env_r': env_r[:, :min_len], 
                'meta': tr['meta']
            })
            
        train_indices, eval_indices = stratified_trial_split(raw_trials, train_ratio=0.8)
        
        raw_train_trials = [raw_trials[i] for i in train_indices]
        raw_eval_trials = [raw_trials[i] for i in eval_indices]
        
        print("  [Calibration] Extracting Toeplitz Matrices...", flush=True)
        calib_train_idx, calib_val_idx = stratified_trial_split(raw_train_trials, train_ratio=0.8)
        
        calib_train = [raw_train_trials[i] for i in calib_train_idx]
        calib_val = [raw_train_trials[i] for i in calib_val_idx]
        
        X_train_cv, Y_train_cv = extract_mtrf_matrices(calib_train)
        
        best_lam = None
        best_val_auc = 0
        calibration_invert = False
        
        print("  [Calibration] Sweeping Ridge parameters...", flush=True)
        for lam in LAMBDAS:
            W = fit_ridge(X_train_cv, Y_train_cv, lam)
            val_preds, val_labels = evaluate_mtrf(W, calib_val)
            
            if len(np.unique(val_labels)) > 1:
                auc = roc_auc_score(val_labels, val_preds)
                is_inverted = (auc < 0.5)
                effective_auc = (1.0 - auc) if is_inverted else auc
                
                print(f"    - Lambda {lam}: AUROC = {effective_auc:.4f} {'(Inverted)' if is_inverted else ''}")
                if effective_auc > best_val_auc:
                    best_val_auc = effective_auc
                    best_lam = lam
                    calibration_invert = is_inverted
                    
        print(f"  [Calibration] Winner: >> Lambda {best_lam} << (Invert: {calibration_invert})", flush=True)
        
        print("  [Deployment] Training mTRF on full Training Set...", flush=True)
        X_train_full, Y_train_full = extract_mtrf_matrices(raw_train_trials)
        W_final = fit_ridge(X_train_full, Y_train_full, best_lam)
        
        print("  [Deployment] Evaluating on Unseen Eval Set...", flush=True)
        eval_preds, eval_labels = evaluate_mtrf(W_final, raw_eval_trials)
        
        deployment_best_auc = 0
        if len(np.unique(eval_labels)) > 1:
            auc = roc_auc_score(eval_labels, eval_preds)
            deployment_best_auc = (1.0 - auc) if calibration_invert else auc
            
        print(f"  [Deployment] Final Deployment AUROC: {deployment_best_auc:.4f} {'(Inverted via Calibration)' if calibration_invert else ''}")
        final_results[subj_name] = {'Lambda': best_lam, 'AUROC': deployment_best_auc}

    print("\n\n=======================================================")
    print(" PHASE 108 mTRF PIPELINE RESULTS")
    print("=======================================================")
    print(f"{'Subject':<10} {'Lambda':<15} {'Deployment AUROC':<10}")
    for subj, res in final_results.items():
        print(f"{subj:<10} {res['Lambda']:<15} {res['AUROC']:.4f}")

if __name__ == '__main__':
    main()
