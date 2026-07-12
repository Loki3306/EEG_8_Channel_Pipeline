import os
import time
import numpy as np
import torch
from pathlib import Path
from scipy import signal
import concurrent.futures
import multiprocessing as mp

from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score

try:
    import pyriemann
except ImportError:
    import os
    os.system("pip install pyriemann")
    import pyriemann

from pyriemann.classification import MDM
from pyriemann.tangentspace import TangentSpace

# -------------------------------------------------------------------------
# CONSTANTS & CONFIGURATION
# -------------------------------------------------------------------------
SR = 128
EAR_CHANNEL_INDICES = [23, 31, 32, 40, 14, 22, 41, 49]
BROADBAND = (0.5, 8.0)

PRE_SWITCH_SAMPLES = int(0.5 * SR)
POST_SWITCH_SAMPLES = int(1.0 * SR)
MIN_SEGMENT_SAMPLES = int(3.0 * SR)

ALPHA = 0.05 # Exponential Moving Average factor for covariance

def fractional_matrix_power(M, power, reg=1e-6):
    M_reg = M + np.eye(M.shape[0]) * reg
    w, v = np.linalg.eigh(M_reg)
    w = np.maximum(w, 1e-12)
    return v @ np.diag(w ** power) @ v.T

def compute_covariance(X):
    X_centered = X - np.mean(X, axis=1, keepdims=True)
    cov = (X_centered @ X_centered.T) / (X.shape[1] - 1)
    return cov

def apply_modulation_filter(env, lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype='band')
    return signal.filtfilt(b, a, env, axis=1)

def get_masks(sp, length):
    mask_true = np.zeros(length, dtype=np.float32)
    mask_valid = np.ones(length, dtype=bool)
    
    if len(sp) == 0:
        mask_true[:] = 1.0
        return mask_true, mask_valid
        
    current_state = 1.0 if sp[0][0] == 'R' else 0.0 
    last_idx = 0
    for spk, idx in sp:
        end_idx = min(idx, length)
        mask_true[last_idx:end_idx] = current_state
        current_state = 1.0 if spk == 'L' else 0.0
        last_idx = end_idx
        
        b_start = max(0, idx - PRE_SWITCH_SAMPLES)
        b_end = min(length, idx + POST_SWITCH_SAMPLES)
        mask_valid[b_start:b_end] = False
        
        if last_idx >= length:
            break
            
    if last_idx < length:
        mask_true[last_idx:] = current_state
        
    return mask_true, mask_valid

def extract_segments(mask_valid):
    padded = np.concatenate([[False], mask_valid, [False]])
    diff = np.diff(padded.astype(int))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    return list(zip(starts, ends))

def process_subject(cache_file):
    subj_name = cache_file.stem.split('_')[0]
    cached = torch.load(cache_file, map_location='cpu', weights_only=False)['raw']
    
    segments = []
    
    for tr_idx, tr in enumerate(cached):
        eeg_raw = tr['eeg'].numpy()[EAR_CHANNEL_INDICES, :] 
        min_len = eeg_raw.shape[1]
        
        eeg_f = apply_modulation_filter(eeg_raw, BROADBAND[0], BROADBAND[1], SR)
        eeg_f = (eeg_f - np.mean(eeg_f, axis=1, keepdims=True)) / (np.std(eeg_f, axis=1, keepdims=True) + 1e-8)
        
        mask_t, mask_v = get_masks(tr['meta'].get('switch_points', []), min_len)
        segs = extract_segments(mask_v)
        
        for start, end in segs:
            if (end - start) < MIN_SEGMENT_SAMPLES:
                continue
                
            label = 1 if mask_t[start] == 1.0 else 0 
            X_seg = eeg_f[:, start:end] # [C, T]
            
            segments.append({
                'X': X_seg,
                'label': label,
                'trial_idx': tr_idx
            })
            
    return subj_name, segments

def evaluate_uca_for_subject(subj_name, segments):
    if len(segments) < 10:
        return subj_name, None
        
    trials = np.array([seg['trial_idx'] for seg in segments])
    
    # ---------------------------------------------------------
    # 1. CALIBRATION (First 2 minutes)
    # ---------------------------------------------------------
    train_mask = trials < 2
    test_mask = trials >= 2
    
    if np.sum(train_mask) < 2 or np.sum(test_mask) < 2:
        return subj_name, None
        
    calib_segments = [segments[i] for i in range(len(segments)) if train_mask[i]]
    track_segments = [segments[i] for i in range(len(segments)) if test_mask[i]]
    
    # Compute base Covariances
    for seg in calib_segments:
        seg['cov'] = compute_covariance(seg['X'])
        
    # Reference Covariance (C_0)
    C_0 = np.mean([seg['cov'] for seg in calib_segments], axis=0)
    
    # Static Whitening (Baseline Phase 162)
    # In phase 162 we just trained on raw covariances, but let's align calibration to I
    A_0 = fractional_matrix_power(C_0, -0.5)
    
    X_train_cov = []
    y_train = []
    for seg in calib_segments:
        X_aligned = A_0 @ seg['X']
        cov_aligned = compute_covariance(X_aligned)
        cov_aligned += np.eye(cov_aligned.shape[0]) * 1e-5
        X_train_cov.append(cov_aligned)
        y_train.append(seg['label'])
        
    X_train_cov = np.array(X_train_cov)
    y_train = np.array(y_train)
    
    # Train fixed Classifier
    clf = make_pipeline(TangentSpace(), LogisticRegression(max_iter=1000))
    clf.fit(X_train_cov, y_train)
    
    # ---------------------------------------------------------
    # 2. ONLINE TRACKING (UCA)
    # ---------------------------------------------------------
    y_test = []
    y_pred_probs_uca = []
    y_preds_uca = []
    
    y_pred_probs_fixed = []
    y_preds_fixed = []
    
    C_t = C_0.copy()
    
    for seg in track_segments:
        X_test = seg['X']
        y_true = seg['label']
        y_test.append(y_true)
        
        # 2a. Fixed Baseline (No Covariance Updating - similar to Phase 162)
        X_fixed = A_0 @ X_test
        cov_fixed = compute_covariance(X_fixed)
        cov_fixed += np.eye(cov_fixed.shape[0]) * 1e-5
        prob_fixed = clf.predict_proba(np.expand_dims(cov_fixed, 0))[0, 1]
        pred_fixed = clf.predict(np.expand_dims(cov_fixed, 0))[0]
        y_pred_probs_fixed.append(prob_fixed)
        y_preds_fixed.append(pred_fixed)
        
        # 2b. Unsupervised Covariance Alignment (UCA)
        current_cov = compute_covariance(X_test)
        
        # Update running EMA covariance
        C_t = (1.0 - ALPHA) * C_t + ALPHA * current_cov
        
        # Compute online whitening transform
        A_t = fractional_matrix_power(C_t, -0.5)
        
        # Align test data
        X_aligned = A_t @ X_test
        cov_aligned = compute_covariance(X_aligned)
        cov_aligned += np.eye(cov_aligned.shape[0]) * 1e-5
        
        # Predict using FIXED classifier
        prob_uca = clf.predict_proba(np.expand_dims(cov_aligned, 0))[0, 1]
        pred_uca = clf.predict(np.expand_dims(cov_aligned, 0))[0]
        
        y_pred_probs_uca.append(prob_uca)
        y_preds_uca.append(pred_uca)
        
    # Metrics
    metrics = {
        'Fixed (Phase 162)': {
            'auc': roc_auc_score(y_test, y_pred_probs_fixed),
            'acc': accuracy_score(y_test, y_preds_fixed)
        },
        'UCA (Phase 163)': {
            'auc': roc_auc_score(y_test, y_pred_probs_uca),
            'acc': accuracy_score(y_test, y_preds_uca)
        }
    }
    
    return subj_name, metrics

def main():
    print("=======================================================")
    print(" PHASE 163: UNSUPERVISED COVARIANCE ALIGNMENT (UCA)")
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
    
    print("Extracting EEG segments chronologically...")
    start_time = time.time()
    
    results_all = {}
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=mp.cpu_count()) as executor:
        futures = {executor.submit(process_subject, cf): cf for cf in cache_files}
        all_segments = {}
        for future in concurrent.futures.as_completed(futures):
            subj, segs = future.result()
            all_segments[subj] = segs
            
    print(f"Extraction Time: {time.time() - start_time:.2f}s\n")
    print(f"Running Unsupervised Alignment Tracking (Alpha = {ALPHA})...\n")
    
    global_metrics = {
        'Fixed (Phase 162)': {'auc': [], 'acc': []},
        'UCA (Phase 163)': {'auc': [], 'acc': []}
    }
    
    subjects_sorted = sorted(all_segments.keys())
    
    for subj in subjects_sorted:
        _, metrics = evaluate_uca_for_subject(subj, all_segments[subj])
        if metrics is None:
            continue
            
        print(f"--- Subject: {subj} ---")
        for name, m in metrics.items():
            print(f"  {name:<18} | AUROC: {m['auc']:.3f} | Acc: {m['acc']:.3f}")
            global_metrics[name]['auc'].append(m['auc'])
            global_metrics[name]['acc'].append(m['acc'])
        print()
        
    print("\n=======================================================")
    print(" GLOBAL UCA TRACKING AVERAGES")
    print("=======================================================")
    for name in global_metrics.keys():
        m_auc = np.mean(global_metrics[name]['auc'])
        m_acc = np.mean(global_metrics[name]['acc'])
        print(f"{name:<18} | Mean AUROC: {m_auc:.3f} | Mean Acc: {m_acc:.3f}")

if __name__ == '__main__':
    main()
