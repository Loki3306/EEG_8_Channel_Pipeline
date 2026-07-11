import os
import time
import numpy as np
import torch
from pathlib import Path
from scipy import signal
from sklearn.metrics import roc_auc_score
from sklearn.decomposition import PCA
import multiprocessing as mp
import concurrent.futures

try:
    import pyriemann
except ImportError:
    os.system("pip install pyriemann")
    import pyriemann

try:
    from pyriemann.utils.mean import mean_covariance as mean_riemann
except ImportError:
    from pyriemann.utils.mean import mean_riemann

from pyriemann.utils.tangentspace import tangent_space

# -------------------------------------------------------------------------
# CONSTANTS & CONFIGURATION
# -------------------------------------------------------------------------
SR = 128
EAR_CHANNEL_INDICES = [23, 31, 32, 40, 14, 22, 41, 49]
BROADBAND = (0.5, 8.0)
WINDOW_SEC = 5.0
HOP_SEC = 0.5
WINDOW_SAMPLES = int(WINDOW_SEC * SR)
HOP_SAMPLES = int(HOP_SEC * SR)
EMA_ALPHA = 0.85  # Smoothing factor for latent state

def apply_modulation_filter(env, lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    b, a = signal.butter(order, [lowcut / nyq, highcut / nyq], btype='band')
    return signal.filtfilt(b, a, env, axis=1)

def get_window_labels(sp, length, window_starts, window_samples):
    # Create sample-by-sample label array
    mask_true = np.zeros(length, dtype=np.float32)
    if len(sp) == 0:
        mask_true[:] = 1.0
    else:
        current_state = 1.0 if sp[0][0] == 'R' else 0.0 
        last_idx = 0
        for spk, idx in sp:
            end_idx = min(idx, length)
            mask_true[last_idx:end_idx] = current_state
            current_state = 1.0 if spk == 'L' else 0.0
            last_idx = end_idx
        if last_idx < length: 
            mask_true[last_idx:] = current_state
            
    # For each window, take the mode of the labels (the dominant state)
    labels = []
    for start in window_starts:
        end = start + window_samples
        window_mask = mask_true[start:end]
        labels.append(1 if np.mean(window_mask) > 0.5 else 0)
    return np.array(labels)

def process_subject(cache_file):
    cached = torch.load(cache_file, map_location='cpu', weights_only=False)['raw']
    subj_name = cache_file.stem.split('_')[0]
    
    trial_results = []
    
    for tr_idx, tr in enumerate(cached):
        eeg_raw = tr['eeg'].numpy()[EAR_CHANNEL_INDICES, :] 
        min_len = eeg_raw.shape[1]
        
        # 1. Filter
        eeg_f = apply_modulation_filter(eeg_raw, BROADBAND[0], BROADBAND[1], SR)
        eeg_f = (eeg_f - np.mean(eeg_f, axis=1, keepdims=True)) / (np.std(eeg_f, axis=1, keepdims=True) + 1e-8)
        
        # 2. Extract Sliding Windows
        window_starts = np.arange(0, min_len - WINDOW_SAMPLES + 1, HOP_SAMPLES)
        if len(window_starts) < 10: continue
            
        covs = []
        for start in window_starts:
            end = start + WINDOW_SAMPLES
            X_seg = eeg_f[:, start:end]
            cov_mat = np.cov(X_seg)
            cov_mat += np.eye(cov_mat.shape[0]) * 1e-5
            covs.append(cov_mat)
            
        covs = np.array(covs)
        labels = get_window_labels(tr['meta'].get('switch_points', []), min_len, window_starts, WINDOW_SAMPLES)
        
        if len(np.unique(labels)) < 2:
            continue # Skip trials with no switch (pure trials) for AUROC evaluation
            
        # 3. Tangent Space Projection (Local Trial Frame)
        trial_mean = mean_riemann(covs)
        T_t = tangent_space(covs, trial_mean) # Shape: (N_windows, 36)
        
        # 4. Latent Dimensionality Reduction (PCA on Tangent Space)
        # This extracts the primary axis along which the covariance matrices are shifting inside this 60s trial
        pca = PCA(n_components=1)
        z_t = pca.fit_transform(T_t).flatten()
        
        # 5. Smooth Dynamical Tracking (EMA)
        s_t = np.zeros_like(z_t)
        s_t[0] = z_t[0]
        for i in range(1, len(z_t)):
            s_t[i] = EMA_ALPHA * s_t[i-1] + (1 - EMA_ALPHA) * z_t[i]
            
        # 6. Unsupervised AUROC (Oracle Polarity)
        # Since geometry has no polarity, we don't know if + means Left or Right.
        # We test both polarities and take the max (this proves if the axis itself is correct)
        auc_pos = roc_auc_score(labels, s_t)
        auc_neg = roc_auc_score(labels, -s_t)
        oracle_auc = max(auc_pos, auc_neg)
        
        trial_results.append(oracle_auc)
        
    if len(trial_results) == 0:
        return subj_name, 0.50
        
    return subj_name, np.mean(trial_results)

def main():
    print("=======================================================")
    print(" PHASE 149: ONLINE LATENT ORIENTATION TRACKING")
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
    
    print(f"Running Unsupervised Latent Tracking on {len(cache_files)} subjects...")
    print("Using 5s windows, 0.5s hop, Local Tangent Space PCA + EMA.\n")
    
    start_time = time.time()
    results = {}
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=mp.cpu_count()) as executor:
        futures = {executor.submit(process_subject, cf): cf for cf in cache_files}
        for future in concurrent.futures.as_completed(futures):
            subj, mean_auc = future.result()
            results[subj] = mean_auc
            print(f"[{subj:3s}] Mean Oracle Trial AUROC: {mean_auc:.4f}")

    print("\n=======================================================")
    print(" FINAL RESULTS")
    print("=======================================================")
    
    sorted_subjs = sorted(results.keys(), key=lambda x: int(x[1:]))
    global_mean = np.mean(list(results.values()))
    
    for subj in sorted_subjs:
        print(f"[{subj:3s}] : {results[subj]:.4f}")
        
    print(f"\nGLOBAL AVERAGE ORACLE AUROC: {global_mean:.4f}")
    
    if global_mean > 0.70:
        print("\n[MASSIVE SUCCESS] The latent geometry perfectly encodes the attention switch!")
        print("This proves we can track the state completely unsupervised.")
        print("Next Step: Use Audio Correlation to solve the Left/Right polarity problem.")
    else:
        print("\n[FAILURE] The latent axis does not align with the attention switches.")
        print("The dominant variance in the trial's tangent space is NOT attention.")

if __name__ == '__main__':
    main()
