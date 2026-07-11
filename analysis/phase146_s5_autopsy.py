import os
import time
import numpy as np
import torch
from pathlib import Path
from scipy import signal
import matplotlib.pyplot as plt

try:
    import pyriemann
except ImportError:
    import os
    os.system("pip install pyriemann")
    import pyriemann

from pyriemann.utils.distance import distance_riemann
from pyriemann.utils.mean import mean_riemann
from pyriemann.utils.variance import variance_riemann

# -------------------------------------------------------------------------
# CONSTANTS & CONFIGURATION
# -------------------------------------------------------------------------
SR = 128
EAR_CHANNEL_INDICES = [23, 31, 32, 40, 14, 22, 41, 49]
BROADBAND = (0.5, 8.0)
PRE_SWITCH_SAMPLES = int(0.5 * SR)
POST_SWITCH_SAMPLES = int(1.0 * SR)
MIN_SEGMENT_SAMPLES = int(3.0 * SR)

# The Subjects we want to autopsy
TARGET_SUBJECTS = ['S5', 'S12', 'S10'] 

def apply_modulation_filter(env, lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    b, a = signal.butter(order, [lowcut / nyq, highcut / nyq], btype='band')
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
        if last_idx >= length: break
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
            if (end - start) < MIN_SEGMENT_SAMPLES: continue
            label = 1 if mask_t[start] == 1.0 else 0
            X_seg = eeg_f[:, start:end]
            cov_mat = np.cov(X_seg)
            cov_mat += np.eye(cov_mat.shape[0]) * 1e-5
            segments.append({'cov': cov_mat, 'label': label, 'trial_idx': tr_idx})
    return segments

def main():
    print("=======================================================")
    print(" PHASE 146: THE SUBJECT 5 AUTOPSY (FORENSICS)")
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
            
    print(f"Loading data for subjects: {TARGET_SUBJECTS}")
    
    results = {}
    
    for subj in TARGET_SUBJECTS:
        cache_file = cache_dir / f"{subj}_multiband.pt"
        if not cache_file.exists():
            print(f"Warning: Could not find data for {subj}. Skipping.")
            continue
            
        segments = process_subject(cache_file)
        if len(segments) < 10:
            continue
            
        covs_L = np.array([s['cov'] for s in segments if s['label'] == 1])
        covs_R = np.array([s['cov'] for s in segments if s['label'] == 0])
        
        # 1. Eigenspectrum & Condition Number
        mean_L = mean_riemann(covs_L)
        mean_R = mean_riemann(covs_R)
        
        eigvals_L = np.sort(np.linalg.eigvalsh(mean_L))[::-1]
        eigvals_R = np.sort(np.linalg.eigvalsh(mean_R))[::-1]
        
        cond_L = eigvals_L[0] / (eigvals_L[-1] + 1e-8)
        cond_R = eigvals_R[0] / (eigvals_R[-1] + 1e-8)
        
        # 2. Riemannian Signal-to-Noise Ratio (SNR)
        dist_between = distance_riemann(mean_L, mean_R)
        
        var_L = variance_riemann(covs_L, mean_L)
        var_R = variance_riemann(covs_R, mean_R)
        var_within = (var_L + var_R) / 2.0
        
        snr = dist_between / (var_within + 1e-8)
        
        # 3. Trial-to-Trial Stability
        trials = np.unique([s['trial_idx'] for s in segments])
        trial_means_L = []
        for tr in trials:
            tr_covs = np.array([s['cov'] for s in segments if s['label'] == 1 and s['trial_idx'] == tr])
            if len(tr_covs) > 0:
                trial_means_L.append(mean_riemann(tr_covs))
                
        if len(trial_means_L) > 1:
            global_mean_L = mean_riemann(np.array(trial_means_L))
            trial_stability_variance = variance_riemann(np.array(trial_means_L), global_mean_L)
        else:
            trial_stability_variance = 0.0
            
        results[subj] = {
            'eig_L': eigvals_L,
            'eig_R': eigvals_R,
            'cond': (cond_L + cond_R)/2.0,
            'dist_between': dist_between,
            'var_within': var_within,
            'snr': snr,
            'trial_stability_var': trial_stability_variance,
            'n_segments': len(segments)
        }
        
    print("\n=======================================================")
    print(" AUTOPSY RESULTS")
    print("=======================================================")
    
    for subj in TARGET_SUBJECTS:
        if subj not in results: continue
        res = results[subj]
        print(f"\n--- {subj} ---")
        print(f"Total Segments:       {res['n_segments']}")
        print(f"Condition Number:     {res['cond']:.2f}")
        print(f"Distance (Between):   {res['dist_between']:.4f}")
        print(f"Variance (Within):    {res['var_within']:.4f}")
        print(f"Riemannian SNR:       {res['snr']:.4f}  <-- THE KEY METRIC")
        print(f"Trial Drift Var:      {res['trial_stability_var']:.4f} (Lower = More stable across trials)")
        print(f"Top 3 Eigenvalues(L): {res['eig_L'][0]:.4f}, {res['eig_L'][1]:.4f}, {res['eig_L'][2]:.4f}")
        
    # Quick Conclusion Print
    print("\n=======================================================")
    print(" FORENSIC ANALYSIS SUMMARY")
    print("=======================================================")
    if 'S5' in results and 'S10' in results:
        snr_ratio = results['S5']['snr'] / (results['S10']['snr'] + 1e-8)
        cond_ratio = results['S5']['cond'] / (results['S10']['cond'] + 1e-8)
        drift_ratio = results['S10']['trial_stability_var'] / (results['S5']['trial_stability_var'] + 1e-8)
        
        print(f"1. Signal-to-Noise: S5 has {snr_ratio:.1f}x higher Riemannian SNR than S10.")
        print(f"2. Condition Num:   S5's matrices are {cond_ratio:.1f}x compared to S10.")
        print(f"3. Trial Drift:     S10 drifts {drift_ratio:.1f}x more across trials than S5.")
        
        print("\nINTERPRETATION:")
        if snr_ratio > 2.0:
            print("S5 simply has a radically cleaner spatial signal. The attention effect size is massively larger.")
        if drift_ratio > 2.0:
            print("S10's failure is likely due to 'Spatial Drift' - their brain geometry changes wildly over time, confusing the classifier.")
            
if __name__ == '__main__':
    main()
