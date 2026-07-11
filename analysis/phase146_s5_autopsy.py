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
from pyriemann.utils.tangentspace import tangent_space

# -------------------------------------------------------------------------
# CONSTANTS & CONFIGURATION
# -------------------------------------------------------------------------
SR = 128
EAR_CHANNEL_INDICES = [23, 31, 32, 40, 14, 22, 41, 49]
BROADBAND = (0.5, 8.0)
PRE_SWITCH_SAMPLES = int(0.5 * SR)
POST_SWITCH_SAMPLES = int(1.0 * SR)
MIN_SEGMENT_SAMPLES = int(3.0 * SR)

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

def get_cosine_similarity(v1, v2):
    norm = np.linalg.norm(v1) * np.linalg.norm(v2)
    return np.dot(v1, v2) / (norm + 1e-8)

def main():
    print("=======================================================")
    print(" PHASE 146 (REVISED): ADVANCED REPRESENTATION FORENSICS")
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
            
    results = {}
    
    for subj in TARGET_SUBJECTS:
        cache_file = cache_dir / f"{subj}_multiband.pt"
        if not cache_file.exists(): continue
            
        segments = process_subject(cache_file)
        if len(segments) < 10: continue
            
        covs_all = np.array([s['cov'] for s in segments])
        covs_L = np.array([s['cov'] for s in segments if s['label'] == 1])
        covs_R = np.array([s['cov'] for s in segments if s['label'] == 0])
        
        global_mean = mean_riemann(covs_all)
        mean_L = mean_riemann(covs_L)
        mean_R = mean_riemann(covs_R)
        
        # 1. Riemannian SNR
        dist_between = distance_riemann(mean_L, mean_R)
        var_L = variance_riemann(covs_L, mean_L)
        var_R = variance_riemann(covs_R, mean_R)
        var_within = (var_L + var_R) / 2.0
        snr = dist_between / (var_within + 1e-8)
        
        # 2. Principal Eigenvector Rotation
        wL, vL = np.linalg.eigh(mean_L)
        wR, vR = np.linalg.eigh(mean_R)
        top_eig_L = vL[:, np.argmax(wL)]
        top_eig_R = vR[:, np.argmax(wR)]
        
        # Absolute dot product to handle sign ambiguity
        cos_angle = np.clip(np.abs(np.dot(top_eig_L, top_eig_R)), 0.0, 1.0)
        angle_deg = np.degrees(np.arccos(cos_angle))
        
        # 3. Separation Direction Consistency (Tangent Space)
        trials = np.unique([s['trial_idx'] for s in segments])
        trial_deltas = []
        
        for tr in trials:
            tr_covs_L = np.array([s['cov'] for s in segments if s['label'] == 1 and s['trial_idx'] == tr])
            tr_covs_R = np.array([s['cov'] for s in segments if s['label'] == 0 and s['trial_idx'] == tr])
            
            if len(tr_covs_L) > 0 and len(tr_covs_R) > 0:
                tr_mean_L = mean_riemann(tr_covs_L)
                tr_mean_R = mean_riemann(tr_covs_R)
                
                # Project onto global tangent space
                # tangent_space expects shape (n_matrices, channels, channels)
                # returns shape (n_matrices, channels * (channels + 1) / 2)
                vec_L = tangent_space(np.array([tr_mean_L]), global_mean)[0]
                vec_R = tangent_space(np.array([tr_mean_R]), global_mean)[0]
                
                delta_vector = vec_L - vec_R
                trial_deltas.append(delta_vector)
                
        # Compute mean pairwise cosine similarity of displacement vectors
        cosine_sims = []
        n_deltas = len(trial_deltas)
        for i in range(n_deltas):
            for j in range(i + 1, n_deltas):
                sim = get_cosine_similarity(trial_deltas[i], trial_deltas[j])
                cosine_sims.append(sim)
                
        mean_directional_consistency = np.mean(cosine_sims) if cosine_sims else 0.0
        
        results[subj] = {
            'n_segments': len(segments),
            'snr': snr,
            'var_within': var_within,
            'angle_deg': angle_deg,
            'dir_consistency': mean_directional_consistency
        }
        
    print("\n=======================================================")
    print(" ADVANCED AUTOPSY RESULTS")
    print("=======================================================")
    
    for subj in TARGET_SUBJECTS:
        if subj not in results: continue
        res = results[subj]
        print(f"\n--- {subj} ---")
        print(f"Total Segments:               {res['n_segments']}")
        print(f"Riemannian SNR (10/10):       {res['snr']:.4f}")
        print(f"Within-Class Compactness:     {res['var_within']:.4f} (Lower = Tighter Clusters)")
        print(f"Principal Dipole Rotation:    {res['angle_deg']:.2f} degrees")
        print(f"Directional Consistency:      {res['dir_consistency']:.4f} (Higher = Stable L-R direction across trials)")
        
    if 'S5' in results and 'S10' in results:
        print("\n=======================================================")
        print(" FINAL SCIENTIFIC CONCLUSION")
        print("=======================================================")
        dir_ratio = results['S5']['dir_consistency'] / (results['S10']['dir_consistency'] + 1e-8)
        print(f"S5 has {dir_ratio:.1f}x higher Directional Consistency than S10.")
        
        if results['S10']['dir_consistency'] < 0.1:
            print("\n[CRITICAL FAILURE MODE DETECTED IN S10]")
            print("S10's Directional Consistency is near zero. This means the 'Left vs Right' displacement")
            print("vector in the brain completely rotates randomly every single trial.")
            print("A static spatial classifier (like CSP or Tangent-Space) is mathematically guaranteed")
            print("to fail on S10 because it tries to draw a single, fixed boundary.")
            print("To solve AAD for the general population, we must build DYNAMIC ADAPTIVE classifiers")
            print("that recalculate the displacement vector on the fly (e.g., Unsupervised CCA or Phase-Locked adaptation).")
            
if __name__ == '__main__':
    main()
