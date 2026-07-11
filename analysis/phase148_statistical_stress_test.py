import os
import time
import numpy as np
import torch
from pathlib import Path
from scipy import signal
import scipy.stats as stats
from sklearn.linear_model import HuberRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import multiprocessing as mp
import concurrent.futures

try:
    import pyriemann
except ImportError:
    os.system("pip install pyriemann")
    import pyriemann

try:
    from pyriemann.utils.distance import distance_riemann
    from pyriemann.utils.mean import mean_riemann
except ImportError:
    from pyriemann.utils.distance import distance as distance_riemann
    from pyriemann.utils.mean import mean_covariance as mean_riemann

def variance_riemann(covmats, Cref):
    return np.mean([distance_riemann(C, Cref)**2 for C in covmats])

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

# Phase 145 AUROC scores (Tangent-Space + LR) for the 18 subjects
AUROC_SCORES = {
    'S12': 0.661, 'S13': 0.547, 'S11': 0.532, 'S10': 0.535,
    'S15': 0.498, 'S14': 0.456, 'S17': 0.601, 'S16': 0.482,
    'S18': 0.677, 'S2':  0.613, 'S1':  0.571, 'S4':  0.494,
    'S3':  0.505, 'S7':  0.515, 'S6':  0.620, 'S8':  0.577,
    'S5':  0.816, 'S9':  0.598
}

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
    if last_idx < length: mask_true[last_idx:] = current_state
    return mask_true, mask_valid

def extract_segments(mask_valid):
    padded = np.concatenate([[False], mask_valid, [False]])
    diff = np.diff(padded.astype(int))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    return list(zip(starts, ends))

def get_cosine_similarity(v1, v2):
    norm = np.linalg.norm(v1) * np.linalg.norm(v2)
    return np.dot(v1, v2) / (norm + 1e-8)

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

def analyze_subject(subj_name, segments):
    if len(segments) < 10: return None
    
    covs_all = np.array([s['cov'] for s in segments])
    covs_L = np.array([s['cov'] for s in segments if s['label'] == 1])
    covs_R = np.array([s['cov'] for s in segments if s['label'] == 0])
    
    global_mean = mean_riemann(covs_all)
    mean_L = mean_riemann(covs_L)
    mean_R = mean_riemann(covs_R)
    
    dist_between = distance_riemann(mean_L, mean_R)
    var_L = variance_riemann(covs_L, mean_L)
    var_R = variance_riemann(covs_R, mean_R)
    var_within = (var_L + var_R) / 2.0
    snr = dist_between / (var_within + 1e-8)
    
    trials = np.unique([s['trial_idx'] for s in segments])
    trial_deltas = []
    
    for tr in trials:
        tr_covs_L = np.array([s['cov'] for s in segments if s['label'] == 1 and s['trial_idx'] == tr])
        tr_covs_R = np.array([s['cov'] for s in segments if s['label'] == 0 and s['trial_idx'] == tr])
        if len(tr_covs_L) > 0 and len(tr_covs_R) > 0:
            tr_mean_L = mean_riemann(tr_covs_L)
            tr_mean_R = mean_riemann(tr_covs_R)
            vec_L = tangent_space(np.array([tr_mean_L]), global_mean)[0]
            vec_R = tangent_space(np.array([tr_mean_R]), global_mean)[0]
            trial_deltas.append(vec_L - vec_R)
            
    cosine_sims = []
    n_deltas = len(trial_deltas)
    for i in range(n_deltas):
        for j in range(i + 1, n_deltas):
            cosine_sims.append(get_cosine_similarity(trial_deltas[i], trial_deltas[j]))
    dir_consistency = np.mean(cosine_sims) if cosine_sims else 0.0

    return {
        'subj': subj_name,
        'auroc': AUROC_SCORES.get(subj_name, 0.50),
        'snr': snr,
        'dir_consist': dir_consistency
    }

def main():
    print("=======================================================")
    print(" PHASE 148: STATISTICAL STRESS-TEST")
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
    
    print(f"Extracting Covariance Metrics for {len(cache_files)} subjects...")
    results = []
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=mp.cpu_count()) as executor:
        futures = {executor.submit(process_subject, cf): cf.stem.split('_')[0] for cf in cache_files}
        for future in concurrent.futures.as_completed(futures):
            subj = futures[future]
            segments = future.result()
            res = analyze_subject(subj, segments)
            if res:
                results.append(res)
                
    # Sort for consistent arrays
    results = sorted(results, key=lambda x: x['subj'])
    aurocs = np.array([r['auroc'] for r in results])
    dir_consists = np.array([r['dir_consist'] for r in results])
    subjs = np.array([r['subj'] for r in results])
    
    # ---------------------------------------------------------
    # TEST 1: Spearman Rank Correlation (Immune to outliers)
    # ---------------------------------------------------------
    r_pearson, p_pearson = stats.pearsonr(dir_consists, aurocs)
    r_spearman, p_spearman = stats.spearmanr(dir_consists, aurocs)
    
    print("\n--- TEST 1: SPEARMAN RANK CORRELATION ---")
    print(f"Pearson r  (Linear): {r_pearson:.4f} (p={p_pearson:.5f})")
    print(f"Spearman r (Rank):   {r_spearman:.4f} (p={p_spearman:.5f})")
    if r_spearman > 0.7:
        print("[PASS] The correlation is robust to extreme outliers (like S5).")
    else:
        print("[FAIL] The high correlation was an artifact of extreme outliers.")
        
    # ---------------------------------------------------------
    # TEST 2: Leave-S5-Out Pearson
    # ---------------------------------------------------------
    s5_mask = subjs != 'S5'
    aurocs_no_s5 = aurocs[s5_mask]
    dir_consists_no_s5 = dir_consists[s5_mask]
    
    r_no_s5, p_no_s5 = stats.pearsonr(dir_consists_no_s5, aurocs_no_s5)
    
    print("\n--- TEST 2: LEAVE-S5-OUT CORRELATION ---")
    print(f"Pearson r (excluding S5): {r_no_s5:.4f} (p={p_no_s5:.5f})")
    if r_no_s5 > 0.6:
        print("[PASS] The relationship holds strong even without our Golden Subject.")
    else:
        print("[FAIL] S5 was artificially inflating the regression line.")

    # ---------------------------------------------------------
    # TEST 3: Leave-One-Subject-Out (LOSO) Robust Regression
    # ---------------------------------------------------------
    print("\n--- TEST 3: LOSO AUROC PREDICTION ---")
    
    predicted_aurocs = np.zeros_like(aurocs)
    
    for i in range(len(subjs)):
        # Train on 17 subjects, predict the 18th
        train_mask = np.ones(len(subjs), dtype=bool)
        train_mask[i] = False
        
        X_train = dir_consists[train_mask].reshape(-1, 1)
        y_train = aurocs[train_mask]
        X_test = dir_consists[i].reshape(-1, 1)
        
        # HuberRegressor is robust to outliers in the training set
        reg = HuberRegressor()
        reg.fit(X_train, y_train)
        
        pred = reg.predict(X_test)[0]
        predicted_aurocs[i] = pred
        print(f"Subj {subjs[i]:3s} | True AUROC: {aurocs[i]:.3f} | Predicted: {pred:.3f} | Error: {abs(aurocs[i]-pred):.3f}")
        
    mae = mean_absolute_error(aurocs, predicted_aurocs)
    rmse = np.sqrt(mean_squared_error(aurocs, predicted_aurocs))
    
    print(f"\nOverall Prediction MAE:  {mae:.4f} AUROC")
    print(f"Overall Prediction RMSE: {rmse:.4f} AUROC")
    
    if mae < 0.05:
        print("[PASS] We can accurately predict a subject's decoder performance using ONLY their Directional Consistency!")
    else:
        print("[FAIL] Directional Consistency is not sufficient on its own to predict test performance.")
        
    print("\n=======================================================")
    print(" SCIENTIFIC CONCLUSION")
    print("=======================================================")
    if r_spearman > 0.7 and r_no_s5 > 0.6 and mae < 0.05:
        print("Directional Consistency has SURVIVED the statistical stress test.")
        print("It is a universal law of Ear-EEG: Static classifiers fail because the discriminative subspace rotates.")
        print("We must proceed to Phase 149: Online Latent Subspace Tracking.")
    else:
        print("Directional Consistency FAILED the stress test.")
        print("S5 was likely an artifact. We must find another explanatory variable.")

if __name__ == '__main__':
    main()
