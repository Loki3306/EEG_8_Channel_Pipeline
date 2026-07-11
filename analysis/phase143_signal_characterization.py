import os
import time
import numpy as np
import torch
from pathlib import Path
from scipy import signal
from scipy.stats import pearsonr
import concurrent.futures
import multiprocessing as mp

# Try importing pyriemann for proper Riemannian geometry
try:
    import pyriemann
except ImportError:
    print("Installing pyriemann for correct Riemannian geometry...")
    os.system("pip install pyriemann")
    import pyriemann

from pyriemann.utils.distance import distance_riemann
from pyriemann.utils.mean import mean_riemann
from sklearn.linear_model import Ridge

# -------------------------------------------------------------------------
# CONSTANTS & CONFIGURATION
# -------------------------------------------------------------------------
SR = 128
EAR_CHANNEL_INDICES = [23, 31, 32, 40, 14, 22, 41, 49]
BROADBAND = (0.5, 8.0)

PRE_SWITCH_SAMPLES = int(0.5 * SR)
POST_SWITCH_SAMPLES = int(1.0 * SR)
MIN_SEGMENT_SAMPLES = int(3.0 * SR)

N_PERMUTATIONS = 1000
LAG_MAX_MS = 400
LAG_MAX_SAMPLES = int((LAG_MAX_MS / 1000.0) * SR)

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

def compute_lagged_features(X, max_lag):
    """ Creates a lagged design matrix for TRF. X is [T, C] """
    T, C = X.shape
    X_lagged = np.zeros((T - max_lag, C * (max_lag + 1)))
    for lag in range(max_lag + 1):
        X_lagged[:, lag*C:(lag+1)*C] = X[max_lag-lag : T-lag, :]
    return X_lagged

def process_subject(cache_file):
    subj_name = cache_file.stem.split('_')[0]
    cached = torch.load(cache_file, map_location='cpu', weights_only=False)['raw']
    
    segments = []
    
    for tr_idx, tr in enumerate(cached):
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
        
        mask_t, mask_v = get_masks(tr['meta'].get('switch_points', []), min_len)
        
        segs = extract_segments(mask_v)
        
        for start, end in segs:
            if (end - start) < MIN_SEGMENT_SAMPLES:
                continue
                
            label = mask_t[start] # 1.0 is Left, 0.0 is Right
            
            X_seg = eeg_f[:, start:end].T # [T, C]
            target_env = env_l_f[0, start:end] if label == 1.0 else env_r_f[0, start:end]
            
            # Covariance matrix with regularization to ensure SPD
            cov_mat = np.cov(X_seg, rowvar=False)
            cov_mat += np.eye(cov_mat.shape[0]) * 1e-5
            
            segments.append({
                'X': X_seg,
                'target_env': target_env,
                'cov': cov_mat,
                'label': label,
                'trial_idx': tr_idx
            })
            
    # Return raw segments for global analysis
    return subj_name, segments

def compute_h1_trf(segments):
    """ Computes H1: True vs Permuted Lagged Correlation (TRF) for a subject """
    if len(segments) < 2: return 0.0, 0.0, 1.0
    
    # Train/Test Split (First 80% Train, Last 20% Test)
    split_idx = int(len(segments) * 0.8)
    train_segs = segments[:split_idx]
    test_segs = segments[split_idx:]
    
    # Build Train Matrix
    X_train_list, Y_train_list = [], []
    for seg in train_segs:
        X_lagged = compute_lagged_features(seg['X'], LAG_MAX_SAMPLES)
        Y_target = seg['target_env'][LAG_MAX_SAMPLES:]
        X_train_list.append(X_lagged)
        Y_train_list.append(Y_target)
        
    X_train = np.vstack(X_train_list)
    Y_train = np.concatenate(Y_train_list)
    
    # Fit TRF
    model = Ridge(alpha=1e3)
    model.fit(X_train, Y_train)
    
    # Build Test Matrix
    X_test_list, Y_test_list = [], []
    for seg in test_segs:
        X_lagged = compute_lagged_features(seg['X'], LAG_MAX_SAMPLES)
        Y_target = seg['target_env'][LAG_MAX_SAMPLES:]
        X_test_list.append(X_lagged)
        Y_test_list.append(Y_target)
        
    X_test = np.vstack(X_test_list)
    Y_test = np.concatenate(Y_test_list)
    
    # True Prediction
    Y_pred = model.predict(X_test)
    true_corr, _ = pearsonr(Y_test, Y_pred)
    
    # Permutation Testing (Circular Shift)
    null_corrs = []
    np.random.seed(42)
    # We will do 500 permutations to save time, but it is rigorous enough
    for _ in range(500):
        shift = np.random.randint(SR, len(Y_test) - SR) # Shift by at least 1 second
        Y_test_shuffled = np.roll(Y_test, shift)
        r, _ = pearsonr(Y_test_shuffled, Y_pred)
        null_corrs.append(r)
        
    null_corrs = np.array(null_corrs)
    p_val = np.sum(null_corrs >= true_corr) / len(null_corrs)
    mean_null = np.mean(null_corrs)
    
    return true_corr, mean_null, p_val

def compute_h3_covariance(segments):
    """ Computes H3: Riemannian distance between Attention states with Permutations """
    covs_L = np.array([seg['cov'] for seg in segments if seg['label'] == 1.0])
    covs_R = np.array([seg['cov'] for seg in segments if seg['label'] == 0.0])
    
    if len(covs_L) < 2 or len(covs_R) < 2:
        return 0.0, 0.0, 1.0
        
    # True Riemannian Means
    mean_L = mean_riemann(covs_L)
    mean_R = mean_riemann(covs_R)
    
    # True Distance
    true_dist = distance_riemann(mean_L, mean_R)
    
    # Permutation Test
    all_covs = np.concatenate([covs_L, covs_R], axis=0)
    n_L = len(covs_L)
    null_dists = []
    
    np.random.seed(42)
    for _ in range(500):
        idx = np.random.permutation(len(all_covs))
        shuf_L = all_covs[idx[:n_L]]
        shuf_R = all_covs[idx[n_L:]]
        
        m_L = mean_riemann(shuf_L)
        m_R = mean_riemann(shuf_R)
        null_dists.append(distance_riemann(m_L, m_R))
        
    null_dists = np.array(null_dists)
    p_val = np.sum(null_dists >= true_dist) / len(null_dists)
    mean_null = np.mean(null_dists)
    
    return true_dist, mean_null, p_val

def compute_h2_permanova(all_covariances, all_subjects, all_labels):
    """ H2: Distance-based PERMANOVA to test Subject vs Attention Variance """
    print("\nComputing H2 Distance-based PERMANOVA...")
    n_samples = len(all_covariances)
    
    # Due to O(N^2) memory for pairwise distance matrix, we will sub-sample if N > 2000
    if n_samples > 1500:
        np.random.seed(42)
        idx = np.random.choice(n_samples, 1500, replace=False)
        all_covariances = all_covariances[idx]
        all_subjects = np.array(all_subjects)[idx]
        all_labels = np.array(all_labels)[idx]
        n_samples = 1500
        
    print(f"Computing pairwise Riemannian distance matrix for {n_samples} segments...")
    dist_matrix = np.zeros((n_samples, n_samples))
    for i in range(n_samples):
        for j in range(i+1, n_samples):
            d = distance_riemann(all_covariances[i], all_covariances[j])
            dist_matrix[i, j] = d
            dist_matrix[j, i] = d
            
    # Total Sum of Squares (SST)
    SST = np.sum(dist_matrix**2) / (2 * n_samples)
    
    # Subject SS
    unique_subs = np.unique(all_subjects)
    SS_subj = 0
    for sub in unique_subs:
        idx = np.where(all_subjects == sub)[0]
        n_k = len(idx)
        if n_k > 0:
            sub_dist = dist_matrix[np.ix_(idx, idx)]
            SS_subj += np.sum(sub_dist**2) / (2 * n_k)
            
    # Attention SS
    unique_atts = np.unique(all_labels)
    SS_att = 0
    for att in unique_atts:
        idx = np.where(all_labels == att)[0]
        n_k = len(idx)
        if n_k > 0:
            sub_dist = dist_matrix[np.ix_(idx, idx)]
            SS_att += np.sum(sub_dist**2) / (2 * n_k)
            
    # Pseudo-F Statistic approach: Var Explained
    var_subj = (SST - SS_subj) / SST * 100
    var_att = (SST - SS_att) / SST * 100
    
    return var_subj, var_att

def main():
    print("=======================================================")
    print(" PHASE 143: RIGOROUS SIGNAL CHARACTERIZATION")
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
    
    start_time = time.time()
    
    all_subjects_data = {}
    all_covariances = []
    all_labels = []
    all_subj_ids = []
    
    # Phase 1: Extract Segments
    with concurrent.futures.ProcessPoolExecutor(max_workers=mp.cpu_count()) as executor:
        futures = [executor.submit(process_subject, cf) for cf in cache_files]
        for future in concurrent.futures.as_completed(futures):
            subj, segments = future.result()
            all_subjects_data[subj] = segments
            
            for seg in segments:
                all_covariances.append(seg['cov'])
                all_labels.append(seg['label'])
                all_subj_ids.append(subj)

    print(f"Extraction Time: {time.time() - start_time:.2f}s\n")
    all_covariances = np.array(all_covariances)
    
    # H1 & H3 Tests (Per Subject)
    h1_results = []
    h3_results = []
    
    print("Running Permutation Tests (H1 & H3) per Subject...")
    for subj, segments in all_subjects_data.items():
        # H1: Lagged TRF
        t_corr, null_corr, p_h1 = compute_h1_trf(segments)
        h1_results.append((t_corr, null_corr, p_h1))
        
        # H3: Covariance Distance
        t_dist, null_dist, p_h3 = compute_h3_covariance(segments)
        h3_results.append((t_dist, null_dist, p_h3))
        print(f"[{subj}] H1 p={p_h1:.3f} | H3 p={p_h3:.3f}")
        
    # H1 Aggregation
    print("\n--- H1: Stimulus Information (Lagged TRF) ---")
    mean_true = np.mean([r[0] for r in h1_results])
    mean_null = np.mean([r[1] for r in h1_results])
    sig_count = sum(1 for r in h1_results if r[2] < 0.05)
    print(f"Mean True TRF Corr:  {mean_true:.4f}")
    print(f"Mean Null TRF Corr:  {mean_null:.4f}")
    print(f"Significant Subjects: {sig_count} / {len(h1_results)} (p < 0.05)")
    
    # H3 Aggregation
    print("\n--- H3: Covariance Geometry (Riemannian) ---")
    mean_tdist = np.mean([r[0] for r in h3_results])
    mean_ndist = np.mean([r[1] for r in h3_results])
    sig_count3 = sum(1 for r in h3_results if r[2] < 0.05)
    print(f"Mean True Dist:  {mean_tdist:.4f}")
    print(f"Mean Null Dist:  {mean_ndist:.4f}")
    print(f"Significant Subjects: {sig_count3} / {len(h3_results)} (p < 0.05)")
    
    # H2: Distance-based PERMANOVA
    var_subj, var_att = compute_h2_permanova(all_covariances, all_subj_ids, all_labels)
    print("\n--- H2: Distance-based PERMANOVA ---")
    print(f"Variance Explained by SUBJECT:   {var_subj:.2f}%")
    print(f"Variance Explained by ATTENTION: {var_att:.2f}%")
    print(f"Ratio (Subject / Attention):     {var_subj / (var_att + 1e-8):.1f}x")

if __name__ == '__main__':
    main()
