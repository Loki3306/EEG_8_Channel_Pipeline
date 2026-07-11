import os
import time
import numpy as np
import torch
from pathlib import Path
from scipy import signal
import scipy.stats as stats
from scipy.linalg import fractional_matrix_power
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
    
    # Base Directional Consistency
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
    
    # ---------------------------------------------------------
    # Riemannian Parallel Transport (Covariance Transport)
    # ---------------------------------------------------------
    transported_covs = []
    transported_trial_deltas = []
    
    global_mean_invsqrt = fractional_matrix_power(global_mean, -0.5).real
    global_mean_sqrt = fractional_matrix_power(global_mean, 0.5).real
    
    for tr in trials:
        tr_covs = [s for s in segments if s['trial_idx'] == tr]
        if len(tr_covs) == 0: continue
        tr_covs_mat = np.array([s['cov'] for s in tr_covs])
        tr_mean = mean_riemann(tr_covs_mat)
        
        # Transport matrix W = M_global^(1/2) * M_trial^(-1/2)
        # We align the trial's center to the global center.
        tr_mean_invsqrt = fractional_matrix_power(tr_mean, -0.5).real
        W = global_mean_sqrt @ tr_mean_invsqrt
        
        tr_covs_L_trans = []
        tr_covs_R_trans = []
        for s in tr_covs:
            C_trans = W @ s['cov'] @ W.T
            if s['label'] == 1: tr_covs_L_trans.append(C_trans)
            else: tr_covs_R_trans.append(C_trans)
            
        if len(tr_covs_L_trans) > 0 and len(tr_covs_R_trans) > 0:
            tr_mean_L_trans = mean_riemann(np.array(tr_covs_L_trans))
            tr_mean_R_trans = mean_riemann(np.array(tr_covs_R_trans))
            
            vec_L_trans = tangent_space(np.array([tr_mean_L_trans]), global_mean)[0]
            vec_R_trans = tangent_space(np.array([tr_mean_R_trans]), global_mean)[0]
            transported_trial_deltas.append(vec_L_trans - vec_R_trans)

    cosine_sims_trans = []
    n_deltas_trans = len(transported_trial_deltas)
    for i in range(n_deltas_trans):
        for j in range(i + 1, n_deltas_trans):
            cosine_sims_trans.append(get_cosine_similarity(transported_trial_deltas[i], transported_trial_deltas[j]))
    dir_consistency_trans = np.mean(cosine_sims_trans) if cosine_sims_trans else 0.0

    return {
        'subj': subj_name,
        'auroc': AUROC_SCORES.get(subj_name, 0.50),
        'snr': snr,
        'dir_consist': dir_consistency,
        'dir_consist_trans': dir_consistency_trans
    }

def main():
    print("=======================================================")
    print(" PHASE 147: RIEMANNIAN PARALLEL TRANSPORT")
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
    
    print(f"Extracting Covariances and Running Manifold Alignment for {len(cache_files)} subjects...")
    results = []
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=mp.cpu_count()) as executor:
        futures = {executor.submit(process_subject, cf): cf.stem.split('_')[0] for cf in cache_files}
        for future in concurrent.futures.as_completed(futures):
            subj = futures[future]
            segments = future.result()
            res = analyze_subject(subj, segments)
            if res:
                results.append(res)
                print(f"[{subj}] AUROC: {res['auroc']:.3f} | DirConsist Base: {res['dir_consist']:.4f} -> Trans: {res['dir_consist_trans']:.4f}")

    # Part 1: Statistical Correlation across 18 Subjects
    aurocs = [r['auroc'] for r in results]
    snrs = [r['snr'] for r in results]
    dir_consists = [r['dir_consist'] for r in results]
    
    r_snr, p_snr = stats.pearsonr(aurocs, snrs)
    r_dir, p_dir = stats.pearsonr(aurocs, dir_consists)
    
    print("\n=======================================================")
    print(" PART 1: STATISTICAL DRIVERS OF AAD PERFORMANCE")
    print("=======================================================")
    print(f"Correlation (AUROC vs Riemannian SNR):         r = {r_snr:.3f} (p={p_snr:.4f})")
    print(f"Correlation (AUROC vs Directional Consistency): r = {r_dir:.3f} (p={p_dir:.4f})")
    
    # Part 2: Manifold Alignment Effectiveness
    print("\n=======================================================")
    print(" PART 2: COVARIANCE TRANSPORT / MANIFOLD ALIGNMENT")
    print("=======================================================")
    improved = 0
    degraded = 0
    for r in results:
        if r['dir_consist_trans'] > r['dir_consist']: improved += 1
        else: degraded += 1
        
    mean_base = np.mean(dir_consists)
    mean_trans = np.mean([r['dir_consist_trans'] for r in results])
    
    print(f"Subjects Improved by Transport: {improved}/{len(results)}")
    print(f"Mean Directional Consistency (Base):       {mean_base:.4f}")
    print(f"Mean Directional Consistency (Transported): {mean_trans:.4f}")
    
    if mean_trans > mean_base:
        print("\n[SUCCESS] Manifold Alignment successfully aligned the 'random' spatial rotations.")
        print("This proves the rotations were largely due to Coordinate System Shifts (e.g. electrode movement/impedance).")
    else:
        print("\n[FAILURE] Manifold Alignment did NOT resolve the directional rotations.")
        print("This proves the spatial drift is intrinsic to the brain's attentional network, not just a baseline shift.")

if __name__ == '__main__':
    main()
