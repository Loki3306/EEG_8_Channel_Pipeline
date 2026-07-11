import os
import time
import numpy as np
import torch
from pathlib import Path
from scipy import signal
from sklearn.cross_decomposition import CCA
import concurrent.futures
import multiprocessing as mp

# -------------------------------------------------------------------------
# CONSTANTS & CONFIGURATION
# -------------------------------------------------------------------------
SR = 128
EAR_CHANNEL_INDICES = [23, 31, 32, 40, 14, 22, 41, 49]
BROADBAND = (0.5, 8.0)

PRE_SWITCH_SAMPLES = int(0.5 * SR)
POST_SWITCH_SAMPLES = int(1.0 * SR)
MIN_SEGMENT_SAMPLES = int(3.0 * SR)

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

def riemannian_distance(C1, C2):
    # Log-Euclidean distance metric for SPD matrices
    # d = || logM(C1) - logM(C2) ||_F
    try:
        w1, v1 = np.linalg.eigh(C1)
        logC1 = v1 @ np.diag(np.log(np.clip(w1, 1e-8, None))) @ v1.T
        
        w2, v2 = np.linalg.eigh(C2)
        logC2 = v2 @ np.diag(np.log(np.clip(w2, 1e-8, None))) @ v2.T
        
        return np.linalg.norm(logC1 - logC2, ord='fro')
    except:
        return 0.0

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
            
            # Target envelope
            target_env = env_l_f[0, start:end] if label == 1.0 else env_r_f[0, start:end]
            
            # Covariance matrix of EEG
            cov_mat = np.cov(X_seg, rowvar=False)
            
            segments.append({
                'X': X_seg,
                'target_env': target_env.reshape(-1, 1),
                'cov': cov_mat,
                'label': label
            })
            
    # H1: CCA Test
    true_cca_scores = []
    shuf_cca_scores = []
    
    cca = CCA(n_components=1)
    
    for i, seg in enumerate(segments):
        X = seg['X']
        Y_true = seg['target_env']
        
        # Pick a random segment for shuffled envelope (same length or truncate)
        rand_idx = (i + np.random.randint(1, len(segments)-1)) % len(segments)
        Y_shuf_full = segments[rand_idx]['target_env']
        min_T = min(len(Y_true), len(Y_shuf_full))
        
        try:
            # True CCA
            Xc, Yc = cca.fit_transform(X[:min_T], Y_true[:min_T])
            true_r = np.corrcoef(Xc[:,0], Yc[:,0])[0,1]
            true_cca_scores.append(abs(true_r))
            
            # Shuffled CCA
            Xc_s, Yc_s = cca.fit_transform(X[:min_T], Y_shuf_full[:min_T])
            shuf_r = np.corrcoef(Xc_s[:,0], Yc_s[:,0])[0,1]
            shuf_cca_scores.append(abs(shuf_r))
        except:
            pass
            
    mean_true_cca = np.mean(true_cca_scores) if true_cca_scores else 0
    mean_shuf_cca = np.mean(shuf_cca_scores) if shuf_cca_scores else 0
    
    # H3: Covariance Distance Test
    covs_L = [seg['cov'] for seg in segments if seg['label'] == 1.0]
    covs_R = [seg['cov'] for seg in segments if seg['label'] == 0.0]
    
    mean_cov_L = np.mean(covs_L, axis=0) if covs_L else np.eye(8)
    mean_cov_R = np.mean(covs_R, axis=0) if covs_R else np.eye(8)
    
    true_cov_dist = riemannian_distance(mean_cov_L, mean_cov_R)
    
    # Shuffled covariance distance
    all_covs = covs_L + covs_R
    np.random.shuffle(all_covs)
    split_idx = len(covs_L)
    shuf_cov_L = np.mean(all_covs[:split_idx], axis=0) if split_idx > 0 else np.eye(8)
    shuf_cov_R = np.mean(all_covs[split_idx:], axis=0) if (len(all_covs)-split_idx) > 0 else np.eye(8)
    
    shuf_cov_dist = riemannian_distance(shuf_cov_L, shuf_cov_R)
    
    return subj_name, mean_true_cca, mean_shuf_cca, true_cov_dist, shuf_cov_dist, [s['cov'] for s in segments], [s['label'] for s in segments]

def main():
    print("=======================================================")
    print(" PHASE 143: SIGNAL CHARACTERIZATION")
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
    
    results = {}
    all_covariances = []
    all_labels = [] # L=1, R=0
    all_subjects = []
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=mp.cpu_count()) as executor:
        futures = [executor.submit(process_subject, cf) for cf in cache_files]
        for future in concurrent.futures.as_completed(futures):
            res = future.result()
            subj = res[0]
            results[subj] = {
                'true_cca': res[1],
                'shuf_cca': res[2],
                'true_cov': res[3],
                'shuf_cov': res[4]
            }
            # Collect for H2
            for c in res[5]:
                all_covariances.append(c)
                all_subjects.append(subj)
            for l in res[6]:
                all_labels.append(l)

    print(f"Extraction Time: {time.time() - start_time:.2f}s\n")
    
    # H1 Results
    print("--- H1: Stimulus Information Exists in Ear-EEG (CCA) ---")
    mean_t_cca = np.mean([r['true_cca'] for r in results.values()])
    mean_s_cca = np.mean([r['shuf_cca'] for r in results.values()])
    print(f"Mean True CCA:     {mean_t_cca:.4f}")
    print(f"Mean Shuffled CCA: {mean_s_cca:.4f}")
    print(f"Verdict: {'Information Exists!' if mean_t_cca > mean_s_cca * 1.5 else 'Barely any information.'}\n")
    
    # H3 Results
    print("--- H3: Ear-EEG Covariance Changes with Attention ---")
    mean_t_cov = np.mean([r['true_cov'] for r in results.values()])
    mean_s_cov = np.mean([r['shuf_cov'] for r in results.values()])
    print(f"Mean True Riemann Dist:     {mean_t_cov:.4f}")
    print(f"Mean Shuffled Riemann Dist: {mean_s_cov:.4f}")
    print(f"Verdict: {'Spatial Covariance encodes Attention!' if mean_t_cov > mean_s_cov * 1.5 else 'No strong spatial geometry change.'}\n")
    
    # H2: Variance Decomposition (Approximate PERMANOVA)
    print("--- H2: Subject Variability Exceeds Attention Variability ---")
    # Vectorize upper triangle of covariances
    vecs = []
    for c in all_covariances:
        idx = np.triu_indices(8)
        vecs.append(c[idx])
    vecs = np.array(vecs)
    
    global_mean = np.mean(vecs, axis=0)
    SST = np.sum(np.linalg.norm(vecs - global_mean, axis=1)**2)
    
    # Subject Variance
    unique_subs = list(set(all_subjects))
    SS_subj = 0
    for sub in unique_subs:
        idx = [i for i, s in enumerate(all_subjects) if s == sub]
        if len(idx) > 0:
            sub_mean = np.mean(vecs[idx], axis=0)
            SS_subj += len(idx) * np.linalg.norm(sub_mean - global_mean)**2
            
    # Attention Variance
    unique_atts = [1.0, 0.0]
    SS_att = 0
    for att in unique_atts:
        idx = [i for i, l in enumerate(all_labels) if l == att]
        if len(idx) > 0:
            att_mean = np.mean(vecs[idx], axis=0)
            SS_att += len(idx) * np.linalg.norm(att_mean - global_mean)**2
            
    var_subj = (SS_subj / SST) * 100
    var_att = (SS_att / SST) * 100
    
    print(f"Variance Explained by SUBJECT:   {var_subj:.2f}%")
    print(f"Variance Explained by ATTENTION: {var_att:.2f}%")
    print(f"Ratio (Subject / Attention):     {var_subj / (var_att + 1e-8):.1f}x")
    print(f"Verdict: {'Universal Decoding is scientifically invalid.' if var_subj > var_att * 5 else 'Universal Decoding is possible.'}\n")

if __name__ == '__main__':
    main()
