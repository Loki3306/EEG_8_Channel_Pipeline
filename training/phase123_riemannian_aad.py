import numpy as np
import torch
from pathlib import Path
from sklearn.metrics import roc_auc_score
from scipy import signal
import time
import concurrent.futures
import multiprocessing as mp

try:
    from pyriemann.estimation import Covariances
    from pyriemann.tangentspace import TangentSpace
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
except ImportError:
    print("PyRiemann is not installed. Please run: pip install pyriemann")
    import sys
    sys.exit(1)

# -------------------------------------------------------------------------
# CONSTANTS & CONFIGURATION
# -------------------------------------------------------------------------
SR = 128
EAR_CHANNEL_INDICES = [23, 31, 32, 40, 14, 22, 41, 49]
SEQ_SAMPLES = int(3.5 * SR)
SEQ_HOP = int(0.5 * SR)

BROADBAND = (0.5, 8.0)

def apply_modulation_filter(env, lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype='band')
    return signal.filtfilt(b, a, env, axis=1)

def prepare_trial_data(tr):
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
    
    T = eeg_f.shape[1]
    
    sp = tr['meta']['switch_points']
    boundaries = [0] + [idx for spk, idx in sp]
    boundaries = sorted(set(boundaries))
    if boundaries[-1] != T: boundaries.append(T)
        
    seq_data_L = []
    seq_data_R = []
    seq_labels = [] # 1.0 if attended L, 0.0 if attended R
    
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
            for seq_start in range(safe_start, safe_end - SEQ_SAMPLES + 1, SEQ_HOP):
                if seq_start + SEQ_SAMPLES <= T:
                    s = seq_start
                    e = seq_start + SEQ_SAMPLES
                    
                    # Stack EEG and Audio (9 channels total)
                    X_L = np.vstack([eeg_f[:, s:e], env_l_f[:, s:e]])
                    X_R = np.vstack([eeg_f[:, s:e], env_r_f[:, s:e]])
                    
                    seq_data_L.append(X_L)
                    seq_data_R.append(X_R)
                    seq_labels.append(1.0 if current_spk == 'L' else 0.0)
                
    return {
        'seq_L': np.array(seq_data_L), # (N_seq, 9, T)
        'seq_R': np.array(seq_data_R), # (N_seq, 9, T)
        'labels': np.array(seq_labels)
    }

def process_subject(cache_file):
    subj_name = cache_file.stem.split('_')[0]
    cached = torch.load(cache_file, map_location='cpu', weights_only=False)['raw']
    
    N_TRIALS = len(cached)
    data_cache = []
    
    for t_idx in range(N_TRIALS):
        data_cache.append(prepare_trial_data(cached[t_idx]))
            
    all_eval_diffs = []
    all_eval_labels = []
    
    cov_est = Covariances(estimator='lwf') # Ledoit-Wolf shrinkage
    
    for test_idx in range(N_TRIALS):
        # 1. Build Training Set for this LOTO split
        X_train_match = []
        X_train_mismatch = []
        
        for i in range(N_TRIALS):
            if i == test_idx: continue
            
            tr = data_cache[i]
            if len(tr['labels']) == 0: continue
                
            for s_idx in range(len(tr['labels'])):
                if tr['labels'][s_idx] == 1.0: # Attended L
                    X_train_match.append(tr['seq_L'][s_idx])
                    X_train_mismatch.append(tr['seq_R'][s_idx])
                else: # Attended R
                    X_train_match.append(tr['seq_R'][s_idx])
                    X_train_mismatch.append(tr['seq_L'][s_idx])
                    
        if len(X_train_match) == 0: continue
            
        # Stack and estimate covariances
        X_train = np.array(X_train_match + X_train_mismatch)
        y_train = np.array([1]*len(X_train_match) + [0]*len(X_train_mismatch))
        
        covs_train = cov_est.fit_transform(X_train)
        
        # Train Riemannian Classifier
        clf = make_pipeline(
            TangentSpace(metric='logeuclid'),
            LogisticRegression(max_iter=1000, class_weight='balanced')
        )
        clf.fit(covs_train, y_train)
        
        # 2. Evaluate on Test Trial
        tr_test = data_cache[test_idx]
        if len(tr_test['labels']) == 0: continue
            
        covs_L = cov_est.transform(tr_test['seq_L'])
        covs_R = cov_est.transform(tr_test['seq_R'])
        
        prob_L = clf.predict_proba(covs_L)[:, 1] # Probability that L is Match
        prob_R = clf.predict_proba(covs_R)[:, 1] # Probability that R is Match
        
        diffs = prob_L - prob_R
        
        all_eval_diffs.extend(diffs.tolist())
        all_eval_labels.extend(tr_test['labels'].tolist())
        
    all_eval_diffs = np.array(all_eval_diffs)
    all_eval_labels = np.array(all_eval_labels)
    
    if len(all_eval_diffs) > 0 and len(np.unique(all_eval_labels)) > 1:
        probs = (all_eval_diffs - np.min(all_eval_diffs)) / (np.max(all_eval_diffs) - np.min(all_eval_diffs) + 1e-8)
        global_auc = roc_auc_score(all_eval_labels, probs)
    else:
        global_auc = 0.5
        
    print(f"  [{subj_name}] Finished. Riemannian AUROC: {global_auc:.4f}")
    return subj_name, global_auc

def main():
    mp.set_start_method('spawn', force=True)
    
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
            
    num_workers = mp.cpu_count()
    
    print(f"\n=======================================================")
    print(f" PHASE 123: RIEMANNIAN GEOMETRY AAD (Tangent Space)")
    print(f" CPUs detected: {num_workers}")
    print(f"=======================================================\n", flush=True)
    
    cache_files = sorted(list(cache_dir.glob('*_multiband.pt')))
    
    start_time = time.time()
    final_results = {}
    
    futures = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=min(num_workers, len(cache_files))) as executor:
        for idx, cache_file in enumerate(cache_files):
            futures.append(executor.submit(process_subject, cache_file))
            
        for future in concurrent.futures.as_completed(futures):
            subj_name, auc = future.result()
            final_results[subj_name] = auc

    print(f"\nTotal Pipeline Execution Time: {time.time() - start_time:.2f}s")
    print("\n=======================================================")
    print(" PHASE 123 RIEMANNIAN GEOMETRY AAD RESULTS")
    print("=======================================================")
    print(f"{'Subject':<10} {'LOTO AUROC':<10}")
    
    sorted_results = sorted(final_results.items(), key=lambda x: int(x[0][1:]))
    for subj, auc in sorted_results:
        print(f"{subj:<10} {auc:.4f}")

if __name__ == '__main__':
    main()
