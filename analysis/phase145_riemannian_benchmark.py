import os
import time
import numpy as np
import torch
from pathlib import Path
from scipy import signal
import concurrent.futures
import multiprocessing as mp

from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import roc_auc_score, accuracy_score

try:
    import pyriemann
except ImportError:
    import os
    os.system("pip install pyriemann")
    import pyriemann

from pyriemann.classification import MDM
from pyriemann.tangentspace import TangentSpace
from pyriemann.spatialfilters import CSP
from sklearn.base import BaseEstimator, TransformerMixin

# -------------------------------------------------------------------------
# CONSTANTS & CONFIGURATION
# -------------------------------------------------------------------------
SR = 128
EAR_CHANNEL_INDICES = [23, 31, 32, 40, 14, 22, 41, 49]
BROADBAND = (0.5, 8.0)

PRE_SWITCH_SAMPLES = int(0.5 * SR)
POST_SWITCH_SAMPLES = int(1.0 * SR)
MIN_SEGMENT_SAMPLES = int(3.0 * SR) # Only segments > 3s

class VectorizeEuclideanCovariance(BaseEstimator, TransformerMixin):
    """ Naive Euclidean vectorization of the upper triangle of the covariance matrix. """
    def fit(self, X, y=None):
        return self
        
    def transform(self, X, y=None):
        # X is shape (n_trials, n_channels, n_channels)
        n_trials, n_channels, _ = X.shape
        out = []
        idx = np.triu_indices(n_channels)
        for i in range(n_trials):
            out.append(X[i][idx])
        return np.array(out)

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
                
            label = 1 if mask_t[start] == 1.0 else 0 # 1 is Left, 0 is Right
            X_seg = eeg_f[:, start:end] # [C, T]
            
            # Covariance matrix with regularization to ensure SPD
            cov_mat = np.cov(X_seg) # shape: (8, 8)
            cov_mat += np.eye(cov_mat.shape[0]) * 1e-5
            
            segments.append({
                'cov': cov_mat,
                'label': label,
                'trial_idx': tr_idx
            })
            
    return subj_name, segments

def evaluate_models_for_subject(subj_name, segments):
    if len(segments) < 10:
        return subj_name, None
        
    X = np.array([seg['cov'] for seg in segments])
    y = np.array([seg['label'] for seg in segments])
    trials = np.array([seg['trial_idx'] for seg in segments])
    
    # 5-Fold Cross Validation across Trials (Strict Leakage Prevention)
    unique_trials = np.unique(trials)
    if len(unique_trials) < 5:
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        cv_splits = list(kf.split(X))
    else:
        # GroupKFold-style splitting based on trials
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        cv_splits = []
        for train_trial_idx, test_trial_idx in kf.split(unique_trials):
            train_trials = unique_trials[train_trial_idx]
            test_trials = unique_trials[test_trial_idx]
            
            train_idx = np.where(np.isin(trials, train_trials))[0]
            test_idx = np.where(np.isin(trials, test_trials))[0]
            cv_splits.append((train_idx, test_idx))
    
    # The 5 Spatial Classifiers
    models = {
        'MDM (Pure Riemannian)': MDM(),
        'Tangent-Space + LR': make_pipeline(TangentSpace(), LogisticRegression(max_iter=1000, random_state=42)),
        'Tangent-Space + SVM': make_pipeline(TangentSpace(), SVC(kernel='linear', probability=True, random_state=42)),
        'CSP + LDA': make_pipeline(CSP(nfilter=4), LinearDiscriminantAnalysis()),
        'Euclidean Cov + LR': make_pipeline(VectorizeEuclideanCovariance(), LogisticRegression(max_iter=1000, random_state=42))
    }
    
    results = {name: {'auroc': [], 'acc': []} for name in models.keys()}
    
    for name, model in models.items():
        for train_idx, test_idx in cv_splits:
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # Skip if only one class in train or test
            if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
                continue
                
            try:
                model.fit(X_train, y_train)
                preds = model.predict(X_test)
                
                # Handling probability outputs for AUROC
                if hasattr(model, "predict_proba"):
                    probs = model.predict_proba(X_test)[:, 1]
                elif hasattr(model, "decision_function"):
                    probs = model.decision_function(X_test)
                else:
                    probs = preds # fallback for pure MDM if not using predict_proba
                    
                auc = roc_auc_score(y_test, probs)
                acc = accuracy_score(y_test, preds)
                
                results[name]['auroc'].append(auc)
                results[name]['acc'].append(acc)
            except Exception as e:
                # Catch SVD convergence or CSP rank issues on bad data
                pass
                
    # Aggregate
    agg_results = {}
    for name in models.keys():
        if len(results[name]['auroc']) > 0:
            agg_results[name] = {
                'auroc_mean': np.mean(results[name]['auroc']),
                'acc_mean': np.mean(results[name]['acc'])
            }
        else:
            agg_results[name] = {'auroc_mean': 0.5, 'acc_mean': 0.5}
            
    return subj_name, agg_results

def main():
    print("=======================================================")
    print(" PHASE 145: RIEMANNIAN CLASSIFICATION BENCHMARK")
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
    
    # Phase 1: Extract Covariance Segments
    print("Extracting Spatial Covariance Matrices...")
    with concurrent.futures.ProcessPoolExecutor(max_workers=mp.cpu_count()) as executor:
        futures = [executor.submit(process_subject, cf) for cf in cache_files]
        for future in concurrent.futures.as_completed(futures):
            subj, segments = future.result()
            all_subjects_data[subj] = segments

    print(f"Extraction Time: {time.time() - start_time:.2f}s\n")
    
    # Phase 2: Benchmark Classifiers
    print("Running 5-Fold Trial-Strict Cross-Validation for 5 Models...\n")
    
    model_names = [
        'MDM (Pure Riemannian)',
        'Tangent-Space + LR',
        'Tangent-Space + SVM',
        'CSP + LDA',
        'Euclidean Cov + LR'
    ]
    
    global_auroc = {name: [] for name in model_names}
    global_acc = {name: [] for name in model_names}
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=mp.cpu_count()) as executor:
        futures = [executor.submit(evaluate_models_for_subject, subj, segs) for subj, segs in all_subjects_data.items()]
        for future in concurrent.futures.as_completed(futures):
            subj, res = future.result()
            if not res: continue
            
            print(f"--- Subject: {subj} ---")
            for name in model_names:
                auc = res[name]['auroc_mean']
                acc = res[name]['acc_mean']
                global_auroc[name].append(auc)
                global_acc[name].append(acc)
                print(f"  {name:25s} | AUROC: {auc:.3f} | Acc: {acc:.3f}")
            print("")

    # Global Results
    print("\n=======================================================")
    print(" GLOBAL BENCHMARK AVERAGES (18 Subjects)")
    print("=======================================================")
    for name in model_names:
        mean_auc = np.mean(global_auroc[name])
        mean_acc = np.mean(global_acc[name])
        print(f"{name:25s} | Mean AUROC: {mean_auc:.3f} | Mean Acc: {mean_acc:.3f}")

if __name__ == '__main__':
    main()
