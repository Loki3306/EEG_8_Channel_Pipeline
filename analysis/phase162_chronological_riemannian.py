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

def evaluate_chronological_for_subject(subj_name, segments):
    if len(segments) < 10:
        return subj_name, None
        
    X = np.array([seg['cov'] for seg in segments])
    y = np.array([seg['label'] for seg in segments])
    trials = np.array([seg['trial_idx'] for seg in segments])
    
    # Chronological Split (First 2 minutes for training, rest for testing)
    # Each trial is approx 1 minute, so trial_idx 0 and 1 are the calibration set
    train_mask = trials < 2
    test_mask = trials >= 2
    
    if np.sum(train_mask) < 2 or np.sum(test_mask) < 2:
        return subj_name, None
        
    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test = X[test_mask], y[test_mask]
    
    # Models to test
    models = {
        'MDM (Pure Riemannian)': MDM(),
        'Tangent-Space + LR': make_pipeline(TangentSpace(), LogisticRegression(max_iter=1000)),
        'Tangent-Space + SVM': make_pipeline(TangentSpace(), SVC(probability=True)),
        'CSP + LDA': make_pipeline(CSP(nfilter=4), LinearDiscriminantAnalysis()),
        'Euclidean Cov + LR': make_pipeline(VectorizeEuclideanCovariance(), LogisticRegression(max_iter=1000))
    }
    
    metrics = {}
    
    for name, clf in models.items():
        try:
            clf.fit(X_train, y_train)
            if hasattr(clf, "predict_proba"):
                y_pred_prob = clf.predict_proba(X_test)[:, 1]
            elif hasattr(clf, "decision_function"):
                y_pred_prob = clf.decision_function(X_test)
            else:
                y_pred_prob = clf.predict(X_test)
                
            y_pred = clf.predict(X_test)
            
            auc = roc_auc_score(y_test, y_pred_prob)
            acc = accuracy_score(y_test, y_pred)
            
            # If MDM learns reversed classes, AUC might be < 0.5. Since it's symmetric, we can take max(AUC, 1-AUC)
            # Actually, standard CV didn't do this, so we leave it as is to be comparable to Phase 145
        except Exception as e:
            auc = 0.5
            acc = 0.5
            
        metrics[name] = {'auc': auc, 'acc': acc}
        
    return subj_name, metrics

def main():
    print("=======================================================")
    print(" PHASE 162: CHRONOLOGICAL RIEMANNIAN BENCHMARK")
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
    
    print("Extracting Spatial Covariance Matrices chronologically...")
    start_time = time.time()
    
    results_all = {}
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=mp.cpu_count()) as executor:
        futures = {executor.submit(process_subject, cf): cf for cf in cache_files}
        all_segments = {}
        for future in concurrent.futures.as_completed(futures):
            subj, segs = future.result()
            all_segments[subj] = segs
            
    print(f"Extraction Time: {time.time() - start_time:.2f}s\n")
    print("Running Chronological Evaluation (Train on First 2min, Test on Rest)...\n")
    
    global_metrics = {
        'MDM (Pure Riemannian)': {'auc': [], 'acc': []},
        'Tangent-Space + LR': {'auc': [], 'acc': []},
        'Tangent-Space + SVM': {'auc': [], 'acc': []},
        'CSP + LDA': {'auc': [], 'acc': []},
        'Euclidean Cov + LR': {'auc': [], 'acc': []}
    }
    
    # Process subjects sequentially to print output cleanly
    subjects_sorted = sorted(all_segments.keys())
    
    for subj in subjects_sorted:
        _, metrics = evaluate_chronological_for_subject(subj, all_segments[subj])
        if metrics is None:
            continue
            
        print(f"--- Subject: {subj} ---")
        for name, m in metrics.items():
            print(f"  {name:<25} | AUROC: {m['auc']:.3f} | Acc: {m['acc']:.3f}")
            global_metrics[name]['auc'].append(m['auc'])
            global_metrics[name]['acc'].append(m['acc'])
        print()
        
    print("\n=======================================================")
    print(" GLOBAL BENCHMARK AVERAGES (Chronological Eval)")
    print("=======================================================")
    for name in global_metrics.keys():
        m_auc = np.mean(global_metrics[name]['auc'])
        m_acc = np.mean(global_metrics[name]['acc'])
        print(f"{name:<25} | Mean AUROC: {m_auc:.3f} | Mean Acc: {m_acc:.3f}")

if __name__ == '__main__':
    main()
