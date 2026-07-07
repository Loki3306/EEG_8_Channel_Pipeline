import os
import sys
import torch
import numpy as np
import scipy.io
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

# -------------------------------------------------------------------------
# CONSTANTS & CONFIG
# -------------------------------------------------------------------------
SR = 128
LAG_SAMPLES = 103  # ~800ms
WINDOW_LENS = {'1.0s': int(1.0 * SR), '1.5s': int(1.5 * SR), '2.0s': int(2.0 * SR), '2.5s': int(2.5 * SR)}
HOP_LEN = int(0.25 * SR)  # More fine-grained hopping

def build_ground_truth_envelope(trial, swap_triggers=False, delay_samples=0):
    eeg_full = trial['eeg'].numpy()
    env_l = trial['env_l'].numpy()
    env_r = trial['env_r'].numpy()
    switch_points = trial['meta']['switch_points']
    
    att = np.zeros(eeg_full.shape[1], dtype=np.float32)
    unatt = np.zeros(eeg_full.shape[1], dtype=np.float32)
    
    def get_dir(raw_dir):
        if not swap_triggers: return raw_dir
        return 'R' if raw_dir == 'L' else 'L'

    if len(switch_points) == 0: 
        switch_points = [('R', 0)]
        
    initial_state = get_dir('R') if (switch_points[0][1] > 0 and get_dir(switch_points[0][0]) == 'L') else get_dir(switch_points[0][0])
    if switch_points[0][1] > 0:
        initial_state = get_dir('R') if get_dir(switch_points[0][0]) == 'L' else get_dir('L')
        
    current_state = initial_state
    prev_idx = 0
    for raw_state, raw_idx in switch_points:
        state = get_dir(raw_state)
        # Apply cognitive delay
        idx = min(raw_idx + delay_samples, eeg_full.shape[1])
        if idx > prev_idx:
            if current_state == 'L':
                att[prev_idx:idx] = env_l[prev_idx:idx]
                unatt[prev_idx:idx] = env_r[prev_idx:idx]
            else:
                att[prev_idx:idx] = env_r[prev_idx:idx]
                unatt[prev_idx:idx] = env_l[prev_idx:idx]
        prev_idx, current_state = idx, state
        
    if current_state == 'L':
        att[prev_idx:] = env_l[prev_idx:]
        unatt[prev_idx:] = env_r[prev_idx:]
    else:
        att[prev_idx:] = env_r[prev_idx:]
        unatt[prev_idx:] = env_l[prev_idx:]
        
    # Align to network output (clip end)
    att = att[:-(LAG_SAMPLES - 1)]
    unatt = unatt[:-(LAG_SAMPLES - 1)]
    return att, unatt

def build_lagged_eeg(eeg_data):
    """
    Creates a sliding window view of the EEG.
    eeg_data: (60, Time)
    Returns: X shape (Time - lag + 1, 60 * lag)
    """
    C, T = eeg_data.shape
    new_T = T - LAG_SAMPLES + 1
    
    shape = (new_T, C, LAG_SAMPLES)
    strides = (eeg_data.strides[1], eeg_data.strides[0], eeg_data.strides[1])
    X = np.lib.stride_tricks.as_strided(eeg_data, shape=shape, strides=strides)
    
    return X.reshape(new_T, -1)

def compute_trial_auroc(pred, att, unatt, switch_points, delay_samples):
    results = {}
    for win_name, win_len in WINDOW_LENS.items():
        num_windows = (len(pred) - win_len) // HOP_LEN + 1
        if num_windows <= 0:
            results[win_name] = 0.5
            continue
            
        sa, su = [], []
        for i in range(num_windows):
            start = i * HOP_LEN
            end = start + win_len
            
            # EXCLUDE windows that overlap the Cognitive Gap!
            # The gap is from raw_idx to raw_idx + delay_samples.
            overlap = False
            for _, raw_idx in switch_points:
                gap_start = raw_idx
                gap_end = raw_idx + delay_samples
                if max(start, gap_start) < min(end, gap_end):
                    overlap = True
                    break
                    
            if overlap:
                continue
            
            p = pred[start:end]
            a = att[start:end]
            u = unatt[start:end]
            
            if np.var(p) < 1e-8 or np.var(a) < 1e-8 or np.var(u) < 1e-8:
                sa.append(0)
                su.append(0)
            else:
                sa.append(np.corrcoef(p, a)[0, 1])
                su.append(np.corrcoef(p, u)[0, 1])
                
        if len(sa) == 0:
            results[win_name] = 0.5
            continue
            
        y_true = [1] * len(sa) + [0] * len(su)
        y_scores = sa + su
        if len(np.unique(y_true)) < 2: 
            results[win_name] = 0.5
        else:
            results[win_name] = roc_auc_score(y_true, y_scores)
            
    return results

def main():
    cache_path = Path('/kaggle/working/eeg_cache/S1_processed.pt')
    if not cache_path.exists():
        print(f"Error: Cache not found at {cache_path}")
        return
        
    print("Loading S1 data into memory...")
    cached = torch.load(cache_path, map_location='cpu', weights_only=False)
    trials = cached['raw']
    print(f"Loaded {len(trials)} trials for Subject S1.")
    
    print("\nPre-computing lagged EEG matrices (this saves massive time)...")
    X_all = []
    for i, tr in enumerate(trials):
        X_all.append(build_lagged_eeg(tr['eeg'].numpy()))
        
    # Sweep Parameters
    trigger_swaps = [False, True]
    delay_ms_list = [1500, 1750, 2000, 2250, 2500]
    
    # 5-Fold Cross Validation
    num_trials = len(trials)
    folds = 5
    fold_size = num_trials // folds
    
    print("\n=======================================================")
    print(" PHASE 48: RIDGE REGRESSION SWEEP ON S1 (WITHIN-SUBJECT) ")
    print("=======================================================")
    
    best_config = None
    best_auroc = 0.0
    
    for swap in trigger_swaps:
        for delay_ms in delay_ms_list:
            delay_samples = int((delay_ms / 1000.0) * SR)
            print(f"\n--- Testing Config: Swap Triggers={swap} | Cognitive Delay={delay_ms}ms ---")
            
            # Rebuild envelopes for this config
            att_all, unatt_all = [], []
            for tr in trials:
                a, u = build_ground_truth_envelope(tr, swap_triggers=swap, delay_samples=delay_samples)
                att_all.append(a)
                unatt_all.append(u)
                
            fold_results = {k: [] for k in WINDOW_LENS.keys()}
            
            # 5-Fold CV
            for f in range(folds):
                test_start = f * fold_size
                test_end = test_start + fold_size if f < folds - 1 else num_trials
                
                # Split indices
                test_idx = list(range(test_start, test_end))
                train_idx = [i for i in range(num_trials) if i not in test_idx]
                
                # Build Training Set
                X_train = np.vstack([X_all[i] for i in train_idx])
                y_train = np.concatenate([att_all[i] for i in train_idx])
                
                # Fit fast Analytical Ridge
                # CRITICAL: We must use massive regularization for EEG (1e5)
                # because we have 6180 features. alpha=100 causes severe overfitting.
                ridge = Ridge(alpha=1e5, solver='cholesky')
                ridge.fit(X_train, y_train)
                
                # Evaluate Test Set
                for idx in test_idx:
                    X_test = X_all[idx]
                    y_att = att_all[idx]
                    y_unatt = unatt_all[idx]
                    switch_points = trials[idx]['meta']['switch_points']
                    
                    pred = ridge.predict(X_test)
                    
                    trial_aurocs = compute_trial_auroc(pred, y_att, y_unatt, switch_points, delay_samples)
                    for k in fold_results.keys():
                        fold_results[k].append(trial_aurocs[k])
                        
            print("Average AUROC across 5-Fold CV:")
            for k in fold_results.keys():
                mean_auc = np.mean(fold_results[k])
                print(f"  {k} Window: {mean_auc:.4f}")
                
            # Track best based on 2.0s window
            if np.mean(fold_results['2.0s']) > best_auroc:
                best_auroc = np.mean(fold_results['2.0s'])
                best_config = (swap, delay_ms)
                
    print("\n=======================================================")
    print(f" BEST CONFIGURATION FOUND: ")
    print(f" Swap Triggers: {best_config[0]}")
    print(f" Cognitive Delay: {best_config[1]}ms")
    print(f" Best 2.0s AUROC: {best_auroc:.4f}")
    print("=======================================================")

if __name__ == "__main__":
    main()
