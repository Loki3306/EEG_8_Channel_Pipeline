import os
import sys
import time
import numpy as np
import scipy.io
import scipy.signal
import glob
from pathlib import Path
from sklearn.linear_model import Ridge

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from training.phase29_cross_subject_train import load_aasd_subject
from baselines.ridge_aad import lagged_eeg_matrix

def safe_corr_np(x, y):
    if len(x) == 0 or len(y) == 0: return 0.0
    x_c = x - np.mean(x)
    y_c = y - np.mean(y)
    var_x = np.sum(x_c**2)
    var_y = np.sum(y_c**2)
    if var_x == 0 or var_y == 0: return 0.0
    return np.sum(x_c * y_c) / np.sqrt(var_x * var_y)

def run_loso_ridge_benchmark():
    print("--- 1. Loading Complete AASD Dataset ---")
    mat_files = glob.glob('/kaggle/input/datasets/lokeshgile/aasd-processed-eeg/Processed EEG/*/*.mat')
    if not mat_files:
        print("ERROR: No .mat files found. Please run on Kaggle.")
        return

    b, a = scipy.signal.butter(4, [1.0/64.0, 8.0/64.0], btype='band')
    sel_idx = [23, 28, 22, 41, 36, 0, 40, 25] # fallback map
    audio_dir = '/kaggle/input/datasets/lokeshgile/aasd-audio-gammatones'

    # Extract distinct subjects from filenames
    # Example filename: /kaggle/input/datasets/.../S18.mat
    subjects = []
    subject_paths = {}
    for p in mat_files:
        basename = os.path.basename(p)
        sub = basename.split('.')[0]
        subjects.append(sub)
        subject_paths[sub] = p
        
    subjects = sorted(list(set(subjects)))
    print(f"Found {len(subjects)} subjects.")

    # Load all subjects into memory
    subject_trials = {}
    for sub in subjects:
        trials = load_aasd_subject(subject_paths[sub], b, a, sel_idx, audio_dir)
        subject_trials[sub] = trials
        
    # Lags: 0 to 250ms at 64Hz
    # 250ms = 0.25 * 64 = 16 samples
    lags = 16
    alpha = 1e3 # Standard regularization for Ridge on EEG
    
    all_results = []
    
    print("\n--- 2. Commencing LOSO Ridge Benchmark ---")
    start_total = time.time()
    
    for test_sub in subjects:
        print(f"\n==================================================")
        print(f"Testing Subject: {test_sub}")
        print(f"==================================================")
        
        # 1. Build Train Set
        X_train_list = []
        Y_train_list = []
        
        for train_sub in subjects:
            if train_sub == test_sub:
                continue
            
            for t in subject_trials[train_sub]:
                # eeg: [Channels, Time]
                eeg = t['eeg'].numpy().T # [Time, Channels]
                env_l = t['env_l'].numpy()
                env_r = t['env_r'].numpy()
                
                # Dynamic labels construction
                # We need a continuous 'attended' envelope
                switch_points = t['meta']['switch_points']
                att_env = np.zeros_like(env_l)
                unatt_env = np.zeros_like(env_r)
                
                current_state = switch_points[0][0]
                state_idx = 1
                for i in range(len(eeg)):
                    if state_idx < len(switch_points) and i >= switch_points[state_idx][1]:
                        current_state = switch_points[state_idx][0]
                        state_idx += 1
                        
                    if current_state == 1:
                        att_env[i] = env_l[i]
                        unatt_env[i] = env_r[i]
                    else:
                        att_env[i] = env_r[i]
                        unatt_env[i] = env_l[i]
                        
                X_lagged = lagged_eeg_matrix(eeg, lags=lags) # [Time, Channels * lags]
                X_train_list.append(X_lagged)
                Y_train_list.append(att_env)
                
        X_train = np.vstack(X_train_list)
        Y_train = np.concatenate(Y_train_list)
        
        # 2. Fit Ridge
        print(f"Fitting Ridge on {X_train.shape[0]} samples with {X_train.shape[1]} features...")
        fit_start = time.time()
        model = Ridge(alpha=alpha, solver='cholesky')
        model.fit(X_train, Y_train)
        print(f"Fit completed in {time.time() - fit_start:.1f}s")
        
        # 3. Evaluate Train
        # Sample a subset of train data for train metric (too slow to do all windows)
        # We will just evaluate train correlation on the whole concatenated train block.
        train_pred = model.predict(X_train)
        train_p = safe_corr_np(train_pred, Y_train)
        
        # 4. Evaluate Test using the same windowed logic as Conformer
        window_len = 128
        hop_len = 64
        transition_margin = 0 # Match Conformer benchmark
        
        test_att = []
        test_unatt = []
        
        for t in subject_trials[test_sub]:
            eeg = t['eeg'].numpy().T # [Time, Channels]
            env_l = t['env_l'].numpy()
            env_r = t['env_r'].numpy()
            switch_points = t['meta']['switch_points']
            
            X_lagged = lagged_eeg_matrix(eeg, lags=lags)
            pred = model.predict(X_lagged)
            
            for start in range(0, len(eeg) - window_len + 1, hop_len):
                end = start + window_len
                
                # Check for transitions
                is_trans = False
                for state, s_idx in switch_points:
                    t_start, t_end = s_idx - transition_margin, s_idx + transition_margin
                    if max(start, t_start) < min(end, t_end):
                        is_trans = True
                        break
                        
                if is_trans:
                    continue # Skip transition windows
                    
                w_pred = pred[start:end]
                w_env_l = env_l[start:end]
                w_env_r = env_r[start:end]
                
                w_env_l = (w_env_l - w_env_l.mean()) / (w_env_l.std() + 1e-8)
                w_env_r = (w_env_r - w_env_r.mean()) / (w_env_r.std() + 1e-8)
                
                mid_point = start + window_len // 2
                current_state = switch_points[0][0]
                for state, s_idx in switch_points:
                    if mid_point >= s_idx: current_state = state
                    
                w_att_env = w_env_l if current_state == 1 else w_env_r
                w_unatt_env = w_env_r if current_state == 1 else w_env_l
                
                test_att.append(safe_corr_np(w_pred, w_att_env))
                test_unatt.append(safe_corr_np(w_pred, w_unatt_env))
                
        test_pearson = np.mean(test_att)
        test_acc = np.mean(np.array(test_att) > np.array(test_unatt))
        
        print(f"Train P: {train_p:.4f} | Test P: {test_pearson:.4f} | Test Acc: {test_acc*100:.1f}%")
        
        all_results.append({
            "Subject": test_sub,
            "Train P": train_p,
            "Test P": test_pearson,
            "Test Acc": test_acc
        })
        
    print("\n\n" + "="*80)
    print("PHASE 31: RIDGE AASD LOSO BENCHMARK SUMMARY")
    print("="*80)
    print(f"{'Subject':<10} | {'Train P':<10} | {'Test P':<10} | {'Test Acc':<10}")
    print("-" * 80)
    
    avg_test_p = 0
    avg_test_acc = 0
    
    for r in all_results:
        print(f"{r['Subject']:<10} | {r['Train P']:<10.4f} | {r['Test P']:<10.4f} | {r['Test Acc']*100:>5.1f}%")
        avg_test_p += r['Test P']
        avg_test_acc += r['Test Acc']
        
    avg_test_p /= len(all_results)
    avg_test_acc /= len(all_results)
    
    print("-" * 80)
    print(f"{'AVERAGE':<10} | {'-':<10} | {avg_test_p:<10.4f} | {avg_test_acc*100:>5.1f}%")
    print(f"Total Time: {time.time() - start_total:.1f}s")
    
if __name__ == "__main__":
    run_loso_ridge_benchmark()
