import os
import sys
import numpy as np
import scipy.io
import scipy.io.wavfile
import scipy.signal
from sklearn.metrics import roc_auc_score
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.phase32_5_spatial_fix import load_aasd_subject_trials

def build_lagged_matrix(eeg, max_lag):
    """
    Builds a time-lagged EEG matrix for mTRF stimulus reconstruction.
    eeg: [Channels, Time]
    max_lag: int (e.g., 16 for 0-16 lags)
    """
    lags = np.arange(0, max_lag + 1)
    C, T = eeg.shape
    num_lags = len(lags)
    
    out_T = T - max_lag
    X = np.zeros((out_T, C * num_lags), dtype=np.float32)
    
    for i, lag in enumerate(lags):
        start_idx = max_lag - lag
        end_idx = T - lag
        X[:, i*C:(i+1)*C] = eeg[:, start_idx:end_idx].T
        
    return X

def run_ridge_sweep():
    print("--- 1. Loading AASD Dataset ---")
    data_root = '/kaggle/input/datasets/lokeshgile/aasd-processed-eeg/Processed EEG'
    mat_files = []
    for root, dirs, files in os.walk(data_root):
        for file in files:
            if file.endswith('.mat') and not file.startswith('._'):
                mat_files.append(os.path.join(root, file))
    
    if not mat_files:
        print("ERROR: No .mat files found. Please run on Kaggle.")
        return
        
    sub_path = next((p for p in mat_files if 'S18' in p), mat_files[0])
    
    b, a = scipy.signal.butter(4, [1.0/64.0, 8.0/64.0], btype='band')
    audio_dir = '/kaggle/input/datasets/lokeshgile/aasd-audio-gammatones'
    wav_dir = '/kaggle/input/datasets/lokeshgile/aasd-audio/Stimuli Audio'
    
    trials = load_aasd_subject_trials(sub_path, b, a, audio_dir, wav_dir)
    print(f"Loaded {len(trials)} trials from {os.path.basename(sub_path)}")
    
    train_trials = trials[:40]
    test_trials = trials[40:]
    
    # Grid Search Parameters
    lambda_vals = [1e-3, 1e-2, 1e-1, 1.0, 10.0, 1e2, 1e3, 1e4, 1e5]
    lag_vals = [8, 16, 24, 32] # 125ms, 250ms, 375ms, 500ms
    
    results = []
    
    # To avoid re-building the massive matrix every lambda step, we build it once per lag
    for max_lag in lag_vals:
        print(f"\n--- Testing Max Lag: {max_lag} ({max_lag/64.0 * 1000:.0f} ms) ---")
        X_train_list, Y_train_list = [], []
        
        for trial in train_trials:
            eeg = trial['eeg'].numpy()
            env_l = trial['env_l'].numpy()
            env_r = trial['env_r'].numpy()
            switch_points = trial['meta']['switch_points']
            
            eeg = (eeg - eeg.mean(axis=1, keepdims=True)) / (eeg.std(axis=1, keepdims=True) + 1e-8)
            env_l = (env_l - env_l.mean()) / (env_l.std() + 1e-8)
            env_r = (env_r - env_r.mean()) / (env_r.std() + 1e-8)
            
            X = build_lagged_matrix(eeg, max_lag)
            
            att = np.zeros(eeg.shape[1], dtype=np.float32)
            if len(switch_points) == 0:
                switch_points = [('R', 0)]
                
            if switch_points[0][1] > 0:
                initial_state = 'R' if switch_points[0][0] == 'L' else 'L'
            else:
                initial_state = switch_points[0][0]
                
            current_state = initial_state
            prev_idx = 0
            for state, idx_64 in switch_points:
                if idx_64 > prev_idx:
                    if current_state == 'L':
                        att[prev_idx:idx_64] = env_l[prev_idx:idx_64]
                    else:
                        att[prev_idx:idx_64] = env_r[prev_idx:idx_64]
                prev_idx, current_state = idx_64, state
                
            if current_state == 'L':
                att[prev_idx:] = env_l[prev_idx:]
            else:
                att[prev_idx:] = env_r[prev_idx:]
                
            Y = att[max_lag:]
            
            X_train_list.append(X)
            Y_train_list.append(Y)
            
        X_train = np.vstack(X_train_list)
        Y_train = np.concatenate(Y_train_list)
        
        cov_X = X_train.T @ X_train
        cov_XY = X_train.T @ Y_train
        
        # Build test matrices
        test_matrices = []
        for trial in test_trials:
            eeg = trial['eeg'].numpy()
            env_l = trial['env_l'].numpy()
            env_r = trial['env_r'].numpy()
            switch_points = trial['meta']['switch_points']
            
            eeg = (eeg - eeg.mean(axis=1, keepdims=True)) / (eeg.std(axis=1, keepdims=True) + 1e-8)
            env_l = (env_l - env_l.mean()) / (env_l.std() + 1e-8)
            env_r = (env_r - env_r.mean()) / (env_r.std() + 1e-8)
            
            X_test = build_lagged_matrix(eeg, max_lag)
            
            att = np.zeros(eeg.shape[1], dtype=np.float32)
            unatt = np.zeros(eeg.shape[1], dtype=np.float32)
            if len(switch_points) == 0:
                switch_points = [('R', 0)]
                
            if switch_points[0][1] > 0:
                initial_state = 'R' if switch_points[0][0] == 'L' else 'L'
            else:
                initial_state = switch_points[0][0]
                
            current_state = initial_state
            prev_idx = 0
            for state, idx_64 in switch_points:
                if idx_64 > prev_idx:
                    if current_state == 'L':
                        att[prev_idx:idx_64] = env_l[prev_idx:idx_64]
                        unatt[prev_idx:idx_64] = env_r[prev_idx:idx_64]
                    else:
                        att[prev_idx:idx_64] = env_r[prev_idx:idx_64]
                        unatt[prev_idx:idx_64] = env_l[prev_idx:idx_64]
                prev_idx, current_state = idx_64, state
                
            if current_state == 'L':
                att[prev_idx:] = env_l[prev_idx:]
                unatt[prev_idx:] = env_r[prev_idx:]
            else:
                att[prev_idx:] = env_r[prev_idx:]
                unatt[prev_idx:] = env_l[prev_idx:]
                
            test_matrices.append((X_test, att[max_lag:], unatt[max_lag:]))
            
        for alpha in lambda_vals:
            ridge_matrix = cov_X + alpha * np.eye(cov_X.shape[0])
            W = np.linalg.solve(ridge_matrix, cov_XY)
            
            sim_att, sim_unatt = [], []
            
            # Use a 5s window with 1s hop
            window_len = 64 * 5
            hop_len = 64 * 1
            
            for X_test, att, unatt in test_matrices:
                pred = X_test @ W
                for start in range(0, len(pred) - window_len + 1, hop_len):
                    end = start + window_len
                    
                    p = pred[start:end]
                    a = att[start:end]
                    u = unatt[start:end]
                    
                    c_a = np.corrcoef(p, a)[0, 1]
                    c_u = np.corrcoef(p, u)[0, 1]
                    
                    if not np.isnan(c_a) and not np.isnan(c_u):
                        sim_att.append(c_a)
                        sim_unatt.append(c_u)
                        
            sim_att = np.array(sim_att)
            sim_unatt = np.array(sim_unatt)
            
            margin = sim_att - sim_unatt
            acc = np.mean(margin > 0)
            y_true = np.concatenate([np.ones(len(sim_att)), np.zeros(len(sim_unatt))])
            y_scores = np.concatenate([sim_att, sim_unatt])
            auroc = roc_auc_score(y_true, y_scores)
            
            print(f" Lambda: {alpha:<8} | AUROC: {auroc:.4f} | Acc: {acc*100:.1f}% | Margin Mean: {margin.mean():.4f}")
            results.append({
                'lag': max_lag,
                'lambda': alpha,
                'auroc': auroc,
                'acc': acc,
                'margin': margin.mean()
            })
            
    # Find best
    best = max(results, key=lambda x: x['auroc'])
    print("\n" + "="*50)
    print(f"🏆 BEST COMBINATION: Lag={best['lag']} ({best['lag']/64.0*1000:.0f}ms), Lambda={best['lambda']}")
    print(f"🏆 BEST AUROC: {best['auroc']:.4f}")
    print(f"🏆 BEST ACCURACY: {best['acc']*100:.1f}%")
    print("="*50)

if __name__ == "__main__":
    run_ridge_sweep()
