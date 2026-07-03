import os
import sys
import numpy as np
import scipy.io
import scipy.io.wavfile
import scipy.signal
from sklearn.metrics import roc_auc_score
import time
from pathlib import Path
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Reuse the robust loading logic we just built
from training.phase32_5_spatial_fix import load_aasd_subject_trials

def build_lagged_matrix(eeg, lags):
    """
    Builds a time-lagged EEG matrix for mTRF stimulus reconstruction.
    eeg: [Channels, Time]
    Returns: [Time - max_lag, Channels * Lags]
    """
    C, T = eeg.shape
    num_lags = len(lags)
    max_lag = max(lags)
    
    # We will lose the first 'max_lag' samples
    out_T = T - max_lag
    X = np.zeros((out_T, C * num_lags), dtype=np.float32)
    
    for i, lag in enumerate(lags):
        # If lag is 0, we take from max_lag to end
        # If lag is max_lag, we take from 0 to end - max_lag
        start_idx = max_lag - lag
        end_idx = T - lag
        X[:, i*C:(i+1)*C] = eeg[:, start_idx:end_idx].T
        
    return X, max_lag

def run_linear_baseline():
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
    
    # Split: 40 train, 20 test (Wait, the data has 60 trials total)
    train_trials = trials[:40]
    test_trials = trials[40:]
    
    # Lags: 0 to 250ms at 64Hz = 0 to 16 samples
    lags = np.arange(0, 17)
    
    print("\n--- 2. Building Training Matrices ---")
    X_train_list, Y_train_list = [], []
    
    for trial in train_trials:
        # Convert to numpy
        eeg = trial['eeg'].numpy()
        env_l = trial['env_l'].numpy()
        env_r = trial['env_r'].numpy()
        switch_points = trial['meta']['switch_points']
        
        # Normalize
        eeg = (eeg - eeg.mean(axis=1, keepdims=True)) / (eeg.std(axis=1, keepdims=True) + 1e-8)
        env_l = (env_l - env_l.mean()) / (env_l.std() + 1e-8)
        env_r = (env_r - env_r.mean()) / (env_r.std() + 1e-8)
        
        X, max_lag = build_lagged_matrix(eeg, lags)
        
        # Build attended envelope
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
    
    print(f"Training matrix shape: {X_train.shape}")
    
    print("\n--- 3. Training Ridge Decoder (mTRF) ---")
    start_time = time.time()
    
    # X^T X
    cov_X = X_train.T @ X_train
    # X^T Y
    cov_XY = X_train.T @ Y_train
    
    # Regularization (Ridge)
    alpha = 1000.0
    ridge_matrix = cov_X + alpha * np.eye(cov_X.shape[0])
    
    # Solve for weights
    W = np.linalg.solve(ridge_matrix, cov_XY)
    
    print(f"Decoder trained in {time.time() - start_time:.2f}s")
    
    print("\n--- 4. Testing ---")
    sim_att = []
    sim_unatt = []
    
    for trial in test_trials:
        eeg = trial['eeg'].numpy()
        env_l = trial['env_l'].numpy()
        env_r = trial['env_r'].numpy()
        switch_points = trial['meta']['switch_points']
        
        eeg = (eeg - eeg.mean(axis=1, keepdims=True)) / (eeg.std(axis=1, keepdims=True) + 1e-8)
        env_l = (env_l - env_l.mean()) / (env_l.std() + 1e-8)
        env_r = (env_r - env_r.mean()) / (env_r.std() + 1e-8)
        
        X_test, max_lag = build_lagged_matrix(eeg, lags)
        
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
            
        # Segment into evaluation windows (e.g., 5 seconds)
        window_len = 64 * 5
        hop_len = 64 * 1
        
        pred_full = X_test @ W
        att_full = att[max_lag:]
        unatt_full = unatt[max_lag:]
        
        for start in range(0, len(pred_full) - window_len + 1, hop_len):
            end = start + window_len
            pred_w = pred_full[start:end]
            att_w = att_full[start:end]
            unatt_w = unatt_full[start:end]
            
            corr_att = np.corrcoef(pred_w, att_w)[0, 1]
            corr_unatt = np.corrcoef(pred_w, unatt_w)[0, 1]
            
            if not np.isnan(corr_att) and not np.isnan(corr_unatt):
                sim_att.append(corr_att)
                sim_unatt.append(corr_unatt)
                
    sim_att = np.array(sim_att)
    sim_unatt = np.array(sim_unatt)
    
    margin = sim_att - sim_unatt
    acc = np.mean(margin > 0)
    
    y_true = np.concatenate([np.ones(len(sim_att)), np.zeros(len(sim_unatt))])
    y_scores = np.concatenate([sim_att, sim_unatt])
    auroc = roc_auc_score(y_true, y_scores)
    
    print(f"Test P(Att): {sim_att.mean():.4f}")
    print(f"Test P(Unatt): {sim_unatt.mean():.4f}")
    print(f"Margin Mean: {margin.mean():.4f}")
    print(f"Margin Std: {margin.std():.4f}")
    print(f"Test Accuracy: {acc*100:.1f}%")
    print(f"Test AUROC: {auroc:.4f}")

if __name__ == "__main__":
    run_linear_baseline()
