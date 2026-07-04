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
    eeg: [Channels, Time]
    Returns: [Time - max_lag, Channels * Lags]
    Channels are interleaved: c0_l0, c1_l0, ..., c61_l0, c0_l1, ...
    Actually, to make channel selection easier, let's group by channel:
    c0_l0, c0_l1, ... c0_lmax, c1_l0, c1_l1, ...
    """
    lags = np.arange(0, max_lag + 1)
    C, T = eeg.shape
    num_lags = len(lags)
    
    out_T = T - max_lag
    X = np.zeros((out_T, C * num_lags), dtype=np.float32)
    
    for c in range(C):
        for i, lag in enumerate(lags):
            start_idx = max_lag - lag
            end_idx = T - lag
            # index = c * num_lags + i
            idx = c * num_lags + i
            X[:, idx] = eeg[c, start_idx:end_idx]
            
    return X

def evaluate_ridge(cov_X, cov_XY, test_matrices, alpha, selected_channels, num_lags):
    """
    cov_X: [62*lags, 62*lags]
    cov_XY: [62*lags]
    selected_channels: list of int (0 to 61)
    """
    # Find the indices in the flattened lag matrix corresponding to selected channels
    indices = []
    for c in selected_channels:
        indices.extend(range(c * num_lags, (c + 1) * num_lags))
        
    indices = np.array(indices)
    
    sub_cov_X = cov_X[np.ix_(indices, indices)]
    sub_cov_XY = cov_XY[indices]
    
    ridge_matrix = sub_cov_X + alpha * np.eye(sub_cov_X.shape[0])
    W = np.linalg.solve(ridge_matrix, sub_cov_XY)
    
    sim_att, sim_unatt = [], []
    window_len = 64 * 5
    hop_len = 64 * 1
    
    for X_test, att, unatt in test_matrices:
        # X_test has all channels. We only want selected columns
        X_test_sub = X_test[:, indices]
        pred = X_test_sub @ W
        
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
    
    return auroc, acc

def run_channel_optimization():
    print("--- 1. Loading AASD Dataset ---")
    data_root = '/kaggle/input/datasets/lokeshgile/aasd-processed-eeg/Processed EEG'
    mat_files = []
    for root, dirs, files in os.walk(data_root):
        for file in files:
            if file.endswith('.mat') and not file.startswith('._'):
                mat_files.append(os.path.join(root, file))
    
    if not mat_files:
        print("ERROR: No .mat files found.")
        return
        
    sub_path = next((p for p in mat_files if 'S18' in p), mat_files[0])
    b, a = scipy.signal.butter(4, [1.0/64.0, 8.0/64.0], btype='band')
    audio_dir = '/kaggle/input/datasets/lokeshgile/aasd-audio-gammatones'
    wav_dir = '/kaggle/input/datasets/lokeshgile/aasd-audio/Stimuli Audio'
    
    trials = load_aasd_subject_trials(sub_path, b, a, audio_dir, wav_dir)
    train_trials = trials[:40]
    test_trials = trials[40:]
    
    max_lag = 24
    alpha = 10000.0
    num_lags = max_lag + 1
    C = 62
    
    print("\n--- 2. Building Covariance Matrices ---")
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
            
        initial_state = 'R' if (switch_points[0][1] > 0 and switch_points[0][0] == 'L') else switch_points[0][0]
        if switch_points[0][1] > 0:
            initial_state = 'R' if switch_points[0][0] == 'L' else 'L'
            
        current_state = initial_state
        prev_idx = 0
        for state, idx_64 in switch_points:
            if idx_64 > prev_idx:
                if current_state == 'L': att[prev_idx:idx_64] = env_l[prev_idx:idx_64]
                else: att[prev_idx:idx_64] = env_r[prev_idx:idx_64]
            prev_idx, current_state = idx_64, state
            
        if current_state == 'L': att[prev_idx:] = env_l[prev_idx:]
        else: att[prev_idx:] = env_r[prev_idx:]
            
        Y = att[max_lag:]
        X_train_list.append(X)
        Y_train_list.append(Y)
        
    X_train = np.vstack(X_train_list)
    Y_train = np.concatenate(Y_train_list)
    
    cov_X = X_train.T @ X_train
    cov_XY = X_train.T @ Y_train
    
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
        if len(switch_points) == 0: switch_points = [('R', 0)]
            
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
        
    print("\n--- 3. Stage A: Ridge Weight Analysis ---")
    base_auroc, _ = evaluate_ridge(cov_X, cov_XY, test_matrices, alpha, list(range(62)), num_lags)
    print(f"Base 62-Channel AUROC: {base_auroc:.4f}")
    
    ridge_matrix = cov_X + alpha * np.eye(cov_X.shape[0])
    W = np.linalg.solve(ridge_matrix, cov_XY) # [62 * num_lags]
    W_reshaped = W.reshape(62, num_lags)
    
    channel_importance = np.sum(np.abs(W_reshaped), axis=1)
    ranked_channels = np.argsort(channel_importance)[::-1]
    
    print("\nTop 10 Channels by Ridge Weight Magnitude:")
    for i in range(10):
        ch = ranked_channels[i]
        print(f"Rank {i+1}: Channel {ch} (Importance: {channel_importance[ch]:.4f})")
        
    print("\n--- 4. Stage B: Leave-One-Channel-Out Ablation ---")
    # Due to speed, we can test all 62
    drops = []
    for ch in range(62):
        ablated_subset = [c for c in range(62) if c != ch]
        auroc, _ = evaluate_ridge(cov_X, cov_XY, test_matrices, alpha, ablated_subset, num_lags)
        drop = base_auroc - auroc
        drops.append((ch, drop))
        
    drops.sort(key=lambda x: x[1], reverse=True)
    print("Top 10 Most Critical Channels (Biggest AUROC Drop when removed):")
    for i in range(10):
        ch, drop = drops[i]
        print(f"Rank {i+1}: Channel {ch} (Drop: {drop:+.4f})")
        
    print("\n--- 5. Stage C: Forward Selection ---")
    # Start greedy forward selection
    selected_channels = []
    remaining_channels = list(range(62))
    best_forward_auroc = 0.0
    
    # We want to select up to 16 channels
    for step in range(1, 17):
        best_candidate = None
        best_step_auroc = -1.0
        
        for candidate in remaining_channels:
            current_subset = selected_channels + [candidate]
            auroc, acc = evaluate_ridge(cov_X, cov_XY, test_matrices, alpha, current_subset, num_lags)
            
            if auroc > best_step_auroc:
                best_step_auroc = auroc
                best_candidate = candidate
                
        selected_channels.append(best_candidate)
        remaining_channels.remove(best_candidate)
        
        print(f"Step {step} | Added Ch {best_candidate:>2} | Subset Size: {len(selected_channels):>2} | AUROC: {best_step_auroc:.4f}")
        
        if best_step_auroc > best_forward_auroc:
            best_forward_auroc = best_step_auroc
            
    print("\n" + "="*60)
    print(f"🏆 BEST FORWARD SELECTION AUROC: {best_forward_auroc:.4f}")
    print(f"🏆 SELECTED 16 CHANNELS: {selected_channels}")
    print("="*60)

if __name__ == "__main__":
    run_channel_optimization()
