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
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.phase32_5_spatial_fix import load_aasd_subject_trials
from training.phase35_neural_ridge import NeuralRidgeDecoder, build_lagged_matrix, pearson_loss

def run_test_suite(model, test_trials, selected_channels, device, max_lag, hop_len, window_len, suite_mode):
    sim_att, sim_unatt = [], []
    for trial_idx, trial in enumerate(test_trials):
        eeg_full = trial['eeg'].numpy()
        env_l = trial['env_l'].numpy()
        env_r = trial['env_r'].numpy()
        switch_points = trial['meta']['switch_points']
        
        if suite_mode == "SHUFFLE_AUDIO":
            alt_trial = test_trials[(trial_idx + 1) % len(test_trials)]
            env_l = alt_trial['env_l'].numpy()
            env_r = alt_trial['env_r'].numpy()
            switch_points = alt_trial['meta']['switch_points']
            
            min_len = min(eeg_full.shape[1], len(env_l))
            eeg_full = eeg_full[:, :min_len]
            env_l = env_l[:min_len]
            env_r = env_r[:min_len]
            
        eeg_full = (eeg_full - eeg_full.mean(axis=1, keepdims=True)) / (eeg_full.std(axis=1, keepdims=True) + 1e-8)
        env_l = (env_l - env_l.mean()) / (env_l.std() + 1e-8)
        env_r = (env_r - env_r.mean()) / (env_r.std() + 1e-8)
        
        eeg = eeg_full[selected_channels, :]
        
        if suite_mode == "REVERSE_EEG":
            eeg = np.flip(eeg, axis=1).copy()
        elif suite_mode == "NOISE_EEG":
            eeg = np.random.randn(*eeg.shape).astype(np.float32)
            
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
            idx_64 = min(idx_64, eeg.shape[1])
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
            
        for start in range(0, eeg.shape[1] - window_len + 1, hop_len):
            end = start + window_len
            
            eeg_w = torch.from_numpy(eeg[:, start:end]).unsqueeze(0).to(device)
            att_w = att[start:end]
            unatt_w = unatt[start:end]
            
            pred_w = model(eeg_w).squeeze().cpu().numpy()
            
            c_a = np.corrcoef(pred_w, att_w)[0, 1]
            c_u = np.corrcoef(pred_w, unatt_w)[0, 1]
            
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
    
    return auroc, acc, sim_att.mean(), sim_unatt.mean()

def run_falsification():
    print("--- 1. Loading AASD Dataset ---")
    data_root = '/kaggle/input/datasets/lokeshgile/aasd-processed-eeg/Processed EEG'
    mat_files = []
    for root, dirs, files in os.walk(data_root):
        for file in files:
            if file.endswith('.mat') and not file.startswith('._'):
                mat_files.append(os.path.join(root, file))
    
    sub_path = next((p for p in mat_files if 'S18' in p), mat_files[0])
    b, a = scipy.signal.butter(4, [1.0/64.0, 8.0/64.0], btype='band')
    audio_dir = '/kaggle/input/datasets/lokeshgile/aasd-audio-gammatones'
    wav_dir = '/kaggle/input/datasets/lokeshgile/aasd-audio/Stimuli Audio'
    
    trials = load_aasd_subject_trials(sub_path, b, a, audio_dir, wav_dir)
    print(f"Loaded {len(trials)} trials from {os.path.basename(sub_path)}")
    
    train_trials = trials[:40]
    test_trials = trials[40:]
    
    max_lag = 24
    alpha_ridge = 10000.0
    selected_channels = [29, 26, 31, 5, 13, 21, 22, 44, 55, 41, 45, 15, 17, 56, 61, 60]
    num_lags = max_lag + 1
    C = len(selected_channels)
    
    window_len = 64 * 5
    hop_len = 64 * 1
    
    print("\n--- 2. Computing Analytical Ridge Solution ---")
    X_train_list, Y_train_list = [], []
    X_train_raw, Y_train_raw = [], [] 
    
    for trial in train_trials:
        eeg_full = trial['eeg'].numpy()
        env_l = trial['env_l'].numpy()
        env_r = trial['env_r'].numpy()
        switch_points = trial['meta']['switch_points']
        
        eeg_full = (eeg_full - eeg_full.mean(axis=1, keepdims=True)) / (eeg_full.std(axis=1, keepdims=True) + 1e-8)
        env_l = (env_l - env_l.mean()) / (env_l.std() + 1e-8)
        env_r = (env_r - env_r.mean()) / (env_r.std() + 1e-8)
        
        eeg = eeg_full[selected_channels, :]
        
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
        
        for start in range(0, eeg.shape[1] - window_len + 1, hop_len):
            end = start + window_len
            is_transition = False
            for _, sw_idx in switch_points:
                if start < sw_idx < end:
                    is_transition = True
                    break
            
            if not is_transition:
                X_train_raw.append(torch.from_numpy(eeg[:, start:end]))
                Y_train_raw.append(torch.from_numpy(att[start:end]))
        
    X_train = np.vstack(X_train_list)
    Y_train = np.concatenate(Y_train_list)
    
    cov_X = X_train.T @ X_train
    cov_XY = X_train.T @ Y_train
    
    ridge_matrix = cov_X + alpha_ridge * np.eye(cov_X.shape[0])
    W_analytical = np.linalg.solve(ridge_matrix, cov_XY)
    
    print("\n--- 3. Training Neural Ridge Hybrid ---")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = NeuralRidgeDecoder(in_channels=C, lags=max_lag).to(device)
    model.load_analytical_weights(W_analytical)
    
    X_nn = torch.stack(X_train_raw)
    Y_nn = torch.stack(Y_train_raw)
    dataset = TensorDataset(X_nn, Y_nn)
    dataloader = DataLoader(dataset, batch_size=256, shuffle=True)
    
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3, weight_decay=1e-2)
    model.train()
    for epoch in range(1, 51):  # 50 epochs is enough
        for bx, by in dataloader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            pred = model(bx)
            loss = pearson_loss(pred, by)
            loss.backward()
            optimizer.step()
            
    print("Neural Hybrid Training completed.")
    
    print("\n--- 4. Executing Falsification Suites ---")
    model.eval()
    
    suites = [
        ("Suite A: Ground Truth Baseline", "NORMAL"),
        ("Suite B: Shuffled Audio Envelopes", "SHUFFLE_AUDIO"),
        ("Suite C: Time-Reversed EEG", "REVERSE_EEG"),
        ("Suite D: Gaussian Noise EEG", "NOISE_EEG")
    ]
    
    with torch.no_grad():
        for name, mode in suites:
            print(f"\nEvaluating {name}...")
            auroc, acc, p_att, p_unatt = run_test_suite(
                model, test_trials, selected_channels, device, 
                max_lag, hop_len, window_len, mode
            )
            print(f"  AUROC: {auroc:.4f}")
            print(f"  Accuracy: {acc*100:.1f}%")
            print(f"  P(Att): {p_att:.4f} | P(Unatt): {p_unatt:.4f}")
            
    print("\n========================================================")
    print("FALSIFICATION COMPLETE. If Suites B, C, D collapsed to ~0.50,")
    print("then the model is scientifically sound!")
    print("========================================================")

if __name__ == "__main__":
    run_falsification()
