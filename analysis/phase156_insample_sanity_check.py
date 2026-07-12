import os
import torch
import numpy as np
import scipy.signal
import scipy.stats
import random

CACHE_DIR = "/kaggle/input/datasets/lowkieee/multiband-cache/kaggle/working/multiband_cache"
SR = 128
EAR_CHANNEL_INDICES = [0, 1, 2, 3, 27, 28, 29, 30]
BROADBAND = (0.5, 8.0)
MAX_LAG_MS = 400
SEQ_SAMPLES = int(3.0 * SR)  # 3s window
RIDGE_LAMBDA = 100.0

def apply_modulation_filter(envelope, lowcut, highcut, sr):
    nyq = 0.5 * sr
    low = lowcut / nyq
    high = highcut / nyq
    b, a = scipy.signal.butter(3, [low, high], btype='band')
    return scipy.signal.filtfilt(b, a, envelope, axis=-1)

def build_toeplitz(eeg, max_lag_samples):
    C, T = eeg.shape
    T_eff = T - max_lag_samples + 1
    X = np.zeros((T_eff, C * max_lag_samples), dtype=np.float32)
    for tau in range(max_lag_samples):
        X[:, tau * C : (tau + 1) * C] = eeg[:, tau : tau + T_eff].T
    return X

def extract_trials(cache_file, num_trials=5, permute_audio=False):
    cached = torch.load(cache_file, map_location='cpu', weights_only=False)['raw']
    max_lag_samples = int(MAX_LAG_MS / 1000.0 * SR) + 1
    
    X_all = []
    Y_L_all = []
    Y_R_all = []
    labels_all = []
    
    for tr_idx, tr in enumerate(cached[:num_trials]):
        eeg_raw = tr['eeg'].numpy()[EAR_CHANNEL_INDICES, :] 
        
        env_l_raw = tr['env_l'].numpy().mean(axis=0, keepdims=True)
        env_r_raw = tr['env_r'].numpy().mean(axis=0, keepdims=True)
        
        min_len = min(eeg_raw.shape[1], env_l_raw.shape[1])
        eeg_raw = eeg_raw[:, :min_len]
        env_l_raw = env_l_raw[:, :min_len]
        env_r_raw = env_r_raw[:, :min_len]
        
        # Z-score EEG
        eeg_raw = (eeg_raw - np.mean(eeg_raw, axis=1, keepdims=True)) / (np.std(eeg_raw, axis=1, keepdims=True) + 1e-8)
        
        # Filter and Z-score envelope
        env_l_f = apply_modulation_filter(env_l_raw, BROADBAND[0], BROADBAND[1], SR)
        env_l_f_z = (env_l_f - np.mean(env_l_f, axis=1, keepdims=True)) / (np.std(env_l_f, axis=1, keepdims=True) + 1e-8)
        
        env_r_f = apply_modulation_filter(env_r_raw, BROADBAND[0], BROADBAND[1], SR)
        env_r_f_z = (env_r_f - np.mean(env_r_f, axis=1, keepdims=True)) / (np.std(env_r_f, axis=1, keepdims=True) + 1e-8)
        
        X_trial = build_toeplitz(eeg_raw, max_lag_samples)
        
        T_eff = X_trial.shape[0]
        # Our Toeplitz matrix row t uses eeg[:, t : t + max_lags]
        # To predict audio at time t. Is that right? The biological response to audio at time t happens at t to t+400ms.
        # So yes, X_trial[t] corresponds to audio at t.
        Y_l_eff = env_l_f_z[0, :T_eff]
        Y_r_eff = env_r_f_z[0, :T_eff]
        
        if permute_audio:
            np.random.shuffle(Y_l_eff)
            np.random.shuffle(Y_r_eff)
            
        sp = tr['meta']['switch_points']
        current_spk = 'L'
        sp_idx = 0
        labels_eff = np.zeros(T_eff, dtype=int)
        for t in range(T_eff):
            if sp_idx < len(sp) and t >= sp[sp_idx][1]:
                current_spk = sp[sp_idx][0]
                sp_idx += 1
            labels_eff[t] = 1 if current_spk == 'L' else 0
        
        for seq_start in range(0, T_eff - SEQ_SAMPLES + 1, SEQ_SAMPLES):
            seq_end = seq_start + SEQ_SAMPLES
            win_labels = labels_eff[seq_start:seq_end]
            
            if np.mean(win_labels) != 0 and np.mean(win_labels) != 1:
                continue
                
            label = int(win_labels[0])
            if tr_idx >= 30:
                label = 1 - label
                
            X_all.append(X_trial[seq_start:seq_end])
            Y_L_all.append(Y_l_eff[seq_start:seq_end])
            Y_R_all.append(Y_r_eff[seq_start:seq_end])
            labels_all.append(label)
            
    return X_all, Y_L_all, Y_R_all, labels_all

def run_experiment(cache_file, permute=False):
    X, Y_L, Y_R, labels = extract_trials(cache_file, num_trials=5, permute_audio=permute)
    
    if not X:
        print("No valid windows found!")
        return
        
    F = X[0].shape[1]
    Rxx = np.zeros((F, F))
    Rxy = np.zeros(F)
    
    total_samples = 0
    for i in range(len(X)):
        Y_true = Y_L[i] if labels[i] == 1 else Y_R[i]
        Rxx += X[i].T @ X[i]
        Rxy += X[i].T @ Y_true
        total_samples += X[i].shape[0]
        
    Rxx /= total_samples
    Rxy /= total_samples
    
    W = np.linalg.solve(Rxx + RIDGE_LAMBDA * np.eye(F), Rxy)
    
    # Evaluate in-sample
    r_att_list = []
    r_unatt_list = []
    correct = 0
    
    for i in range(len(X)):
        y_pred = X[i] @ W
        
        c_L, _ = scipy.stats.pearsonr(y_pred, Y_L[i])
        c_R, _ = scipy.stats.pearsonr(y_pred, Y_R[i])
        
        c_att = c_L if labels[i] == 1 else c_R
        c_unatt = c_R if labels[i] == 1 else c_L
        
        r_att_list.append(c_att)
        r_unatt_list.append(c_unatt)
        
        if (c_L > c_R and labels[i] == 1) or (c_R > c_L and labels[i] == 0):
            correct += 1
            
    r_att_arr = np.array(r_att_list)
    r_unatt_arr = np.array(r_unatt_list)
    delta_r = r_att_arr - r_unatt_arr
    
    acc = correct / len(X)
    print(f"--- Results (Permute: {permute}) ---")
    print(f"Windows: {len(X)}")
    print(f"Mean r_att:   {np.mean(r_att_arr):.4f}")
    print(f"Mean r_unatt: {np.mean(r_unatt_arr):.4f}")
    print(f"Mean Delta_r: {np.mean(delta_r):.4f}")
    print(f"Accuracy:     {acc*100:.1f}%")
    print()

if __name__ == "__main__":
    cache_path = os.path.join(CACHE_DIR, "S10_multiband.pt")
    if not os.path.exists(cache_path):
        print(f"Run locally on valid dataset: {cache_path} not found.")
    else:
        print("Experiment 1: True Alignment")
        run_experiment(cache_path, permute=False)
        
        print("Experiment 2: Permuted Alignment")
        run_experiment(cache_path, permute=True)
