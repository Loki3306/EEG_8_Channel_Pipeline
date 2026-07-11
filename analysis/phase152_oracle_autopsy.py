import os
import time
import numpy as np
import torch
from pathlib import Path
from sklearn.metrics import roc_auc_score
from sklearn.decomposition import PCA
from scipy import signal
import multiprocessing as mp
import concurrent.futures
import matplotlib.pyplot as plt

# -------------------------------------------------------------------------
# CONSTANTS & CONFIGURATION
# -------------------------------------------------------------------------
SR = 128
EAR_CHANNEL_INDICES = [23, 31, 32, 40, 14, 22, 41, 49]
MAX_LAG_SEC = 0.400 
MAX_LAG_SAMPLES = int(MAX_LAG_SEC * SR)
SEQ_SAMPLES = int(3.0 * SR)
SEQ_HOP = int(0.5 * SR)

BROADBAND = (0.5, 8.0)
RIDGE_LAMBDA = 100.0
FORGETTING_FACTOR = 0.98

def apply_modulation_filter(env, lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    b, a = signal.butter(order, [lowcut / nyq, highcut / nyq], btype='band')
    return signal.filtfilt(b, a, env, axis=1)

def create_toeplitz_features_pt(eeg, max_lag_samples):
    C, T = eeg.shape
    T_eff = T - max_lag_samples
    X = torch.zeros((T_eff, C * max_lag_samples), dtype=eeg.dtype, device=eeg.device)
    for tau in range(max_lag_samples):
        X[:, tau*C : (tau+1)*C] = eeg[:, tau : tau+T_eff].T
    return X

def batch_pearsonr_pt(x, y):
    x_mean = x - x.mean(dim=0, keepdim=True)
    y_mean = y - y.mean(dim=0, keepdim=True)
    num = (x_mean * y_mean).sum(dim=0)
    den = torch.sqrt((x_mean**2).sum(dim=0) * (y_mean**2).sum(dim=0))
    return num / (den + 1e-8)

def prepare_subject_windows(cache_file, device):
    cached = torch.load(cache_file, map_location='cpu', weights_only=False)['raw']
    windows = []
    
    for tr in cached:
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
        
        eeg = torch.tensor(eeg_f, dtype=torch.float32, device=device)
        env_l = torch.tensor(env_l_f[0], dtype=torch.float32, device=device)
        env_r = torch.tensor(env_r_f[0], dtype=torch.float32, device=device)
        
        T = eeg.shape[1]
        X_trial = create_toeplitz_features_pt(eeg, MAX_LAG_SAMPLES)
        T_eff = X_trial.shape[0]
        
        Y_l_eff = env_l[:T_eff]
        Y_r_eff = env_r[:T_eff]
        
        sp = tr['meta']['switch_points']
        boundaries = [0] + [idx for spk, idx in sp]
        boundaries = sorted(set(boundaries))
        if boundaries[-1] != T: boundaries.append(T)
            
        for i in range(len(boundaries) - 1):
            start_idx = boundaries[i]
            end_idx = boundaries[i+1]
            
            current_spk = 'L'
            for spk, idx in sp:
                if idx <= start_idx: current_spk = spk
                else: break
                
            safe_start = start_idx + int(1.5 * SR)
            safe_end = end_idx
            
            first_win_in_block = True
            if safe_end - safe_start >= SEQ_SAMPLES:
                for seq_start in range(safe_start, safe_end - SEQ_SAMPLES + 1, SEQ_HOP):
                    if seq_start + SEQ_SAMPLES <= T_eff:
                        X_win = X_trial[seq_start:seq_start + SEQ_SAMPLES]
                        Y_L_win = Y_l_eff[seq_start:seq_start + SEQ_SAMPLES]
                        Y_R_win = Y_r_eff[seq_start:seq_start + SEQ_SAMPLES]
                        label = 1 if current_spk == 'L' else 0
                        
                        is_switch = first_win_in_block and (i > 0) # True for the first window after a real switch
                        
                        windows.append({
                            'X': X_win,
                            'Y_L': Y_L_win,
                            'Y_R': Y_R_win,
                            'label': label,
                            'is_switch': is_switch
                        })
                        first_win_in_block = False
    return windows

def process_subject(cache_file, device_id):
    device = torch.device(f'cuda:{device_id}' if torch.cuda.is_available() else 'cpu')
    subj_name = cache_file.stem.split('_')[0]
    
    windows = prepare_subject_windows(cache_file, device)
    if len(windows) < 200:
        return subj_name, None, None, None
        
    CALIB_WINDOWS = 240
    calib_set = windows[:CALIB_WINDOWS]
    track_set = windows[CALIB_WINDOWS:]
    
    F = windows[0]['X'].shape[1]
    
    # ---------------------------------------------------------
    # 1. CALIBRATION
    # ---------------------------------------------------------
    Rxx_calib = torch.zeros((F, F), device=device)
    Rxy_calib = torch.zeros((F,), device=device)
    
    for w in calib_set:
        X = w['X']
        Y_true = w['Y_L'] if w['label'] == 1 else w['Y_R']
        Rxx_calib += X.T @ X
        Rxy_calib += X.T @ Y_true
        
    I = torch.eye(F, device=device)
    W_0 = torch.linalg.solve(Rxx_calib + RIDGE_LAMBDA * I, Rxy_calib)
    
    # ---------------------------------------------------------
    # 2. ORACLE TRACKER & FORENSIC COLLECTION
    # ---------------------------------------------------------
    Rxx = Rxx_calib.clone()
    Rxy = Rxy_calib.clone()
    W = W_0.clone()
    
    W_history = []
    switch_indices = []
    
    for idx, w in enumerate(track_set):
        if w['is_switch']:
            switch_indices.append(idx)
            
        W_history.append(W.cpu().numpy())
        
        # Oracle Update
        Y_true = w['Y_L'] if w['label'] == 1 else w['Y_R']
        Rxx = FORGETTING_FACTOR * Rxx + w['X'].T @ w['X']
        Rxy = FORGETTING_FACTOR * Rxy + w['X'].T @ Y_true
        W = torch.linalg.solve(Rxx + RIDGE_LAMBDA * I, Rxy)
        
    M = np.array(W_history) # Shape: (T, F)
    
    # ---------------------------------------------------------
    # ANALYSIS A: SVD MANIFOLD (PCA)
    # ---------------------------------------------------------
    pca = PCA()
    pca.fit(M)
    variance_explained = np.cumsum(pca.explained_variance_ratio_)
    
    # ---------------------------------------------------------
    # ANALYSIS B: DECODER VELOCITY AROUND SWITCHES
    # ---------------------------------------------------------
    # Compute velocity: || W_t - W_{t-1} ||_2
    diffs = np.diff(M, axis=0)
    velocities = np.linalg.norm(diffs, axis=1) # Length: T-1
    # Pad first element to make it length T
    velocities = np.insert(velocities, 0, velocities[0])
    
    WINDOW_SIZE = 20 # 20 windows before, 20 after
    switch_trajectories = []
    for s_idx in switch_indices:
        if s_idx - WINDOW_SIZE >= 0 and s_idx + WINDOW_SIZE < len(velocities):
            switch_trajectories.append(velocities[s_idx - WINDOW_SIZE : s_idx + WINDOW_SIZE + 1])
            
    return subj_name, variance_explained, switch_trajectories, M

def main():
    mp.set_start_method('spawn', force=True)
    
    print("=======================================================")
    print(" PHASE 152: THE ORACLE AUTOPSY")
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
    num_gpus = torch.cuda.device_count()
    num_workers = min(mp.cpu_count(), num_gpus if num_gpus > 0 else mp.cpu_count())
    
    start_time = time.time()
    
    global_variances = []
    all_switch_trajectories = []
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(process_subject, cf, idx % max(1, num_gpus)): cf for idx, cf in enumerate(cache_files)}
        for future in concurrent.futures.as_completed(futures):
            subj, var_expl, sw_traj, M = future.result()
            if var_expl is not None:
                global_variances.append(var_expl)
                all_switch_trajectories.extend(sw_traj)
                
                # Print individual PCA stats
                print(f"[{subj:3s}] Variance Explained - 1 PC: {var_expl[0]*100:4.1f}% | 3 PCs: {var_expl[2]*100:4.1f}% | 10 PCs: {var_expl[9]*100:4.1f}%")

    print("\n=======================================================")
    print(" GLOBAL MANIFOLD DIMENSIONALITY")
    print("=======================================================")
    # Pad variances with 1.0s if different lengths to average them
    max_len = max([len(v) for v in global_variances])
    padded_variances = np.array([np.pad(v, (0, max_len - len(v)), constant_values=1.0) for v in global_variances])
    mean_variances = np.mean(padded_variances, axis=0) * 100
    
    print(f"PC 1 : {mean_variances[0]:5.1f}%")
    print(f"PC 2 : {mean_variances[1]:5.1f}%")
    print(f"PC 3 : {mean_variances[2]:5.1f}%")
    print(f"PC 5 : {mean_variances[4]:5.1f}%")
    print(f"PC 10: {mean_variances[9]:5.1f}%")
    print(f"PC 20: {mean_variances[19]:5.1f}%")
    
    if mean_variances[2] > 90.0:
        print("\n[MASSIVE DISCOVERY] The Oracle Decoder lives on a tiny < 3D manifold!")
        print("We do not need 408 weights. We just need to track 3 parameters.")
    else:
        print("\n[OBSERVATION] The Oracle requires many degrees of freedom to track the environment.")

    print("\n=======================================================")
    print(" DECODER VELOCITY DYNAMICS")
    print("=======================================================")
    
    os.makedirs('/kaggle/working/plots', exist_ok=True)
    
    if len(all_switch_trajectories) > 0:
        trajs = np.array(all_switch_trajectories) # (N, 41)
        mean_traj = np.mean(trajs, axis=0)
        std_traj = np.std(trajs, axis=0)
        
        x_axis = np.arange(-20, 21) * 0.5 # Windows to seconds (approx, 0.5s hop)
        
        plt.figure(figsize=(10, 6))
        plt.plot(x_axis, mean_traj, 'b-', linewidth=2, label='Mean Decoder Velocity')
        plt.fill_between(x_axis, mean_traj - std_traj*0.5, mean_traj + std_traj*0.5, color='b', alpha=0.2)
        plt.axvline(x=0, color='r', linestyle='--', linewidth=2, label='True Attention Switch')
        plt.xlabel("Time relative to switch (seconds)")
        plt.ylabel("Decoder Velocity || W_t - W_{t-1} ||_2")
        plt.title("Oracle Decoder Velocity Around Attention Switches")
        plt.legend()
        plt.grid(True)
        
        plot_path = '/kaggle/working/plots/phase152_decoder_velocity.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved Decoder Velocity plot to: {plot_path}")
        
        # Determine if it's event-driven
        pre_switch_vel = np.mean(mean_traj[:15])
        switch_vel = np.mean(mean_traj[18:23])
        if switch_vel > pre_switch_vel * 1.5:
            print("\n[DISCOVERY] The Oracle experiences violent weight jumps exactly during attention switches.")
            print("This proves the adaptation is EVENT-DRIVEN, not just tracking slow drift.")
        else:
            print("\n[DISCOVERY] The Oracle velocity is smooth and independent of attention switches.")
            print("This proves the adaptation is primarily tracking BACKGROUND DRIFT, not the switches.")
            
    else:
        print("No switch trajectories found.")

if __name__ == '__main__':
    main()
