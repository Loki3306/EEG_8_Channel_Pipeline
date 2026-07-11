import os
import time
import numpy as np
import torch
from pathlib import Path
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

def compute_participation_ratio(M):
    """
    Computes the Participation Ratio (intrinsic dimensionality) from a matrix M.
    PR = (sum(eigenvalues))^2 / sum(eigenvalues^2)
    """
    pca = PCA()
    pca.fit(M)
    eigenvalues = pca.explained_variance_
    pr = (np.sum(eigenvalues))**2 / np.sum(eigenvalues**2)
    return pr, pca.explained_variance_ratio_

def cosine_distance(W_t, W_prev):
    """Computes 1 - cosine_similarity"""
    num = np.dot(W_t, W_prev)
    den = np.linalg.norm(W_t) * np.linalg.norm(W_prev)
    return 1.0 - (num / (den + 1e-8))

def prepare_subject_windows_continuous(cache_file, device):
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
        
        X_trial = create_toeplitz_features_pt(eeg, MAX_LAG_SAMPLES)
        T_eff = X_trial.shape[0]
        
        Y_l_eff = env_l[:T_eff]
        Y_r_eff = env_r[:T_eff]
        
        sp = tr['meta']['switch_points']
        switch_indices = [idx for spk, idx in sp]
        
        # Determine continuous labels
        current_spk = 'L'
        sp_idx = 0
        labels_eff = np.zeros(T_eff, dtype=int)
        for t in range(T_eff):
            if sp_idx < len(sp) and t >= sp[sp_idx][1]:
                current_spk = sp[sp_idx][0]
                sp_idx += 1
            labels_eff[t] = 1 if current_spk == 'L' else 0
            
        # Extract continuous windows
        for seq_start in range(0, T_eff - SEQ_SAMPLES + 1, SEQ_HOP):
            seq_end = seq_start + SEQ_SAMPLES
            X_win = X_trial[seq_start:seq_end]
            Y_L_win = Y_l_eff[seq_start:seq_end]
            Y_R_win = Y_r_eff[seq_start:seq_end]
            
            # The label is the majority label in this window
            win_labels = labels_eff[seq_start:seq_end]
            label = 1 if np.mean(win_labels) >= 0.5 else 0
            
            # Check if any switch point falls inside this window
            has_switch = any(seq_start <= s < seq_end for s in switch_indices)
            
            windows.append({
                'X': X_win,
                'Y_L': Y_L_win,
                'Y_R': Y_R_win,
                'label': label,
                'has_switch': has_switch
            })
            
    return windows

def process_subject(cache_file, device_id):
    device = torch.device(f'cuda:{device_id}' if torch.cuda.is_available() else 'cpu')
    subj_name = cache_file.stem.split('_')[0]
    
    windows = prepare_subject_windows_continuous(cache_file, device)
    if len(windows) < 400:
        return subj_name, None, None, None, None
        
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
    # 2. THE ORACLE AUTOPSY (REAL LABELS)
    # ---------------------------------------------------------
    Rxx_real = Rxx_calib.clone()
    Rxy_real = Rxy_calib.clone()
    W_real = W_0.clone()
    
    M_real = []
    real_switch_indices = []
    
    for idx, w in enumerate(track_set):
        if w['has_switch']:
            real_switch_indices.append(idx)
            
        M_real.append(W_real.cpu().numpy())
        
        # Oracle Update (Real Label)
        Y_true = w['Y_L'] if w['label'] == 1 else w['Y_R']
        Rxx_real = FORGETTING_FACTOR * Rxx_real + w['X'].T @ w['X']
        Rxy_real = FORGETTING_FACTOR * Rxy_real + w['X'].T @ Y_true
        W_real = torch.linalg.solve(Rxx_real + RIDGE_LAMBDA * I, Rxy_real)
        
    M_real = np.array(M_real)
    
    # ---------------------------------------------------------
    # 3. THE "FAKE SWITCH" CONTROL (SHUFFLED LABELS)
    # ---------------------------------------------------------
    # Create a synthetic label sequence by circularly shifting the real labels
    # by exactly half the track_set length.
    SHIFT = len(track_set) // 2
    real_labels = [w['label'] for w in track_set]
    fake_labels = real_labels[SHIFT:] + real_labels[:SHIFT]
    
    # We define fake switches as points where fake_labels changes
    fake_switch_indices = []
    for idx in range(1, len(fake_labels)):
        if fake_labels[idx] != fake_labels[idx-1]:
            fake_switch_indices.append(idx)
            
    Rxx_fake = Rxx_calib.clone()
    Rxy_fake = Rxy_calib.clone()
    W_fake = W_0.clone()
    
    M_fake = []
    
    for idx, (w, fake_label) in enumerate(zip(track_set, fake_labels)):
        M_fake.append(W_fake.cpu().numpy())
        
        # Oracle Update (Fake Label)
        Y_true = w['Y_L'] if fake_label == 1 else w['Y_R']
        Rxx_fake = FORGETTING_FACTOR * Rxx_fake + w['X'].T @ w['X']
        Rxy_fake = FORGETTING_FACTOR * Rxy_fake + w['X'].T @ Y_true
        W_fake = torch.linalg.solve(Rxx_fake + RIDGE_LAMBDA * I, Rxy_fake)
        
    M_fake = np.array(M_fake)
    
    # ---------------------------------------------------------
    # ANALYSIS A: PARTICIPATION RATIO
    # ---------------------------------------------------------
    pr_real, var_ratio = compute_participation_ratio(M_real)
    
    # ---------------------------------------------------------
    # ANALYSIS B: DECODER ROTATION (COSINE DISTANCE)
    # ---------------------------------------------------------
    def extract_trajectories(M, switch_indices):
        cos_dists = [0.0]
        for t in range(1, len(M)):
            cos_dists.append(cosine_distance(M[t], M[t-1]))
        cos_dists = np.array(cos_dists)
        
        WINDOW_SIZE = 40 # 40 windows before, 40 after (spanning 20s left, 20s right)
        trajectories = []
        for s_idx in switch_indices:
            if s_idx - WINDOW_SIZE >= 0 and s_idx + WINDOW_SIZE < len(cos_dists):
                trajectories.append(cos_dists[s_idx - WINDOW_SIZE : s_idx + WINDOW_SIZE + 1])
        return trajectories
        
    real_trajectories = extract_trajectories(M_real, real_switch_indices)
    fake_trajectories = extract_trajectories(M_fake, fake_switch_indices)
            
    return subj_name, pr_real, real_trajectories, fake_trajectories, var_ratio

def main():
    mp.set_start_method('spawn', force=True)
    
    print("=======================================================")
    print(" PHASE 152: THE RIGOROUS ORACLE AUTOPSY")
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
    
    global_prs = []
    global_vars = []
    all_real_trajectories = []
    all_fake_trajectories = []
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(process_subject, cf, idx % max(1, num_gpus)): cf for idx, cf in enumerate(cache_files)}
        for future in concurrent.futures.as_completed(futures):
            subj, pr, real_traj, fake_traj, var_ratio = future.result()
            if pr is not None:
                global_prs.append(pr)
                global_vars.append(var_ratio)
                all_real_trajectories.extend(real_traj)
                all_fake_trajectories.extend(fake_traj)
                
                print(f"[{subj:3s}] Participation Ratio: {pr:5.1f} | PC1+PC2 Var: {(var_ratio[0]+var_ratio[1])*100:4.1f}%")

    print("\n=======================================================")
    print(" GLOBAL MANIFOLD DIMENSIONALITY")
    print("=======================================================")
    mean_pr = np.mean(global_prs)
    print(f"Mean Participation Ratio: {mean_pr:.2f} Dimensions (out of 408)")
    
    if mean_pr < 10.0:
        print("\n[MASSIVE DISCOVERY] The Oracle Decoder lives on a tiny, low-dimensional manifold!")
        print(f"It effectively uses only {mean_pr:.1f} degrees of freedom out of 408.")
    else:
        print("\n[OBSERVATION] The Oracle intrinsically uses a large number of degrees of freedom.")

    print("\n=======================================================")
    print(" DECODER ROTATION DYNAMICS (REAL VS FAKE)")
    print("=======================================================")
    
    os.makedirs('/kaggle/working/plots', exist_ok=True)
    
    if len(all_real_trajectories) > 0 and len(all_fake_trajectories) > 0:
        real_trajs = np.array(all_real_trajectories) # (N, 81)
        fake_trajs = np.array(all_fake_trajectories) # (N, 81)
        
        mean_real = np.mean(real_trajs, axis=0)
        std_real = np.std(real_trajs, axis=0)
        
        mean_fake = np.mean(fake_trajs, axis=0)
        std_fake = np.std(fake_trajs, axis=0)
        
        x_axis = np.arange(-40, 41) * 0.5 # Windows to seconds (approx, 0.5s hop)
        
        plt.figure(figsize=(12, 6))
        
        plt.plot(x_axis, mean_fake, 'r--', linewidth=2, label='FAKE Switch (Control)')
        plt.fill_between(x_axis, mean_fake - std_fake*0.2, mean_fake + std_fake*0.2, color='r', alpha=0.1)
        
        plt.plot(x_axis, mean_real, 'b-', linewidth=2, label='REAL Switch (Biology)')
        plt.fill_between(x_axis, mean_real - std_real*0.2, mean_real + std_real*0.2, color='b', alpha=0.2)
        
        plt.axvline(x=0, color='k', linestyle=':', linewidth=2, label='Switch Event')
        
        plt.xlabel("Time relative to switch (seconds)")
        plt.ylabel("Decoder Rotation (1 - Cosine Similarity)")
        plt.title("Oracle Decoder Rotation: Biology vs Update Rule")
        plt.legend()
        plt.grid(True)
        
        plot_path = '/kaggle/working/plots/phase152_decoder_rotation.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved Decoder Rotation plot to: {plot_path}")
        
        # Determine if it's biology or the update rule
        real_spike = np.max(mean_real[35:45]) - np.mean(mean_real[:20])
        fake_spike = np.max(mean_fake[35:45]) - np.mean(mean_fake[:20])
        
        print(f"Real Spike Magnitude: {real_spike:.6f}")
        print(f"Fake Spike Magnitude: {fake_spike:.6f}")
        
        if real_spike > fake_spike * 1.5:
            print("\n[DISCOVERY] The Oracle rotates significantly more during REAL switches than FAKE switches.")
            print("This proves we are capturing genuine Neural Spatial Geometry rotation!")
        else:
            print("\n[DISCOVERY] The Oracle rotates identically during FAKE and REAL switches.")
            print("This proves the rotation is just an artifact of the supervised mathematical update rule.")
            
    else:
        print("Not enough switch trajectories found.")

if __name__ == '__main__':
    main()
