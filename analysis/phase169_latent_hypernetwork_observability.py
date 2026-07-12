import os
# MUST set before importing torch or numpy to prevent thread thrashing in multiprocessing!
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import time
import numpy as np
import torch
from pathlib import Path
from scipy import signal
import concurrent.futures
import multiprocessing as mp
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

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
RIDGE_LAMBDA = 2.0
CALIB_MINUTES = 2.0
CALIB_WINDOWS = int((CALIB_MINUTES * 60) / 0.5)

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

def get_spectral_features(eeg_raw):
    # eeg_raw: (C, T) numpy array
    freqs, psd = signal.welch(eeg_raw, fs=SR, nperseg=SR)
    
    delta_idx = (freqs >= 1) & (freqs < 4)
    theta_idx = (freqs >= 4) & (freqs < 8)
    alpha_idx = (freqs >= 8) & (freqs < 13)
    beta_idx = (freqs >= 13) & (freqs < 30)
    
    delta_power = np.sum(psd[:, delta_idx], axis=1)
    theta_power = np.sum(psd[:, theta_idx], axis=1)
    alpha_power = np.sum(psd[:, alpha_idx], axis=1)
    beta_power = np.sum(psd[:, beta_idx], axis=1)
    
    return np.concatenate([delta_power, theta_power, alpha_power, beta_power])

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
        
        env_l_f = (env_l_f - np.mean(env_l_f, axis=1, keepdims=True)) / (np.std(env_l_f, axis=1, keepdims=True) + 1e-8)
        env_r_f = (env_r_f - np.mean(env_r_f, axis=1, keepdims=True)) / (np.std(env_r_f, axis=1, keepdims=True) + 1e-8)
        
        eeg = torch.tensor(eeg_f.copy(), dtype=torch.float32, device=device)
        env_l = torch.tensor(env_l_f[0].copy(), dtype=torch.float32, device=device)
        env_r = torch.tensor(env_r_f[0].copy(), dtype=torch.float32, device=device)
        
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
            
            if safe_end - safe_start >= SEQ_SAMPLES:
                for seq_start in range(safe_start, safe_end - SEQ_SAMPLES + 1, SEQ_HOP):
                    if seq_start + SEQ_SAMPLES <= T_eff:
                        X_win = X_trial[seq_start:seq_start + SEQ_SAMPLES]
                        Y_L_win = Y_l_eff[seq_start:seq_start + SEQ_SAMPLES]
                        Y_R_win = Y_r_eff[seq_start:seq_start + SEQ_SAMPLES]
                        
                        # Extract raw EEG for spectral analysis (not delayed)
                        raw_eeg_win = eeg_raw[:, seq_start:seq_start + SEQ_SAMPLES]
                        spectral_feats = get_spectral_features(raw_eeg_win)
                        
                        label = 1 if current_spk == 'L' else 0
                        
                        windows.append({
                            'X': X_win,
                            'Y_L': Y_L_win,
                            'Y_R': Y_R_win,
                            'spectral_feats': spectral_feats,
                            'label': label
                        })
    return windows

def process_subject(cache_file):
    torch.set_num_threads(1)
    device = torch.device('cpu')
    subj_name = cache_file.stem.split('_')[0]
    
    windows = prepare_subject_windows(cache_file, device)
    if len(windows) < CALIB_WINDOWS:
        return subj_name, None
        
    F = windows[0]['X'].shape[1]
    I = torch.eye(F, device=device)
    
    calib_set = windows[:CALIB_WINDOWS]
    track_set = windows[CALIB_WINDOWS:]
    
    # 1. Calibration
    Rxx_calib = torch.zeros((F, F), device=device)
    Rxy_calib = torch.zeros((F,), device=device)
    
    count = 0
    for w in calib_set:
        X = w['X']
        Y_true = w['Y_L'] if w['label'] == 1 else w['Y_R']
        Rxx_calib += X.T @ X
        Rxy_calib += X.T @ Y_true
        count += 1
        
    C_xx = Rxx_calib / count
    C_xy = Rxy_calib / count
    
    W_static = torch.linalg.solve(C_xx + RIDGE_LAMBDA * I, C_xy).cpu().numpy()
    
    # 2. Compute Oracle EMA Trajectory W_t and Unsupervised Spectral Stats
    W_oracles = []
    spectral_features = []
    
    for w in track_set:
        X = w['X']
        Y_true = w['Y_L'] if w['label'] == 1 else w['Y_R']
        
        # Oracle EMA (Alpha = 0.02 matches our optimal filter)
        C_xx = (1.0 - 0.02) * C_xx + 0.02 * (X.T @ X)
        C_xy = (1.0 - 0.02) * C_xy + 0.02 * (X.T @ Y_true)
        W_t = torch.linalg.solve(C_xx + RIDGE_LAMBDA * I, C_xy)
        W_oracles.append(W_t.cpu().numpy())
        
        spectral_features.append(w['spectral_feats'])
        
    W_oracles = np.stack(W_oracles)
    spectral_features = np.stack(spectral_features)
    
    # Optional: We can add temporal context to the spectral features
    # since brain states evolve slowly. Let's smooth the spectral features.
    # A simple moving average of 5 windows (2.5 seconds)
    smoothed_features = np.zeros_like(spectral_features)
    for i in range(len(spectral_features)):
        start_idx = max(0, i - 4)
        smoothed_features[i] = np.mean(spectral_features[start_idx:i+1], axis=0)
        
    # 3. TimeSeriesSplit Regression to avoid Leakage
    tscv = TimeSeriesSplit(n_splits=5)
    reconstructed_W = np.zeros_like(W_oracles)
    
    for train_idx, test_idx in tscv.split(smoothed_features):
        X_train, X_test = smoothed_features[train_idx], smoothed_features[test_idx]
        
        # Fit scaler strictly on training fold
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        W_train = W_oracles[train_idx]
        
        # Fit PCA strictly on the training fold to prevent leakage
        pca = PCA(n_components=18)
        a_train = pca.fit_transform(W_train)
        
        # Multi-output Ridge CV
        reg = RidgeCV(alphas=np.logspace(-2, 4, 10))
        reg.fit(X_train, a_train)
        
        # Predict test latent coords and inverse-transform to full space
        a_test_pred = reg.predict(X_test)
        W_test_pred = pca.inverse_transform(a_test_pred)
        reconstructed_W[test_idx] = W_test_pred
        
    # 4. Evaluate downstream accuracy of the Reconstructed W
    predicted_indices = []
    for _, test_idx in tscv.split(smoothed_features):
        predicted_indices.extend(test_idx)
    predicted_indices = np.array(predicted_indices)
    
    if len(predicted_indices) == 0:
        return subj_name, None
        
    correct_recon = 0
    correct_oracle = 0
    correct_static = 0
    total = 0
    
    cos_sims = []
    
    for i in predicted_indices:
        w = track_set[i]
        X = w['X'].cpu().numpy()
        Y_L = w['Y_L'].cpu().numpy()
        Y_R = w['Y_R'].cpu().numpy()
        
        Y_L_c = Y_L - np.mean(Y_L)
        Y_R_c = Y_R - np.mean(Y_R)
        
        # Evaluate Reconstructed W
        W_pred = reconstructed_W[i]
        pred_recon = X @ W_pred
        pred_recon_c = pred_recon - np.mean(pred_recon)
        corr_L_recon = np.sum(pred_recon_c * Y_L_c) / (np.linalg.norm(pred_recon_c) * np.linalg.norm(Y_L_c) + 1e-8)
        corr_R_recon = np.sum(pred_recon_c * Y_R_c) / (np.linalg.norm(pred_recon_c) * np.linalg.norm(Y_R_c) + 1e-8)
        label_recon = 1 if corr_L_recon > corr_R_recon else 0
        if label_recon == w['label']: correct_recon += 1
        
        # Evaluate True Oracle W
        W_oracle = W_oracles[i]
        pred_oracle = X @ W_oracle
        pred_oracle_c = pred_oracle - np.mean(pred_oracle)
        corr_L_oracle = np.sum(pred_oracle_c * Y_L_c) / (np.linalg.norm(pred_oracle_c) * np.linalg.norm(Y_L_c) + 1e-8)
        corr_R_oracle = np.sum(pred_oracle_c * Y_R_c) / (np.linalg.norm(pred_oracle_c) * np.linalg.norm(Y_R_c) + 1e-8)
        label_oracle = 1 if corr_L_oracle > corr_R_oracle else 0
        if label_oracle == w['label']: correct_oracle += 1
        
        # Evaluate Static Baseline W
        pred_static = X @ W_static
        pred_static_c = pred_static - np.mean(pred_static)
        corr_L_static = np.sum(pred_static_c * Y_L_c) / (np.linalg.norm(pred_static_c) * np.linalg.norm(Y_L_c) + 1e-8)
        corr_R_static = np.sum(pred_static_c * Y_R_c) / (np.linalg.norm(pred_static_c) * np.linalg.norm(Y_R_c) + 1e-8)
        label_static = 1 if corr_L_static > corr_R_static else 0
        if label_static == w['label']: correct_static += 1
        
        total += 1
        
        # Compute Cosine Similarity between Reconstructed W and Oracle W
        sim = np.sum(W_pred * W_oracle) / (np.linalg.norm(W_pred) * np.linalg.norm(W_oracle) + 1e-8)
        cos_sims.append(sim)
        
    acc_recon = correct_recon / total
    acc_oracle = correct_oracle / total
    acc_static = correct_static / total
    mean_sim = np.mean(cos_sims)
    
    print(f"[{subj_name}] Acc Hypernetwork: {acc_recon*100:.1f}% | Acc Static: {acc_static*100:.1f}% | Mean W-Sim: {mean_sim:.3f}")
    
    return subj_name, {
        'acc_recon': acc_recon,
        'acc_oracle': acc_oracle,
        'acc_static': acc_static,
        'mean_sim': mean_sim
    }

def main():
    print("=======================================================")
    print(" PHASE 169: LATENT STATE-CONDITIONED HYPERNETWORK")
    print(" Predicting Oracle Drift from Unsupervised Spectral Power")
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
    
    start_time = time.time()
    results = {}
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=mp.cpu_count()) as executor:
        futures = {executor.submit(process_subject, cf): cf for cf in cache_files}
        for future in concurrent.futures.as_completed(futures):
            subj, metrics = future.result()
            if metrics is not None:
                results[subj] = metrics
                
    print(f"\nExtraction & Regression Time: {time.time() - start_time:.2f}s\n")
    
    global_recon = []
    global_static = []
    global_oracle = []
    global_sim = []
    
    subjects_sorted = sorted(results.keys())
    for subj in subjects_sorted:
        metrics = results[subj]
        
        global_recon.append(metrics['acc_recon'] * 100)
        global_static.append(metrics['acc_static'] * 100)
        global_oracle.append(metrics['acc_oracle'] * 100)
        global_sim.append(metrics['mean_sim'])
        
        print(f"--- Subject: {subj} ---")
        print(f"  Hypernetwork Acc       : {metrics['acc_recon']*100:.1f}%")
        print(f"  Static Baseline Acc    : {metrics['acc_static']*100:.1f}%")
        print(f"  Oracle EMA Accuracy    : {metrics['acc_oracle']*100:.1f}%")
        print(f"  Decoder Cos Similarity : {metrics['mean_sim']:.3f}")
        print()
        
    print("=======================================================")
    print(" GLOBAL OBSERVABILITY AVERAGES")
    print("=======================================================")
    print(f"Global Hypernetwork Acc  : {np.mean(global_recon):.2f}%")
    print(f"Global Static Baseline   : {np.mean(global_static):.2f}%")
    print(f"Global Oracle EMA Acc    : {np.mean(global_oracle):.2f}%\n")
    print(f"Global Decoder Similarity: {np.mean(global_sim):.3f}")
    print("=======================================================")
    
if __name__ == '__main__':
    main()
