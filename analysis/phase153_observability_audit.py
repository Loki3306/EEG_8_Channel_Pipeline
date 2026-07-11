import os
import time
import numpy as np
import torch
from pathlib import Path
from scipy import signal
import multiprocessing as mp
import concurrent.futures
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score

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

def get_pearsonr(x, y):
    x_m = x - np.mean(x)
    y_m = y - np.mean(y)
    num = np.dot(x_m, y_m)
    den = np.linalg.norm(x_m) * np.linalg.norm(y_m)
    return num / (den + 1e-8)

def compute_pr_fast(X):
    """
    Computes Participation Ratio of X^T X quickly without SVD.
    PR = (Tr(C))^2 / Tr(C^2)
    """
    C = X.T @ X
    tr_C = np.trace(C)
    tr_C2 = np.sum(C ** 2) # Trace of C^2 is sum of squared elements for symmetric C
    return (tr_C ** 2) / (tr_C2 + 1e-8)

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
        
        current_spk = 'L'
        sp_idx = 0
        labels_eff = np.zeros(T_eff, dtype=int)
        for t in range(T_eff):
            if sp_idx < len(sp) and t >= sp[sp_idx][1]:
                current_spk = sp[sp_idx][0]
                sp_idx += 1
            labels_eff[t] = 1 if current_spk == 'L' else 0
            
        for seq_start in range(0, T_eff - SEQ_SAMPLES + 1, SEQ_HOP):
            seq_end = seq_start + SEQ_SAMPLES
            X_win = X_trial[seq_start:seq_end]
            Y_L_win = Y_l_eff[seq_start:seq_end]
            Y_R_win = Y_r_eff[seq_start:seq_end]
            
            win_labels = labels_eff[seq_start:seq_end]
            label = 1 if np.mean(win_labels) >= 0.5 else 0
            
            last_switch = 0
            for s in switch_indices:
                if s <= seq_start:
                    last_switch = s
                else:
                    break
                    
            time_since_switch = (seq_start - last_switch) / SR
            
            windows.append({
                'X': X_win,
                'Y_L': Y_L_win,
                'Y_R': Y_R_win,
                'label': label,
                'time_since_switch': time_since_switch
            })
            
    return windows

def process_subject(cache_file, device_id):
    device = torch.device(f'cuda:{device_id}' if torch.cuda.is_available() else 'cpu')
    subj_name = cache_file.stem.split('_')[0]
    
    windows = prepare_subject_windows_continuous(cache_file, device)
    if len(windows) < 400:
        return subj_name, None
        
    CALIB_WINDOWS = 240
    calib_set = windows[:CALIB_WINDOWS]
    track_set = windows[CALIB_WINDOWS:]
    
    F = windows[0]['X'].shape[1]
    
    # Calibration
    Rxx_calib = torch.zeros((F, F), device=device)
    Rxy_calib = torch.zeros((F,), device=device)
    
    for w in calib_set:
        X = w['X']
        Y_true = w['Y_L'] if w['label'] == 1 else w['Y_R']
        Rxx_calib += X.T @ X
        Rxy_calib += X.T @ Y_true
        
    I = torch.eye(F, device=device)
    W = torch.linalg.solve(Rxx_calib + RIDGE_LAMBDA * I, Rxy_calib)
    
    Rxx = Rxx_calib.clone()
    Rxy = Rxy_calib.clone()
    
    audit_data = []
    margin_prev = 0.0
    
    for w in track_set:
        X_cpu = w['X'].cpu().numpy()
        YL_cpu = w['Y_L'].cpu().numpy()
        YR_cpu = w['Y_R'].cpu().numpy()
        W_cpu = W.cpu().numpy()
        
        y_true = YL_cpu if w['label'] == 1 else YR_cpu
        y_false = YR_cpu if w['label'] == 1 else YL_cpu
        
        # 1. Oracle Confidence Margin
        preds = X_cpu @ W_cpu
        c_true = get_pearsonr(preds, y_true)
        c_false = get_pearsonr(preds, y_false)
        margin = c_true - c_false
        
        # 2. Richer Environmental Features
        env_var = np.var(y_true) # Modulation Depth
        env_sim = get_pearsonr(YL_cpu, YR_cpu) # Acoustic Similarity
        
        # Fast compute of Participation Ratio for current window
        eeg_pr = compute_pr_fast(X_cpu)
        
        # Log-transform time since switch
        log_time = np.log10(w['time_since_switch'] + 0.5)
        
        audit_data.append({
            'subject': subj_name,
            'margin': margin,
            'margin_prev': margin_prev,
            'env_var': env_var,
            'env_sim': env_sim,
            'eeg_pr': eeg_pr,
            'log_time_since_switch': log_time
        })
        
        margin_prev = margin
        
        # Oracle Update
        Y_true_pt = w['Y_L'] if w['label'] == 1 else w['Y_R']
        Rxx = FORGETTING_FACTOR * Rxx + w['X'].T @ w['X']
        Rxy = FORGETTING_FACTOR * Rxy + w['X'].T @ Y_true_pt
        W = torch.linalg.solve(Rxx + RIDGE_LAMBDA * I, Rxy)
        
    df = pd.DataFrame(audit_data)
    return subj_name, df

def main():
    mp.set_start_method('spawn', force=True)
    
    print("=======================================================")
    print(" PHASE 153: HIERARCHICAL OBSERVABILITY AUDIT")
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
    
    all_dfs = []
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(process_subject, cf, idx % max(1, num_gpus)): cf for idx, cf in enumerate(cache_files)}
        for future in concurrent.futures.as_completed(futures):
            subj, df = future.result()
            if df is not None:
                all_dfs.append(df)
                print(f"[{subj:3s}] Extracted {len(df)} tracking windows.")

    print("\n=======================================================")
    print(" 1. SUBJECT-WISE CORRELATION ANALYSIS")
    print("=======================================================")
    master_df = pd.concat(all_dfs, ignore_index=True)
    features = ['margin_prev', 'env_var', 'env_sim', 'eeg_pr', 'log_time_since_switch']
    
    subject_corrs = {f: [] for f in features}
    for subj in master_df['subject'].unique():
        sdf = master_df[master_df['subject'] == subj]
        for f in features:
            r = get_pearsonr(sdf[f].values, sdf['margin'].values)
            subject_corrs[f].append(r)
            
    print("Mean Within-Subject Pearson R with Oracle Margin:")
    for f in features:
        mean_r = np.mean(subject_corrs[f])
        std_r = np.std(subject_corrs[f])
        print(f" - {f:25s}: {mean_r:+.4f} (std: {std_r:.4f})")

    print("\n=======================================================")
    print(" 2. LEAVE-SUBJECTS-OUT PERMUTATION IMPORTANCE")
    print("=======================================================")
    # Predict binary observability: margin > 0.05
    master_df['observable'] = (master_df['margin'] > 0.05).astype(int)
    
    # Split subjects for generalizable feature importance
    subjects = master_df['subject'].unique()
    np.random.seed(42)
    np.random.shuffle(subjects)
    train_subs = subjects[:14]
    test_subs = subjects[14:]
    
    train_df = master_df[master_df['subject'].isin(train_subs)]
    test_df = master_df[master_df['subject'].isin(test_subs)]
    
    X_train = train_df[features].values
    y_train = train_df['observable'].values
    X_test = test_df[features].values
    y_test = test_df['observable'].values
    
    rf = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42, n_jobs=-1, class_weight='balanced')
    rf.fit(X_train, y_train)
    
    test_auc = roc_auc_score(y_test, rf.predict_proba(X_test)[:, 1])
    print(f"Random Forest AUC on Unseen Subjects: {test_auc:.3f}")
    
    # Permutation Importance
    print("\nPermutation Importance (Drop in Test AUC when feature is scrambled):")
    result = permutation_importance(rf, X_test, y_test, scoring='roc_auc', n_repeats=5, random_state=42, n_jobs=-1)
    
    for i in result.importances_mean.argsort()[::-1]:
        feat = features[i]
        imp = result.importances_mean[i]
        std = result.importances_std[i]
        print(f" - {feat:25s}: {imp:+.4f} (std: {std:.4f})")
        
    # ---------------------------------------------------------
    # 3. OBSERVABILITY MAPPING (VISUALIZATION)
    # ---------------------------------------------------------
    os.makedirs('/kaggle/working/plots', exist_ok=True)
    
    fig, axes = plt.subplots(1, len(features), figsize=(25, 5))
    
    for i, feat in enumerate(features):
        # We use deciles across the whole dataset for visualization
        try:
            master_df['bin'] = pd.qcut(master_df[feat], q=10, duplicates='drop')
            bin_means = master_df.groupby('bin')['observable'].mean()
            bin_stds = master_df.groupby('bin')['observable'].std() / np.sqrt(master_df.groupby('bin')['observable'].count())
            
            x_centers = [interval.mid for interval in bin_means.index]
            
            axes[i].errorbar(x_centers, bin_means.values, yerr=bin_stds.values, fmt='o-', capsize=5)
            axes[i].set_title(f"Observability vs {feat}")
            axes[i].set_xlabel(feat)
            axes[i].set_ylabel("Prob(Margin > 0.05)")
            axes[i].grid(True)
        except Exception as e:
            axes[i].set_title(f"Could not bin {feat}")
            
    plt.tight_layout()
    plot_path = '/kaggle/working/plots/phase153_hierarchical_audit.png'
    plt.savefig(plot_path, dpi=300)
    plt.close()
    
    print(f"\nSaved Hierarchical Observability Map to: {plot_path}")

if __name__ == '__main__':
    main()
