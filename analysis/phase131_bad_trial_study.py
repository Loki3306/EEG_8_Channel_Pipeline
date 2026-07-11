import os
import numpy as np
import torch
import pandas as pd
from pathlib import Path
from scipy import signal

# -------------------------------------------------------------------------
# CONSTANTS & CONFIGURATION
# -------------------------------------------------------------------------
SR = 128
EAR_CHANNEL_INDICES = [23, 31, 32, 40, 14, 22, 41, 49]
MAX_LAG_SEC = 0.400 
MAX_LAG_SAMPLES = int(MAX_LAG_SEC * SR)

BROADBAND = (0.5, 8.0)
RIDGE_LAMBDA = 10.0
PCA_COMPONENTS = 60
N_FOLDS = 4
N_NULL_SHIFTS = 200

def apply_modulation_filter(env, lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype='band')
    return signal.filtfilt(b, a, env, axis=1)

def create_toeplitz_features_pt(eeg, max_lag_samples):
    C, T = eeg.shape
    T_eff = T - max_lag_samples
    X = torch.zeros((T_eff, C * max_lag_samples), dtype=eeg.dtype, device=eeg.device)
    for tau in range(max_lag_samples):
        X[:, tau*C : (tau+1)*C] = eeg[:, tau : tau+T_eff].T
    return X

def batch_pearsonr_pt(x, y):
    x_mean = x - x.mean(dim=1, keepdim=True)
    y_mean = y - y.mean(dim=1, keepdim=True)
    num = (x_mean * y_mean).sum(dim=1)
    den = torch.sqrt((x_mean**2).sum(dim=1) * (y_mean**2).sum(dim=1))
    return num / (den + 1e-8)

def solve_ridge_pt(XTX, XTy, lam=10.0):
    F = XTX.shape[0]
    I = torch.eye(F, device=XTX.device, dtype=XTX.dtype)
    jitter = 1e-6 * torch.randn(F, F, device=XTX.device, dtype=XTX.dtype) * I
    return torch.linalg.solve(XTX + lam * I + jitter, XTy)

def process_trial_diagnostic(t_idx, tr, device):
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
    
    Y_L = env_l[:T_eff]
    Y_R = env_r[:T_eff]
    
    U, S, V = torch.pca_lowrank(X_trial, q=PCA_COMPONENTS)
    X_pca = torch.matmul(X_trial, V)
    
    block_size = T_eff // N_FOLDS
    z_scores_L = []
    z_scores_R = []
    r_scores_L = []
    r_scores_R = []
    
    for fold in range(N_FOLDS):
        test_start = fold * block_size
        test_end = (fold + 1) * block_size if fold < N_FOLDS - 1 else T_eff
        
        train_mask = torch.ones(T_eff, dtype=torch.bool, device=device)
        train_mask[test_start:test_end] = False
        
        X_train = X_pca[train_mask]
        YL_train = Y_L[train_mask]
        YR_train = Y_R[train_mask]
        
        X_test = X_pca[~train_mask]
        YL_test = Y_L[~train_mask]
        YR_test = Y_R[~train_mask]
        
        XTX_train = X_train.T @ X_train
        W_L = solve_ridge_pt(XTX_train, (X_train.T @ YL_train).unsqueeze(-1), lam=RIDGE_LAMBDA)
        W_R = solve_ridge_pt(XTX_train, (X_train.T @ YR_train).unsqueeze(-1), lam=RIDGE_LAMBDA)
        
        Y_hat_L = (X_test @ W_L).squeeze(-1)
        Y_hat_R = (X_test @ W_R).squeeze(-1)
        
        r_L_true = batch_pearsonr_pt(Y_hat_L.unsqueeze(0), YL_test.unsqueeze(0)).item()
        r_R_true = batch_pearsonr_pt(Y_hat_R.unsqueeze(0), YR_test.unsqueeze(0)).item()
        
        T_test = YL_test.shape[0]
        shift_min = SR * 1 
        shift_max = T_test - SR * 1
        shifts = torch.randint(shift_min, shift_max, (N_NULL_SHIFTS,), device=device)
        
        YL_null = torch.stack([torch.roll(YL_test, shifts[i].item()) for i in range(N_NULL_SHIFTS)])
        YR_null = torch.stack([torch.roll(YR_test, shifts[i].item()) for i in range(N_NULL_SHIFTS)])
        
        r_L_null = batch_pearsonr_pt(Y_hat_L.unsqueeze(0).expand(N_NULL_SHIFTS, -1), YL_null)
        r_R_null = batch_pearsonr_pt(Y_hat_R.unsqueeze(0).expand(N_NULL_SHIFTS, -1), YR_null)
        
        z_L = (r_L_true - r_L_null.mean().item()) / (r_L_null.std().item() + 1e-8)
        z_R = (r_R_true - r_R_null.mean().item()) / (r_R_null.std().item() + 1e-8)
        
        z_scores_L.append(z_L)
        z_scores_R.append(z_R)
        r_scores_L.append(r_L_true)
        r_scores_R.append(r_R_true)
        
    final_z_L = np.mean(z_scores_L)
    final_z_R = np.mean(z_scores_R)
    final_r_L = np.mean(r_scores_L)
    final_r_R = np.mean(r_scores_R)
    
    sp = tr['meta']['switch_points']
    attended_side = sp[0][0] if len(sp) > 0 else 'L'
    attended_gender = sp[0][1] if len(sp) > 0 else 'Unknown'
    # Infer unattended gender assuming it's usually opposite in this dataset block
    unattended_gender = 'F' if attended_gender == 'M' else 'M'
    
    predicted_side = 'L' if final_z_L > final_z_R else 'R'
    correct = 1 if predicted_side == attended_side else 0
    
    return {
        'Trial': t_idx,
        'Att_Side': attended_side,
        'Att_Gender': attended_gender,
        'Unatt_Gender': unattended_gender,
        'r_L': final_r_L,
        'r_R': final_r_R,
        'z_L': final_z_L,
        'z_R': final_z_R,
        'Predicted': predicted_side,
        'Correct': correct
    }

def analyze_subject(cache_file, device):
    subj_name = cache_file.stem.split('_')[0]
    cached = torch.load(cache_file, map_location='cpu', weights_only=False)['raw']
    
    results = []
    for t_idx in range(len(cached)):
        res = process_trial_diagnostic(t_idx, cached[t_idx], device)
        results.append(res)
        
    df = pd.DataFrame(results)
    
    print(f"\n=======================================")
    print(f" SUBJECT {subj_name} DIAGNOSTIC STUDY")
    print(f"=======================================")
    print(f"Total Accuracy: {df['Correct'].mean():.2%}")
    
    print("\n[Accuracy by Attended Gender]")
    if 'Att_Gender' in df.columns:
        print(df.groupby('Att_Gender')['Correct'].mean())
        
    print("\n[Accuracy by Attended Side]")
    print(df.groupby('Att_Side')['Correct'].mean())
    
    print("\n[Mean Z-Scores (Correct vs Incorrect)]")
    print(df.groupby('Correct')[['z_L', 'z_R']].mean())
    
    print("\n[Worst 5 Failed Trials (Highest margin for wrong prediction)]")
    # Margin = z_Predicted - z_Actual
    df['Error_Margin'] = df.apply(lambda row: row['z_R'] - row['z_L'] if row['Att_Side'] == 'L' else row['z_L'] - row['z_R'], axis=1)
    failures = df[df['Correct'] == 0].sort_values('Error_Margin', ascending=False)
    print(failures[['Trial', 'Att_Side', 'Att_Gender', 'z_L', 'z_R', 'r_L', 'r_R', 'Error_Margin']].head(5).to_string())

def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
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
            
    # Analyze S14 (Bad) and S16 (Good)
    target_subjects = ['S14', 'S16']
    
    for subj in target_subjects:
        cache_file = cache_dir / f"{subj}_multiband.pt"
        if cache_file.exists():
            analyze_subject(cache_file, device)
        else:
            print(f"Cache file for {subj} not found!")

if __name__ == '__main__':
    main()
