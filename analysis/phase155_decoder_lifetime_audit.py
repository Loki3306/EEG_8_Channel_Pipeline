import os
import numpy as np
import torch
from pathlib import Path
from scipy import signal
import multiprocessing as mp
import concurrent.futures
import pandas as pd
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

WINDOWS_PER_MIN = 120
CALIB_MINUTES = [1, 2, 5, 10, 20]
CALIB_SIZES = [m * WINDOWS_PER_MIN for m in CALIB_MINUTES]
HOLDOUT_MINUTES = 15
HOLDOUT_WINDOWS = HOLDOUT_MINUTES * WINDOWS_PER_MIN

LIFETIME_TRAIN_MIN = 10
LIFETIME_TRAIN_WIN = LIFETIME_TRAIN_MIN * WINDOWS_PER_MIN
LIFETIME_BLOCK_MIN = 5
LIFETIME_BLOCK_WIN = LIFETIME_BLOCK_MIN * WINDOWS_PER_MIN

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

def prepare_subject_windows_continuous(cache_file, device):
    cached = torch.load(cache_file, map_location='cpu', weights_only=False)['raw']
    windows = []
    
    for tr_idx, tr in enumerate(cached):
        eeg_raw = tr['eeg'].numpy()[EAR_CHANNEL_INDICES, :] 
        
        # 1. Average across the 16 Gammatone bands to get the true Broadband Envelope
        env_l_raw = tr['env_l'].numpy().mean(axis=0, keepdims=True)
        env_r_raw = tr['env_r'].numpy().mean(axis=0, keepdims=True)
        
        min_len = min(eeg_raw.shape[1], env_l_raw.shape[1])
        eeg_raw = eeg_raw[:, :min_len]
        env_l_raw = env_l_raw[:, :min_len]
        env_r_raw = env_r_raw[:, :min_len]
        
        eeg_f = apply_modulation_filter(eeg_raw, BROADBAND[0], BROADBAND[1], SR)
        env_l_f = apply_modulation_filter(env_l_raw, BROADBAND[0], BROADBAND[1], SR)
        env_r_f = apply_modulation_filter(env_r_raw, BROADBAND[0], BROADBAND[1], SR)
        
        eeg_f = (eeg_f - np.mean(eeg_f, axis=1, keepdims=True)) / (np.std(eeg_f, axis=1, keepdims=True) + 1e-8)
        env_l_f_z = (env_l_f - np.mean(env_l_f, axis=1, keepdims=True)) / (np.std(env_l_f, axis=1, keepdims=True) + 1e-8)
        env_r_f_z = (env_r_f - np.mean(env_r_f, axis=1, keepdims=True)) / (np.std(env_r_f, axis=1, keepdims=True) + 1e-8)
        
        eeg = torch.tensor(eeg_f, dtype=torch.float32, device=device)
        env_l = torch.tensor(env_l_f_z[0], dtype=torch.float32, device=device)
        env_r = torch.tensor(env_r_f_z[0], dtype=torch.float32, device=device)
        
        X_trial = create_toeplitz_features_pt(eeg, MAX_LAG_SAMPLES)
        T_eff = X_trial.shape[0]
        
        Y_l_eff = env_l[:T_eff]
        Y_r_eff = env_r[:T_eff]
        Y_l_raw_eff = env_l_raw[0, :T_eff]
        Y_r_raw_eff = env_r_raw[0, :T_eff]
        
        sp = tr['meta']['switch_points']
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
            
            win_labels = labels_eff[seq_start:seq_end]
            
            # Skip transition windows that contain a speaker switch
            valid_win = True
            if np.mean(win_labels) != 0 and np.mean(win_labels) != 1:
                valid_win = False
                
            label = int(win_labels[0])
            
            # CRITICAL FIX: The AASD dataset swaps Male/Female speakers between left and right ears 
            # after the first 30 trials. 
            # 179 ('L') means "Attended Male". 184 ('R') means "Attended Female".
            # Trials 0-29: Male is Left. So 'L' (179) means Attended Left. label=1 is correct.
            # Trials 30-59: Male is Right, Female is Left. So 'R' (184) means Attended Left!
            # Therefore, we must invert the label in the second half so that label=1 ALWAYS means "Attending Left Ear".
            if tr_idx >= 30:
                label = 1 - label
            
            X_win = X_trial[seq_start:seq_end]
            
            windows.append({
                'valid': valid_win,
                'X': X_win,
                'Y_L': Y_l_eff[seq_start:seq_end],
                'Y_R': Y_r_eff[seq_start:seq_end],
                'Y_L_raw': Y_l_raw_eff[seq_start:seq_end],
                'Y_R_raw': Y_r_raw_eff[seq_start:seq_end],
                'label': label
            })
            
    return windows

def train_ridge_decoder(windows, device):
    if not windows:
        return None
    F = windows[0]['X'].shape[1]
    Rxx = torch.zeros((F, F), device=device)
    Rxy = torch.zeros((F,), device=device)
    
    total_samples = 0
    for w in windows:
        if not w['valid']: continue
        X = w['X']
        Y_true = w['Y_L'] if w['label'] == 1 else w['Y_R']
        Rxx += X.T @ X
        Rxy += X.T @ Y_true
        total_samples += X.shape[0]
        
    if total_samples > 0:
        Rxx /= total_samples
        Rxy /= total_samples
        
    I = torch.eye(F, device=device)
    W = torch.linalg.solve(Rxx + RIDGE_LAMBDA * I, Rxy).cpu().numpy()
    return W

def evaluate_decoder(W, windows):
    if not windows or W is None:
        return {'acc': 0.5, 'auroc': 0.5, 'mean_margin': 0.0, 'mean_mod': 0.0, 'mean_sim': 0.0}
        
    correct = 0
    margins = []
    mods = []
    sims = []
    preds_L = []
    labels = []
    
    for w in windows:
        if not w['valid']: continue
        X_cpu = w['X'].cpu().numpy()
        YL_cpu = w['Y_L'].cpu().numpy()
        YR_cpu = w['Y_R'].cpu().numpy()
        YL_raw = w['Y_L_raw']
        YR_raw = w['Y_R_raw']
        label = w['label']
        
        preds = X_cpu @ W
        c_L = get_pearsonr(preds, YL_cpu)
        c_R = get_pearsonr(preds, YR_cpu)
        
        margin = c_L - c_R if label == 1 else c_R - c_L
        margins.append(margin)
        preds_L.append(c_L - c_R) # Use evidence for AUROC
        labels.append(label)
        
        mod = np.var(YL_raw) + np.var(YR_raw)
        sim = get_pearsonr(YL_cpu, YR_cpu)
        mods.append(mod)
        sims.append(sim)
        
        if (c_L > c_R and label == 1) or (c_R > c_L and label == 0):
            correct += 1
            
    valid_count = len(labels)
    acc = correct / valid_count if valid_count > 0 else 0.5
    
    try:
        auroc = roc_auc_score(labels, preds_L)
    except:
        auroc = 0.5
        
    return {
        'acc': acc, 
        'auroc': auroc,
        'mean_margin': np.mean(margins) if margins else 0.0,
        'mean_mod': np.mean(mods) if mods else 0.0,
        'mean_sim': np.mean(sims) if sims else 0.0
    }

def process_subject(cache_file, device_id):
    device = torch.device(f'cuda:{device_id}' if torch.cuda.is_available() else 'cpu')
    subj_name = cache_file.stem.split('_')[0]
    
    windows = prepare_subject_windows_continuous(cache_file, device)
    total_windows = len(windows)
    
    expA_results = []
    expB_results = []
    expC_results = []
    
    # Require at least 45 minutes (5400 windows) of data
    if total_windows < 5400:
        return subj_name, expA_results, expB_results, expC_results
        
    # --- EXPERIMENT A: CALIBRATION CURVE (Fixed End, Expanding Start) ---
    # We fix the end of calibration at Minute 30 (window 3600).
    # This ensures the gap to the Test Set (Minute 30-45) is exactly 0 for all sweep sizes!
    CALIB_END_WIN = 30 * WINDOWS_PER_MIN
    TEST_END_WIN = 45 * WINDOWS_PER_MIN
    holdout_set = windows[CALIB_END_WIN:TEST_END_WIN]
    
    for c_min, c_win in zip(CALIB_MINUTES, CALIB_SIZES):
        calib_start_win = CALIB_END_WIN - c_win
        calib_set = windows[calib_start_win:CALIB_END_WIN]
        W = train_ridge_decoder(calib_set, device)
        res = evaluate_decoder(W, holdout_set)
        expA_results.append({
            'subject': subj_name, 
            'calib_min': c_min, 
            'acc': res['acc'],
            'auroc': res['auroc'],
            'mean_margin': res['mean_margin']
        })
        
    # --- EXPERIMENT B: DECODER LIFETIME (Train First 10, Test Rest) ---
    lt_calib_set = windows[:LIFETIME_TRAIN_WIN]
    W_lt = train_ridge_decoder(lt_calib_set, device)
    
    current_idx = LIFETIME_TRAIN_WIN
    block_idx = 0
    while current_idx + LIFETIME_BLOCK_WIN <= total_windows:
        test_block = windows[current_idx : current_idx + LIFETIME_BLOCK_WIN]
        res = evaluate_decoder(W_lt, test_block)
        delta_t_min = LIFETIME_TRAIN_MIN + block_idx * LIFETIME_BLOCK_MIN
        expB_results.append({
            'subject': subj_name, 
            'delta_t_min': delta_t_min, 
            'acc': res['acc'],
            'auroc': res['auroc'],
            'mean_margin': res['mean_margin'],
            'mean_mod': res['mean_mod'],
            'mean_sim': res['mean_sim']
        })
        current_idx += LIFETIME_BLOCK_WIN
        block_idx += 1
        
    # --- EXPERIMENT C: SLIDING LOCAL CALIBRATION ---
    SLIDING_WIN = 10 * WINDOWS_PER_MIN
    current_idx = 0
    while current_idx + 2 * SLIDING_WIN <= total_windows:
        calib_block = windows[current_idx : current_idx + SLIDING_WIN]
        test_block = windows[current_idx + SLIDING_WIN : current_idx + 2 * SLIDING_WIN]
        W_sl = train_ridge_decoder(calib_block, device)
        res = evaluate_decoder(W_sl, test_block)
        
        train_start_min = current_idx // WINDOWS_PER_MIN
        expC_results.append({
            'subject': subj_name,
            'train_start_min': train_start_min,
            'acc': res['acc'],
            'auroc': res['auroc'],
            'mean_margin': res['mean_margin']
        })
        current_idx += SLIDING_WIN
        
    return subj_name, expA_results, expB_results, expC_results

def main():
    mp.set_start_method('spawn', force=True)
    print("=======================================================")
    print(" PHASE 155: DECODER LIFETIME & OBSERVABILITY AUDIT")
    print("=======================================================\n")
    
    cache_dir = Path('/kaggle/working/multiband_cache')
    possible_paths = [
        Path('/kaggle/input/datasets/lokeshgile/aasd-universal-cache-v1'),
        Path('/kaggle/input/datasets/lowkieee/multiband-cache/kaggle/working/multiband_cache'),
        cache_dir
    ]
    for p in possible_paths:
        if p.exists() and len(list(p.glob('*_multiband.pt'))) > 0:
            cache_dir = p
            break
            
    cache_files = sorted(list(cache_dir.glob('*_multiband.pt')))
    num_gpus = torch.cuda.device_count()
    num_workers = mp.cpu_count()
    
    all_expA, all_expB, all_expC = [], [], []
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(process_subject, cf, idx % max(1, num_gpus)): cf for idx, cf in enumerate(cache_files)}
        for future in concurrent.futures.as_completed(futures):
            subj, expA, expB, expC = future.result()
            all_expA.extend(expA)
            all_expB.extend(expB)
            all_expC.extend(expC)
            print(f"[{subj}] Processed: ExpA {len(expA)}, ExpB {len(expB)}, ExpC {len(expC)}")

    if all_expA:
        dfA = pd.DataFrame(all_expA)
        dfB = pd.DataFrame(all_expB)
        dfC = pd.DataFrame(all_expC)
        
        dfA.to_csv("phase155_experimentA_calibration.csv", index=False)
        dfB.to_csv("phase155_experimentB_lifetime.csv", index=False)
        dfC.to_csv("phase155_experimentC_sliding.csv", index=False)
        
        print("\n=======================================================")
        print(" EXPERIMENT A: CALIBRATION CURVE (Fixed Test 30-45m)")
        print("=======================================================")
        meanA = dfA.groupby('calib_min')[['acc', 'auroc']].mean().reset_index()
        for _, row in meanA.iterrows():
            print(f"  Train: {int(row['calib_min']):2d} min  ->  Acc: {row['acc']*100:.1f}% | AUROC: {row['auroc']:.3f}")
            
        print("\n=======================================================")
        print(" EXPERIMENT B: LIFETIME CURVE (Train 0-10m)")
        print("=======================================================")
        meanB = dfB.groupby('delta_t_min')[['acc', 'auroc', 'mean_mod', 'mean_sim']].mean().reset_index()
        for _, row in meanB.iterrows():
            print(f"  Eval @ Min {int(row['delta_t_min']):3d}  ->  Acc: {row['acc']*100:.1f}% | AUROC: {row['auroc']:.3f} | Mod: {row['mean_mod']:.2f}")
            
        print("\n=======================================================")
        print(" EXPERIMENT C: SLIDING CALIBRATION (Train 10m, Test next 10m)")
        print("=======================================================")
        meanC = dfC.groupby('train_start_min')[['acc', 'auroc']].mean().reset_index()
        for _, row in meanC.iterrows():
            print(f"  Train {int(row['train_start_min']):2d}-{int(row['train_start_min'])+10:2d}m  ->  Acc: {row['acc']*100:.1f}% | AUROC: {row['auroc']:.3f}")

if __name__ == '__main__':
    main()
