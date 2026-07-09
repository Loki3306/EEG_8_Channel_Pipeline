import numpy as np
from scipy.linalg import solve
from scipy.stats import pearsonr
import torch
from pathlib import Path
from sklearn.metrics import roc_auc_score
from scipy import signal
import time

# -------------------------------------------------------------------------
# CONSTANTS & CONFIGURATION
# -------------------------------------------------------------------------
SR = 128
EAR_CHANNEL_INDICES = [23, 31, 32, 40, 14, 22, 41, 49]

# mTRF Design Parameters
MAX_LAG_SEC = 0.400 
MAX_LAG_SAMPLES = int(MAX_LAG_SEC * SR)

BANDS = {
    'Delta': (0.5, 4.0),
    'Theta': (4.0, 8.0),
    'Alpha': (8.0, 15.0),
    'Beta': (15.0, 32.0)
}

def apply_modulation_filter(env, lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype='band')
    filtered = signal.filtfilt(b, a, env, axis=1)
    return filtered

def get_trial_dominant_speaker(tr):
    sp = tr['meta']['switch_points']
    T = tr['eeg_bands'][0].shape[1]
    
    boundaries = [0]
    boundaries.extend([idx for spk, idx in sp])
    boundaries = sorted(set(boundaries))
    if boundaries[-1] != T: boundaries.append(T)
        
    l_duration = 0
    r_duration = 0
    
    for i in range(len(boundaries) - 1):
        start_idx = boundaries[i]
        end_idx = boundaries[i+1]
        current_spk = 'L'
        for spk, idx in sp:
            if idx <= start_idx: current_spk = spk
            else: break
            
        if current_spk == 'L': l_duration += (end_idx - start_idx)
        else: r_duration += (end_idx - start_idx)
        
    return 'L' if l_duration >= r_duration else 'R'

def create_toeplitz_features(eeg, max_lag_samples):
    C, T = eeg.shape
    T_eff = T - max_lag_samples
    
    X = np.zeros((T_eff, C * max_lag_samples))
    for tau in range(max_lag_samples):
        X[:, tau*C : (tau+1)*C] = eeg[:, tau : tau+T_eff].T
    return X

def compute_trial_covariance(tr, band_idx):
    eeg = tr['eeg_bands'][band_idx] 
    env_l = tr['env_l_bands'][band_idx][0] 
    env_r = tr['env_r_bands'][band_idx][0] 
    
    T = eeg.shape[1]
    X_trial = create_toeplitz_features(eeg, MAX_LAG_SAMPLES)
    T_eff = X_trial.shape[0]
    
    sp = tr['meta']['switch_points']
    boundaries = [0]
    boundaries.extend([idx for spk, idx in sp])
    boundaries = sorted(set(boundaries))
    if boundaries[-1] != T: boundaries.append(T)
        
    Y_att = np.zeros(T)
    for i in range(len(boundaries) - 1):
        start_idx = boundaries[i]
        end_idx = boundaries[i+1]
        current_spk = 'L'
        for spk, idx in sp:
            if idx <= start_idx: current_spk = spk
            else: break
        if current_spk == 'L':
            Y_att[start_idx:end_idx] = env_l[start_idx:end_idx]
        else:
            Y_att[start_idx:end_idx] = env_r[start_idx:end_idx]
            
    Y_trial = Y_att[:T_eff]
    
    XTX = X_trial.T @ X_trial
    XTy = X_trial.T @ Y_trial
    
    return XTX, XTy

def evaluate_trial_sequence(W_band, tr, band_idx):
    diff_scores = []
    Y_meta = []
    
    SEQ_SAMPLES = int(3.5 * SR)
    seq_hop = int(0.5 * SR)
    
    T = tr['eeg_bands'][0].shape[1]
    sp = tr['meta']['switch_points']
    boundaries = [0]
    boundaries.extend([idx for spk, idx in sp])
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
            for seq_start in range(safe_start, safe_end - SEQ_SAMPLES + 1, seq_hop):
                eeg_seq = tr['eeg_bands'][band_idx][:, seq_start:seq_start + SEQ_SAMPLES]
                env_l_seq = tr['env_l_bands'][band_idx][0][seq_start:seq_start + SEQ_SAMPLES]
                env_r_seq = tr['env_r_bands'][band_idx][0][seq_start:seq_start + SEQ_SAMPLES]
                
                X_seq = create_toeplitz_features(eeg_seq, MAX_LAG_SAMPLES)
                T_eff = X_seq.shape[0]
                Y_l_eff = env_l_seq[:T_eff]
                Y_r_eff = env_r_seq[:T_eff]
                
                Y_hat = X_seq @ W_band
                
                if np.std(Y_hat) < 1e-8 or np.std(Y_l_eff) < 1e-8 or np.std(Y_r_eff) < 1e-8:
                    r_L = 0.0
                    r_R = 0.0
                else:
                    r_L, _ = pearsonr(Y_hat, Y_l_eff)
                    r_R, _ = pearsonr(Y_hat, Y_r_eff)
                
                diff_scores.append(r_L - r_R)
                label_L = 1.0 if current_spk == 'L' else 0.0
                Y_meta.append(label_L)
                
    return np.array(diff_scores), np.array(Y_meta)

def fast_ridge(XTX, XTy, lam):
    I = np.eye(XTX.shape[0])
    return solve(XTX + lam * I, XTy, assume_a='pos')

def main():
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
            
    print("\n=======================================================")
    print(" PHASE 113: LEAVE-ONE-TRIAL-OUT (LOTO) CALIBRATION")
    print(" Nested LOTO for mathematically pure, leak-free evaluation")
    print("=======================================================\n", flush=True)
    
    cache_files = sorted(list(cache_dir.glob('*_multiband.pt')))
    final_results = {}
    
    for cache_file in cache_files:
        subj_name = cache_file.stem.split('_')[0]
        print(f"\n=======================================================")
        print(f" SUBJECT {subj_name}")
        print(f"=======================================================", flush=True)
        
        cached = torch.load(cache_file, map_location='cpu', weights_only=False)['raw']
        
        # Load and Filter
        trials = []
        for i in range(len(cached)):
            tr = cached[i]
            eeg_raw = tr['eeg'].numpy()[EAR_CHANNEL_INDICES, :] 
            env_l_raw = tr['env_l'].numpy()
            env_r_raw = tr['env_r'].numpy()
            
            min_len = min(eeg_raw.shape[1], env_l_raw.shape[1])
            eeg_raw = eeg_raw[:, :min_len]
            env_l_raw = env_l_raw[:, :min_len]
            env_r_raw = env_r_raw[:, :min_len]
            
            eeg_bands, env_l_bands, env_r_bands = [], [], []
            for band_name, (lowcut, highcut) in BANDS.items():
                eeg_f = apply_modulation_filter(eeg_raw, lowcut, highcut, SR)
                env_l_f = apply_modulation_filter(env_l_raw, lowcut, highcut, SR)
                env_r_f = apply_modulation_filter(env_r_raw, lowcut, highcut, SR)
                
                eeg_f = (eeg_f - np.mean(eeg_f, axis=1, keepdims=True)) / (np.std(eeg_f, axis=1, keepdims=True) + 1e-8)
                env_l_f = (env_l_f - np.mean(env_l_f, axis=1, keepdims=True)) / (np.std(env_l_f, axis=1, keepdims=True) + 1e-8)
                env_r_f = (env_r_f - np.mean(env_r_f, axis=1, keepdims=True)) / (np.std(env_r_f, axis=1, keepdims=True) + 1e-8)
                
                eeg_bands.append(eeg_f)
                env_l_bands.append(env_l_f)
                env_r_bands.append(env_r_f)
                
            trials.append({'eeg_bands': eeg_bands, 'env_l_bands': env_l_bands, 'env_r_bands': env_r_bands, 'meta': tr['meta']})
            
        N_TRIALS = len(trials)
        
        # 1. Precompute XTX and XTy for all trials and bands
        print("  [Precompute] Building covariance matrices...", flush=True)
        cov_cache = {b: [] for b in range(len(BANDS))}
        for t_idx in range(N_TRIALS):
            for b_idx in range(len(BANDS)):
                XTX, XTy = compute_trial_covariance(trials[t_idx], b_idx)
                cov_cache[b_idx].append((XTX, XTy))
                
        # 2. Main LOTO Loop
        print(f"  [LOTO] Running {N_TRIALS}-Fold Leave-One-Trial-Out Cross Validation...", flush=True)
        
        all_eval_diffs = []
        all_eval_labels = []
        
        start_time = time.time()
        
        for test_idx in range(N_TRIALS):
            # Inner LOTO for Calibration (on the 59 train trials)
            train_indices = [i for i in range(N_TRIALS) if i != test_idx]
            
            calibration_diffs = {b: [] for b in range(len(BANDS))}
            calibration_labels = {b: [] for b in range(len(BANDS))}
            
            for b_idx in range(len(BANDS)):
                # XTX for all 59 train trials
                XTX_train_full = sum([cov_cache[b_idx][i][0] for i in train_indices])
                XTy_train_full = sum([cov_cache[b_idx][i][1] for i in train_indices])
                
                for val_idx in train_indices:
                    # Inner Leave-One-Out (58 train, 1 val)
                    XTX_subtrain = XTX_train_full - cov_cache[b_idx][val_idx][0]
                    XTy_subtrain = XTy_train_full - cov_cache[b_idx][val_idx][1]
                    
                    W_subtrain = fast_ridge(XTX_subtrain, XTy_subtrain, lam=100.0)
                    
                    diffs, labels = evaluate_trial_sequence(W_subtrain, trials[val_idx], b_idx)
                    calibration_diffs[b_idx].extend(diffs)
                    calibration_labels[b_idx].extend(labels)
            
            # Select Top 2 Bands for this specific LOTO split
            band_aucs = {}
            for b_idx, band_name in enumerate(BANDS.keys()):
                diffs = np.array(calibration_diffs[b_idx])
                Y = np.array(calibration_labels[b_idx])
                probs = (diffs - np.min(diffs)) / (np.max(diffs) - np.min(diffs) + 1e-8)
                auc = roc_auc_score(Y, probs) if len(np.unique(Y)) > 1 else 0.5
                band_aucs[b_idx] = auc
                
            top_2_b_idx = sorted(band_aucs.keys(), key=lambda x: band_aucs[x], reverse=True)[:2]
            
            # Now train the selected Top 2 bands on the full 59 train trials
            test_diffs_ensemble = None
            test_labels = None
            
            for b_idx in top_2_b_idx:
                XTX_train_full = sum([cov_cache[b_idx][i][0] for i in train_indices])
                XTy_train_full = sum([cov_cache[b_idx][i][1] for i in train_indices])
                
                W_final = fast_ridge(XTX_train_full, XTy_train_full, lam=100.0)
                
                diffs, labels = evaluate_trial_sequence(W_final, trials[test_idx], b_idx)
                
                if test_diffs_ensemble is None:
                    test_diffs_ensemble = diffs
                    test_labels = labels
                else:
                    test_diffs_ensemble += diffs
                    
            all_eval_diffs.extend(test_diffs_ensemble)
            all_eval_labels.extend(test_labels)
            
        # Global AUROC across all 60 independent evaluations
        all_eval_diffs = np.array(all_eval_diffs)
        all_eval_labels = np.array(all_eval_labels)
        probs = (all_eval_diffs - np.min(all_eval_diffs)) / (np.max(all_eval_diffs) - np.min(all_eval_diffs) + 1e-8)
        global_auc = roc_auc_score(all_eval_labels, probs) if len(np.unique(all_eval_labels)) > 1 else 0.5
        
        print(f"  [Deployment] LOTO Global AUROC: {global_auc:.4f} (Took {time.time()-start_time:.1f}s)")
        final_results[subj_name] = global_auc

    print("\n\n=======================================================")
    print(" PHASE 113 LOTO CLINICAL CALIBRATION RESULTS")
    print("=======================================================")
    print(f"{'Subject':<10} {'LOTO AUROC':<10}")
    for subj, auc in final_results.items():
        print(f"{subj:<10} {auc:.4f}")

if __name__ == '__main__':
    main()
