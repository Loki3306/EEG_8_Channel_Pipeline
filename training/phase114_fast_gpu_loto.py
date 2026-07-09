import numpy as np
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
MAX_LAG_SEC = 0.400 
MAX_LAG_SAMPLES = int(MAX_LAG_SEC * SR)
SEQ_SAMPLES = int(3.5 * SR)
SEQ_HOP = int(0.5 * SR)

BANDS = {
    'Delta': (0.5, 4.0),
    'Theta': (4.0, 8.0),
    'Alpha': (8.0, 15.0),
    'Beta': (15.0, 32.0)
}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def apply_modulation_filter(env, lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype='band')
    filtered = signal.filtfilt(b, a, env, axis=1)
    return filtered

def create_toeplitz_features_pt(eeg, max_lag_samples):
    C, T = eeg.shape
    T_eff = T - max_lag_samples
    X = torch.zeros((T_eff, C * max_lag_samples), dtype=eeg.dtype, device=eeg.device)
    for tau in range(max_lag_samples):
        X[:, tau*C : (tau+1)*C] = eeg[:, tau : tau+T_eff].T
    return X

def batch_pearsonr_pt(x, y):
    # x, y: (N_seq, T)
    x_mean = x - x.mean(dim=1, keepdim=True)
    y_mean = y - y.mean(dim=1, keepdim=True)
    num = (x_mean * y_mean).sum(dim=1)
    den = torch.sqrt((x_mean**2).sum(dim=1) * (y_mean**2).sum(dim=1))
    return num / (den + 1e-8)

def fast_ridge_pt(XTX, XTy, lam=100.0):
    I = torch.eye(XTX.shape[0], device=XTX.device, dtype=XTX.dtype)
    W = torch.linalg.solve(XTX + lam * I, XTy)
    return W

def prepare_trial_data(tr, band_idx):
    eeg = torch.tensor(tr['eeg_bands'][band_idx], dtype=torch.float32, device=device)
    env_l = torch.tensor(tr['env_l_bands'][band_idx][0], dtype=torch.float32, device=device)
    env_r = torch.tensor(tr['env_r_bands'][band_idx][0], dtype=torch.float32, device=device)
    
    T = eeg.shape[1]
    X_trial = create_toeplitz_features_pt(eeg, MAX_LAG_SAMPLES)
    T_eff = X_trial.shape[0]
    
    Y_l_eff = env_l[:T_eff]
    Y_r_eff = env_r[:T_eff]
    
    sp = tr['meta']['switch_points']
    boundaries = [0]
    boundaries.extend([idx for spk, idx in sp])
    boundaries = sorted(set(boundaries))
    if boundaries[-1] != T: boundaries.append(T)
        
    Y_att = torch.zeros(T, dtype=torch.float32, device=device)
    seq_indices = []
    Y_meta = []
    
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
            
        safe_start = start_idx + int(1.5 * SR)
        safe_end = end_idx
        
        if safe_end - safe_start >= SEQ_SAMPLES:
            for seq_start in range(safe_start, safe_end - SEQ_SAMPLES + 1, SEQ_HOP):
                if seq_start + SEQ_SAMPLES <= T_eff:
                    seq_indices.append((seq_start, seq_start + SEQ_SAMPLES))
                    Y_meta.append(1.0 if current_spk == 'L' else 0.0)
                
    Y_trial = Y_att[:T_eff]
    XTX = X_trial.T @ X_trial
    XTy = X_trial.T @ Y_trial
    
    Y_meta_tensor = torch.tensor(Y_meta, dtype=torch.float32, device=device)
    
    return {
        'X': X_trial,
        'Y_l': Y_l_eff,
        'Y_r': Y_r_eff,
        'XTX': XTX,
        'XTy': XTy,
        'seq_indices': seq_indices,
        'Y_meta': Y_meta_tensor
    }

def evaluate_trial_fast(W_band, tr_info):
    if len(tr_info['seq_indices']) == 0:
        return torch.tensor([], device=device), torch.tensor([], device=device)
        
    Y_hat_full = tr_info['X'] @ W_band
    
    Y_hat_seqs = torch.stack([Y_hat_full[s:e] for s, e in tr_info['seq_indices']])
    Y_l_seqs = torch.stack([tr_info['Y_l'][s:e] for s, e in tr_info['seq_indices']])
    Y_r_seqs = torch.stack([tr_info['Y_r'][s:e] for s, e in tr_info['seq_indices']])
    
    r_L = batch_pearsonr_pt(Y_hat_seqs, Y_l_seqs)
    r_R = batch_pearsonr_pt(Y_hat_seqs, Y_r_seqs)
    
    diff_scores = r_L - r_R
    return diff_scores, tr_info['Y_meta']

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
            
    print(f"\n=======================================================")
    print(f" PHASE 114: FAST GPU LOTO CALIBRATION")
    print(f" Device: {device}")
    print(f"=======================================================\n", flush=True)
    
    cache_files = sorted(list(cache_dir.glob('*_multiband.pt')))
    final_results = {}
    
    for cache_file in cache_files:
        subj_name = cache_file.stem.split('_')[0]
        print(f"\n=======================================================")
        print(f" SUBJECT {subj_name}")
        print(f"=======================================================", flush=True)
        
        cached = torch.load(cache_file, map_location='cpu', weights_only=False)['raw']
        
        # CPU Pre-filtering to save GPU memory overhead
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
                
                # Zero-leakage Per-Trial Normalization
                eeg_f = (eeg_f - np.mean(eeg_f, axis=1, keepdims=True)) / (np.std(eeg_f, axis=1, keepdims=True) + 1e-8)
                env_l_f = (env_l_f - np.mean(env_l_f, axis=1, keepdims=True)) / (np.std(env_l_f, axis=1, keepdims=True) + 1e-8)
                env_r_f = (env_r_f - np.mean(env_r_f, axis=1, keepdims=True)) / (np.std(env_r_f, axis=1, keepdims=True) + 1e-8)
                
                eeg_bands.append(eeg_f)
                env_l_bands.append(env_l_f)
                env_r_bands.append(env_r_f)
                
            trials.append({'eeg_bands': eeg_bands, 'env_l_bands': env_l_bands, 'env_r_bands': env_r_bands, 'meta': tr['meta']})
            
        N_TRIALS = len(trials)
        
        print("  [GPU] Precomputing covariance matrices and sequences...", flush=True)
        start_prep = time.time()
        
        # Precompute everything on GPU
        # data_cache[band_idx][trial_idx]
        data_cache = {b: [] for b in range(len(BANDS))}
        for t_idx in range(N_TRIALS):
            for b_idx in range(len(BANDS)):
                tr_info = prepare_trial_data(trials[t_idx], b_idx)
                data_cache[b_idx].append(tr_info)
                
        print(f"  [GPU] Preparation took {time.time() - start_prep:.2f}s", flush=True)
                
        print(f"  [LOTO] Running Vectorized {N_TRIALS}-Fold LOTO Cross Validation...", flush=True)
        start_time = time.time()
        
        all_eval_diffs = []
        all_eval_labels = []
        
        for test_idx in range(N_TRIALS):
            train_indices = [i for i in range(N_TRIALS) if i != test_idx]
            
            calibration_diffs = {b: [] for b in range(len(BANDS))}
            calibration_labels = {b: [] for b in range(len(BANDS))}
            
            for b_idx in range(len(BANDS)):
                XTX_train_full = sum([data_cache[b_idx][i]['XTX'] for i in train_indices])
                XTy_train_full = sum([data_cache[b_idx][i]['XTy'] for i in train_indices])
                
                for val_idx in train_indices:
                    XTX_subtrain = XTX_train_full - data_cache[b_idx][val_idx]['XTX']
                    XTy_subtrain = XTy_train_full - data_cache[b_idx][val_idx]['XTy']
                    
                    W_subtrain = fast_ridge_pt(XTX_subtrain, XTy_subtrain, lam=100.0)
                    
                    diffs, labels = evaluate_trial_fast(W_subtrain, data_cache[b_idx][val_idx])
                    if len(diffs) > 0:
                        calibration_diffs[b_idx].extend(diffs.cpu().tolist())
                        calibration_labels[b_idx].extend(labels.cpu().tolist())
            
            band_aucs = {}
            for b_idx, band_name in enumerate(BANDS.keys()):
                diffs = np.array(calibration_diffs[b_idx])
                Y = np.array(calibration_labels[b_idx])
                if len(diffs) > 0 and len(np.unique(Y)) > 1:
                    probs = (diffs - np.min(diffs)) / (np.max(diffs) - np.min(diffs) + 1e-8)
                    band_aucs[b_idx] = roc_auc_score(Y, probs)
                else:
                    band_aucs[b_idx] = 0.5
                
            top_2_b_idx = sorted(band_aucs.keys(), key=lambda x: band_aucs[x], reverse=True)[:2]
            
            test_diffs_ensemble = None
            test_labels = None
            
            for b_idx in top_2_b_idx:
                XTX_train_full = sum([data_cache[b_idx][i]['XTX'] for i in train_indices])
                XTy_train_full = sum([data_cache[b_idx][i]['XTy'] for i in train_indices])
                
                W_final = fast_ridge_pt(XTX_train_full, XTy_train_full, lam=100.0)
                diffs, labels = evaluate_trial_fast(W_final, data_cache[b_idx][test_idx])
                
                if test_diffs_ensemble is None:
                    test_diffs_ensemble = diffs
                    test_labels = labels
                else:
                    test_diffs_ensemble += diffs
                    
            if test_diffs_ensemble is not None and len(test_diffs_ensemble) > 0:
                all_eval_diffs.extend(test_diffs_ensemble.cpu().tolist())
                all_eval_labels.extend(test_labels.cpu().tolist())
            
        all_eval_diffs = np.array(all_eval_diffs)
        all_eval_labels = np.array(all_eval_labels)
        if len(all_eval_diffs) > 0 and len(np.unique(all_eval_labels)) > 1:
            probs = (all_eval_diffs - np.min(all_eval_diffs)) / (np.max(all_eval_diffs) - np.min(all_eval_diffs) + 1e-8)
            global_auc = roc_auc_score(all_eval_labels, probs)
        else:
            global_auc = 0.5
        
        print(f"  [Deployment] LOTO Global AUROC: {global_auc:.4f} (Took {time.time()-start_time:.2f}s)")
        final_results[subj_name] = global_auc

    print("\n\n=======================================================")
    print(" PHASE 114 FAST GPU LOTO CLINICAL CALIBRATION RESULTS")
    print("=======================================================")
    print(f"{'Subject':<10} {'LOTO AUROC':<10}")
    for subj, auc in final_results.items():
        print(f"{subj:<10} {auc:.4f}")

if __name__ == '__main__':
    main()
