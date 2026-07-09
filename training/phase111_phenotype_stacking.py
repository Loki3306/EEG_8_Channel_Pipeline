import numpy as np
from scipy.linalg import solve
from scipy.stats import pearsonr
import torch
from pathlib import Path
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
import random
from scipy import signal

# -------------------------------------------------------------------------
# CONSTANTS & CONFIGURATION
# -------------------------------------------------------------------------
SR = 128
EAR_CHANNEL_INDICES = [23, 31, 32, 40, 14, 22, 41, 49]
TARGET_SUBJECTS = ['S05', 'S08', 'S10', 'S11', 'S13', 'S16']

# mTRF Design Parameters
MAX_LAG_SEC = 0.400 
MAX_LAG_SAMPLES = int(MAX_LAG_SEC * SR)

# Frequency Bands (Filter-Bank)
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

def stratified_trial_split(trials, train_ratio=0.8):
    l_trials = []
    r_trials = []
    
    for i, tr in enumerate(trials):
        if get_trial_dominant_speaker(tr) == 'L':
            l_trials.append(i)
        else:
            r_trials.append(i)
            
    random.seed(42)
    random.shuffle(l_trials)
    random.shuffle(r_trials)
    
    l_split = int(len(l_trials) * train_ratio)
    r_split = int(len(r_trials) * train_ratio)
    
    train_indices = l_trials[:l_split] + r_trials[:r_split]
    eval_indices = l_trials[l_split:] + r_trials[r_split:]
    
    random.shuffle(train_indices)
    random.shuffle(eval_indices)
    
    return train_indices, eval_indices

def create_toeplitz_features(eeg, max_lag_samples):
    C, T = eeg.shape
    T_eff = T - max_lag_samples
    
    X = np.zeros((T_eff, C * max_lag_samples))
    for tau in range(max_lag_samples):
        X[:, tau*C : (tau+1)*C] = eeg[:, tau : tau+T_eff].T
    return X

def extract_mtrf_matrices(trials, band_idx):
    X_list = []
    Y_attended_list = []
    
    for tr in trials:
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
        
        X_list.append(X_trial)
        Y_attended_list.append(Y_trial)
        
    return np.vstack(X_list), np.concatenate(Y_attended_list)

def fit_ridge(X, y, lam):
    XTX = X.T @ X
    XTy = X.T @ y
    I = np.eye(XTX.shape[0])
    W = solve(XTX + lam * I, XTy, assume_a='pos')
    return W

def evaluate_filterbank_mtrf_features(W_dict, trials):
    X_meta = []
    Y_meta = []
    
    SEQ_SAMPLES = int(3.5 * SR)
    seq_hop = int(0.5 * SR)
    
    for tr in trials:
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
                    
                    seq_features = []
                    
                    for band_idx, band_name in enumerate(BANDS.keys()):
                        eeg_seq = tr['eeg_bands'][band_idx][:, seq_start:seq_start + SEQ_SAMPLES]
                        env_l_seq = tr['env_l_bands'][band_idx][0][seq_start:seq_start + SEQ_SAMPLES]
                        env_r_seq = tr['env_r_bands'][band_idx][0][seq_start:seq_start + SEQ_SAMPLES]
                        
                        X_seq = create_toeplitz_features(eeg_seq, MAX_LAG_SAMPLES)
                        T_eff = X_seq.shape[0]
                        
                        Y_l_eff = env_l_seq[:T_eff]
                        Y_r_eff = env_r_seq[:T_eff]
                        
                        Y_hat = X_seq @ W_dict[band_name]
                        
                        if np.std(Y_hat) < 1e-8 or np.std(Y_l_eff) < 1e-8 or np.std(Y_r_eff) < 1e-8:
                            r_L = 0.0
                            r_R = 0.0
                        else:
                            r_L, _ = pearsonr(Y_hat, Y_l_eff)
                            r_R, _ = pearsonr(Y_hat, Y_r_eff)
                        
                        seq_features.extend([r_L, r_R, r_L - r_R])
                    
                    label_L = 1.0 if current_spk == 'L' else 0.0
                    
                    X_meta.append(seq_features)
                    Y_meta.append(label_L)
                    
    return np.array(X_meta), np.array(Y_meta)

def main():
    possible_paths = [
        Path('/kaggle/input/datasets/lowkieee/multiband-cache/kaggle/working/multiband_cache'),
        Path('/kaggle/input/multiband-cache/kaggle/working/multiband_cache'),
        Path('/kaggle/input/aasd-universal-cache-v1/kaggle/working/multiband_cache'),
        Path('/kaggle/working/multiband_cache')
    ]
    
    cache_dir = Path('/kaggle/working/multiband_cache')
    for p in possible_paths:
        if p.exists() and len(list(p.glob('*_multiband.pt'))) > 0:
            cache_dir = p
            break
            
    print(f"\n=======================================================")
    print(f" PHASE 111: SUBJECT-LEVEL PHENOTYPE STACKING")
    print(f" Out-of-Fold Logistic Regression over Frequency Bands")
    print(f"=======================================================\n", flush=True)
    
    cache_files = sorted(list(cache_dir.glob('*_multiband.pt')))
    filtered_files = [f for f in cache_files if f.stem.split('_')[0] in TARGET_SUBJECTS]
    
    final_results = {}
    
    for cache_file in filtered_files:
        subj_name = cache_file.stem.split('_')[0]
        print(f"\n=======================================================")
        print(f" SUBJECT {subj_name}")
        print(f"=======================================================", flush=True)
        
        cached = torch.load(cache_file, map_location='cpu', weights_only=False)['raw']
        
        raw_trials = []
        for i in range(len(cached)):
            tr = cached[i]
            eeg_raw = tr['eeg'].numpy()[EAR_CHANNEL_INDICES, :] 
            env_l_raw = tr['env_l'].numpy()
            env_r_raw = tr['env_r'].numpy()
            
            eeg_bands = []
            env_l_bands = []
            env_r_bands = []
            
            min_len = min(eeg_raw.shape[1], env_l_raw.shape[1])
            eeg_raw = eeg_raw[:, :min_len]
            env_l_raw = env_l_raw[:, :min_len]
            env_r_raw = env_r_raw[:, :min_len]
            
            for band_name, (lowcut, highcut) in BANDS.items():
                eeg_f = apply_modulation_filter(eeg_raw, lowcut, highcut, SR)
                env_l_f = apply_modulation_filter(env_l_raw, lowcut, highcut, SR)
                env_r_f = apply_modulation_filter(env_r_raw, lowcut, highcut, SR)
                
                # Z-score normalization per band
                eeg_f = (eeg_f - np.mean(eeg_f, axis=1, keepdims=True)) / (np.std(eeg_f, axis=1, keepdims=True) + 1e-8)
                env_l_f = (env_l_f - np.mean(env_l_f, axis=1, keepdims=True)) / (np.std(env_l_f, axis=1, keepdims=True) + 1e-8)
                env_r_f = (env_r_f - np.mean(env_r_f, axis=1, keepdims=True)) / (np.std(env_r_f, axis=1, keepdims=True) + 1e-8)
                
                eeg_bands.append(eeg_f)
                env_l_bands.append(env_l_f)
                env_r_bands.append(env_r_f)
                
            raw_trials.append({
                'eeg_bands': eeg_bands, 
                'env_l_bands': env_l_bands, 
                'env_r_bands': env_r_bands, 
                'meta': tr['meta']
            })
            
        train_indices, eval_indices = stratified_trial_split(raw_trials, train_ratio=0.8)
        
        raw_train_trials = [raw_trials[i] for i in train_indices]
        raw_eval_trials = [raw_trials[i] for i in eval_indices]
        
        # ---------------------------------------------------------
        # STAGE 1: OUT-OF-FOLD (OOF) PREDICTIONS ON TRAIN SET
        # ---------------------------------------------------------
        print("  [Stage 1] Generating Unbiased OOF Features (5-Fold CV)...", flush=True)
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        
        X_meta_train_list = []
        Y_meta_train_list = []
        
        for fold, (kf_train_idx, kf_val_idx) in enumerate(kf.split(raw_train_trials)):
            fold_train_trials = [raw_train_trials[i] for i in kf_train_idx]
            fold_val_trials = [raw_train_trials[i] for i in kf_val_idx]
            
            W_dict_fold = {}
            for band_idx, band_name in enumerate(BANDS.keys()):
                X_train_fold, Y_train_fold = extract_mtrf_matrices(fold_train_trials, band_idx)
                W_dict_fold[band_name] = fit_ridge(X_train_fold, Y_train_fold, lam=100.0)
                
            X_meta_fold, Y_meta_fold = evaluate_filterbank_mtrf_features(W_dict_fold, fold_val_trials)
            X_meta_train_list.append(X_meta_fold)
            Y_meta_train_list.append(Y_meta_fold)
            
        X_meta_train = np.vstack(X_meta_train_list)
        Y_meta_train = np.concatenate(Y_meta_train_list)
        
        # ---------------------------------------------------------
        # STAGE 2: TRAIN LOGISTIC REGRESSION PHENOTYPE MODEL
        # ---------------------------------------------------------
        print(f"  [Stage 2] Training Logistic Regression on {X_meta_train.shape[1]} OOF Features...", flush=True)
        # We use a modest C=0.5 to prevent overfitting to the train set
        clf = LogisticRegression(C=0.5, class_weight='balanced', random_state=42, max_iter=1000)
        clf.fit(X_meta_train, Y_meta_train)
        
        # Print learned weights for intuition
        print("    Learned Phenotype Weights:")
        for band_idx, band_name in enumerate(BANDS.keys()):
            w_rL = clf.coef_[0][band_idx * 3]
            w_rR = clf.coef_[0][band_idx * 3 + 1]
            w_diff = clf.coef_[0][band_idx * 3 + 2]
            print(f"      {band_name:5s}: rL={w_rL:+.3f}, rR={w_rR:+.3f}, Δr={w_diff:+.3f}")
            
        # ---------------------------------------------------------
        # STAGE 3: FULL mTRF TRAINING
        # ---------------------------------------------------------
        print("  [Stage 3] Fitting Final mTRFs on Full Train Set...", flush=True)
        W_dict_full = {}
        for band_idx, band_name in enumerate(BANDS.keys()):
            X_train_full, Y_train_full = extract_mtrf_matrices(raw_train_trials, band_idx)
            W_dict_full[band_name] = fit_ridge(X_train_full, Y_train_full, lam=100.0)
            
        # ---------------------------------------------------------
        # STAGE 4: EVALUATION ON UNSEEN DATA
        # ---------------------------------------------------------
        print("  [Stage 4] Deploying on Unseen Eval Set...", flush=True)
        X_meta_eval, Y_meta_eval = evaluate_filterbank_mtrf_features(W_dict_full, raw_eval_trials)
        
        deployment_best_auc = 0
        if len(np.unique(Y_meta_eval)) > 1:
            y_pred_proba = clf.predict_proba(X_meta_eval)[:, 1]
            auc = roc_auc_score(Y_meta_eval, y_pred_proba)
            deployment_best_auc = auc
            
        print(f"  [Deployment] Final Stacked AUROC: {deployment_best_auc:.4f}")
        final_results[subj_name] = deployment_best_auc

    print("\n\n=======================================================")
    print(" PHASE 111 SUBJECT-LEVEL PHENOTYPE STACKING RESULTS")
    print("=======================================================")
    print(f"{'Subject':<10} {'Deployment AUROC':<10}")
    for subj, auc in final_results.items():
        print(f"{subj:<10} {auc:.4f}")

if __name__ == '__main__':
    main()
