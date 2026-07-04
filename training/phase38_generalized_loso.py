import os
import sys
import numpy as np
import scipy.io
import scipy.io.wavfile
import scipy.signal
from sklearn.metrics import roc_auc_score
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.phase32_5_spatial_fix import load_aasd_subject_trials
from training.phase35_neural_ridge import build_lagged_matrix

def safe_pearson(x, y):
    if np.var(x) < 1e-8 or np.var(y) < 1e-8:
        return np.nan
    return np.corrcoef(x, y)[0, 1]

def run_classical_test_suite(W, test_trials, max_lag, hop_len, window_len, suite_mode, offset_sec=0.0):
    sim_att, sim_unatt = [], []
    for trial_idx, trial in enumerate(test_trials):
        eeg_full = trial['eeg'].numpy()
        env_l = trial['env_l'].numpy()
        env_r = trial['env_r'].numpy()
        switch_points = trial['meta']['switch_points']
        
        if suite_mode == "SHUFFLE_AUDIO":
            alt_trial = test_trials[(trial_idx + 1) % len(test_trials)]
            env_l = alt_trial['env_l'].numpy()
            env_r = alt_trial['env_r'].numpy()
            switch_points = alt_trial['meta']['switch_points']
            
            min_len = min(eeg_full.shape[1], len(env_l))
            eeg_full = eeg_full[:, :min_len]
            env_l = env_l[:min_len]
            env_r = env_r[:min_len]
            
        eeg = (eeg_full - eeg_full.mean(axis=1, keepdims=True)) / (eeg_full.std(axis=1, keepdims=True) + 1e-8)
        env_l = (env_l - env_l.mean()) / (env_l.std() + 1e-8)
        env_r = (env_r - env_r.mean()) / (env_r.std() + 1e-8)
        
        if suite_mode == "REVERSE_EEG":
            eeg = np.flip(eeg, axis=1).copy()
        elif suite_mode == "NOISE_EEG":
            eeg = np.random.randn(*eeg.shape).astype(np.float32)
            
        att = np.zeros(eeg.shape[1], dtype=np.float32)
        unatt = np.zeros(eeg.shape[1], dtype=np.float32)
        if len(switch_points) == 0: switch_points = [('R', 0)]
            
        if switch_points[0][1] > 0:
            initial_state = 'R' if switch_points[0][0] == 'L' else 'L'
        else:
            initial_state = switch_points[0][0]
            
        current_state = initial_state
        prev_idx = 0
        for state, idx_64 in switch_points:
            idx_64 = min(idx_64, eeg.shape[1])
            if idx_64 > prev_idx:
                if current_state == 'L':
                    att[prev_idx:idx_64] = env_l[prev_idx:idx_64]
                    unatt[prev_idx:idx_64] = env_r[prev_idx:idx_64]
                else:
                    att[prev_idx:idx_64] = env_r[prev_idx:idx_64]
                    unatt[prev_idx:idx_64] = env_l[prev_idx:idx_64]
            prev_idx, current_state = idx_64, state
            
        if current_state == 'L':
            att[prev_idx:] = env_l[prev_idx:]
            unatt[prev_idx:] = env_r[prev_idx:]
        else:
            att[prev_idx:] = env_r[prev_idx:]
            unatt[prev_idx:] = env_l[prev_idx:]
            
        if suite_mode == "TEMPORAL_OFFSET":
            offset_samples = int(offset_sec * 64)
            if offset_samples > 0:
                att = np.concatenate([np.zeros(offset_samples), att[:-offset_samples]])
                unatt = np.concatenate([np.zeros(offset_samples), unatt[:-offset_samples]])
            elif offset_samples < 0:
                att = np.concatenate([att[-offset_samples:], np.zeros(-offset_samples)])
                unatt = np.concatenate([unatt[-offset_samples:], np.zeros(-offset_samples)])
                
        # Build predictions globally for speed
        X_trial = build_lagged_matrix(eeg, max_lag)
        # Pad beginning with zeros to align with original EEG indices
        pred_full = np.concatenate([np.zeros(max_lag), X_trial @ W])
                
        for start in range(0, eeg.shape[1] - window_len + 1, hop_len):
            end = start + window_len
            
            pred_w = pred_full[start:end]
            att_w = att[start:end]
            unatt_w = unatt[start:end]
            
            c_a = safe_pearson(pred_w, att_w)
            c_u = safe_pearson(pred_w, unatt_w)
            
            if not np.isnan(c_a) and not np.isnan(c_u):
                sim_att.append(c_a)
                sim_unatt.append(c_u)
                
    if len(sim_att) == 0:
        return 0.5, 0.5, 0.0, 0.0
        
    sim_att = np.array(sim_att)
    sim_unatt = np.array(sim_unatt)
    
    margin = sim_att - sim_unatt
    acc = np.mean(margin > 0)
    y_true = np.concatenate([np.ones(len(sim_att)), np.zeros(len(sim_unatt))])
    y_scores = np.concatenate([sim_att, sim_unatt])
    
    try:
        auroc = roc_auc_score(y_true, y_scores)
    except ValueError:
        auroc = 0.5
    
    return auroc, acc, sim_att.mean(), sim_unatt.mean()


def run_generalized_loso():
    print("=======================================================")
    print(" PHASE 38: GENERALIZED CLASSICAL RIDGE LOSO VALIDATION ")
    print("=======================================================")
    
    data_root = '/kaggle/input/datasets/lokeshgile/aasd-processed-eeg/Processed EEG'
    mat_files = []
    for root, dirs, files in os.walk(data_root):
        for file in files:
            if file.endswith('.mat') and not file.startswith('._'):
                mat_files.append(os.path.join(root, file))
                
    if not mat_files:
        print("ERROR: No .mat files found. Please run on Kaggle.")
        return
        
    b, a = scipy.signal.butter(4, [1.0/64.0, 8.0/64.0], btype='band')
    audio_dir = '/kaggle/input/datasets/lokeshgile/aasd-audio-gammatones'
    wav_dir = '/kaggle/input/datasets/lokeshgile/aasd-audio/Stimuli Audio'
    
    max_lag = 24
    
    # 1. FIX: Scale Alpha linearly with the number of training subjects
    num_train_subjects = 17 
    alpha_base = 10000.0
    alpha_ridge = alpha_base * num_train_subjects
    print(f"Mathematical Fix: Scaled Ridge Alpha = {alpha_ridge} (Base {alpha_base} x {num_train_subjects} subjects)")
    
    # 2. FIX: Use all 62 channels instead of overfitting to S18's topology
    print("Spatial Fix: Using ALL 62 EEG Channels for generalized spatial filtering.")
    C = 62
    
    window_len = 64 * 5
    hop_len = 64 * 1
    
    subject_data = {}
    
    print("\n--- 1. Precomputing Subject Covariance Matrices ---")
    start_time = time.time()
    for sub_path in mat_files:
        sub_id = os.path.basename(sub_path).split('.')[0]
        trials = load_aasd_subject_trials(sub_path, b, a, audio_dir, wav_dir)
        
        X_train_list, Y_train_list = [], []
        
        for trial in trials:
            eeg = trial['eeg'].numpy()
            env_l = trial['env_l'].numpy()
            env_r = trial['env_r'].numpy()
            switch_points = trial['meta']['switch_points']
            
            eeg = (eeg - eeg.mean(axis=1, keepdims=True)) / (eeg.std(axis=1, keepdims=True) + 1e-8)
            env_l = (env_l - env_l.mean()) / (env_l.std() + 1e-8)
            env_r = (env_r - env_r.mean()) / (env_r.std() + 1e-8)
            
            X = build_lagged_matrix(eeg, max_lag)
            att = np.zeros(eeg.shape[1], dtype=np.float32)
            if len(switch_points) == 0:
                switch_points = [('R', 0)]
                
            initial_state = 'R' if (switch_points[0][1] > 0 and switch_points[0][0] == 'L') else switch_points[0][0]
            if switch_points[0][1] > 0:
                initial_state = 'R' if switch_points[0][0] == 'L' else 'L'
                
            current_state = initial_state
            prev_idx = 0
            for state, idx_64 in switch_points:
                if idx_64 > prev_idx:
                    if current_state == 'L': att[prev_idx:idx_64] = env_l[prev_idx:idx_64]
                    else: att[prev_idx:idx_64] = env_r[prev_idx:idx_64]
                prev_idx, current_state = idx_64, state
                
            if current_state == 'L': att[prev_idx:] = env_l[prev_idx:]
            else: att[prev_idx:] = env_r[prev_idx:]
                
            Y = att[max_lag:]
            X_train_list.append(X)
            Y_train_list.append(Y)
            
        X_mat = np.vstack(X_train_list)
        Y_mat = np.concatenate(Y_train_list)
        
        cov_X = X_mat.T @ X_mat
        cov_XY = X_mat.T @ Y_mat
        
        subject_data[sub_id] = {
            'cov_X': cov_X,
            'cov_XY': cov_XY,
            'trials': trials
        }
        print(f"  Processed {sub_id} | CovX Trace: {np.trace(cov_X):.2e}")
        
    print(f"Data loading complete in {time.time() - start_time:.1f}s")
    
    print("\n--- 2. Leave-One-Subject-Out (LOSO) Execution ---")
    
    global_results = {
        "NORMAL": [],
        "SHUFFLE_AUDIO": [],
        "REVERSE_EEG": [],
        "NOISE_EEG": []
    }
    
    for test_sub in sorted(subject_data.keys()):
        print(f"\n[{test_sub}] Starting Evaluation Fold...")
        
        train_cov_X = np.zeros_like(subject_data[test_sub]['cov_X'])
        train_cov_XY = np.zeros_like(subject_data[test_sub]['cov_XY'])
        
        for sub_id, data in subject_data.items():
            if sub_id != test_sub:
                train_cov_X += data['cov_X']
                train_cov_XY += data['cov_XY']
                
        ridge_matrix = train_cov_X + alpha_ridge * np.eye(train_cov_X.shape[0])
        W_analytical = np.linalg.solve(ridge_matrix, train_cov_XY)
        
        test_trials = subject_data[test_sub]['trials']
        
        suites = [
            ("NORMAL", "NORMAL", 0.0),
            ("SHUFFLE_AUDIO", "SHUFFLE_AUDIO", 0.0),
            ("REVERSE_EEG", "REVERSE_EEG", 0.0),
            ("NOISE_EEG", "NOISE_EEG", 0.0)
        ]
        
        for suite_name, mode, offset in suites:
            auroc, acc, _, _ = run_classical_test_suite(
                W_analytical, test_trials, max_lag, hop_len, window_len, mode, offset
            )
            global_results[suite_name].append(auroc)
            
            if suite_name == "NORMAL":
                print(f"  -> NORMAL AUROC: {auroc:.4f}")
                
    print("\n=======================================================")
    print("                 FINAL LOSO RESULTS                    ")
    print("=======================================================")
    
    for suite_name in global_results.keys():
        aurocs = np.array(global_results[suite_name])
        mean_auroc = aurocs.mean()
        std_auroc = aurocs.std()
        print(f"{suite_name:<15} | Mean AUROC: {mean_auroc:.4f} ± {std_auroc:.4f}")
        
    print("=======================================================")

if __name__ == "__main__":
    run_generalized_loso()
