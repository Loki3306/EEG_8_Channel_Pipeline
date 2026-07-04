import os
import sys
import numpy as np
import scipy.io
import scipy.io.wavfile
import scipy.signal
from sklearn.metrics import roc_auc_score
import time
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.phase32_5_spatial_fix import load_aasd_subject_trials
from training.phase35_neural_ridge import NeuralRidgeDecoder, build_lagged_matrix, pearson_loss

def run_test_suite(model, test_trials, selected_channels, device, max_lag, hop_len, window_len, suite_mode, offset_sec=0.0):
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
            
        eeg_full = (eeg_full - eeg_full.mean(axis=1, keepdims=True)) / (eeg_full.std(axis=1, keepdims=True) + 1e-8)
        env_l = (env_l - env_l.mean()) / (env_l.std() + 1e-8)
        env_r = (env_r - env_r.mean()) / (env_r.std() + 1e-8)
        
        eeg = eeg_full[selected_channels, :]
        
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
                
        starts = list(range(0, eeg.shape[1] - window_len + 1, hop_len))
        if not starts:
            continue
            
        eeg_windows = [eeg[:, s:s+window_len] for s in starts]
        eeg_batch = torch.from_numpy(np.stack(eeg_windows)).to(device)
        
        with torch.no_grad():
            preds = model(eeg_batch).squeeze(1).cpu().numpy()
            
        for idx, start in enumerate(starts):
            end = start + window_len
            
            pred_w = preds[idx]
            att_w = att[start:end]
            unatt_w = unatt[start:end]
            
            c_a = np.corrcoef(pred_w, att_w)[0, 1]
            c_u = np.corrcoef(pred_w, unatt_w)[0, 1]
            
            if not np.isnan(c_a) and not np.isnan(c_u):
                sim_att.append(c_a)
                sim_unatt.append(c_u)
                
    sim_att = np.array(sim_att)
    sim_unatt = np.array(sim_unatt)
    
    margin = sim_att - sim_unatt
    acc = np.mean(margin > 0)
    y_true = np.concatenate([np.ones(len(sim_att)), np.zeros(len(sim_unatt))])
    y_scores = np.concatenate([sim_att, sim_unatt])
    auroc = roc_auc_score(y_true, y_scores)
    
    return auroc, acc, sim_att.mean(), sim_unatt.mean()

def run_loso_validation():
    print("=======================================================")
    print("   PHASE 37: MASSIVE LEAVE-ONE-SUBJECT-OUT VALIDATION  ")
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
    alpha_ridge = 10000.0
    selected_channels = [29, 26, 31, 5, 13, 21, 22, 44, 55, 41, 45, 15, 17, 56, 61, 60]
    num_lags = max_lag + 1
    C = len(selected_channels)
    
    window_len = 64 * 5
    hop_len = 64 * 1
    
    subject_data = {}
    
    print("\n--- 1. Precomputing Subject Covariance Matrices & Tensors ---")
    start_time = time.time()
    for sub_path in mat_files:
        sub_id = os.path.basename(sub_path).split('.')[0]
        trials = load_aasd_subject_trials(sub_path, b, a, audio_dir, wav_dir)
        
        X_train_list, Y_train_list = [], []
        X_nn_list, Y_nn_list = [], []
        
        for trial in trials:
            eeg_full = trial['eeg'].numpy()
            env_l = trial['env_l'].numpy()
            env_r = trial['env_r'].numpy()
            switch_points = trial['meta']['switch_points']
            
            eeg_full = (eeg_full - eeg_full.mean(axis=1, keepdims=True)) / (eeg_full.std(axis=1, keepdims=True) + 1e-8)
            env_l = (env_l - env_l.mean()) / (env_l.std() + 1e-8)
            env_r = (env_r - env_r.mean()) / (env_r.std() + 1e-8)
            
            eeg = eeg_full[selected_channels, :]
            
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
            
            for start in range(0, eeg.shape[1] - window_len + 1, hop_len):
                end = start + window_len
                is_transition = False
                for _, sw_idx in switch_points:
                    if start < sw_idx < end:
                        is_transition = True
                        break
                
                if not is_transition:
                    X_nn_list.append(torch.from_numpy(eeg[:, start:end]))
                    Y_nn_list.append(torch.from_numpy(att[start:end]))
                    
        X_mat = np.vstack(X_train_list)
        Y_mat = np.concatenate(Y_train_list)
        
        cov_X = X_mat.T @ X_mat
        cov_XY = X_mat.T @ Y_mat
        
        subject_data[sub_id] = {
            'cov_X': cov_X,
            'cov_XY': cov_XY,
            'X_nn': torch.stack(X_nn_list),
            'Y_nn': torch.stack(Y_nn_list),
            'trials': trials
        }
        print(f"  Processed {sub_id} | Windows: {len(X_nn_list)}")
        
    print(f"Data loading complete in {time.time() - start_time:.1f}s")
    
    print("\n--- 2. Leave-One-Subject-Out (LOSO) Execution ---")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    global_results = {
        "NORMAL": [],
        "SHUFFLE_AUDIO": [],
        "REVERSE_EEG": [],
        "NOISE_EEG": [],
        "OFFSET_+5S": [],
        "OFFSET_-5S": []
    }
    
    for test_sub in sorted(subject_data.keys()):
        print(f"\n[{test_sub}] Starting Evaluation Fold...")
        
        # 1. Aggregate Training Data
        train_cov_X = np.zeros_like(subject_data[test_sub]['cov_X'])
        train_cov_XY = np.zeros_like(subject_data[test_sub]['cov_XY'])
        
        train_X_nn = []
        train_Y_nn = []
        
        for sub_id, data in subject_data.items():
            if sub_id != test_sub:
                train_cov_X += data['cov_X']
                train_cov_XY += data['cov_XY']
                train_X_nn.append(data['X_nn'])
                train_Y_nn.append(data['Y_nn'])
                
        train_X_nn = torch.cat(train_X_nn, dim=0).to(device)
        train_Y_nn = torch.cat(train_Y_nn, dim=0).to(device)
        
        # 2. Solve Analytical Ridge
        ridge_matrix = train_cov_X + alpha_ridge * np.eye(train_cov_X.shape[0])
        W_analytical = np.linalg.solve(ridge_matrix, train_cov_XY)
        
        # 3. Train Neural Hybrid
        model = NeuralRidgeDecoder(in_channels=C, lags=max_lag).to(device)
        model.load_analytical_weights(W_analytical)
        
        # PRE-TRAINING EVALUATION (Pure Classical Ridge)
        model.eval()
        with torch.no_grad():
            auroc_base, _, _, _ = run_test_suite(
                model, subject_data[test_sub]['trials'], selected_channels, device, 
                max_lag, hop_len, window_len, "NORMAL", 0.0
            )
        print(f"  -> PRE-TRAIN (Pure Ridge) AUROC: {auroc_base:.4f}")
        
        dataset = TensorDataset(train_X_nn, train_Y_nn)
        dataloader = DataLoader(dataset, batch_size=4096, shuffle=True)
        
        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3, weight_decay=1e-2)
        model.train()
        
        for epoch in range(1, 16): 
            epoch_loss = 0.0
            for bx, by in dataloader:
                optimizer.zero_grad()
                pred = model(bx)
                loss = pearson_loss(pred, by)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                
            if epoch == 15:
                print(f"  -> Final Neural Epoch Loss: {epoch_loss / len(dataloader):.4f}")
                
        # 4. Falsification Testing
        model.eval()
        test_trials = subject_data[test_sub]['trials']
        
        suites = [
            ("NORMAL", "NORMAL", 0.0),
            ("SHUFFLE_AUDIO", "SHUFFLE_AUDIO", 0.0),
            ("REVERSE_EEG", "REVERSE_EEG", 0.0),
            ("NOISE_EEG", "NOISE_EEG", 0.0),
            ("OFFSET_+5S", "TEMPORAL_OFFSET", 5.0),
            ("OFFSET_-5S", "TEMPORAL_OFFSET", -5.0)
        ]
        
        with torch.no_grad():
            for suite_name, mode, offset in suites:
                auroc, acc, _, _ = run_test_suite(
                    model, test_trials, selected_channels, device, 
                    max_lag, hop_len, window_len, mode, offset
                )
                global_results[suite_name].append(auroc)
                
                if suite_name == "NORMAL":
                    print(f"  -> {suite_name}: AUROC = {auroc:.4f}")
                
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
    run_loso_validation()
