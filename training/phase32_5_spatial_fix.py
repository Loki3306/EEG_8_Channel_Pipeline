import os
import sys
import numpy as np
import scipy.io
import scipy.io.wavfile
import scipy.signal
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score
import time
from pathlib import Path
import glob

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.contrastive_aad import ContrastiveMatchNet

def get_spatial_assignment(wav_path, env_a, env_b, target_fs=64):
    """
    Dynamically determine which envelope corresponds to the Left ear by cross-correlating 
    with the raw left channel of the HRTF mixed stereo file.
    """
    if not os.path.exists(wav_path):
        return 'a_is_left'
        
    try:
        fs, data = scipy.io.wavfile.read(wav_path)
    except:
        return 'a_is_left'
        
    if len(data.shape) == 1:
        return 'a_is_left'
        
    left = np.abs(data[:, 0].astype(np.float32))
    down_factor = fs // target_fs
    left_ds = left[::down_factor]
    
    min_len = min(len(left_ds), len(env_a), len(env_b))
    left_ds = left_ds[:min_len]
    env_a_clip = env_a[:min_len]
    env_b_clip = env_b[:min_len]
    
    if len(left_ds) < 10:
        return 'a_is_left'
        
    corr_a = np.corrcoef(env_a_clip, left_ds)[0, 1]
    corr_b = np.corrcoef(env_b_clip, left_ds)[0, 1]
    
    return 'a_is_left' if corr_a > corr_b else 'b_is_left'

def load_aasd_subject_trials(sub_path, b, a, sel_idx, audio_dir, wav_dir):
    mat = scipy.io.loadmat(sub_path)
    data_all = mat['data']
    events = mat['events']
    audio_markers = mat['audio_markers'].flatten()
    
    trials = []
    for epoch_idx in range(1, data_all.shape[2] + 1):
        audio_marker_val = int(audio_markers[epoch_idx - 1])
        npz_path = os.path.join(audio_dir, f"{audio_marker_val}.npz")
        wav_path = os.path.join(wav_dir, f"mixed_{audio_marker_val - 10:03d}.wav")
        if not os.path.exists(npz_path): continue
            
        epoch_start_lat_128 = (epoch_idx - 1) * 7680 + 1
        switch_points = []
        for ev in events:
            if len(ev) >= 5:
                t_str, epoch_val = str(ev[0]).strip(), str(ev[4]).strip()
                if epoch_val == str(epoch_idx) and t_str in ['179', '184', '254', '255']:
                    abs_lat = float(ev[1])
                    rel_lat_128 = abs_lat - epoch_start_lat_128
                    idx_64 = max(0, int(rel_lat_128 / 2.0) - 4)
                    switch_points.append(('R' if t_str in ['179', '254'] else 'L', idx_64))
        switch_points.sort(key=lambda x: x[1])
        
        trial_eeg = data_all[:, :, epoch_idx - 1]
        trial_eeg_filt = scipy.signal.filtfilt(b, a, trial_eeg, axis=1)
        trial_eeg_8 = scipy.signal.resample_poly(trial_eeg_filt, 1, 2, axis=1)[sel_idx, 4:]
        
        audio_data = np.load(npz_path)
        env_a, env_b = audio_data['env_l'][:-4], audio_data['env_r'][:-4]
        
        spatial = get_spatial_assignment(wav_path, env_a, env_b)
        if spatial == 'a_is_left':
            env_l, env_r = env_a, env_b
        else:
            env_l, env_r = env_b, env_a
            
        min_len = min(trial_eeg_8.shape[1], len(env_l))
        trials.append({
            'eeg': torch.from_numpy(trial_eeg_8[:, :min_len]).float(),
            'env_l': torch.from_numpy(env_l[:min_len]).float(),
            'env_r': torch.from_numpy(env_r[:min_len]).float(),
            'meta': {'switch_points': switch_points}
        })
    return trials

def generate_windows(trials, window_len=64, hop_len=8, transition_margin=32, mask_transitions=True):
    X_eeg, X_att, X_unatt = [], [], []
    for trial in trials:
        eeg_full = trial['eeg']
        env_l_full = trial['env_l']
        env_r_full = trial['env_r']
        switch_points = trial['meta']['switch_points']
        
        env_l_full = (env_l_full - env_l_full.mean()) / (env_l_full.std() + 1e-8)
        env_r_full = (env_r_full - env_r_full.mean()) / (env_r_full.std() + 1e-8)
        eeg_full = (eeg_full - eeg_full.mean(dim=1, keepdim=True)) / (eeg_full.std(dim=1, keepdim=True) + 1e-8)
        
        for start in range(0, eeg_full.shape[1] - window_len + 1, hop_len):
            end = start + window_len
            is_trans = False
            if mask_transitions:
                for state, s_idx in switch_points:
                    t_start, t_end = s_idx - transition_margin, s_idx + transition_margin
                    if max(start, t_start) < min(end, t_end):
                        is_trans = True
                        break
            if is_trans: continue
            
            mid_point = start + window_len // 2
            initial_state = 'L'
            if switch_points and switch_points[0][1] > 0:
                initial_state = 'R' if switch_points[0][0] == 'L' else 'L'
            elif switch_points:
                initial_state = switch_points[0][0]
                
            current_state = initial_state
            for state, s_idx in switch_points:
                if mid_point >= s_idx: current_state = state
                    
            eeg_w = eeg_full[:, start:end].clone()
            env_l_w = env_l_full[start:end].clone()
            env_r_w = env_r_full[start:end].clone()
            
            if current_state == 'L':
                X_att.append(env_l_w); X_unatt.append(env_r_w)
            else:
                X_att.append(env_r_w); X_unatt.append(env_l_w)
            X_eeg.append(eeg_w)
            
    if not X_eeg:
        return None, None, None
        
    return torch.stack(X_eeg).float(), torch.stack(X_att).float(), torch.stack(X_unatt).float()

def pairwise_softmax_loss(z_e, z_att, z_unatt, tau=0.1):
    sim_att = (z_e * z_att).sum(dim=-1) / tau
    sim_unatt = (z_e * z_unatt).sum(dim=-1) / tau
    
    # logits: [B, 2]
    logits = torch.stack([sim_att, sim_unatt], dim=1)
    labels = torch.zeros(z_e.size(0), dtype=torch.long, device=z_e.device)
    
    loss = nn.functional.cross_entropy(logits, labels)
    return loss

def run_pairwise_softmax_no_norm():
    print("--- 1. Loading Data for Phase 32D No-Norm Fix ---")
    
    mat_files = glob.glob('/kaggle/input/datasets/lokeshgile/aasd-processed-eeg/Processed EEG/*/*.mat')
    if not mat_files:
        print("ERROR: No .mat files found. Please run on Kaggle.")
        return
        
    sub_path = next((p for p in mat_files if 'S18' in p), mat_files[0])
    
    b, a = scipy.signal.butter(4, [1.0/64.0, 8.0/64.0], btype='band')
    sel_idx = [23, 28, 22, 41, 36, 0, 40, 25]
    audio_dir = '/kaggle/input/datasets/lokeshgile/aasd-audio-gammatones'
    wav_dir = '/kaggle/input/datasets/lokeshgile/aasd-audio/Stimuli Audio'
    
    # Load trials
    print("\n--- 1. Loading Data for Phase 32D No-Norm Fix ---")
    trials = load_aasd_subject_trials(sub_path, b, a, sel_idx, audio_dir, wav_dir)
    print(f"Loaded {len(trials)} trials from {os.path.basename(sub_path)}")
    
    # Split: 40 train, 10 test
    train_trials = trials[:40]
    test_trials = trials[40:]
    
    print("Generating training windows (masked transitions)...")
    X_train_eeg, X_train_att, X_train_unatt = generate_windows(
        train_trials, window_len=64, hop_len=8, transition_margin=32, mask_transitions=True
    )
    print(f"Generated {len(X_train_eeg)} training windows.")
    
    print("Generating testing windows (no masking)...")
    X_test_eeg, X_test_att, X_test_unatt = generate_windows(
        test_trials, window_len=64, hop_len=8, transition_margin=0, mask_transitions=False
    )
    print(f"Generated {len(X_test_eeg)} testing windows.")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    model = ContrastiveMatchNet().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=150)
    
    # Setup DataLoaders
    train_dataset = torch.utils.data.TensorDataset(X_train_eeg, X_train_att, X_train_unatt)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=256, shuffle=True)
    
    test_dataset = torch.utils.data.TensorDataset(X_test_eeg, X_test_att, X_test_unatt)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=256, shuffle=False)
    
    epochs = 150
    
    print("\n--- 2. Training ---")
    start_time = time.time()
    
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0
        
        for b_eeg, b_att, b_unatt in train_loader:
            b_eeg, b_att, b_unatt = b_eeg.to(device), b_att.to(device), b_unatt.to(device)
            
            optimizer.zero_grad()
            z_e, z_a, z_b = model(b_eeg, b_att, b_unatt)
            
            loss = pairwise_softmax_loss(z_e, z_a, z_b)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * len(b_eeg)
            
        scheduler.step()
        train_loss = total_loss / len(train_dataset)
        
        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d} | Train Pairwise Softmax Loss: {train_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.6f}")
            
    print(f"Training completed in {time.time() - start_time:.1f}s")
    
    print("\n--- 3. Testing ---")
    model.eval()
    all_sim_att = []
    all_sim_unatt = []
    
    with torch.no_grad():
        for b_eeg, b_att, b_unatt in test_loader:
            b_eeg, b_att, b_unatt = b_eeg.to(device), b_att.to(device), b_unatt.to(device)
            
            z_e, z_a, z_b = model(b_eeg, b_att, b_unatt)
            sim_att = (z_e * z_a).sum(dim=-1).cpu().numpy()
            sim_unatt = (z_e * z_b).sum(dim=-1).cpu().numpy()
            
            all_sim_att.extend(sim_att)
            all_sim_unatt.extend(sim_unatt)
            
    sim_att = np.array(all_sim_att)
    sim_unatt = np.array(all_sim_unatt)
    
    margin = sim_att - sim_unatt
    acc = np.mean(margin > 0)
    
    y_true = np.concatenate([np.ones(len(sim_att)), np.zeros(len(sim_unatt))])
    y_scores = np.concatenate([sim_att, sim_unatt])
    auroc = roc_auc_score(y_true, y_scores)
    
    print(f"Test P(Att): {sim_att.mean():.4f}")
    print(f"Test P(Unatt): {sim_unatt.mean():.4f}")
    print(f"Margin Mean: {margin.mean():.4f}")
    print(f"Margin Std: {margin.std():.4f}")
    print(f"Test Accuracy: {acc*100:.1f}%")
    print(f"Test AUROC: {auroc:.4f}")

if __name__ == "__main__":
    run_pairwise_softmax_no_norm()
