import os
import sys
import torch
import torch.optim as optim
import numpy as np
import scipy.signal
import glob
from pathlib import Path
from sklearn.metrics import roc_auc_score
import time

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from training.phase29_cross_subject_train import load_aasd_subject
from models.contrastive_aad import ContrastiveMatchNet, contrastive_loss

def generate_windows(trials, window_len=64, hop_len=8, transition_margin=32, mask_transitions=True):
    X_eeg = []
    X_att = []
    X_unatt = []
    
    for trial in trials:
        eeg_full = trial['eeg']
        env_l_full = trial['env_l']
        env_r_full = trial['env_r']
        switch_points = trial['meta']['switch_points']
        
        for start in range(0, eeg_full.shape[1] - window_len + 1, hop_len):
            end = start + window_len
            
            # Check for transitions
            is_trans = False
            if mask_transitions:
                for state, s_idx in switch_points:
                    t_start, t_end = s_idx - transition_margin, s_idx + transition_margin
                    if max(start, t_start) < min(end, t_end):
                        is_trans = True
                        break
            
            if is_trans:
                continue
                
            # Determine center label
            mid_point = start + window_len // 2
            current_state = switch_points[0][0]
            for state, s_idx in switch_points:
                if mid_point >= s_idx:
                    current_state = state
                    
            eeg_w = eeg_full[:, start:end].clone()
            env_l_w = env_l_full[start:end].clone()
            env_r_w = env_r_full[start:end].clone()
            
            # Normalize audio envelopes individually
            env_l_w = (env_l_w - env_l_w.mean()) / (env_l_w.std() + 1e-8)
            env_r_w = (env_r_w - env_r_w.mean()) / (env_r_w.std() + 1e-8)
            
            if current_state == 1:
                X_att.append(env_l_w)
                X_unatt.append(env_r_w)
            else:
                X_att.append(env_r_w)
                X_unatt.append(env_l_w)
                
            X_eeg.append(eeg_w)
            
    if not X_eeg:
        return None, None, None
        
    return torch.stack(X_eeg).float(), torch.stack(X_att).float(), torch.stack(X_unatt).float()

def run_within_subject():
    print("--- 1. Loading Data for Phase 32B Within-Subject ---")
    
    mat_files = glob.glob('/kaggle/input/datasets/lokeshgile/aasd-processed-eeg/Processed EEG/*/*.mat')
    if not mat_files:
        print("ERROR: No .mat files found. Please run on Kaggle.")
        return
        
    sub_path = next((p for p in mat_files if 'S18' in p), mat_files[0])
    
    b, a = scipy.signal.butter(4, [1.0/64.0, 8.0/64.0], btype='band')
    sel_idx = [23, 28, 22, 41, 36, 0, 40, 25]
    audio_dir = '/kaggle/input/datasets/lokeshgile/aasd-audio-gammatones'
    
    trials = load_aasd_subject(sub_path, b, a, sel_idx, audio_dir)
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
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    
    # Setup DataLoaders
    train_dataset = torch.utils.data.TensorDataset(X_train_eeg, X_train_att, X_train_unatt)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=256, shuffle=True)
    
    test_dataset = torch.utils.data.TensorDataset(X_test_eeg, X_test_att, X_test_unatt)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=256, shuffle=False)
    
    epochs = 30
    
    print("\n--- 2. Training ---")
    start_time = time.time()
    
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0
        total_info = 0
        total_marg = 0
        
        for b_eeg, b_att, b_unatt in train_loader:
            b_eeg, b_att, b_unatt = b_eeg.to(device), b_att.to(device), b_unatt.to(device)
            
            optimizer.zero_grad()
            z_e, z_a, z_b = model(b_eeg, b_att, b_unatt)
            
            loss, l_info, l_marg = contrastive_loss(z_e, z_a, z_b)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * len(b_eeg)
            total_info += l_info.item() * len(b_eeg)
            total_marg += l_marg.item() * len(b_eeg)
            
        train_loss = total_loss / len(train_dataset)
        train_info = total_info / len(train_dataset)
        train_marg = total_marg / len(train_dataset)
        
        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d} | Train Loss: {train_loss:.4f} (Info: {train_info:.4f}, Marg: {train_marg:.4f})")
            
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
    run_within_subject()
