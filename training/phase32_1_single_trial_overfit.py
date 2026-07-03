import os
import sys
import torch
import torch.optim as optim
import numpy as np
import scipy.signal
import glob
from pathlib import Path
from sklearn.metrics import roc_auc_score

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from training.phase29_cross_subject_train import load_aasd_subject
from models.contrastive_aad import ContrastiveMatchNet, contrastive_loss

def run_single_trial_overfit():
    print("--- 1. Loading Single Trial for Phase 32A Overfit ---")
    
    # Kaggle dataset paths
    mat_files = glob.glob('/kaggle/input/datasets/lokeshgile/aasd-processed-eeg/Processed EEG/*/*.mat')
    if not mat_files:
        print("ERROR: No .mat files found. Please run on Kaggle.")
        return
        
    # We will just pick S18
    sub_path = next((p for p in mat_files if 'S18' in p), mat_files[0])
    
    b, a = scipy.signal.butter(4, [1.0/64.0, 8.0/64.0], btype='band')
    sel_idx = [23, 28, 22, 41, 36, 0, 40, 25]
    audio_dir = '/kaggle/input/datasets/lokeshgile/aasd-audio-gammatones'
    
    trials = load_aasd_subject(sub_path, b, a, sel_idx, audio_dir)
    trial = trials[0] # Pick the very first trial
    
    print(f"Loaded trial from {os.path.basename(sub_path)}")
    
    # 2. Generate Sliding Windows
    eeg_full = trial['eeg'] # [Channels, Time]
    env_l_full = trial['env_l']
    env_r_full = trial['env_r']
    switch_points = trial['meta']['switch_points']
    
    window_len = 64 # 1.0 seconds
    hop_len = 8     # 125 ms
    
    X_eeg = []
    X_att = []
    X_unatt = []
    
    for start in range(0, eeg_full.shape[1] - window_len + 1, hop_len):
        end = start + window_len
        
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
        
    X_eeg = torch.stack(X_eeg).float() # [B, 8, 64]
    X_att = torch.stack(X_att).float() # [B, 64]
    X_unatt = torch.stack(X_unatt).float() # [B, 64]
    
    print(f"Generated {len(X_eeg)} overlapping windows.")
    
    # 3. Model Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    model = ContrastiveMatchNet().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    
    X_eeg = X_eeg.to(device)
    X_att = X_att.to(device)
    X_unatt = X_unatt.to(device)
    
    epochs = 200
    batch_size = 64 # Use full batch if possible, or mini-batches
    num_samples = len(X_eeg)
    
    print("\n--- 3. Training Single Trial Overfit ---")
    
    for epoch in range(1, epochs + 1):
        model.train()
        
        # Shuffle indices
        indices = torch.randperm(num_samples)
        
        total_loss = 0
        total_info = 0
        total_marg = 0
        
        for i in range(0, num_samples, batch_size):
            idx = indices[i:i+batch_size]
            b_eeg = X_eeg[idx]
            b_att = X_att[idx]
            b_unatt = X_unatt[idx]
            
            optimizer.zero_grad()
            z_e, z_a, z_b = model(b_eeg, b_att, b_unatt)
            
            loss, l_info, l_marg = contrastive_loss(z_e, z_a, z_b)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * len(idx)
            total_info += l_info.item() * len(idx)
            total_marg += l_marg.item() * len(idx)
            
        if epoch % 10 == 0 or epoch == 1:
            model.eval()
            with torch.no_grad():
                z_e, z_a, z_b = model(X_eeg, X_att, X_unatt)
                sim_att = (z_e * z_a).sum(dim=-1).cpu().numpy()
                sim_unatt = (z_e * z_b).sum(dim=-1).cpu().numpy()
                
                margin = sim_att - sim_unatt
                acc = np.mean(margin > 0)
                
                # AUROC: we want sim(att) to be higher than sim(unatt).
                # We can formulate this as a binary classification problem:
                # 1 if positive pair (sim_att), 0 if negative pair (sim_unatt)
                y_true = np.concatenate([np.ones(len(sim_att)), np.zeros(len(sim_unatt))])
                y_scores = np.concatenate([sim_att, sim_unatt])
                auroc = roc_auc_score(y_true, y_scores)
                
                print(f"Epoch {epoch:3d} | Loss: {total_loss/num_samples:.4f} "
                      f"(Info: {total_info/num_samples:.4f}, Marg: {total_marg/num_samples:.4f}) | "
                      f"Sim(Att): {sim_att.mean():.4f} | Sim(Unatt): {sim_unatt.mean():.4f} | "
                      f"Acc: {acc*100:.1f}% | AUROC: {auroc:.4f}")
                      
    print("\n--- 4. Final Evaluation ---")
    print(f"Final Margin Mean: {margin.mean():.4f}")
    print(f"Final Margin Std: {margin.std():.4f}")
    print("If Accuracy is 100% and AUROC is 1.0, the model successfully overfit and the contrastive logic is bug-free.")

if __name__ == "__main__":
    run_single_trial_overfit()
