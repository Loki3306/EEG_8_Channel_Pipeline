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
from models.aad_conformer import AADConformer

def safe_corr_torch(x, y, eps=1e-8):
    """Batched Pearson correlation in PyTorch. x, y: (Batch, Time)"""
    x_mean = x.mean(dim=-1, keepdim=True)
    y_mean = y.mean(dim=-1, keepdim=True)
    x_centered = x - x_mean
    y_centered = y - y_mean
    
    cov = (x_centered * y_centered).sum(dim=-1)
    x_var = (x_centered ** 2).sum(dim=-1)
    y_var = (y_centered ** 2).sum(dim=-1)
    
    corr = cov / (torch.sqrt(x_var * y_var) + eps)
    return corr

def pearson_loss(pred, target):
    corr = safe_corr_torch(pred, target)
    return 1.0 - corr.mean()

def load_pretrained_conformer(checkpoint_path, in_channels=62, device='cpu'):
    # Initialize the model with 62 channels
    model = AADConformer(in_channels=in_channels, temporal_filters=32, spatial_filters=64, embed_dim=64, num_heads=4, num_layers=2).to(device)
    
    if not os.path.exists(checkpoint_path):
        print(f"WARNING: Checkpoint {checkpoint_path} not found. Training from scratch.")
        return model
        
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
    
    model_dict = model.state_dict()
    pretrained_dict = {}
    
    for k, v in state_dict.items():
        if k in model_dict:
            if model_dict[k].shape == v.shape:
                pretrained_dict[k] = v
            else:
                print(f"Skipping {k} due to shape mismatch: {v.shape} vs {model_dict[k].shape}")
        else:
            print(f"Skipping {k} (not in current model)")
            
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    print(f"Loaded {len(pretrained_dict)}/{len(model_dict)} matching layers from KUL checkpoint.")
    
    # Freeze all layers EXCEPT the spatial_conv (which maps the 62 channels) and the final head
    for name, param in model.named_parameters():
        if 'spatial_conv' not in name and 'head' not in name:
            param.requires_grad = False
        else:
            print(f"Fine-tuning layer: {name}")
            
    return model

def train_transfer_learning():
    print("--- 1. Loading AASD Dataset ---")
    data_root = '/kaggle/input/datasets/lokeshgile/aasd-processed-eeg/Processed EEG'
    mat_files = []
    for root, dirs, files in os.walk(data_root):
        for file in files:
            if file.endswith('.mat') and not file.startswith('._'):
                mat_files.append(os.path.join(root, file))
    
    if not mat_files:
        print("ERROR: No .mat files found. Please run on Kaggle.")
        return
        
    sub_path = next((p for p in mat_files if 'S18' in p), mat_files[0])
    
    b, a = scipy.signal.butter(4, [1.0/64.0, 8.0/64.0], btype='band')
    audio_dir = '/kaggle/input/datasets/lokeshgile/aasd-audio-gammatones'
    wav_dir = '/kaggle/input/datasets/lokeshgile/aasd-audio/Stimuli Audio'
    
    trials = load_aasd_subject_trials(sub_path, b, a, audio_dir, wav_dir)
    print(f"Loaded {len(trials)} trials from {os.path.basename(sub_path)}")
    
    train_trials = trials[:40]
    test_trials = trials[40:]
    
    window_len = 64 * 5
    hop_len = 64 * 1
    
    print("\n--- 2. Building Training Windows ---")
    X_train, Y_train = [], []
    for trial in train_trials:
        eeg = trial['eeg']
        env_l = trial['env_l']
        env_r = trial['env_r']
        switch_points = trial['meta']['switch_points']
        
        att = torch.zeros_like(env_l)
        if len(switch_points) == 0:
            switch_points = [('R', 0)]
            
        if switch_points[0][1] > 0:
            initial_state = 'R' if switch_points[0][0] == 'L' else 'L'
        else:
            initial_state = switch_points[0][0]
            
        current_state = initial_state
        prev_idx = 0
        for state, idx_64 in switch_points:
            if idx_64 > prev_idx:
                if current_state == 'L':
                    att[prev_idx:idx_64] = env_l[prev_idx:idx_64]
                else:
                    att[prev_idx:idx_64] = env_r[prev_idx:idx_64]
            prev_idx, current_state = idx_64, state
            
        if current_state == 'L':
            att[prev_idx:] = env_l[prev_idx:]
        else:
            att[prev_idx:] = env_r[prev_idx:]
            
        for start in range(0, eeg.shape[1] - window_len + 1, hop_len):
            end = start + window_len
            
            is_transition = False
            for _, sw_idx in switch_points:
                if start < sw_idx < end:
                    is_transition = True
                    break
            
            if not is_transition:
                eeg_w = eeg[:, start:end].clone()
                eeg_w = (eeg_w - eeg_w.mean(dim=1, keepdim=True)) / (eeg_w.std(dim=1, keepdim=True) + 1e-8)
                
                att_w = att[start:end].clone()
                att_w = (att_w - att_w.mean()) / (att_w.std() + 1e-8)
                
                X_train.append(eeg_w)
                Y_train.append(att_w)

    X_train = torch.stack(X_train)
    Y_train = torch.stack(Y_train)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_eeg_channels = X_train.size(1)
    print(f"Device: {device} | Channels: {num_eeg_channels}")
    
    dataset = TensorDataset(X_train, Y_train)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True)
    
    # Path to the uploaded KUL checkpoint on Kaggle
    kul_checkpoint_path = '/kaggle/input/datasets/lowkieee/eeg-aad-conformer-seed1-checkpoints/checkpoints/seed_123/model_S1.pt'
    
    model = load_pretrained_conformer(kul_checkpoint_path, in_channels=num_eeg_channels, device=device)
    
    # Only optimize the un-frozen parameters
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=5e-4, weight_decay=1e-3)
    
    print("\n--- 3. Fine-tuning AADConformer ---")
    start_time = time.time()
    model.train()
    for epoch in range(1, 41):
        total_loss = 0.0
        for batch_eeg, batch_att in dataloader:
            # AADConformer expects [B, C, T]
            batch_eeg = batch_eeg.to(device)
            batch_att = batch_att.to(device)
            
            optimizer.zero_grad()
            pred = model(batch_eeg)
            loss = pearson_loss(pred.squeeze(1), batch_att)
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d} | Train Pearson Loss: {total_loss / len(dataloader):.4f}")
            
    print(f"Fine-tuning completed in {time.time() - start_time:.1f}s")
    
    print("\n--- 4. Testing ---")
    model.eval()
    sim_att, sim_unatt = [], []
    
    with torch.no_grad():
        for trial in test_trials:
            eeg = trial['eeg']
            env_l = trial['env_l']
            env_r = trial['env_r']
            switch_points = trial['meta']['switch_points']
            
            att = torch.zeros_like(env_l)
            unatt = torch.zeros_like(env_l)
            
            if len(switch_points) == 0:
                switch_points = [('R', 0)]
                
            if switch_points[0][1] > 0:
                initial_state = 'R' if switch_points[0][0] == 'L' else 'L'
            else:
                initial_state = switch_points[0][0]
                
            current_state = initial_state
            prev_idx = 0
            for state, idx_64 in switch_points:
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
                
            for start in range(0, eeg.shape[1] - window_len + 1, hop_len):
                end = start + window_len
                
                eeg_w = eeg[:, start:end].clone()
                eeg_w = (eeg_w - eeg_w.mean(dim=1, keepdim=True)) / (eeg_w.std(dim=1, keepdim=True) + 1e-8)
                
                att_w = att[start:end].clone()
                unatt_w = unatt[start:end].clone()
                
                # Predict (expecting [B, C, T])
                pred_w = model(eeg_w.unsqueeze(0).to(device)).squeeze().cpu()
                
                # Pearson corr
                corr_att = np.corrcoef(pred_w.numpy(), att_w.numpy())[0, 1]
                corr_unatt = np.corrcoef(pred_w.numpy(), unatt_w.numpy())[0, 1]
                
                if not np.isnan(corr_att) and not np.isnan(corr_unatt):
                    sim_att.append(corr_att)
                    sim_unatt.append(corr_unatt)
                    
    sim_att = np.array(sim_att)
    sim_unatt = np.array(sim_unatt)
    
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
    train_transfer_learning()
