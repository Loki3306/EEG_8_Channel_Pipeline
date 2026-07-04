import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
from sklearn.metrics import roc_auc_score
from pathlib import Path
import gc
import random

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.phase35_neural_ridge import pearson_loss

def safe_pearson(x, y):
    if np.var(x) < 1e-8 or np.var(y) < 1e-8:
        return 0.0
    return np.corrcoef(x, y)[0, 1]

class ShallowNeuralRidge(nn.Module):
    def __init__(self, channels=8, temporal_filters=32, spatial_filters=64, max_lag=24):
        super().__init__()
        self.max_lag = max_lag
        self.selected_channels = [0, 2, 5, 13, 23, 31, 41, 49]
        
        self.temporal_conv = nn.Conv2d(1, temporal_filters, kernel_size=(1, 33), padding=(0, 16), bias=False)
        self.temporal_norm = nn.BatchNorm2d(temporal_filters)
        
        self.spatial_conv = nn.Conv2d(temporal_filters, spatial_filters, kernel_size=(channels, 1), 
                                      groups=temporal_filters, bias=False)
        self.spatial_norm = nn.BatchNorm2d(spatial_filters)
        self.stem_act = nn.SiLU()
        self.stem_dropout = nn.Dropout(0.3)
        
        self.ridge_decoder = nn.Conv1d(
            in_channels=spatial_filters,
            out_channels=1,
            kernel_size=max_lag + 1,
            padding=max_lag,
            bias=False
        )

    def forward(self, x):
        x_8ch = x[:, self.selected_channels, :]
        x_8ch = x_8ch.unsqueeze(1)
        
        x_t = self.temporal_conv(x_8ch)
        x_t = self.temporal_norm(x_t)
        
        x_s = self.spatial_conv(x_t)
        x_s = self.spatial_norm(x_s)
        x_s = self.stem_act(x_s)
        x_s = self.stem_dropout(x_s)
        
        x_s = x_s.squeeze(2)
        out_padded = self.ridge_decoder(x_s)
        
        if self.max_lag > 0:
            out = out_padded[:, :, :-self.max_lag]
        else:
            out = out_padded
            
        return out.squeeze(1)

def run_test_suite(model, test_trials, device, max_lag, hop_len, window_len):
    sim_att, sim_unatt = [], []
    for trial in test_trials:
        eeg_full = trial['eeg'].numpy()
        env_l = trial['env_l'].numpy()
        env_r = trial['env_r'].numpy()
        switch_points = trial['meta']['switch_points']
        
        eeg = (eeg_full - eeg_full.mean(axis=1, keepdims=True)) / (eeg_full.std(axis=1, keepdims=True) + 1e-8)
        env_l = (env_l - env_l.mean()) / (env_l.std() + 1e-8)
        env_r = (env_r - env_r.mean()) / (env_r.std() + 1e-8)
        
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
            
        eeg_tensor = torch.from_numpy(eeg).float().unsqueeze(0).to(device)
        with torch.no_grad():
            pred = model(eeg_tensor).squeeze(0).cpu().numpy()
            
        num_windows = (eeg.shape[1] - window_len) // hop_len + 1
        if num_windows <= 0: continue
        
        for i in range(num_windows):
            start = i * hop_len
            end = start + window_len
            
            pred_w = pred[start:end]
            att_w = att[start:end]
            unatt_w = unatt[start:end]
            
            var_pred = np.var(pred_w)
            if var_pred > 1e-8 and np.var(att_w) > 1e-8:
                sim_att.append(safe_pearson(pred_w, att_w))
            else:
                sim_att.append(0.0)
                
            if var_pred > 1e-8 and np.var(unatt_w) > 1e-8:
                sim_unatt.append(safe_pearson(pred_w, unatt_w))
            else:
                sim_unatt.append(0.0)
                
    if not sim_att: return 0.5
    y_true = np.concatenate([np.ones(len(sim_att)), np.zeros(len(sim_unatt))])
    y_scores = np.concatenate([sim_att, sim_unatt])
    return roc_auc_score(y_true, y_scores)

def collate_fn(batch):
    max_len = max(x.size(1) for x, y in batch)
    x_padded = torch.zeros(len(batch), batch[0][0].size(0), max_len)
    y_padded = torch.zeros(len(batch), max_len)
    for i, (x, y) in enumerate(batch):
        x_padded[i, :, :x.size(1)] = x
        y_padded[i, :y.size(0)] = y
    return x_padded, y_padded

def main():
    print("=======================================================")
    print(" PHASE 44: PURE WITHIN-SUBJECT BENCHMARK               ")
    print("=======================================================")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    cache_dir = Path('/kaggle/working/eeg_cache')
    if not cache_dir.exists():
        print("ERROR: Cache directory not found! Run Phase 41 first to generate cache.")
        return
        
    subject_ids = []
    for pt_file in cache_dir.glob("*_processed.pt"):
        subject_ids.append(pt_file.name.split('_')[0])
    subject_ids.sort()
    
    max_lag = 24
    window_len = 64 * 5
    hop_len = 64 * 1
    
    print(f"Found {len(subject_ids)} subjects. Establishing Information Ceiling...\n")
    
    all_auroc = []
    
    for subject_id in subject_ids:
        print(f"==========================================")
        print(f" SUBJECT: {subject_id}")
        
        cached = torch.load(cache_dir / f"{subject_id}_processed.pt", weights_only=False)
        all_X = cached['X']
        all_Y = cached['Y']
        all_raw = cached['raw']
        
        # Shuffle trials deterministically for reproducibility
        indices = list(range(len(all_X)))
        random.seed(42)
        random.shuffle(indices)
        
        # 75% Train, 25% Test split (45 train, 15 test for 60 trials)
        split_idx = int(len(indices) * 0.75)
        train_indices = indices[:split_idx]
        test_indices = indices[split_idx:]
        
        train_data = [(all_X[i], all_Y[i]) for i in train_indices]
        test_raw = [all_raw[i] for i in test_indices]
        
        train_loader = DataLoader(train_data, batch_size=4, shuffle=True, collate_fn=collate_fn)
        
        model = ShallowNeuralRidge(channels=8, temporal_filters=32, spatial_filters=64, max_lag=max_lag).to(device)
        optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
        
        epochs = 30
        print(f"  -> Training from scratch on {len(train_data)} trials for {epochs} epochs...")
        model.train()
        for epoch in range(1, epochs + 1):
            epoch_loss = 0.0
            for bx, by in train_loader:
                bx, by = bx.to(device), by.to(device)
                optimizer.zero_grad()
                pred = model(bx)
                loss = pearson_loss(pred, by)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                epoch_loss += loss.item()
            if epoch % 10 == 0 or epoch == epochs:
                print(f"     Epoch {epoch:02d} | Loss: {epoch_loss/len(train_loader):.4f}")
                
        model.eval()
        auroc = run_test_suite(model, test_raw, device, max_lag, hop_len, window_len)
        all_auroc.append(auroc)
        print(f"  -> Within-Subject AUROC: {auroc:.4f}\n")
        
        del cached, all_X, all_Y, all_raw, train_data, train_loader, test_raw, model, optimizer
        gc.collect()
        
    print("=======================================================")
    print(" SUMMARY: PHASE 44 WITHIN-SUBJECT (INFORMATION CEILING)")
    print("=======================================================")
    print(f"{'Subject':<10} | {'AUROC':<12}")
    print("-" * 25)
    for sid, auroc in zip(subject_ids, all_auroc):
        print(f"{sid:<10} | {auroc:.4f}")
    print("-" * 25)
    print(f"MEAN       : {np.mean(all_auroc):.4f}")
    print(f"STD        : {np.std(all_auroc):.4f}")
    
if __name__ == '__main__':
    main()
