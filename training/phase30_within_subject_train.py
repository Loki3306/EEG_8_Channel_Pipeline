import os
import sys
import numpy as np
import scipy.io
import scipy.signal
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, precision_recall_curve, auc, confusion_matrix
import glob
from pathlib import Path
import time
import random
import argparse
import matplotlib.pyplot as plt
import pandas as pd

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.aad_conformer import AADConformer
from models.pearson_aad import NegativePearsonLoss

def pearson_corr(x, y, dim=-1):
    x_centered = x - x.mean(dim=dim, keepdim=True)
    y_centered = y - y.mean(dim=dim, keepdim=True)
    cov = (x_centered * y_centered).sum(dim=dim)
    var_x = (x_centered ** 2).sum(dim=dim)
    var_y = (y_centered ** 2).sum(dim=dim)
    return cov / torch.sqrt(var_x * var_y + 1e-8)

class TrialDataset(Dataset):
    def __init__(self, trials, window_len=128, hop_len=64, censor_margin=256):
        self.windows = []
        for trial_idx, trial in enumerate(trials):
            eeg = trial['eeg']
            att = trial['att'].unsqueeze(0)
            unatt = trial['unatt'].unsqueeze(0)
            switch_points = trial.get('meta', {}).get('switch_points', [])
            
            for start in range(0, eeg.shape[1] - window_len + 1, hop_len):
                end = start + window_len
                
                is_censored = False
                for state, s_idx in switch_points:
                    if s_idx > 0:
                        if (start < s_idx + censor_margin) and (end > s_idx):
                            is_censored = True
                            break
                            
                if is_censored:
                    continue
                    
                w_eeg = eeg[:, start:end].clone()
                w_att = att[:, start:end].clone()
                w_unatt = unatt[:, start:end].clone()
                
                # Restore window-level normalization to match checkpoint distribution
                w_eeg_mean = w_eeg.mean(dim=1, keepdim=True)
                w_eeg_std = w_eeg.std(dim=1, keepdim=True) + 1e-8
                w_eeg = (w_eeg - w_eeg_mean) / w_eeg_std
                
                w_att_mean = w_att.mean(dim=1, keepdim=True)
                w_att_std = w_att.std(dim=1, keepdim=True) + 1e-8
                w_att = (w_att - w_att_mean) / w_att_std
                
                w_unatt_mean = w_unatt.mean(dim=1, keepdim=True)
                w_unatt_std = w_unatt.std(dim=1, keepdim=True) + 1e-8
                w_unatt = (w_unatt - w_unatt_mean) / w_unatt_std
                
                self.windows.append({
                    'eeg': w_eeg,
                    'att': w_att,
                    'unatt': w_unatt,
                    'trial_idx': trial_idx,
                    'start': start,
                    'end': end
                })
                
    def __len__(self):
        return len(self.windows)
        
    def __getitem__(self, idx):
        return self.windows[idx]

def load_aasd_subject_trials(mat_path, b, a, sel_idx, audio_dir):
    mat = scipy.io.loadmat(mat_path, squeeze_me=True, struct_as_record=False)
    eeg_var = [k for k in mat.keys() if not k.startswith('__')][0]
    data_all, events = mat[eeg_var].data, mat[eeg_var].event
    
    trials = []
    for epoch_idx in range(1, 61):
        audio_marker_val = None
        for ev in events:
            if len(ev) >= 5:
                t_str, epoch_val = str(ev[0]).strip(), str(ev[4]).strip()
                if epoch_val == str(epoch_idx) and t_str.isdigit() and 11 <= int(t_str) <= 70:
                    audio_marker_val = int(t_str)
                    break
        if audio_marker_val is None: continue
            
        npz_path = os.path.join(audio_dir, f"{audio_marker_val}.npz")
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
        
        # Revert back to CAR (Common Average Reference across channels) as used in Phase 29
        trial_eeg = trial_eeg - trial_eeg.mean(axis=0, keepdims=True)
        trial_eeg_filt = scipy.signal.filtfilt(b, a, trial_eeg, axis=1)
        trial_eeg_8 = scipy.signal.resample_poly(trial_eeg_filt, 1, 2, axis=1)[sel_idx, 4:]
        
        audio_data = np.load(npz_path)
        env_l, env_r = audio_data['env_l'][:-4], audio_data['env_r'][:-4]
        
        min_len = min(trial_eeg_8.shape[1], len(env_l))
        trial_eeg_8 = trial_eeg_8[:, :min_len]
        env_l, env_r = env_l[:min_len], env_r[:min_len]
        
        att, unatt = np.zeros_like(env_l), np.zeros_like(env_r)
        if len(switch_points) == 0: switch_points = [('R', 0)]
        
        current_state = switch_points[0][0]
        prev_idx = 0
        for state, idx_64 in switch_points:
            if idx_64 > prev_idx:
                if current_state == 'R':
                    att[prev_idx:idx_64], unatt[prev_idx:idx_64] = env_r[prev_idx:idx_64], env_l[prev_idx:idx_64]
                else:
                    att[prev_idx:idx_64], unatt[prev_idx:idx_64] = env_l[prev_idx:idx_64], env_r[prev_idx:idx_64]
            prev_idx, current_state = idx_64, state
            
        if current_state == 'R':
            att[prev_idx:], unatt[prev_idx:] = env_r[prev_idx:], env_l[prev_idx:]
        else:
            att[prev_idx:], unatt[prev_idx:] = env_l[prev_idx:], env_r[prev_idx:]
            
        trials.append({
            'meta': {'switch_points': switch_points},
            'eeg': torch.FloatTensor(trial_eeg_8),
            'env_l': torch.FloatTensor(env_l),
            'env_r': torch.FloatTensor(env_r),
            'att': torch.FloatTensor(att),
            'unatt': torch.FloatTensor(unatt)
        })
    return trials

def evaluate_model(model, loader, device):
    model.eval()
    all_margins = []
    all_att_corr = []
    all_unatt_corr = []
    
    trial_preds = {}
    
    with torch.no_grad():
        for batch in loader:
            eeg = batch['eeg'].to(device)
            att = batch['att'].squeeze(1).to(device)
            unatt = batch['unatt'].squeeze(1).to(device)
            trial_idx = batch['trial_idx'].numpy()
            
            env_pred = model(eeg) # [B, T]
            
            sim_att = pearson_corr(env_pred, att, dim=-1).cpu().numpy()
            sim_unatt = pearson_corr(env_pred, unatt, dim=-1).cpu().numpy()
            margin = sim_att - sim_unatt
            
            all_margins.extend(margin)
            all_att_corr.extend(sim_att)
            all_unatt_corr.extend(sim_unatt)
            
            for i in range(len(margin)):
                t_idx = trial_idx[i]
                if t_idx not in trial_preds:
                    trial_preds[t_idx] = []
                trial_preds[t_idx].append(1 if margin[i] > 0 else 0)
                
    # Window-level metrics
    y_true = [1] * len(all_margins) + [0] * len(all_margins)
    y_scores = list(all_margins) + [-m for m in all_margins]
    
    auc_val = roc_auc_score(y_true, y_scores)
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    auprc_val = auc(recall, precision)
    
    y_pred = [1 if s > 0 else 0 for s in y_scores]
    bacc = balanced_accuracy_score(y_true, y_pred)
    window_acc = np.mean([1 if m > 0 else 0 for m in all_margins])
    
    cm = confusion_matrix(y_true, y_pred)
    
    # Trial-level accuracy (majority vote)
    trial_accs = []
    for t_idx, preds in trial_preds.items():
        trial_accs.append(1 if np.mean(preds) > 0.5 else 0)
    trial_acc = np.mean(trial_accs) if trial_accs else 0.0
    
    return {
        'mean_att': np.mean(all_att_corr),
        'mean_unatt': np.mean(all_unatt_corr),
        'mean_margin': np.mean(all_margins),
        'auc': auc_val,
        'auprc': auprc_val,
        'bacc': bacc,
        'window_acc': window_acc,
        'trial_acc': trial_acc,
        'margins': all_margins,
        'cm': cm
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type=str, default="S18", help="Target subject")
    parser.add_argument("--checkpoint", type=str, default="", help="Path to KUL pretrained checkpoint")
    parser.add_argument("--censor_margin", type=int, default=256, help="Samples to censor after switch (256=4s)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--epochs", type=int, default=30, help="Training epochs")
    parser.add_argument("--out_dir", type=str, default="results/phase30")
    args = parser.parse_args()

    seed_everything(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    
    mat_path = glob.glob(f'/kaggle/input/datasets/lokeshgile/aasd-processed-eeg/Processed EEG/*/{args.subject}.mat')
    if not mat_path:
        mat_path = glob.glob(f'data/*/{args.subject}.mat')
    if not mat_path:
        print(f"ERROR: Could not find {args.subject}.mat.")
        return
        
    mat_path = mat_path[0]
    audio_dir = '/kaggle/input/datasets/lokeshgile/aasd-audio-gammatones'
    if not os.path.exists(audio_dir):
        audio_dir = 'data/audio_features'
        
    fs_eeg = 256
    nyq = 0.5 * fs_eeg
    b, a = scipy.signal.butter(4, [1.0/nyq, 8.0/nyq], btype='band')
    target_channels = ['T7', 'C2', 'FT8', 'P7', 'CPz', 'Fp1', 'TP8', 'C3']
    channel_names = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'FC5', 'FC1', 'FC2', 'FC6', 'T7', 'C3', 'Cz', 'C4', 'T8', 'CP5', 'CP1', 'CP2', 'CP6', 'P7', 'P3', 'Pz', 'P4', 'P8', 'PO9', 'O1', 'Oz', 'O2', 'PO10', 'AF7', 'AF3', 'AF4', 'AF8', 'F5', 'F1', 'F2', 'F6', 'FT7', 'FC3', 'FC4', 'FT8', 'C5', 'C1', 'C2', 'C6', 'TP7', 'CP3', 'CPz', 'CP4', 'TP8', 'P5', 'P1', 'P2', 'P6', 'PO7', 'PO3', 'POz', 'PO4', 'PO8', 'O9', 'O10', 'Iz', 'Cz']
    sel_idx = [channel_names.index(tc) for tc in target_channels]
    
    trials = load_aasd_subject_trials(mat_path, b, a, sel_idx, audio_dir)
    
    if len(trials) == 0:
        print("ERROR: No trials loaded.")
        return
        
    # Trial-level Chronological Split (80/20)
    split_idx = int(len(trials) * 0.8)
    train_trials = trials[:split_idx]
    test_trials = trials[split_idx:]
    
    train_ds = TrialDataset(train_trials, censor_margin=args.censor_margin)
    test_ds = TrialDataset(test_trials, censor_margin=args.censor_margin)
    
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = AADConformer(in_channels=8).to(device)
    if args.checkpoint and os.path.exists(args.checkpoint):
        state_dict = torch.load(args.checkpoint, map_location=device)
        
        # Diagnostic Checkpoint Loading
        model_dict = model.state_dict()
        loaded_keys = []
        skipped_keys = []
        for k, v in state_dict.items():
            if k in model_dict:
                if v.shape == model_dict[k].shape:
                    model_dict[k] = v
                    loaded_keys.append(k)
                else:
                    skipped_keys.append((k, v.shape, model_dict[k].shape))
        
        model.load_state_dict(model_dict)
        print(f"Loaded checkpoint: {args.checkpoint}")
        print(f"[DIAGNOSTIC] Successfully loaded {len(loaded_keys)} keys.")
        if skipped_keys:
            print(f"[DIAGNOSTIC] WARNING! Skipped {len(skipped_keys)} keys due to shape mismatch:")
            for k, s1, s2 in skipped_keys:
                print(f"  --> {k}: Checkpoint shape {s1} != Model shape {s2}")
        
    criterion = NegativePearsonLoss().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    
    history = {'train_auc': [], 'test_auc': [], 'train_pearson': [], 'test_pearson': []}
    
    best_test_auc = 0.0
    best_metrics = None
    best_epoch = 0
    
    # MINI OVERFIT TEST DIAGNOSTIC
    print("\n[DIAGNOSTIC] RUNNING SINGLE-BATCH OVERFIT TEST (50 Epochs)")
    overfit_batch = next(iter(train_loader))
    
    for epoch in range(1, 51):
        model.train()
        eeg = overfit_batch['eeg'].to(device)
        att = overfit_batch['att'].squeeze(1).to(device)
        
        optimizer.zero_grad()
        env_pred = model(eeg)
        
        # Variance Diagnostic
        pred_var = env_pred.var(dim=-1).mean().item()
        
        loss = criterion(env_pred, att)
        loss.backward()
        
        # Gradient Diagnostic
        spatial_grad = model.spatial_conv.weight.grad.norm().item() if model.spatial_conv.weight.grad is not None else 0.0
        head_grad = model.head.weight.grad.norm().item() if model.head.weight.grad is not None else 0.0
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        if epoch % 10 == 0 or epoch == 1:
            print(f"Overfit Epoch {epoch:02d} | Loss: {loss.item():.4f} | Pred Var: {pred_var:.6f} | Spatial Grad: {spatial_grad:.4f} | Head Grad: {head_grad:.4f}")
            
    print("[DIAGNOSTIC] Overfit test complete. Exiting diagnostic mode.\n")
    sys.exit(0)
    
    for epoch in range(1, args.epochs + 1):
        model.train()
        for batch in train_loader:
            eeg = batch['eeg'].to(device)
            att = batch['att'].squeeze(1).to(device)
            
            optimizer.zero_grad()
            env_pred = model(eeg)
            loss = criterion(env_pred, att)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
        train_metrics = evaluate_model(model, train_loader, device)
        test_metrics = evaluate_model(model, test_loader, device)
        
        history['train_auc'].append(train_metrics['auc'])
        history['test_auc'].append(test_metrics['auc'])
        history['train_pearson'].append(train_metrics['mean_att'])
        history['test_pearson'].append(test_metrics['mean_att'])
        
        if test_metrics['auc'] >= best_test_auc:
            best_test_auc = test_metrics['auc']
            best_metrics = {'train': train_metrics, 'test': test_metrics}
            best_epoch = epoch
            
    # Produce Plots
    plt.figure()
    plt.plot(history['train_pearson'], label='Train Pearson')
    plt.plot(history['test_pearson'], label='Test Pearson')
    plt.legend()
    plt.savefig(f"{args.out_dir}/pearson_curve.png")
    plt.close()
    
    plt.figure()
    plt.plot(history['train_auc'], label='Train AUROC')
    plt.plot(history['test_auc'], label='Test AUROC')
    plt.legend()
    plt.savefig(f"{args.out_dir}/auroc_curve.png")
    plt.close()
    
    plt.figure()
    plt.bar(['Train Att', 'Train Unatt', 'Test Att', 'Test Unatt'], 
            [best_metrics['train']['mean_att'], best_metrics['train']['mean_unatt'], 
             best_metrics['test']['mean_att'], best_metrics['test']['mean_unatt']])
    plt.savefig(f"{args.out_dir}/correlations.png")
    plt.close()
    
    plt.figure()
    m = np.array(best_metrics['test']['margins'])
    plt.hist(m[m>0], bins=30, alpha=0.5, label='Correct (Margin>0)')
    plt.hist(m[m<=0], bins=30, alpha=0.5, label='Incorrect (Margin<=0)')
    plt.legend()
    plt.savefig(f"{args.out_dir}/margin_dist.png")
    plt.close()

    # Save Confusion Matrix & Metrics
    pd.DataFrame(best_metrics['test']['cm']).to_csv(f"{args.out_dir}/confusion_matrix.csv", index=False)
    
    metrics_df = pd.DataFrame([{
        'Train Pearson': best_metrics['train']['mean_att'],
        'Test Pearson': best_metrics['test']['mean_att'],
        'Train AUROC': best_metrics['train']['auc'],
        'Test AUROC': best_metrics['test']['auc'],
        'Train BAcc': best_metrics['train']['bacc'],
        'Test BAcc': best_metrics['test']['bacc'],
        'Train AUPRC': best_metrics['train']['auprc'],
        'Test AUPRC': best_metrics['test']['auprc']
    }])
    metrics_df.to_csv(f"{args.out_dir}/test_metrics.csv", index=False)

    # Determine Conclusion
    overfitting = "YES" if (best_metrics['train']['auc'] - best_metrics['test']['auc']) > 0.1 else "NO"
    learning = "YES" if best_metrics['train']['auc'] > 0.55 else "NO"
    validated = "YES" if (best_metrics['test']['auc'] > 0.55 and overfitting == "NO") else "NO"

    # Stdout block
    print("==================================================")
    print("PHASE 30 RESULTS")
    print("==================================================")
    print(f"Subject             {args.subject}")
    print(f"Train Trials        {len(train_trials)}")
    print(f"Test Trials         {len(test_trials)}")
    print(f"Training Windows    {len(train_ds)}")
    print(f"Testing Windows     {len(test_ds)}")
    print(f"Best Epoch          {best_epoch}")
    print(f"Train Pearson       {best_metrics['train']['mean_att']:.4f}")
    print(f"Test Pearson        {best_metrics['test']['mean_att']:.4f}")
    print(f"Train AUROC         {best_metrics['train']['auc']:.4f}")
    print(f"Test AUROC          {best_metrics['test']['auc']:.4f}")
    print(f"Train Accuracy      {best_metrics['train']['trial_acc']:.4f}")
    print(f"Test Accuracy       {best_metrics['test']['trial_acc']:.4f}")
    print(f"Attended Corr       {best_metrics['test']['mean_att']:.4f}")
    print(f"Unattended Corr     {best_metrics['test']['mean_unatt']:.4f}")
    print(f"Generalization Gap  {best_metrics['train']['auc'] - best_metrics['test']['auc']:.4f}")
    print("==================================================")
    print("Conclusion:")
    print(f"Learning:           {learning}")
    print(f"Overfitting:        {overfitting}")
    print(f"Pipeline validated: {validated}")
    print("==================================================")

if __name__ == "__main__":
    main()
