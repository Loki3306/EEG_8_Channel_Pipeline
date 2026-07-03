import os
import sys
import numpy as np
import scipy.io
import scipy.signal
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
import glob
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.aad_conformer import AADConformer

def cross_corr(x, y):
    if len(x) == 0 or len(y) == 0: return 0.0
    x = (x - np.mean(x)) / (np.std(x) + 1e-8)
    y = (y - np.mean(y)) / (np.std(y) + 1e-8)
    c = np.corrcoef(x, y)[0, 1]
    return c if not np.isnan(c) else 0.0

def pearson_loss(pred, target):
    pred = pred - pred.mean(dim=-1, keepdim=True)
    target = target - target.mean(dim=-1, keepdim=True)
    cov = (pred * target).sum(dim=-1)
    var_pred = (pred ** 2).sum(dim=-1)
    var_target = (target ** 2).sum(dim=-1)
    corr = cov / torch.sqrt(var_pred * var_target + 1e-8)
    return 1.0 - corr.mean()

class WindowedDataset(Dataset):
    def __init__(self, trials, window_len=128, hop_len=64):
        self.windows = []
        for trial in trials:
            eeg = trial['eeg']
            att = trial['att'][0]
            unatt = trial['unatt'][0]
            
            for start in range(0, eeg.shape[1] - window_len + 1, hop_len):
                w_eeg = eeg[:, start:start+window_len]
                w_att = att[start:start+window_len]
                w_unatt = unatt[start:start+window_len]
                self.windows.append((w_eeg, w_att, w_unatt))
                
    def __len__(self):
        return len(self.windows)
        
    def __getitem__(self, idx):
        return self.windows[idx]

def compute_metrics(preds, env_l_list, env_r_list, true_att_states):
    if len(preds) == 0:
        return 0, 0, 0, 0, 0, 0
        
    corrs_att = []
    corrs_unatt = []
    scores = []
    labels = []
    
    for p, el, er, state in zip(preds, env_l_list, env_r_list, true_att_states):
        cl = cross_corr(p, el)
        cr = cross_corr(p, er)
        
        if state == 'L':
            corrs_att.append(cl)
            corrs_unatt.append(cr)
            scores.append(cl - cr)
            labels.append(1)
        else:
            corrs_att.append(cr)
            corrs_unatt.append(cl)
            scores.append(cl - cr)
            labels.append(0)
            
    mean_att = np.mean(corrs_att)
    mean_unatt = np.mean(corrs_unatt)
    
    preds_class = [1 if s > 0 else 0 for s in scores]
    
    if len(np.unique(labels)) > 1:
        auc = roc_auc_score(labels, scores)
        bacc = balanced_accuracy_score(labels, preds_class)
    else:
        auc = np.nan
        bacc = np.nan
        
    # Concatenated (60s equivalent) correlation
    if len(preds) > 1:
        cat_p = np.concatenate(preds)
        cat_el = np.concatenate(env_l_list)
        cat_er = np.concatenate(env_r_list)
        
        cat_cl = cross_corr(cat_p, cat_el)
        cat_cr = cross_corr(cat_p, cat_er)
        
        # We can't do single state if there are multiple states in the concatenated array
        # Instead, build the exact concatenated true att and unatt
        cat_att = []
        cat_unatt = []
        for el, er, state in zip(env_l_list, env_r_list, true_att_states):
            if state == 'L':
                cat_att.append(el)
                cat_unatt.append(er)
            else:
                cat_att.append(er)
                cat_unatt.append(el)
                
        cat_att = np.concatenate(cat_att)
        cat_unatt = np.concatenate(cat_unatt)
        
        cat_c_att = cross_corr(cat_p, cat_att)
        cat_c_unatt = cross_corr(cat_p, cat_unatt)
    else:
        cat_c_att = mean_att
        cat_c_unatt = mean_unatt
        
    return mean_att, cat_c_att, cat_c_unatt, bacc, auc, len(preds)

def evaluate_audit(model, trials, device, window_len=128, hop_len=64, transition_margin=192):
    model.eval()
    
    stable_preds, stable_env_l, stable_env_r, stable_states = [], [], [], []
    trans_preds, trans_env_l, trans_env_r, trans_states = [], [], [], []
    
    with torch.no_grad():
        for trial in trials:
            eeg = trial['eeg']
            env_l = trial['env_l']
            env_r = trial['env_r']
            switch_points = trial['meta']['switch_points']
            
            for start in range(0, eeg.shape[1] - window_len + 1, hop_len):
                end = start + window_len
                
                # Check overlap with ANY transition region
                is_trans = False
                for state, s_idx in switch_points:
                    t_start = s_idx - transition_margin
                    t_end = s_idx + transition_margin
                    if max(start, t_start) < min(end, t_end):
                        is_trans = True
                        break
                        
                # Determine ground truth state at the center of the window
                center = start + window_len // 2
                current_state = switch_points[0][0]
                for state, s_idx in switch_points:
                    if center >= s_idx:
                        current_state = state
                        
                # Extract and predict
                w_eeg = eeg[:, start:end].unsqueeze(0).to(device)
                pred, _ = model(w_eeg, return_features=True)
                pred = pred.squeeze().cpu().numpy()
                
                w_env_l = env_l[start:end].numpy()
                w_env_r = env_r[start:end].numpy()
                
                if is_trans:
                    trans_preds.append(pred)
                    trans_env_l.append(w_env_l)
                    trans_env_r.append(w_env_r)
                    trans_states.append(current_state)
                else:
                    stable_preds.append(pred)
                    stable_env_l.append(w_env_l)
                    stable_env_r.append(w_env_r)
                    stable_states.append(current_state)
                    
    s_2s, s_catt, s_cunatt, s_bacc, s_auc, s_n = compute_metrics(stable_preds, stable_env_l, stable_env_r, stable_states)
    t_2s, t_catt, t_cunatt, t_bacc, t_auc, t_n = compute_metrics(trans_preds, trans_env_l, trans_env_r, trans_states)
    
    return {
        'stable': {'n': s_n, '2s_att': s_2s, 'cat_att': s_catt, 'cat_unatt': s_cunatt, 'bacc': s_bacc, 'auc': s_auc},
        'trans': {'n': t_n, '2s_att': t_2s, 'cat_att': t_catt, 'cat_unatt': t_cunatt, 'bacc': t_bacc, 'auc': t_auc}
    }

def print_res(name, res):
    print(f"\n[{name}]")
    print("Stable windows:")
    print(f"    Number: {res['stable']['n']}")
    print(f"    2s Att Corr: {res['stable']['2s_att']:.4f}")
    print(f"    Concat Att Corr: {res['stable']['cat_att']:.4f}")
    print(f"    Concat Unatt Corr: {res['stable']['cat_unatt']:.4f}")
    print(f"    Balanced Acc: {res['stable']['bacc']:.4f}")
    print(f"    AUROC: {res['stable']['auc']:.4f}")
    
    print("\nTransition windows:")
    print(f"    Number: {res['trans']['n']}")
    print(f"    2s Att Corr: {res['trans']['2s_att']:.4f}")
    print(f"    Concat Att Corr: {res['trans']['cat_att']:.4f}")
    print(f"    Concat Unatt Corr: {res['trans']['cat_unatt']:.4f}")
    print(f"    Balanced Acc: {res['trans']['bacc']:.4f}")
    print(f"    AUROC: {res['trans']['auc']:.4f}")

def main():
    print("==================================================")
    print("=== PHASE 28.21 STABLE VS TRANSITION AUDIT =======")
    print("==================================================\n")
    
    # 1. Build Ground-Truth Cache (Same as Phase 28.20)
    S1_mat = glob.glob('/kaggle/input/datasets/lokeshgile/aasd-processed-eeg/Processed EEG/*/*.mat')[0]
    mat = scipy.io.loadmat(S1_mat, squeeze_me=True, struct_as_record=False)
    eeg_var = [k for k in mat.keys() if not k.startswith('__')][0]
    data_all, events = mat[eeg_var].data, mat[eeg_var].event
    
    target_channels = ['T7', 'C2', 'FT8', 'P7', 'CPz', 'Fp1', 'TP8', 'C3']
    fallback_map = {'T7': 23, 'C2': 28, 'FT8': 22, 'P7': 41, 'CPz': 36, 'Fp1': 0, 'TP8': 40, 'C3': 25}
    sel_idx = [fallback_map[tc] for tc in target_channels]
    
    b, a = scipy.signal.butter(4, [1.0/64.0, 8.0/64.0], btype='band')
    audio_dir = '/kaggle/input/datasets/lokeshgile/aasd-audio-gammatones'
    
    new_trials = []
    
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
                    idx_64 = max(0, int(rel_lat_128 / 2.0))
                    switch_points.append(('R' if t_str in ['179', '254'] else 'L', idx_64))
        switch_points.sort(key=lambda x: x[1])
        
        trial_eeg = data_all[:, :, epoch_idx - 1]
        trial_eeg = trial_eeg - trial_eeg.mean(axis=0, keepdims=True)
        trial_eeg_filt = scipy.signal.filtfilt(b, a, trial_eeg, axis=1)
        trial_eeg_8 = scipy.signal.resample_poly(trial_eeg_filt, 1, 2, axis=1)[sel_idx, 4:] # Lag correct
        
        audio_data = np.load(npz_path)
        env_l, env_r = audio_data['env_l'][:-4], audio_data['env_r'][:-4]
        
        min_len = min(trial_eeg_8.shape[1], len(env_l))
        trial_eeg_8 = trial_eeg_8[:, :min_len]
        env_l, env_r = env_l[:min_len], env_r[:min_len]
        
        trial_eeg_8 = trial_eeg_8 - trial_eeg_8.mean(axis=1, keepdims=True)
        trial_eeg_8 = trial_eeg_8 / (trial_eeg_8.std(axis=1, keepdims=True) + 1e-12)
        env_l = (env_l - env_l.mean()) / (env_l.std() + 1e-12)
        env_r = (env_r - env_r.mean()) / (env_r.std() + 1e-12)
        
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
            
        new_trials.append({
            'meta': {'switch_points': switch_points},
            'eeg': torch.FloatTensor(trial_eeg_8),
            'env_l': torch.FloatTensor(env_l),
            'env_r': torch.FloatTensor(env_r),
            'att': torch.FloatTensor(att).unsqueeze(0),
            'unatt': torch.FloatTensor(unatt).unsqueeze(0)
        })
        
    print(f"[INFO] Data loaded. Trials: {len(new_trials)}")
    
    # 2. Train Model (fast 50 epochs)
    split_idx = int(len(new_trials) * 0.8)
    train_trials, test_trials = new_trials[:split_idx], new_trials[split_idx:]
    
    train_ds = WindowedDataset(train_trials)
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AADConformer(in_channels=8).to(device)
    
    kul_path = '/kaggle/input/datasets/lowkieee/eeg-aad-conformer-seed1-checkpoints/checkpoints/seed_1/model_S1.pt'
    if os.path.exists(kul_path):
        ckpt = torch.load(kul_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt, strict=False)
        
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    print("\n[INFO] Fast-training for 50 epochs to recreate Phase 28.20 model...")
    for epoch in range(1, 51):
        model.train()
        for eeg_w, att_w, _ in train_loader:
            eeg_w, att_w = eeg_w.to(device), att_w.to(device)
            optimizer.zero_grad()
            pred, _ = model(eeg_w, return_features=True)
            loss = pearson_loss(pred.squeeze(1), att_w)
            loss.backward()
            optimizer.step()
            
    print("[INFO] Training complete. Running Audit...\n")
    
    os.makedirs('results/phase28_21', exist_ok=True)
    
    train_res = evaluate_audit(model, train_trials, device)
    test_res = evaluate_audit(model, test_trials, device)
    
    print_res("TRAIN RESULTS", train_res)
    print_res("TEST RESULTS", test_res)
    
    # Save to file
    with open('results/phase28_21/audit_results.txt', 'w') as f:
        f.write("TRAIN RESULTS\n")
        f.write(str(train_res) + "\n\n")
        f.write("TEST RESULTS\n")
        f.write(str(test_res) + "\n")

if __name__ == "__main__":
    main()
