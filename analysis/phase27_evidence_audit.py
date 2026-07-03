import numpy as np
import os
import torch
import glob
import pandas as pd
from scipy import signal
import scipy.io
import collections
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score, average_precision_score, balanced_accuracy_score
from sklearn.feature_selection import mutual_info_classif

import sys
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

from models.aad_conformer import AADConformer

def safe_corr_np(x, y, eps=1e-8):
    x_mean = x.mean(axis=1, keepdims=True)
    y_mean = y.mean(axis=1, keepdims=True)
    num = np.sum((x - x_mean) * (y - y_mean), axis=1)
    den = np.sqrt(np.sum((x - x_mean)**2, axis=1) * np.sum((y - y_mean)**2, axis=1))
    return num / (den + eps)

def get_ev_attr(e, attr_name, array_idx=0):
    try:
        if hasattr(e, attr_name):
            return getattr(e, attr_name)
        if isinstance(e, np.ndarray):
            if e.size == 1 and hasattr(e.flat[0], attr_name):
                return getattr(e.flat[0], attr_name)
            return e[array_idx]
    except:
        pass
    return ''

def generate_gt_state(t_array, raw_evs, target_speaker):
    gt = np.zeros(len(t_array))
    if len(raw_evs) == 0:
        return gt
    st_times = []
    types = []
    for ev_t, ev_lat in raw_evs:
        if ev_t in ['179', '184', '254', '255']:
            st_times.append(ev_lat / 128.0)
            if target_speaker == 'A':
                types.append('L' if ev_t in ['179', '254'] else 'R')
            else:
                types.append('R' if ev_t in ['179', '254'] else 'L')
    if len(types) == 0:
        return gt
    current_state = 1 if types[0] == 'R' else 0
    for i, t in enumerate(t_array):
        state = current_state
        for st, s_type in zip(st_times, types):
            if t >= st:
                state = 1 if s_type == 'L' else 0
        gt[i] = state
    return gt

def cohens_d(x, y):
    nx = len(x)
    ny = len(y)
    dof = nx + ny - 2
    if dof <= 0: return 0.0
    pooled_std = np.sqrt(((nx-1)*np.std(x, ddof=1)**2 + (ny-1)*np.std(y, ddof=1)**2) / dof)
    if pooled_std == 0: return 0.0
    return (np.mean(x) - np.mean(y)) / pooled_std

def bhattacharyya_distance(x, y):
    mean_x, mean_y = np.mean(x), np.mean(y)
    var_x, var_y = np.var(x), np.var(y)
    if var_x == 0 or var_y == 0: return float('inf')
    var_pooled = (var_x + var_y) / 2
    return 0.125 * ((mean_x - mean_y)**2) / var_pooled + 0.5 * np.log(var_pooled / np.sqrt(var_x * var_y))

def main():
    print("[INFO] Starting Phase 27 Evidence Stream Information Audit")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AADConformer(in_channels=8).to(device)
    ckpt_path = '/kaggle/input/datasets/lowkieee/eeg-aad-conformer-seed1-checkpoints/checkpoints/seed_1/model_S1.pt'
    if not os.path.exists(ckpt_path):
        print(f"[ERROR] Model not found at {ckpt_path}")
        return
        
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    if 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'], strict=False)
    else:
        model.load_state_dict(ckpt, strict=False)
    model.eval()

    eeg_dir = '/kaggle/input/datasets/lokeshgile/aasd-processed-eeg/Processed EEG'
    audio_dir = '/kaggle/input/datasets/lokeshgile/aasd-audio-gammatones'
    out_dir = os.path.join(project_root, 'results', 'phase27')
    os.makedirs(out_dir, exist_ok=True)

    target_channels = ['T7', 'C2', 'FT8', 'P7', 'CPz', 'Fp1', 'TP8', 'C3']
    fallback_map = {'T7': 23, 'C2': 28, 'FT8': 22, 'P7': 41, 'CPz': 36, 'Fp1': 0, 'TP8': 40, 'C3': 25}
    sel_idx = [fallback_map[tc] for tc in target_channels]
    b, a = signal.butter(4, [1.0/32.0, 8.0/32.0], btype='band')
    
    subjects = ['S1', 'S14', 'S18']
    
    all_margins = []
    all_gts = []
    
    for sub in subjects:
        print(f"\n[INFO] Processing Subject {sub}...")
        mf = glob.glob(os.path.join(eeg_dir, '*', f'{sub}.mat'))
        if not mf:
            continue
        mf = mf[0]
        
        mat = scipy.io.loadmat(mf, squeeze_me=True, struct_as_record=False)
        eeg_var = [k for k in mat.keys() if not k.startswith('__')][0]
        eeg_obj = mat[eeg_var]
        data_all = eeg_obj.data
        events = eeg_obj.event

        if len(data_all.shape) == 3:
            data_all = np.concatenate([data_all[:, :, i] for i in range(data_all.shape[2])], axis=1)
            
        eeg_filt = signal.filtfilt(b, a, data_all, axis=1)
        eeg_64 = signal.resample_poly(eeg_filt, 1, 2, axis=1)
        eeg_8 = eeg_64[sel_idx, :]

        trial_starts = []
        for i, ev in enumerate(events):
            t_str = str(get_ev_attr(ev, 'type', 0)).strip()
            if t_str and t_str not in ['179', '184', '254', '255']:
                try:
                    lat = float(get_ev_attr(ev, 'latency', 1))
                    trial_starts.append((i, t_str, lat))
                except:
                    pass

        sub_margins = []
        sub_gt = []
        trial_data = []
        
        for idx_ev, (ev_idx, audio_marker, trial_start_lat) in enumerate(trial_starts):
            npz_path = os.path.join(audio_dir, f"{int(audio_marker)}.npz")
            if not os.path.exists(npz_path):
                continue
                
            audio_data = np.load(npz_path)
            env_l_1d = audio_data['env_l']
            env_r_1d = audio_data['env_r']

            next_start_lat = trial_starts[idx_ev+1][2] if idx_ev+1 < len(trial_starts) else data_all.shape[1]
            if next_start_lat - trial_start_lat < 128 * 10: 
                continue
                
            raw_evs = []
            for ev in events[ev_idx:]:
                try:
                    lat = float(get_ev_attr(ev, 'latency', 1))
                    if lat >= next_start_lat:
                        break
                    t_str = str(get_ev_attr(ev, 'type', 0)).strip()
                    raw_evs.append((t_str, lat - trial_start_lat))
                except:
                    pass
                    
            start_64 = int(trial_start_lat // 2)
            end_64 = int(next_start_lat // 2)
            trial_eeg_8 = eeg_8[:, start_64:end_64]

            win_len = 128
            hop = 64
            t_array = np.arange(0, min(trial_eeg_8.shape[1], len(env_l_1d)) - win_len, hop) / 64.0 + 1.0
            gt_B = generate_gt_state(t_array, raw_evs, 'B')
            
            margins = []
            for start in range(0, min(trial_eeg_8.shape[1], len(env_l_1d)) - win_len, hop):
                win_eeg = trial_eeg_8[:, start:start+win_len]
                win_eeg = (win_eeg - win_eeg.mean(axis=1, keepdims=True)) / (win_eeg.std(axis=1, keepdims=True) + 1e-8)
                win_l = env_l_1d[start:start+win_len]
                win_r = env_r_1d[start:start+win_len]
                win_l = (win_l - win_l.mean()) / (win_l.std() + 1e-8)
                win_r = (win_r - win_r.mean()) / (win_r.std() + 1e-8)
                
                eeg_t = torch.tensor(win_eeg[np.newaxis, ...], dtype=torch.float32).to(device)
                with torch.no_grad():
                    out, _ = model(eeg_t, return_features=True)
                    pred_env = out.squeeze(1).cpu().numpy()
                    
                c_l = safe_corr_np(pred_env, win_l[np.newaxis, ...])[0]
                c_r = safe_corr_np(pred_env, win_r[np.newaxis, ...])[0]
                margins.append(c_r - c_l)
                
            sub_margins.extend(margins)
            sub_gt.extend(gt_B)
            all_margins.extend(margins)
            all_gts.extend(gt_B)
            
            if len(trial_data) < 3: # Save a few trials for plotting over time
                trial_data.append((t_array, gt_B, margins))
                
        # --- Plotting per subject ---
        sub_margins = np.array(sub_margins)
        sub_gt = np.array(sub_gt)
        
        # 1. Margin vs Time
        plt.figure(figsize=(15, 6))
        for t_idx, (t_array, gt, m) in enumerate(trial_data):
            plt.subplot(3, 1, t_idx+1)
            plt.plot(t_array, m, label='Margin (c_r - c_l)', color='blue', alpha=0.7)
            plt.plot(t_array, gt - 0.5, label='GT (1=Right, 0=Left)', color='red', linestyle='--', linewidth=2)
            plt.axhline(0, color='black', linewidth=1)
            plt.title(f"{sub} - Trial {t_idx}")
            if t_idx == 0: plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{sub}_margin_vs_time.png"))
        plt.close()
        
        # 2. Histogram
        plt.figure(figsize=(10, 6))
        m_left = sub_margins[sub_gt == 0]
        m_right = sub_margins[sub_gt == 1]
        sns.histplot(m_left, color='blue', label='GT=Left (0)', stat='density', alpha=0.5, bins=50)
        sns.histplot(m_right, color='red', label='GT=Right (1)', stat='density', alpha=0.5, bins=50)
        plt.title(f"{sub} - Margin Distributions")
        plt.legend()
        plt.savefig(os.path.join(out_dir, f"{sub}_margin_histogram.png"))
        plt.close()

    print("\n" + "="*80)
    print("=== GLOBAL EVIDENCE AUDIT (ALL SUBJECTS) ===")
    print("="*80)
    
    all_margins = np.array(all_margins)
    all_gts = np.array(all_gts)
    
    m_l = all_margins[all_gts == 0]
    m_r = all_margins[all_gts == 1]
    
    auroc = roc_auc_score(all_gts, all_margins)
    auprc = average_precision_score(all_gts, all_margins)
    mi = mutual_info_classif(all_margins.reshape(-1, 1), all_gts, random_state=42)[0]
    cd = cohens_d(m_r, m_l)
    bd = bhattacharyya_distance(m_r, m_l)
    
    print(f"AUROC:                   {auroc:.4f}")
    print(f"AUPRC:                   {auprc:.4f}")
    print(f"Mutual Information:      {mi:.6f} nats")
    print(f"Cohen's d:               {cd:.4f}")
    print(f"Bhattacharyya Distance:  {bd:.4f}")
    
    # 3. Threshold Sweep
    thresholds = np.linspace(-1.0, 1.0, 100)
    best_thresh = 0
    best_bal_acc = 0
    bal_accs = []
    
    for t in thresholds:
        preds = (all_margins > t).astype(int)
        acc = balanced_accuracy_score(all_gts, preds)
        bal_accs.append(acc)
        if acc > best_bal_acc:
            best_bal_acc = acc
            best_thresh = t
            
    print(f"\n[Static Threshold Bound]")
    print(f"Best Static Threshold:   {best_thresh:.4f}")
    print(f"Max Balanced Accuracy:   {best_bal_acc:.4f} (at best static threshold)")
    
    # 4. Oracle Controller (Non-causal centered moving average)
    print(f"\n[Oracle Controller Bound]")
    print("Simulating non-causal moving average smoothing (hindsight).")
    oracle_accs = []
    # Test window sizes from 1 to 20 seconds
    for window in [1, 5, 10, 15, 20]: 
        smoothed_margins = np.convolve(all_margins, np.ones(window)/window, mode='same')
        preds = (smoothed_margins > best_thresh).astype(int)
        acc = balanced_accuracy_score(all_gts, preds)
        oracle_accs.append(acc)
        print(f"MA Window {window:>2}s: Acc = {acc:.4f}")
        
    print("\nCONCLUSION:")
    if auroc < 0.6:
        print("The decoder margins contain almost no mutual information with attention.")
        print("The KUL zero-shot weights have failed to transfer to AASD.")
        print("Controller engineering cannot fix this. Focus must pivot to Domain Adaptation.")
    else:
        print("The decoder is providing a usable signal, but the controller is failing to exploit it.")

if __name__ == "__main__":
    main()
