import numpy as np
import os
import torch
import glob
import pandas as pd
from scipy import signal

import sys
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

from models.aad_conformer import AADConformer
from decision_engine.context_aware_engine import ContextAwarePolicyEngine
from decision_engine.strategies import InfiniteAccumulator

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

def main():
    print("[INFO] Starting Phase 25B.5 Forensic Audit")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AADConformer(in_channels=8).to(device)
    ckpt_path = '/kaggle/input/datasets/lowkieee/eeg-aad-conformer-seed1-checkpoints/checkpoints/seed_1/model_S1.pt'
    if not os.path.exists(ckpt_path):
        print(f"[ERROR] Model not found")
        return
        
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    if 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'], strict=False)
    else:
        model.load_state_dict(ckpt, strict=False)
    model.eval()

    eeg_dir = '/kaggle/input/datasets/lokeshgile/aasd-processed-eeg/Processed EEG'
    audio_dir = '/kaggle/input/datasets/lokeshgile/aasd-audio-gammatones'

    target_channels = ['T7', 'C2', 'FT8', 'P7', 'CPz', 'Fp1', 'TP8', 'C3']
    fallback_map = {'T7': 23, 'C2': 28, 'FT8': 22, 'P7': 41, 'CPz': 36, 'Fp1': 0, 'TP8': 40, 'C3': 25}
    sel_idx = [fallback_map[tc] for tc in target_channels]
    b, a = signal.butter(4, [1.0/32.0, 8.0/32.0], btype='band')
    
    mf = glob.glob(os.path.join(eeg_dir, '*', 'S18.mat'))[0]
    import scipy.io
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
        
        engine = ContextAwarePolicyEngine(
            base_threshold=0.85, 
            active_heuristics=['cooldown'],
            strategy=InfiniteAccumulator()
        )
        
        trace = []
        print(f"{'Idx':<4} | {'Time(s)':<7} | {'GT':<3} | {'Margin':<7} | {'Conf':<5} | {'Acc(Ev)':<7} | {'Decision':<14} | {'Switch Emitted?':<16} | {'True/False'}")
        print("-" * 105)
        
        for i, start in enumerate(range(0, min(trial_eeg_8.shape[1], len(env_l_1d)) - win_len, hop)):
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
            
            margin_B = c_r - c_l
            prob = 1.0 / (1.0 + np.exp(-5.0 * margin_B))
            
            res = engine.update(prob, margin_B)
            
            active_lock = 1 if res['action'] == 'SWITCH_LEFT' else (0 if res['action'] == 'SWITCH_RIGHT' else None)
            is_switch = res['action'] in ['SWITCH_LEFT', 'SWITCH_RIGHT']
            
            tf_str = ""
            if is_switch:
                tf_str = "True Positive" if active_lock == gt_B[i] else "False Positive"
                
            print(f"{i:<4} | {t_array[i]:<7.1f} | {int(gt_B[i]):<3} | {margin_B:<7.3f} | {res['confidence']:<5.2f} | {res['evidence']:<7.2f} | {res['decision']:<14} | {str(is_switch):<16} | {tf_str}")
            
            trace.append({
                'timestamp_sec': t_array[i],
                'window_idx': i, 
                'ground_truth': int(gt_B[i]),
                'cumulative_evidence': res['evidence'], 
                'confidence': res['confidence'],
                'active_threshold': res['threshold_used'], 
                'state': res['state'],
                'decision': res['decision'], 
                'active_lock': active_lock
            })
            
        print("-" * 105)
        
        trace_df = pd.DataFrame(trace)
        trace_df['trial_id'] = "S18_0"
        trace_df['active_lock'] = trace_df['active_lock'].ffill()
        
        emitted_switches = sum([1 for t in trace if t['decision'] in ['SWITCH_LEFT', 'SWITCH_RIGHT']])
        print(f"Total emitted switches (actions): {emitted_switches}")
        
        trace_df['wrong_lock_block'] = (trace_df['active_lock'] != trace_df['ground_truth']) & trace_df['active_lock'].notna()
        trace_df['block_id'] = (trace_df['wrong_lock_block'] != trace_df['wrong_lock_block'].shift(1)).cumsum()
        false_switches = trace_df[trace_df['wrong_lock_block']]['block_id'].nunique()
        
        old_duration_hours = (trace_df['timestamp_sec'].max() - trace_df['timestamp_sec'].min()) / 3600.0 if len(trace_df) > 0 else 0
        
        print("\n--- Phase 25B Metric Audit ---")
        print(f"Trace length (windows): {len(trace_df)}")
        print(f"Total false lock sequences (wrong lock blocks): {false_switches}")
        print(f"Old Duration Hours logic: (max_t - min_t)/3600 = {old_duration_hours:.6f}")
        print(f"Old False Switches/hr = {false_switches / old_duration_hours if old_duration_hours > 0 else 0:.2f}")
        
        new_duration_hours = len(trace_df) / 3600.0 if len(trace_df) > 0 else 0
        print(f"New Duration Hours logic: (n_windows * 1s)/3600 = {new_duration_hours:.6f}")
        print(f"New False Switches/hr = {false_switches / new_duration_hours if new_duration_hours > 0 else 0:.2f}")
        
        # Calculate real true positive / false positive breakdown
        # Count true switches in ground truth
        trace_df['gt_switch'] = (trace_df['ground_truth'] != trace_df['ground_truth'].shift(1)) & (trace_df.index > 0)
        gt_switches = trace_df['gt_switch'].sum()
        print(f"Ground Truth Switches: {gt_switches}")
        
        tp = 0
        fp = 0
        # Check every emitted switch
        switch_idx = trace_df[trace_df['decision'].isin(['SWITCH_LEFT', 'SWITCH_RIGHT'])].index
        for idx in switch_idx:
            # Check if this switch correctly aligns with ground truth at that window
            # Actually, a True Positive switch is when the controller switches to the CORRECT stream.
            act_lock = trace_df.loc[idx, 'active_lock']
            gt_state = trace_df.loc[idx, 'ground_truth']
            if act_lock == gt_state:
                tp += 1
            else:
                fp += 1
                
        print(f"True Positive Switches: {tp}")
        print(f"False Positive Switches: {fp}")
        
        break

if __name__ == "__main__":
    main()
