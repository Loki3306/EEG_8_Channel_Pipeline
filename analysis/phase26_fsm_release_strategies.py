import numpy as np
import os
import torch
import glob
import pandas as pd
from scipy import signal
import json
import collections

import sys
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

from models.aad_conformer import AADConformer

class State:
    INITIALIZING = "INITIALIZING"
    WAITING = "WAITING"
    LOCKED = "LOCKED"
    SWITCHING = "SWITCHING"
    UNCERTAIN = "UNCERTAIN"

class Action:
    WAIT = "WAIT"
    HOLD = "HOLD"
    SWITCH_LEFT = "SWITCH_LEFT"
    SWITCH_RIGHT = "SWITCH_RIGHT"
    REJECT = "REJECT"

class CUSUMDetector:
    def __init__(self, target_mean=0.0, drift=0.1, threshold=2.0):
        self.g_pos = 0.0
        self.g_neg = 0.0
        self.target_mean = target_mean
        self.drift = drift
        self.threshold = threshold
        
    def update(self, value):
        s_pos = value - self.target_mean - self.drift
        s_neg = self.target_mean - value - self.drift
        self.g_pos = max(0.0, self.g_pos + s_pos)
        self.g_neg = max(0.0, self.g_neg + s_neg)
        if self.g_pos > self.threshold or self.g_neg > self.threshold:
            self.reset()
            return True
        return False
        
    def reset(self):
        self.g_pos = 0.0
        self.g_neg = 0.0

class ComparativePolicyEngine:
    def __init__(self, strategy_name='baseline'):
        self.strategy_name = strategy_name
        self.config = {
            'base_threshold': 0.85,
            'minimum_lock_duration': 5,
            'minimum_switch_gap': 10,
            'minimum_consecutive_windows': 3,
            'maximum_wait_time': 15,
            'uncertainty_threshold': 0.15,
        }
        self.cusum = CUSUMDetector(drift=0.5, threshold=3.0)
        self.reset()
        
    def reset(self):
        self.state = State.INITIALIZING
        self.decision = None
        self.evidence = 0.0
        self.window_index = 0
        self.time_in_state = 0
        self.last_switch_time = -9999
        self.consecutive_agreement_count = 0
        self.last_candidate = None
        self.cusum.reset()
        self.uncertainty_ticks = 0

    def update(self, probability, margin):
        self.window_index += 1
        self.time_in_state += 1
        
        p = np.clip(probability, 1e-5, 1 - 1e-5)
        llr = np.log(p / (1 - p))
        self.evidence += llr
        
        confidence = 1.0 / (1.0 + np.exp(np.clip(-self.evidence, -500, 500)))
        cusum_alarm = self.cusum.update(margin)
        
        active_threshold = self.config['base_threshold']
        active_consecutive = self.config['minimum_consecutive_windows']
        active_switch_gap = self.config['minimum_switch_gap']
        
        if confidence >= active_threshold:
            candidate = 1
        elif confidence <= (1.0 - active_threshold):
            candidate = 0
        else:
            candidate = None
            
        if candidate is not None and candidate == self.last_candidate:
            self.consecutive_agreement_count += 1
        else:
            self.consecutive_agreement_count = 1 if candidate is not None else 0
        self.last_candidate = candidate
        
        is_uncertain = (0.5 - self.config['uncertainty_threshold']) <= confidence <= (0.5 + self.config['uncertainty_threshold'])
        
        if is_uncertain:
            self.uncertainty_ticks += 1
        else:
            self.uncertainty_ticks = 0

        action = Action.WAIT
        prev_state = self.state

        if self.state == State.INITIALIZING:
            if self.window_index >= 5: 
                self.state = State.WAITING
            action = Action.WAIT
            
        elif self.state == State.WAITING or self.state == State.UNCERTAIN:
            if candidate is not None and self.consecutive_agreement_count >= active_consecutive:
                self.state = State.LOCKED
                self.decision = candidate
                self.last_switch_time = self.window_index
                action = Action.SWITCH_LEFT if candidate == 1 else Action.SWITCH_RIGHT
            elif self.time_in_state >= self.config['maximum_wait_time']:
                self.state = State.UNCERTAIN
                action = Action.REJECT
            else:
                action = Action.WAIT
                
        elif self.state == State.LOCKED:
            action = Action.HOLD
            if self.time_in_state >= self.config['minimum_lock_duration']:
                release_triggered = False
                
                if self.strategy_name == 'baseline':
                    release_triggered = is_uncertain
                elif self.strategy_name == 'hold':
                    release_triggered = False
                elif self.strategy_name == 'timed':
                    release_triggered = is_uncertain and self.uncertainty_ticks >= 5
                elif self.strategy_name == 'hybrid':
                    opposing_evidence = (self.decision == 1 and self.evidence < 0) or (self.decision == 0 and self.evidence > 0)
                    release_triggered = is_uncertain and opposing_evidence and abs(self.evidence) > 2.0
                elif self.strategy_name == 'cusum':
                    release_triggered = is_uncertain and cusum_alarm

                if release_triggered:
                    self.state = State.UNCERTAIN
                    self.decision = None
                    action = Action.REJECT
                elif candidate is not None and candidate != self.decision:
                    if self.consecutive_agreement_count >= active_consecutive:
                        if (self.window_index - self.last_switch_time) >= active_switch_gap:
                            self.state = State.SWITCHING
                            action = Action.HOLD
                            
        elif self.state == State.SWITCHING:
            self.state = State.LOCKED 
            self.decision = candidate
            self.last_switch_time = self.window_index
            action = Action.SWITCH_LEFT if candidate == 1 else Action.SWITCH_RIGHT
            
        if self.state != prev_state:
            self.time_in_state = 0
            
        return {
            'action': action,
            'state': self.state,
            'decision': self.decision,
            'evidence': self.evidence,
            'confidence': confidence
        }

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

def compute_metrics(trace_df):
    if len(trace_df) == 0:
        return {}
        
    trace_df['active_lock'] = trace_df['active_lock'].ffill()
    
    total_windows = len(trace_df)
    n_hours = total_windows / 3600.0  # Proper duration calculation!
    
    valid_locks = trace_df.dropna(subset=['active_lock']).copy()
    valid_locks['active_lock'] = valid_locks['active_lock'].astype(int)
    
    correct_locks = (valid_locks['active_lock'] == valid_locks['ground_truth']).sum()
    coverage = correct_locks / total_windows if total_windows > 0 else 0
    availability = len(valid_locks) / total_windows if total_windows > 0 else 0
    
    gt_switches = (trace_df['ground_truth'].diff().fillna(0) != 0).sum()
    
    # Calculate False Switches block by block
    valid_locks['lock_group'] = (valid_locks['active_lock'] != valid_locks['active_lock'].shift()).cumsum()
    false_switches = 0
    latencies = []
    
    for _, group in valid_locks.groupby('lock_group'):
        group_lock = group['active_lock'].iloc[0]
        group_gt = group['ground_truth'].mode()[0] if len(group) > 0 else -1
        
        if group_lock != group_gt:
            false_switches += 1
            
        gt_changes = group[group['ground_truth'] != group['ground_truth'].shift()]
        if len(gt_changes) > 0:
            for idx in gt_changes.index:
                actual_gt = group.loc[idx, 'ground_truth']
                if actual_gt == group_lock:
                    latencies.append(idx - group.index[0])
                    
    false_switches_hr = false_switches / n_hours if n_hours > 0 else 0
    mean_lat = np.mean(latencies) if latencies else 0.0
    
    uncertain_time = (trace_df['state'] == State.UNCERTAIN).sum()
    
    return {
        'coverage': coverage * 100,
        'false_switches_hr': false_switches_hr,
        'mean_latency': mean_lat,
        'availability': availability * 100,
        'uncertain_time': uncertain_time,
        'gt_switches': gt_switches
    }

def main():
    print("[INFO] Starting Phase 26 FSM Release Policy Comparative Evaluation")
    
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
    
    subjects = ['S1', 'S14', 'S18']
    strategies = ['baseline', 'hold', 'timed', 'hybrid', 'cusum']
    
    results = {s: [] for s in strategies}
    
    for sub in subjects:
        mf = glob.glob(os.path.join(eeg_dir, '*', f'{sub}.mat'))
        if not mf:
            continue
        mf = mf[0]
        
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

        # Precompute model outputs for the subject to ensure identical inputs to all strategies
        print(f"Precomputing margins for {sub}...")
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
                
            trial_data.append({
                't_array': t_array,
                'gt': gt_B,
                'margins': margins
            })
            
        print(f"Running strategies for {sub}...")
        for strat in strategies:
            strat_trace = []
            for t_data in trial_data:
                engine = ComparativePolicyEngine(strategy_name=strat)
                active_lock = None
                
                for i in range(len(t_data['margins'])):
                    margin = t_data['margins'][i]
                    prob = 1.0 / (1.0 + np.exp(-5.0 * margin))
                    res = engine.update(prob, margin)
                    
                    if res['action'] == Action.SWITCH_LEFT:
                        active_lock = 1
                    elif res['action'] == Action.SWITCH_RIGHT:
                        active_lock = 0
                    elif res['action'] == Action.REJECT:
                        active_lock = None
                        
                    strat_trace.append({
                        'ground_truth': t_data['gt'][i],
                        'state': res['state'],
                        'active_lock': active_lock
                    })
                    
            df = pd.DataFrame(strat_trace)
            metrics = compute_metrics(df)
            results[strat].append(metrics)
            
    print("\n" + "="*80)
    print(f"{'Strategy':<15} | {'Coverage':<10} | {'Avail':<10} | {'Uncertain':<10} | {'False Swt/hr':<15} | {'Latency':<10}")
    print("-" * 80)
    
    for strat in strategies:
        avg_cov = np.mean([r.get('coverage', 0) for r in results[strat]])
        avg_avail = np.mean([r.get('availability', 0) for r in results[strat]])
        avg_unc = np.mean([r.get('uncertain_time', 0) for r in results[strat]])
        avg_fsh = np.mean([r.get('false_switches_hr', 0) for r in results[strat]])
        avg_lat = np.mean([r.get('mean_latency', 0) for r in results[strat]])
        
        print(f"{strat:<15} | {avg_cov:>8.2f}% | {avg_avail:>8.2f}% | {avg_unc:>8.0f}s | {avg_fsh:>13.2f} | {avg_lat:>8.2f}s")
        
if __name__ == "__main__":
    main()
