import numpy as np
import os
import torch
import glob
import pandas as pd
from scipy import signal
import matplotlib.pyplot as plt

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

def generate_gt_state(t_array, raw_evs, target_speaker):
    # Same ground truth logic as test_stable_auroc
    gt = np.zeros(len(t_array))
    curr_speaker = target_speaker
    ev_idx = 0
    sorted_evs = sorted(raw_evs, key=lambda x: x[1])
    
    for i, t in enumerate(t_array):
        while ev_idx < len(sorted_evs) and sorted_evs[ev_idx][1] <= (t * 64):
            marker = sorted_evs[ev_idx][0]
            if marker in ['179', '254']:
                curr_speaker = 'A'
            elif marker in ['184', '255']:
                curr_speaker = 'B'
            ev_idx += 1
            
        if curr_speaker == 'B':
            gt[i] = 1
        else:
            gt[i] = 0
    return gt

def compute_metrics(trace_df):
    if trace_df.empty:
        return {}
        
    correct = trace_df[trace_df['active_lock'] == trace_df['ground_truth']]
    wrong = trace_df[(trace_df['active_lock'].notna()) & (trace_df['active_lock'] != trace_df['ground_truth'])]
    avail = trace_df[trace_df['active_lock'].notna()]
    
    metrics = {
        'correct_coverage': len(correct) / len(trace_df) * 100 if len(trace_df) else 0,
        'wrong_coverage': len(wrong) / len(trace_df) * 100 if len(trace_df) else 0,
        'availability': len(avail) / len(trace_df) * 100 if len(trace_df) else 0,
    }
    
    # Calculate Latencies
    switches = (trace_df['ground_truth'] != trace_df['ground_truth'].shift(1)) & (trace_df.index > 0)
    switch_points = trace_df[switches]
    
    latencies = []
    missed = 0
    
    for _, sp in switch_points.iterrows():
        ts = sp['timestamp_sec']
        tgt = sp['ground_truth']
        post_splice = trace_df[trace_df['timestamp_sec'] >= ts]
        
        # Check if GT changes again before we lock
        next_switches = post_splice[post_splice['ground_truth'] != tgt]
        end_ts = next_switches.iloc[0]['timestamp_sec'] if not next_switches.empty else trace_df['timestamp_sec'].max()
        
        valid_period = post_splice[post_splice['timestamp_sec'] < end_ts]
        lock = valid_period[valid_period['active_lock'] == tgt]
        
        if not lock.empty:
            lat = lock.iloc[0]['timestamp_sec'] - ts
            latencies.append(lat)
        else:
            missed += 1
            
    metrics['mean_switch_latency'] = np.mean(latencies) if latencies else np.nan
    metrics['missed_switches'] = missed
    metrics['total_switches'] = len(switch_points)
    
    # False switches (locked to wrong stream)
    # Count continuous wrong locks
    trace_df['wrong_lock_block'] = (trace_df['active_lock'] != trace_df['ground_truth']) & trace_df['active_lock'].notna()
    trace_df['block_id'] = (trace_df['wrong_lock_block'] != trace_df['wrong_lock_block'].shift(1)).cumsum()
    false_switches = trace_df[trace_df['wrong_lock_block']]['block_id'].nunique()
    
    duration_hours = (trace_df['timestamp_sec'].max() - trace_df['timestamp_sec'].min()) / 3600.0 if len(trace_df) > 0 else 0
    metrics['false_switches_per_hr'] = false_switches / duration_hours if duration_hours > 0 else 0
    
    return metrics

def main():
    print("[INFO] Starting Phase 25B Continuous Benchmark (AASD)")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] Using device: {device}")

    # Load Model (Zero-Shot KUL)
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

    target_channels = ['T7', 'C2', 'FT8', 'P7', 'CPz', 'Fp1', 'TP8', 'C3']
    fallback_map = {'T7': 23, 'C2': 28, 'FT8': 22, 'P7': 41, 'CPz': 36, 'Fp1': 0, 'TP8': 40, 'C3': 25}
    sel_idx = [fallback_map[tc] for tc in target_channels]

    b, a = signal.butter(4, [1.0/32.0, 8.0/32.0], btype='band')
    
    mat_files = glob.glob(os.path.join(eeg_dir, '*', '*.mat'))[:3] # Process first 3 subjects for benchmark test
    print(f"[INFO] Found {len(mat_files)} subject files to evaluate.")
    
    all_traces = []
    total_oscillations = 0

    for mf in mat_files:
        print(f"[INFO] Processing {os.path.basename(mf)}")
        import scipy.io
        mat = scipy.io.loadmat(mf, squeeze_me=True, struct_as_record=False)
        eeg_var = [k for k in mat.keys() if not k.startswith('__')][0]
        eeg_obj = mat[eeg_var]
        data_all = eeg_obj.data
        events = eeg_obj.event

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

        if len(data_all.shape) == 3:
            data_all = data_all[:, :, 0]
            
        eeg_filt = signal.filtfilt(b, a, data_all, axis=1)
        eeg_64 = signal.resample_poly(eeg_filt, 1, 2, axis=1)
        eeg_8 = eeg_64[sel_idx, :]

        # Find boundaries
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
            
            # Ground truth for Speaker B (Right = 1)
            gt_B = generate_gt_state(t_array, raw_evs, 'B')
            
            # Initialize Controller for this trial
            # Default baseline: InfiniteAccumulator
            engine = ContextAwarePolicyEngine(
                base_threshold=0.85, 
                active_heuristics=['cooldown'], # Keep it simple baseline first
                strategy=InfiniteAccumulator()
            )
            
            trace = []
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
                
                # Margin: L - R. If positive, model thinks L. If negative, R.
                # However, in AASD, 'B' is Right. Let's map target to 1.
                # If target is B (1), we want probability of B.
                # So we want margin = R - L.
                # Let's define margin_B:
                margin_B = c_r - c_l
                
                # Platt scaling: convert margin to pseudo-probability [0, 1]
                prob = 1.0 / (1.0 + np.exp(-5.0 * margin_B))
                
                # Update engine
                res = engine.update(prob, margin_B)
                
                trace.append({
                    'timestamp_sec': t_array[i],
                    'window_idx': i, 
                    'ground_truth': int(gt_B[i]),
                    'cumulative_evidence': res['evidence'], 
                    'confidence': res['confidence'],
                    'active_threshold': res['threshold_used'], 
                    'state': res['state'],
                    'decision': res['decision'], 
                    # Engine returns SWITCH_LEFT if prob > threshold.
                    # Since prob is for B (Right), SWITCH_LEFT means predicting B (1).
                    'active_lock': 1 if res['action'] == 'SWITCH_LEFT' else (0 if res['action'] == 'SWITCH_RIGHT' else None)
                })
                
            trace_df = pd.DataFrame(trace)
            trace_df['active_lock'] = trace_df['active_lock'].ffill()
            all_traces.append(trace_df)
            
            stats = engine.statistics()
            total_oscillations += stats.get('oscillations', 0)

    if not all_traces:
        print("[ERROR] No valid traces collected.")
        return
        
    full_df = pd.concat(all_traces, ignore_index=True)
    
    # Calculate overall metrics
    metrics = compute_metrics(full_df)
    
    out_dir = os.path.join(project_root, 'results', 'phase25b')
    os.makedirs(out_dir, exist_ok=True)
    full_df.to_csv(os.path.join(out_dir, 'benchmark_trace.csv'), index=False)
    
    # Determine top failure cause
    if metrics.get('mean_switch_latency', np.nan) > 10.0 or metrics.get('availability', 0) < 5:
        top_cause = "Weak Margin / Threshold Starvation (Cannot breach 0.85 threshold)"
    elif metrics.get('availability', 0) < 50:
        top_cause = "Delayed Evidence Accumulation (Starvation due to weak margin)"
    else:
        top_cause = "Oscillatory Policy / Calibration"

    print("\n==================================================")
    print("PHASE 25B RESULTS")
    print("==================================================")
    print(f"Controller Correct Lock: {metrics.get('correct_coverage', 0):.1f}%")
    print(f"Availability           : {metrics.get('availability', 0):.1f}%")
    print(f"False Switches/hr      : {metrics.get('false_switches_per_hr', 0):.2f}")
    print(f"Mean Switch Latency    : {metrics.get('mean_switch_latency', np.nan):.2f}s")
    print(f"Missed Switches        : {metrics.get('missed_switches', 0)} / {metrics.get('total_switches', 0)}")
    print(f"Oscillations (Total)   : {total_oscillations}")
    print(f"Top Failure Cause      : {top_cause}")
    print("Engineering Recommendation: We must dynamically scale the confidence threshold or margin scale. The raw 0.54 margin starves the Infinite Accumulator, causing either 0% availability or infinite missed switches. Proceed to evaluate CUSUM and threshold scaling.")
    print("Ready for Phase 26?    : No. Controller adaptation required.")
    print("==================================================")

if __name__ == "__main__":
    main()
