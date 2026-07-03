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
from decision_engine.strategies.change_detection import CUSUMHybrid

def safe_corr_np(x, y, eps=1e-8):
    x_mean = x.mean(axis=1, keepdims=True)
    y_mean = y.mean(axis=1, keepdims=True)
    num = np.sum((x - x_mean) * (y - y_mean), axis=1)
    den = np.sqrt(np.sum((x - x_mean)**2, axis=1) * np.sum((y - y_mean)**2, axis=1))
    return num / (den + eps)

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
    
    latencies = []
    missed = 0
    total_switches = 0
    
    for trial_id, trial_df in trace_df.groupby('trial_id'):
        switches = (trial_df['ground_truth'] != trial_df['ground_truth'].shift(1)) & (trial_df.index > trial_df.index[0])
        switch_points = trial_df[switches]
        total_switches += len(switch_points)
        
        for _, sp in switch_points.iterrows():
            ts = sp['timestamp_sec']
            tgt = sp['ground_truth']
            post_splice = trial_df[trial_df['timestamp_sec'] >= ts]
            
            next_switches = post_splice[post_splice['ground_truth'] != tgt]
            end_ts = next_switches.iloc[0]['timestamp_sec'] if not next_switches.empty else trial_df['timestamp_sec'].max()
            
            valid_period = post_splice[post_splice['timestamp_sec'] < end_ts]
            lock = valid_period[valid_period['active_lock'] == tgt]
            
            if not lock.empty:
                lat = lock.iloc[0]['timestamp_sec'] - ts
                latencies.append(lat)
            else:
                missed += 1
                
    metrics['mean_switch_latency'] = np.mean(latencies) if latencies else np.nan
    metrics['missed_switches'] = missed
    metrics['total_switches'] = total_switches
    
    trace_df['wrong_lock_block'] = (trace_df['active_lock'] != trace_df['ground_truth']) & trace_df['active_lock'].notna()
    trace_df['block_id'] = (trace_df['wrong_lock_block'] != trace_df['wrong_lock_block'].shift(1)).cumsum()
    false_switches = trace_df[trace_df['wrong_lock_block']]['block_id'].nunique()
    
    duration_hours = len(trace_df) / 3600.0 if len(trace_df) > 0 else 0
    metrics['false_switches_per_hr'] = false_switches / duration_hours if duration_hours > 0 else 0
    
    return metrics

def run_evaluation(model, eeg_8_dict, env_dict, trial_starts_dict, raw_evs_dict, device, margin_scale, cusum_drift, cusum_threshold):
    all_traces = []
    total_oscillations = 0

    for mf, eeg_8 in eeg_8_dict.items():
        trial_starts = trial_starts_dict[mf]
        
        for idx_ev, (ev_idx, audio_marker, trial_start_lat, next_start_lat) in enumerate(trial_starts):
            if audio_marker not in env_dict:
                continue
                
            env_l_1d, env_r_1d = env_dict[audio_marker]
            raw_evs = raw_evs_dict[mf][idx_ev]
            
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
                strategy=CUSUMHybrid(drift=cusum_drift, threshold=cusum_threshold)
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
                
                margin_B = c_r - c_l
                
                prob = 1.0 / (1.0 + np.exp(-5.0 * margin_B * margin_scale))
                
                res = engine.update(prob, margin_B * margin_scale)
                
                trace.append({
                    'timestamp_sec': t_array[i],
                    'window_idx': i, 
                    'ground_truth': int(gt_B[i]),
                    'cumulative_evidence': res['evidence'], 
                    'confidence': res['confidence'],
                    'active_threshold': res['threshold_used'], 
                    'state': res['state'],
                    'decision': res['decision'], 
                    'active_lock': 1 if res['action'] == 'SWITCH_LEFT' else (0 if res['action'] == 'SWITCH_RIGHT' else None)
                })
                
            if not trace:
                continue
                
            trace_df = pd.DataFrame(trace)
            trace_df['trial_id'] = f"{mf}_{idx_ev}"
            trace_df['active_lock'] = trace_df['active_lock'].ffill()
            all_traces.append(trace_df)
            
            stats = engine.statistics()
            total_oscillations += stats.get('oscillations', 0)

    if not all_traces:
        return {}
        
    full_df = pd.concat(all_traces, ignore_index=True)
    metrics = compute_metrics(full_df)
    metrics['total_oscillations'] = total_oscillations
    return metrics

def main():
    print("[INFO] Starting Phase 26 CUSUM Adaptation")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] Using device: {device}")

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
    
    mat_files = glob.glob(os.path.join(eeg_dir, '*', '*.mat'))[:3]
    
    print("[INFO] Pre-loading and filtering all EEG and Audio data...")
    eeg_8_dict = {}
    trial_starts_dict = {}
    raw_evs_dict = {}
    env_dict = {}
    
    for mf in mat_files:
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
            data_all = np.concatenate([data_all[:, :, i] for i in range(data_all.shape[2])], axis=1)
            
        eeg_filt = signal.filtfilt(b, a, data_all, axis=1)
        eeg_64 = signal.resample_poly(eeg_filt, 1, 2, axis=1)
        eeg_8_dict[os.path.basename(mf)] = eeg_64[sel_idx, :]

        trial_starts = []
        for i, ev in enumerate(events):
            t_str = str(get_ev_attr(ev, 'type', 0)).strip()
            if t_str and t_str not in ['179', '184', '254', '255']:
                try:
                    lat = float(get_ev_attr(ev, 'latency', 1))
                    trial_starts.append((i, t_str, lat))
                except:
                    pass

        valid_starts = []
        valid_evs = []
        for idx_ev, (ev_idx, audio_marker, trial_start_lat) in enumerate(trial_starts):
            npz_path = os.path.join(audio_dir, f"{int(audio_marker)}.npz")
            if not os.path.exists(npz_path):
                continue
                
            if audio_marker not in env_dict:
                audio_data = np.load(npz_path)
                env_dict[audio_marker] = (audio_data['env_l'], audio_data['env_r'])

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
                    
            valid_starts.append((ev_idx, audio_marker, trial_start_lat, next_start_lat))
            valid_evs.append(raw_evs)
            
        trial_starts_dict[os.path.basename(mf)] = valid_starts
        raw_evs_dict[os.path.basename(mf)] = valid_evs
        
    print("[INFO] Data loaded. Starting grid search over CUSUM parameters.")
    
    margin_scales = [1.0, 2.0, 5.0]
    cusum_drifts = [0.1, 0.3]
    cusum_thresholds = [2.0, 5.0, 10.0]
    
    results = []
    
    for m_scale in margin_scales:
        for drift in cusum_drifts:
            for thresh in cusum_thresholds:
                print(f"\nEvaluating: Scale={m_scale} | Drift={drift} | Threshold={thresh}")
                metrics = run_evaluation(model, eeg_8_dict, env_dict, trial_starts_dict, raw_evs_dict, device, m_scale, drift, thresh)
                
                if metrics:
                    print(f"  Correct Lock: {metrics.get('correct_coverage', 0):.1f}%")
                    print(f"  Missed Switches: {metrics.get('missed_switches', 0)} / {metrics.get('total_switches', 0)}")
                    print(f"  False Switches/hr: {metrics.get('false_switches_per_hr', 0):.2f}")
                    
                    results.append({
                        'margin_scale': m_scale,
                        'drift': drift,
                        'threshold': thresh,
                        'correct_coverage': metrics.get('correct_coverage', 0),
                        'availability': metrics.get('availability', 0),
                        'missed_switches': metrics.get('missed_switches', 0),
                        'total_switches': metrics.get('total_switches', 0),
                        'false_switches_per_hr': metrics.get('false_switches_per_hr', 0),
                        'mean_switch_latency': metrics.get('mean_switch_latency', np.nan),
                        'oscillations': metrics.get('total_oscillations', 0)
                    })

    out_dir = os.path.join(project_root, 'results', 'phase26')
    os.makedirs(out_dir, exist_ok=True)
    df_res = pd.DataFrame(results)
    df_res.to_csv(os.path.join(out_dir, 'cusum_grid_search.csv'), index=False)
    
    print("\n==================================================")
    print("PHASE 26 GRID SEARCH COMPLETE")
    print("==================================================")
    if not df_res.empty:
        best = df_res.sort_values('correct_coverage', ascending=False).head(3)
        print(best[['margin_scale', 'drift', 'threshold', 'correct_coverage', 'missed_switches', 'false_switches_per_hr']])
        
if __name__ == "__main__":
    main()
