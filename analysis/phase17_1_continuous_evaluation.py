import os
import sys
import json
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.pipeline.session_generator import ContinuousSessionGenerator, KULAdapter
from decision_engine.context_aware_engine import ContextAwarePolicyEngine, Action, State
from models.aad_conformer import AADConformer

def safe_corr_np(x, y):
    x_mean = x.mean()
    y_mean = y.mean()
    x_std = x.std()
    y_std = y.std()
    if x_std < 1e-8 or y_std < 1e-8:
        return 0.0
    cov = np.mean((x - x_mean) * (y - y_mean))
    return cov / (x_std * y_std)

def extract_predictions(generator, scenario_path, model, device, out_path):
    """Runs the raw streaming windows through the neural network and saves probabilities."""
    if os.path.exists(out_path):
        print(f"Skipping {Path(scenario_path).name} (already computed)")
        return pd.read_csv(out_path)
        
    stream = generator.generate_stream(scenario_path)
    
    batch_size = 128
    batch_eeg = []
    batch_ya = []
    batch_yb = []
    batch_meta = []
    
    results = []
    
    model.eval()
    with torch.no_grad():
        for window in tqdm(stream, desc=f"Predicting {Path(scenario_path).name}"):
            batch_eeg.append(window['eeg_window'])
            batch_ya.append(window['audio_a_window'])
            batch_yb.append(window['audio_b_window'])
            
            batch_meta.append({
                'timestamp_sec': window['timestamp_sec'],
                'ground_truth': window['ground_truth'],
                'scene_name': window['scene_name'],
                'scenario_name': window['scenario_name'],
                'window_idx': window['window_idx']
            })
            
            if len(batch_eeg) == batch_size:
                x = torch.FloatTensor(np.array(batch_eeg)).to(device)
                
                # Normalize EEG
                x_mean = x.mean(dim=2, keepdim=True)
                x_std = x.std(dim=2, keepdim=True) + 1e-8
                x_norm = (x - x_mean) / x_std
                
                preds = model(x_norm).squeeze(1).cpu().numpy() # [B, 128]
                
                ya = np.array(batch_ya)
                yb = np.array(batch_yb)
                
                for i in range(batch_size):
                    c_att = safe_corr_np(preds[i], ya[i])
                    c_unatt = safe_corr_np(preds[i], yb[i])
                    margin = c_att - c_unatt
                    
                    # Approximate Platt scaling (maps margin to 0..1 probability)
                    prob = 1.0 / (1.0 + np.exp(-5.0 * margin))
                    
                    res = batch_meta[i]
                    res['c_att'] = c_att
                    res['c_unatt'] = c_unatt
                    res['margin'] = margin
                    res['prob'] = prob
                    results.append(res)
                    
                batch_eeg, batch_ya, batch_yb, batch_meta = [], [], [], []
                
        # Flush remaining
        if len(batch_eeg) > 0:
            x = torch.FloatTensor(np.array(batch_eeg)).to(device)
            x_mean = x.mean(dim=2, keepdim=True)
            x_std = x.std(dim=2, keepdim=True) + 1e-8
            x_norm = (x - x_mean) / x_std
            
            preds = model(x_norm).squeeze(1).cpu().numpy()
            ya = np.array(batch_ya)
            yb = np.array(batch_yb)
            
            for i in range(len(batch_eeg)):
                c_att = safe_corr_np(preds[i], ya[i])
                c_unatt = safe_corr_np(preds[i], yb[i])
                margin = c_att - c_unatt
                prob = 1.0 / (1.0 + np.exp(-5.0 * margin))
                
                res = batch_meta[i]
                res['c_att'] = c_att
                res['c_unatt'] = c_unatt
                res['margin'] = margin
                res['prob'] = prob
                results.append(res)
                
    df = pd.DataFrame(results)
    df.to_csv(out_path, index=False)
    return df

def extract_json_splices(df):
    """Finds exact transition timestamps where JSON scene changes."""
    splices = []
    current_scene = None
    for idx, row in df.iterrows():
        if row['scene_name'] != current_scene:
            if current_scene is not None:
                splices.append({
                    'timestamp_sec': row['timestamp_sec'],
                    'old_scene': current_scene,
                    'new_scene': row['scene_name'],
                    'new_gt': int(row['ground_truth'])
                })
            current_scene = row['scene_name']
    return splices

def compute_metrics(predictions_df, policy_trace, splices):
    """Computes Phase 17 product metrics."""
    total_duration = predictions_df['timestamp_sec'].max() - predictions_df['timestamp_sec'].min()
    
    # Coverage & UNCERTAIN time
    uncertain_windows = 0
    correct_locked_windows = 0
    total_windows = len(policy_trace)
    
    state_flips = 0
    prev_state = None
    for t in policy_trace:
        st = t['state']
        gt = t['true_label']
        lock = t['active_lock']
        
        if lock != prev_state and prev_state is not None:
            state_flips += 1
        prev_state = lock
        
        if lock is None:
            uncertain_windows += 1
        else:
            # Check if locked onto correct speaker
            if lock == gt:
                correct_locked_windows += 1
                
    coverage_pct = (correct_locked_windows / total_windows) * 100
    uncertain_pct = (uncertain_windows / total_windows) * 100
    
    oscillation_freq = (state_flips / (total_duration / 60.0)) if total_duration > 0 else 0
    
    # Latency calculation
    latencies = []
    recovery_times = []
    
    for splice in splices:
        splice_ts = splice['timestamp_sec']
        target_gt = splice['new_gt']
        
        # Find first time after splice where state hits target
        for t in policy_trace:
            if t['timestamp_sec'] >= splice_ts:
                if t['active_lock'] == target_gt:
                    latency = t['timestamp_sec'] - splice_ts
                    latencies.append(latency)
                    break
                    
    mean_latency = np.mean(latencies) if latencies else 0.0
    
    # False Switch Rate
    false_switches = 0
    for t in policy_trace:
        action = t.get('action')
        gt = t['true_label']
        if action == 'SWITCH_LEFT' and gt == 0:
            false_switches += 1
        elif action == 'SWITCH_RIGHT' and gt == 1:
            false_switches += 1
                
    fsr = (false_switches / (total_duration / 3600.0)) if total_duration > 0 else 0
    
    return {
        'total_duration_s': round(total_duration, 2),
        'mean_lock_latency_s': round(mean_latency, 2),
        'false_switch_rate_per_hr': round(fsr, 2),
        'coverage_pct': round(coverage_pct, 1),
        'uncertain_pct': round(uncertain_pct, 1),
        'oscillation_freq_per_min': round(oscillation_freq, 2),
        'splices_evaluated': len(splices)
    }

def main():
    print("====================================================")
    print("PHASE 17.1: CONTINUOUS HEARING AID EVALUATION")
    print("====================================================")
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Path to trained Conformer checkpoint")
    args = parser.parse_args()
    
    out_dir = Path("results/phase17_1")
    (out_dir / "scenario_streams").mkdir(parents=True, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    print("Loading Model...")
    model = AADConformer(
        in_channels=8,
        temporal_filters=32,
        spatial_filters=64,
        embed_dim=64,
        num_heads=4,
        num_layers=2,
        dropout=0.3,
        stride=4
    ).to(device)
    model.load_state_dict(torch.load(args.model, map_location=device), strict=False)
    model.eval()
    
    print("Initializing Generator...")
    kul_adapter = KULAdapter(cache_dir="/kaggle/input/datasets/lokeshgile/kul-processed/data/processed_kul")
    generator = ContinuousSessionGenerator(adapters={'KUL': kul_adapter})
    
    scenarios_dir = Path("scenarios")
    scenario_files = sorted(list(scenarios_dir.glob("*.json")))
    
    engine = ContextAwarePolicyEngine(base_threshold=0.85, 
        active_heuristics=['difficulty', 'growth_rate', 'hysteresis', 'oscillation_penalty', 'cooldown'])
        
    all_metrics = []
    
    for sf in scenario_files:
        scen_name = Path(sf).stem
        stream_path = out_dir / "scenario_streams" / f"{scen_name}_predictions.csv"
        
        # 1. Generate Deep Learning Stream
        df = extract_predictions(generator, sf, model, device, stream_path)
        
        # 2. Extract JSON Splices
        splices = extract_json_splices(df)
        
        # 3. Simulate Policy Engine
        trace = []
        engine.reset()
        active_lock = None
        for idx, row in df.iterrows():
            res = engine.update(row['prob'], row['margin'])
            action = str(res['action'])
            st = str(res['state'])
            
            if action == 'SWITCH_LEFT':
                active_lock = 1
            elif action == 'SWITCH_RIGHT':
                active_lock = 0
            elif st in ['UNCERTAIN', 'INITIALIZING', 'WAITING']:
                active_lock = None
                
            trace.append({
                'timestamp_sec': row['timestamp_sec'],
                'state': st,
                'action': action,
                'active_lock': active_lock,
                'true_label': int(row['ground_truth'])
            })
            
        # 4. Compute Phase 17 Metrics
        metrics = compute_metrics(df, trace, splices)
        metrics['scenario'] = scen_name
        all_metrics.append(metrics)
        
    metrics_df = pd.DataFrame(all_metrics)
    
    print("\n--- PHASE 17.1 RESULTS ---")
    print(metrics_df.to_markdown(index=False))
    
    metrics_df.to_csv(out_dir / "product_metrics.csv", index=False)
    
    with open(out_dir / "product_metrics_summary.md", "w") as f:
        f.write("# Phase 17.1 Continuous Evaluation\n\n")
        f.write(metrics_df.to_markdown(index=False) + "\n")
        
    print(f"\nDone. Saved to {out_dir}/product_metrics_summary.md")
    print("====================================================")

if __name__ == "__main__":
    main()
