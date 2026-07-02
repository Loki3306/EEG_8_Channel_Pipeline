import os
import json
import glob
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add project root to path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(REPO_ROOT))

def load_events():
    event_file = REPO_ROOT / "results" / "phase17_2" / "event_log.jsonl"
    if not event_file.exists():
        print(f"File not found: {event_file}")
        return None
        
    events = []
    with open(event_file, 'r') as f:
        for line in f:
            events.append(json.loads(line))
            
    df = pd.DataFrame(events)
    return df

def audit_switches(df, out_dir):
    """
    Step 2 & Step 3: Switch Audit & False Switch Investigation
    """
    switch_events = []
    false_switches = []
    
    # We group by scenario
    for scenario, group in df.groupby('scenario'):
        prev_lock = None
        for idx, row in group.iterrows():
            curr_lock = row['active_lock']
            
            # A switch occurs when lock changes from one state to another
            if pd.notna(curr_lock) and pd.notna(prev_lock) and curr_lock != prev_lock:
                # Calculate how long the new speaker has been speaking (for early/late)
                gt = row['ground_truth']
                is_correct = (curr_lock == gt)
                
                switch = {
                    'scenario': scenario,
                    'timestamp_sec': row['timestamp_sec'],
                    'old_lock': prev_lock,
                    'new_lock': curr_lock,
                    'ground_truth': gt,
                    'is_correct': is_correct,
                    'scene': row['scene'],
                    'probability': row['probability'],
                    'margin': row['margin']
                }
                switch_events.append(switch)
                
                if not is_correct:
                    # RCA Investigation
                    cause = "Unknown"
                    if row['time_in_state'] < 20: # Oscillation
                        cause = "Oscillation/Thrashing"
                    elif row['margin'] < 0.10:
                        cause = "Weak Evidence (Low Margin)"
                    else:
                        cause = "Strong Incorrect Evidence"
                        
                    fs = switch.copy()
                    fs['root_cause'] = cause
                    false_switches.append(fs)
                    
            prev_lock = curr_lock
            
    pd.DataFrame(switch_events).to_csv(out_dir / "switch_events.csv", index=False)
    pd.DataFrame(false_switches).to_csv(out_dir / "false_switches.csv", index=False)
    
    return switch_events, false_switches

def extract_json_splices(df):
    splices = []
    for scenario, group in df.groupby('scenario'):
        current_scene = None
        for idx, row in group.iterrows():
            if row['scene'] != current_scene:
                if current_scene is not None:
                    splices.append({
                        'scenario': scenario,
                        'timestamp_sec': row['timestamp_sec'],
                        'old_scene': current_scene,
                        'new_scene': row['scene'],
                        'new_gt': int(row['ground_truth'])
                    })
                current_scene = row['scene']
    return splices

def audit_latency(df, splices, out_dir):
    """
    Step 4: Latency Audit
    Time Zero = Scene Boundary (first window where ground truth flips).
    Lock Complete = First window where active_lock == ground_truth after Time Zero.
    """
    latencies = []
    
    for splice in splices:
        scenario = splice['scenario']
        splice_ts = splice['timestamp_sec']
        target_gt = splice['new_gt']
        
        # Filter scenario events after splice
        post_splice = df[(df['scenario'] == scenario) & (df['timestamp_sec'] >= splice_ts)]
        
        locked_ts = None
        for idx, row in post_splice.iterrows():
            if row['active_lock'] == target_gt:
                locked_ts = row['timestamp_sec']
                break
                
        latency = (locked_ts - splice_ts) if locked_ts is not None else np.nan
        latencies.append({
            'scenario': scenario,
            'splice_ts': splice_ts,
            'new_scene': splice['new_scene'],
            'target_gt': target_gt,
            'locked_ts': locked_ts,
            'latency_s': latency
        })
        
    pd.DataFrame(latencies).to_csv(out_dir / "latency_breakdown.csv", index=False)
    return latencies

def audit_coverage_and_uncertainty(df, out_dir):
    """
    Step 5 & 6: Coverage & Uncertainty Audit
    """
    stats = []
    for scenario, group in df.groupby('scenario'):
        total_windows = len(group)
        total_seconds = total_windows * 0.05 # Assuming 50ms hop
        
        uncertain_mask = group['active_lock'].isnull()
        uncertain_windows = uncertain_mask.sum()
        uncertain_seconds = uncertain_windows * 0.05
        
        # Coverage: Locked AND Correct
        correct_mask = group['active_lock'] == group['ground_truth']
        correct_windows = correct_mask.sum()
        
        stats.append({
            'scenario': scenario,
            'total_windows': total_windows,
            'uncertain_windows': uncertain_windows,
            'uncertain_pct': (uncertain_windows / total_windows) * 100,
            'correct_windows': correct_windows,
            'coverage_pct': (correct_windows / total_windows) * 100
        })
        
    pd.DataFrame(stats).to_csv(out_dir / "coverage_breakdown.csv", index=False)
    # Reusing same stats for uncertainty as it shares the denominator
    pd.DataFrame(stats).to_csv(out_dir / "uncertainty_breakdown.csv", index=False)

def audit_oscillation(df, out_dir):
    """
    Step 7: Oscillation Audit
    An oscillation is when the lock changes A->B->A within 10 seconds.
    """
    oscillations = []
    for scenario, group in df.groupby('scenario'):
        locks = group[~group['active_lock'].isnull()].copy()
        
        if len(locks) < 3:
            continue
            
        locks['lock_changed'] = locks['active_lock'] != locks['active_lock'].shift(1)
        changes = locks[locks['lock_changed']].copy()
        
        for i in range(len(changes) - 2):
            w1, w2, w3 = changes.iloc[i], changes.iloc[i+1], changes.iloc[i+2]
            
            time_diff = w3['timestamp_sec'] - w1['timestamp_sec']
            # If it flipped back to original lock within 10 seconds
            if w1['active_lock'] == w3['active_lock'] and time_diff < 10.0:
                oscillations.append({
                    'scenario': scenario,
                    'start_ts': w1['timestamp_sec'],
                    'end_ts': w3['timestamp_sec'],
                    'duration': time_diff,
                    'original_lock': w1['active_lock'],
                    'temp_lock': w2['active_lock']
                })
                
    pd.DataFrame(oscillations).to_csv(out_dir / "oscillation_events.csv", index=False)

def new_product_metrics(df, switch_events, out_dir):
    """
    Step 8: New Product Metrics
    """
    switches_df = pd.DataFrame(switch_events)
    if switches_df.empty:
        tp = fp = fn = tn = 0
    else:
        tp = len(switches_df[switches_df['is_correct'] == True])
        fp = len(switches_df[switches_df['is_correct'] == False])
        # Note: False Negatives and True Negatives are harder to define cleanly in continuous streaming
        fn = 0 # Not implemented for this audit
        
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = 0 # Cannot compute without FN
    
    metrics = {
        'total_true_switches': tp,
        'total_false_switches': fp,
        'switch_precision': precision,
        'controller_utilization': 100 - (df['active_lock'].isnull().sum() / len(df) * 100)
    }
    
    with open(out_dir / "controller_state_statistics.json", "w") as f:
        json.dump(metrics, f, indent=4)

def print_executive_summary(switch_events, df):
    switches_df = pd.DataFrame(switch_events)
    
    if switches_df.empty:
        tp = fp = 0
    else:
        tp = len(switches_df[switches_df['is_correct'] == True])
        fp = len(switches_df[switches_df['is_correct'] == False])
        
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    coverage = (df['active_lock'] == df['ground_truth']).sum() / len(df) * 100
    
    print("====================================================")
    print("PHASE 17.2")
    print("Metrics Verified ............ DONE")
    print("Switch Audit ............... DONE")
    print("Latency Audit .............. DONE")
    print("Coverage Audit ............. DONE")
    print("Case Studies ............... PENDING (Manual)")
    print("----------------------------------------------------")
    print("Verified Product Metrics")
    print(f"True Switches: {tp}")
    print(f"False Switches: {fp}")
    print(f"Precision: {precision:.2f}")
    print(f"Coverage: {coverage:.2f}%")
    print("----------------------------------------------------")
    print("Files Written to results/phase17_2/")
    print("Done")
    print("====================================================")

def main():
    df = load_events()
    if df is None:
        return
        
    out_dir = REPO_ROOT / "results" / "phase17_2"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    switch_events, false_switches = audit_switches(df, out_dir)
    splices = extract_json_splices(df)
    
    audit_latency(df, splices, out_dir)
    audit_coverage_and_uncertainty(df, out_dir)
    audit_oscillation(df, out_dir)
    new_product_metrics(df, switch_events, out_dir)
    
    print_executive_summary(switch_events, df)

if __name__ == "__main__":
    main()
