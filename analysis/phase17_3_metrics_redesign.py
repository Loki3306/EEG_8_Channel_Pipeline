import pandas as pd
import numpy as np
import json
from pathlib import Path
import sys

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

def generate_output_events(df, stability_window=20):
    """
    Step 1 & 6: Output State Machine & Event Collapsing
    Convert internal states to User-Visible Output States:
    NO_OUTPUT, LOCK_LEFT (0), LOCK_RIGHT (1), SWITCHING
    """
    output_events = []
    
    for scenario, group in df.groupby('scenario'):
        current_output = "NO_OUTPUT"
        unstable_counter = 0
        stable_counter = 0
        candidate_lock = None
        
        for idx, row in group.iterrows():
            internal_lock = row['active_lock']
            
            if pd.isna(internal_lock):
                unstable_counter += 1
                if unstable_counter >= stability_window:
                    current_output = "NO_OUTPUT"
                    stable_counter = 0
                    candidate_lock = None
                else:
                    if current_output not in ["NO_OUTPUT"]:
                        current_output = "SWITCHING"
            else:
                unstable_counter = 0
                if internal_lock == candidate_lock:
                    stable_counter += 1
                else:
                    candidate_lock = internal_lock
                    stable_counter = 1
                    
                if stable_counter >= stability_window:
                    current_output = f"LOCK_{int(internal_lock)}"
                else:
                    if current_output not in ["NO_OUTPUT"]:
                        current_output = "SWITCHING"
            
            output_events.append({
                'scenario': scenario,
                'timestamp_sec': row['timestamp_sec'],
                'internal_lock': internal_lock,
                'output_state': current_output,
                'ground_truth': row['ground_truth'],
                'scene': row['scene']
            })
            
    return pd.DataFrame(output_events)

def redesign_switches(out_df, out_dir):
    """
    Step 2: Redesign False Switch Metric
    Extract Audible Switches from collapsed events
    """
    audible_switches = []
    
    for scenario, group in out_df.groupby('scenario'):
        prev_state = "NO_OUTPUT"
        
        for idx, row in group.iterrows():
            curr_state = row['output_state']
            
            if curr_state != prev_state and curr_state.startswith("LOCK_"):
                # We reached a committed lock
                is_correct = (curr_state == f"LOCK_{int(row['ground_truth'])}")
                
                audible_switches.append({
                    'scenario': scenario,
                    'timestamp_sec': row['timestamp_sec'],
                    'from_state': prev_state,
                    'to_state': curr_state,
                    'ground_truth': row['ground_truth'],
                    'is_correct': is_correct
                })
            
            prev_state = curr_state
            
    pd.DataFrame(audible_switches).to_csv(out_dir / "audible_switches.csv", index=False)
    return audible_switches

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

def redesign_latency(out_df, splices, out_dir):
    """
    Step 3: Redesign Latency
    Initial Acquisition, Switch, Recovery, Already Correct
    """
    latencies = []
    
    for splice in splices:
        scenario = splice['scenario']
        splice_ts = splice['timestamp_sec']
        target_gt_state = f"LOCK_{splice['new_gt']}"
        
        # What was the state exactly before the splice?
        pre_splice = out_df[(out_df['scenario'] == scenario) & (out_df['timestamp_sec'] < splice_ts)]
        if pre_splice.empty:
            continue
        state_at_splice = pre_splice.iloc[-1]['output_state']
        
        category = "Unknown"
        if state_at_splice == target_gt_state:
            category = "Already Correct"
            latency = 0.0
        elif state_at_splice == "NO_OUTPUT":
            category = "Initial Acquisition"
        else:
            category = "Switch/Recovery"
            
        if category != "Already Correct":
            # Find the next time it hits target_gt_state
            post_splice = out_df[(out_df['scenario'] == scenario) & (out_df['timestamp_sec'] >= splice_ts)]
            locked_ts = None
            for idx, row in post_splice.iterrows():
                if row['output_state'] == target_gt_state:
                    locked_ts = row['timestamp_sec']
                    break
            
            latency = (locked_ts - splice_ts) if locked_ts is not None else np.nan
        
        latencies.append({
            'scenario': scenario,
            'splice_ts': splice_ts,
            'category': category,
            'state_at_splice': state_at_splice,
            'latency_s': latency
        })
        
    pd.DataFrame(latencies).to_csv(out_dir / "latency_breakdown.csv", index=False)
    return latencies

def redesign_coverage(out_df, out_dir):
    """
    Step 4: Redesign Coverage
    Decision Availability, Stable Output Time, Correct Stable Output Time
    """
    stats = []
    for scenario, group in out_df.groupby('scenario'):
        total_windows = len(group)
        
        decision_avail = group['output_state'] != "NO_OUTPUT"
        stable_output = group['output_state'].str.startswith("LOCK_")
        
        # Correct stable output
        correct_state_series = "LOCK_" + group['ground_truth'].astype(int).astype(str)
        correct_stable = (group['output_state'] == correct_state_series)
        
        stats.append({
            'scenario': scenario,
            'total_seconds': total_windows * 0.05,
            'decision_availability_pct': (decision_avail.sum() / total_windows) * 100,
            'stable_output_pct': (stable_output.sum() / total_windows) * 100,
            'correct_stable_output_pct': (correct_stable.sum() / total_windows) * 100
        })
        
    pd.DataFrame(stats).to_csv(out_dir / "coverage_breakdown.csv", index=False)
    return stats

def compute_ux_metrics(out_df, audible_switches, latencies, coverage, out_dir):
    switches_df = pd.DataFrame(audible_switches)
    
    total_hours = len(out_df) * 0.05 / 3600
    
    if switches_df.empty:
        tp = fp = 0
    else:
        tp = len(switches_df[switches_df['is_correct'] == True])
        fp = len(switches_df[switches_df['is_correct'] == False])
        
    lat_df = pd.DataFrame(latencies)
    acq_lat = lat_df[lat_df['category'] == 'Initial Acquisition']['latency_s'].mean() if not lat_df.empty else np.nan
    sw_lat = lat_df[lat_df['category'] == 'Switch/Recovery']['latency_s'].mean() if not lat_df.empty else np.nan
    
    overall_correct_coverage = np.mean([s['correct_stable_output_pct'] for s in coverage])
    overall_avail = np.mean([s['decision_availability_pct'] for s in coverage])
    
    metrics = {
        'total_duration_hours': total_hours,
        'audible_false_switches_per_hour': fp / total_hours if total_hours > 0 else 0,
        'audible_correct_switches_per_hour': tp / total_hours if total_hours > 0 else 0,
        'mean_acquisition_latency_s': acq_lat,
        'mean_switch_recovery_latency_s': sw_lat,
        'decision_availability_pct': overall_avail,
        'correct_stable_output_pct': overall_correct_coverage,
        'output_downtime_pct': 100 - overall_avail
    }
    
    with open(out_dir / "ux_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)
        
    print("====================================================")
    print("PHASE 17.3")
    print("Internal Events ............... DONE")
    print("Output Events ................. DONE")
    print("Metric Redesign ............... DONE")
    print("----------------------------------------------------")
    print("New UX Metrics")
    print(f"Audible False Switches/hr: {metrics['audible_false_switches_per_hour']:.2f}")
    print(f"Decision Availability: {metrics['decision_availability_pct']:.2f}%")
    print(f"Correct Lock Coverage: {metrics['correct_stable_output_pct']:.2f}%")
    print(f"Acquisition Latency: {metrics['mean_acquisition_latency_s']:.2f}s")
    print(f"Switch/Recovery Latency: {metrics['mean_switch_recovery_latency_s']:.2f}s")
    print("====================================================")

def main():
    df = load_events()
    if df is None:
        return
        
    out_dir = REPO_ROOT / "results" / "phase17_3"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    out_df = generate_output_events(df, stability_window=20)
    out_df.to_csv(out_dir / "collapsed_output_events.csv", index=False)
    
    audible_switches = redesign_switches(out_df, out_dir)
    splices = extract_json_splices(df)
    
    latencies = redesign_latency(out_df, splices, out_dir)
    coverage = redesign_coverage(out_df, out_dir)
    
    compute_ux_metrics(out_df, audible_switches, latencies, coverage, out_dir)

if __name__ == "__main__":
    main()
