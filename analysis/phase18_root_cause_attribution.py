import pandas as pd
import numpy as np
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(REPO_ROOT))

def load_data():
    event_file = REPO_ROOT / "results" / "phase17_2" / "event_log.jsonl"
    switches_file = REPO_ROOT / "results" / "phase17_3" / "audible_switches.csv"
    latency_file = REPO_ROOT / "results" / "phase17_3" / "latency_breakdown.csv"
    
    if not event_file.exists() or not switches_file.exists() or not latency_file.exists():
        print("Missing Phase 17 outputs. Run Phase 17 pipeline first.")
        return None, None, None
        
    events = []
    with open(event_file, 'r') as f:
        for line in f:
            events.append(json.loads(line))
    trace_df = pd.DataFrame(events)
    
    switches_df = pd.read_csv(switches_file)
    latency_df = pd.read_csv(latency_file)
    
    return trace_df, switches_df, latency_df

def classify_event(trace_window, target_decision, event_type):
    if trace_window.empty:
        return "10. Unknown"
        
    mean_margin = trace_window['margin'].mean()
    margin_decision = 1 if mean_margin > 0 else 0
    mean_prob = trace_window['probability'].mean()
    prob_decision = 1 if mean_prob > 0.5 else 0
    mean_threshold = trace_window['threshold_used'].mean()
    
    if event_type == "FALSE_SWITCH":
        # Target decision is the WRONG decision it made
        if margin_decision == target_decision:
            if abs(mean_margin) > 0.05:
                return "1. Decoder Error"
            else:
                return "2. Weak Margin"
                
        # Margin pointed correctly, did probability point wrongly?
        if prob_decision == target_decision:
            return "3. Calibration Compression"
            
        # Probability pointed correctly, why did it switch?
        # Check evidence oscillations or policy artifacts
        zero_crossings = len(np.where(np.diff(np.signbit(trace_window['evidence'])))[0])
        if zero_crossings > 3:
            return "4. Evidence Accumulation Failure"
            
        return "9. Metric / Evaluation Artifact"
        
    elif event_type == "DELAY":
        # Target decision is the CORRECT decision it eventually made or failed to make
        if margin_decision != target_decision:
            if abs(mean_margin) > 0.05:
                return "1. Decoder Error"
            else:
                return "2. Weak Margin"
                
        if abs(mean_margin) < 0.05:
            return "2. Weak Margin"
            
        if prob_decision != target_decision or (target_decision == 1 and mean_prob < 0.65) or (target_decision == 0 and mean_prob > 0.35):
            return "3. Calibration Compression"
            
        zero_crossings = len(np.where(np.diff(np.signbit(trace_window['evidence'])))[0])
        if zero_crossings > 3:
            return "4. Evidence Accumulation Failure"
            
        if mean_threshold >= 0.90:
            return "5. Difficulty Predictor Error"
            
        max_conf_towards_target = trace_window['confidence'].max() if target_decision == 1 else (1.0 - trace_window['confidence'].min())
        if max_conf_towards_target < mean_threshold:
            return "6. Threshold Too Conservative"
            
        if (trace_window['state'] == 'COOLDOWN').sum() > len(trace_window) * 0.3 or \
           (trace_window['state'] == 'STABILIZING').sum() > len(trace_window) * 0.3:
            return "7. Cooldown / Hysteresis Delay"
            
        return "8. Policy Logic"

def main():
    trace_df, switches_df, latency_df = load_data()
    if trace_df is None: return
    
    out_dir = REPO_ROOT / "results" / "phase18"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    attributed_events = []
    
    # 1. Process False Switches
    false_switches = switches_df[switches_df['is_correct'] == False]
    for idx, row in false_switches.iterrows():
        scenario = row['scenario']
        ts = row['timestamp_sec']
        
        # Target decision for False Switch is what it switched TO (which is wrong)
        to_state = row['to_state']
        target_decision = 1 if to_state == "LOCK_1" else 0
        
        # 5 seconds prior to the switch
        window = trace_df[(trace_df['scenario'] == scenario) & 
                          (trace_df['timestamp_sec'] > ts - 5.0) & 
                          (trace_df['timestamp_sec'] <= ts)]
                          
        cause = classify_event(window, target_decision, "FALSE_SWITCH")
        attributed_events.append({
            'scenario': scenario,
            'timestamp_sec': ts,
            'event_type': 'FALSE_SWITCH',
            'duration_s': 5.0,
            'root_cause': cause
        })
        
    # 2. Process Delays
    delays = latency_df[(latency_df['category'] != 'Already Correct') & (latency_df['latency_s'] > 0)]
    for idx, row in delays.iterrows():
        scenario = row['scenario']
        start_ts = row['splice_ts']
        latency = row['latency_s']
        end_ts = start_ts + latency
        
        # Assuming ground truth changes to a new speaker
        # Let's get the target decision from trace_df at end_ts
        pt = trace_df[(trace_df['scenario'] == scenario) & (trace_df['timestamp_sec'] >= end_ts)]
        if pt.empty: continue
        target_decision = int(pt.iloc[0]['ground_truth'])
        
        window = trace_df[(trace_df['scenario'] == scenario) & 
                          (trace_df['timestamp_sec'] >= start_ts) & 
                          (trace_df['timestamp_sec'] <= end_ts)]
                          
        cause = classify_event(window, target_decision, "DELAY")
        attributed_events.append({
            'scenario': scenario,
            'timestamp_sec': start_ts,
            'event_type': row['category'],
            'duration_s': latency,
            'root_cause': cause
        })
        
    attr_df = pd.DataFrame(attributed_events)
    attr_df.to_csv(out_dir / "root_cause_events.csv", index=False)
    
    # Taxonomy
    taxonomy = attr_df['root_cause'].value_counts().reset_index()
    taxonomy.columns = ['Root Cause', 'Event Count']
    taxonomy.to_csv(out_dir / "failure_taxonomy.csv", index=False)
    
    # Subsystem Contributions
    total_events = len(attr_df)
    total_delay_s = attr_df[attr_df['event_type'] != 'FALSE_SWITCH']['duration_s'].sum()
    
    subsystems = attr_df.groupby('root_cause').agg({
        'scenario': 'count',
        'duration_s': lambda x: x[attr_df.loc[x.index, 'event_type'] != 'FALSE_SWITCH'].sum()
    }).reset_index()
    
    subsystems.columns = ['Root Cause', 'Count', 'Total Delay (s)']
    subsystems['Pct Events'] = (subsystems['Count'] / total_events) * 100 if total_events > 0 else 0
    subsystems['Pct Delay'] = (subsystems['Total Delay (s)'] / total_delay_s) * 100 if total_delay_s > 0 else 0
    
    with open(out_dir / "subsystem_contributions.json", "w") as f:
        json.dump(subsystems.to_dict('records'), f, indent=4)
        
    # Generate Report
    with open(out_dir / "root_cause_report.md", "w") as f:
        f.write("# Phase 18: End-to-End Root Cause Attribution\n\n")
        
        f.write("## Taxonomy Summary\n")
        f.write(subsystems[['Root Cause', 'Count', 'Pct Events', 'Total Delay (s)', 'Pct Delay']].to_markdown(index=False))
        f.write("\n\n")
        
        f.write("## Ranked Bottlenecks\n")
        sorted_by_delay = subsystems.sort_values('Total Delay (s)', ascending=False)
        for idx, row in sorted_by_delay.iterrows():
            f.write(f"1. **{row['Root Cause']}** ({row['Pct Delay']:.1f}% of delay, {row['Count']} events)\n")
            
        f.write("\n## Engineering Recommendation\n")
        if not sorted_by_delay.empty:
            top_cause = sorted_by_delay.iloc[0]['Root Cause']
            f.write(f"The primary bottleneck is **{top_cause}**. Fixing this subsystem will produce the largest improvement in real-world hearing-aid performance.\n")

if __name__ == '__main__':
    main()
