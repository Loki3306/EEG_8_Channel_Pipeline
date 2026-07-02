import pandas as pd
import numpy as np
import os
import sys
import json
import matplotlib.pyplot as plt
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(REPO_ROOT))

try:
    from decision_engine.context_aware_engine import ContextAwarePolicyEngine
except ImportError:
    print("WARNING: Could not import ContextAwarePolicyEngine.")
    ContextAwarePolicyEngine = None

def find_prediction_files():
    search_paths = [
        REPO_ROOT / "results" / "phase17_1" / "scenario_streams",
        REPO_ROOT / "EEG_8_Channel_Pipeline" / "results" / "phase17_1" / "scenario_streams",
        Path("/kaggle/working/EEG_8_Channel_Pipeline/results/phase17_1/scenario_streams"),
        Path("/kaggle/working/EEG_8_Channel_Pipeline/EEG_8_Channel_Pipeline/results/phase17_1/scenario_streams")
    ]
    for path in search_paths:
        if path.exists():
            files = list(path.glob("*predictions.csv"))
            if len(files) > 0:
                print(f"Found {len(files)} prediction files in {path}")
                return files
    print("ERROR: Could not find Phase 17.1 predictions.csv files.")
    return []

def extract_splices(df):
    splices = []
    current_scene = None
    for idx, row in df.iterrows():
        if row['scene_name'] != current_scene:
            if current_scene is not None:
                splices.append({
                    'timestamp_sec': row['timestamp_sec'],
                    'old_gt': current_scene,
                    'new_gt': int(row['ground_truth'])
                })
            current_scene = row['scene_name']
    return splices

def run_forensics(files, out_dir):
    all_timelines = []
    delay_budgets = []
    reset_events = []
    
    # Store aggregated counts for the final budget
    budget_counts = {
        'Weak Evidence': 0,
        'Evidence Reset': 0,
        'Threshold Waiting': 0,
        'Evidence Integration': 0,
        'Cooldown / Hysteresis': 0,
        'Release Delay': 0
    }
    
    total_delayed_switches = 0
    total_latencies_s = 0.0
    
    for f in files:
        df = pd.read_csv(f)
        scenario = f.stem
        splices = extract_splices(df)
        
        engine = ContextAwarePolicyEngine(base_threshold=0.85, 
            active_heuristics=['difficulty', 'growth_rate', 'hysteresis', 'oscillation_penalty', 'cooldown'])
        
        trace = []
        for idx, row in df.iterrows():
            p = row['prob']
            m = row['margin']
            
            p_clip = np.clip(p, 1e-5, 1 - 1e-5)
            llr = np.log(p_clip / (1 - p_clip))
            
            old_evidence = engine.evidence
            res = engine.update(p, m)
            
            # Record Reset
            if engine.evidence < old_evidence - 1.0 or (old_evidence > 0 and engine.evidence <= 0):
                reset_events.append({
                    'scenario': scenario,
                    'timestamp_sec': row['timestamp_sec'],
                    'old_evidence': old_evidence,
                    'new_evidence': engine.evidence,
                    'state': res['state'],
                    'margin': m
                })
            
            trace.append({
                'scenario': scenario,
                'timestamp_sec': row['timestamp_sec'],
                'window_idx': idx,
                'ground_truth': int(row['ground_truth']),
                'margin': m,
                'prob': p,
                'llr': llr,
                'cumulative_evidence': res['evidence'],
                'confidence': res['confidence'],
                'active_threshold': res['threshold_used'],
                'state': res['state'],
                'decision': res['decision'],
                'active_lock': 1 if res['action'] == 'SWITCH_LEFT' else (0 if res['action'] == 'SWITCH_RIGHT' else None)
            })
            
        trace_df = pd.DataFrame(trace)
        
        # Carry forward the last known lock to know what the current active lock is in every frame
        trace_df['active_lock'] = trace_df['active_lock'].ffill()
        
        all_timelines.append(trace_df)
        
        time_delta = trace_df['timestamp_sec'].diff().median() if len(trace_df) > 1 else 0.5
        
        # Calculate Delay Budget for this scenario
        for sp in splices:
            ts = sp['timestamp_sec']
            tgt = sp['new_gt']
            
            # Find lock time
            post_splice = trace_df[trace_df['timestamp_sec'] >= ts]
            lock_row = post_splice[post_splice['active_lock'] == tgt]
            if lock_row.empty: continue
            
            lock_ts = lock_row.iloc[0]['timestamp_sec']
            latency = lock_ts - ts
            if latency <= 0: continue
            
            total_delayed_switches += 1
            total_latencies_s += latency
            
            # Analyze frames in the delay period
            delay_frames = post_splice[post_splice['timestamp_sec'] < lock_ts]
            
            for _, frame in delay_frames.iterrows():
                st = frame['state']
                cf = frame['confidence']
                th = frame['active_threshold']
                m = frame['margin']
                
                # Check target direction. If tgt=1, m should be > 0. If tgt=0, m should be < 0.
                margin_correct_dir = (m > 0) if tgt == 1 else (m < 0)
                
                if st in ['COOLDOWN', 'STABILIZING']:
                    budget_counts['Cooldown / Hysteresis'] += 1
                elif st == 'UNCERTAIN' or frame['active_lock'] == (1 - tgt):
                    budget_counts['Release Delay'] += 1
                else:
                    if not margin_correct_dir:
                        budget_counts['Evidence Reset'] += 1
                    elif abs(m) < 0.05:
                        budget_counts['Weak Evidence'] += 1
                    elif cf >= 0.85 and cf < th:
                        budget_counts['Threshold Waiting'] += 1
                    else:
                        budget_counts['Evidence Integration'] += 1

    full_timeline = pd.concat(all_timelines, ignore_index=True)
    full_timeline.to_csv(out_dir / "evidence_timelines.csv", index=False)
    
    pd.DataFrame(reset_events).to_csv(out_dir / "reset_events.csv", index=False)
    
    # Save the budget
    total_budget_frames = sum(budget_counts.values())
    budget_df = pd.DataFrame([{
        'Category': k,
        'Frames': v,
        'Time (s)': v * 0.5, # approx
        'Percentage': (v / total_budget_frames * 100) if total_budget_frames > 0 else 0
    } for k, v in budget_counts.items()])
    
    budget_df = budget_df.sort_values('Percentage', ascending=False)
    budget_df.to_csv(out_dir / "delay_budget.csv", index=False)
    
    return full_timeline, budget_df, total_delayed_switches, total_latencies_s

def audit_sprt(full_timeline, out_dir):
    margins = full_timeline['margin'].values
    
    # Calculate autocorrelations
    mean_m = np.mean(margins)
    var_m = np.var(margins)
    if var_m == 0: var_m = 1
    
    def autocorr(x, lag=1):
        return np.sum((x[:-lag] - mean_m) * (x[lag:] - mean_m)) / (len(x) - lag) / var_m
        
    lag1 = autocorr(margins, 1)
    lag5 = autocorr(margins, 5)
    
    with open(out_dir / "sprt_assumption_audit.md", "w") as f:
        f.write("# SPRT Assumption Audit\n\n")
        f.write("Theoretical SPRT assumes that each incoming log-likelihood ratio (evidence update) is independently sampled from the underlying distribution. This allows evidence to grow linearly and variance to drop efficiently.\n\n")
        
        f.write(f"**Empirical Lag-1 Autocorrelation (next window):** {lag1:.3f}\n")
        f.write(f"**Empirical Lag-5 Autocorrelation (2.5s later):** {lag5:.3f}\n\n")
        
        if lag1 > 0.5:
            f.write("### Conclusion: ASSUMPTION VIOLATED\n")
            f.write("Because the EEG windows heavily overlap (typically 64Hz features spanning a 2-second window with high overlap), the margins are highly correlated. ")
            f.write("This means when the signal is weak or wrong, it *stays* weak or wrong for multiple consecutive frames, causing massive evidence collapses or plateaus that SPRT does not expect.\n")
        else:
            f.write("### Conclusion: ASSUMPTION HOLDS\n")

def main():
    print("====================================================")
    print("PHASE 20: TEMPORAL EVIDENCE DYNAMICS & DELAY FORENSICS")
    print("====================================================")
    
    files = find_prediction_files()
    if not files: return
    
    out_dir = REPO_ROOT / "results" / "phase20"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print("Running Forensics (Timeline, Resets, Budget)...")
    full_timeline, budget_df, total_switches, total_latency = run_forensics(files, out_dir)
    
    print("Auditing SPRT Assumptions...")
    audit_sprt(full_timeline, out_dir)
    
    # Generate final report
    print("Generating Final Report...")
    with open(out_dir / "phase20_report.md", "w") as f:
        f.write("# Phase 20: Temporal Delay Budget Forensics\n\n")
        
        f.write(f"**Total Analyzed Delayed Switches:** {total_switches}\n")
        f.write(f"**Total Accumulated Latency (s):** {total_latency:.1f}\n\n")
        
        f.write("## The Delay Budget (Where did the seconds go?)\n")
        f.write(budget_df.to_markdown(index=False))
        f.write("\n\n")
        
        largest_contributor = budget_df.iloc[0]['Category']
        largest_pct = budget_df.iloc[0]['Percentage']
        
        f.write(f"## Largest Contributor\n")
        f.write(f"**{largest_contributor}** ({largest_pct:.1f}%)\n\n")
        
        f.write("## Engineering Recommendation\n")
        if largest_contributor == "Evidence Reset":
            f.write("Evidence actively drops during the waiting period due to highly correlated negative noise (SPRT violation). You must fix **Evidence Integration (e.g. replacing it with a heavily smoothed moving average or median filter)** rather than traditional accumulation.\n")
        elif largest_contributor == "Release Delay":
            f.write("The system takes too long to realize the old speaker is gone. You must aggressively tune the **Release Logic (e.g. fast decay on negative evidence)**.\n")
        elif largest_contributor == "Cooldown / Hysteresis":
            f.write("The system artificially blocks itself. You should remove or heavily shorten the **Cooldown/Hysteresis parameters** in the policy engine.\n")
        elif largest_contributor == "Threshold Waiting":
            f.write("The dynamic threshold is outrunning the evidence. You must relax the **Adaptive Difficulty Scaling**.\n")
        else:
            f.write("The fundamental margin is simply too weak to integrate quickly. You must improve the **Decoder or Calibration**.\n")
            
    print(f"Done! Phase 20 artifacts generated in {out_dir}")

if __name__ == '__main__':
    main()
