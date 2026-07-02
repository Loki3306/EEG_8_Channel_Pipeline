import pandas as pd
import numpy as np
import os
import sys
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

def run_release_dissection(files, out_dir):
    budget_counts = {
        'Insufficient Negative Evidence': 0,
        'Minimum Lock Constraint': 0,
        'Consecutive Confirmation Constraint': 0,
        'Switch Gap Constraint': 0,
        'Cooldown Constraint': 0,
        'Unknown Blockage': 0
    }
    budget_time = {k: 0.0 for k in budget_counts.keys()}
    
    peak_evidences = []
    
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
            
            res = engine.update(p, m)
            
            trace.append({
                'timestamp_sec': row['timestamp_sec'],
                'window_idx': idx, 'ground_truth': int(row['ground_truth']),
                'cumulative_evidence': res['evidence'], 'confidence': res['confidence'],
                'active_threshold': res['threshold_used'], 'state': res['state'],
                'decision': res['decision'], 'time_in_state': engine.time_in_state,
                'is_uncertain': (0.5 - engine.config['uncertainty_threshold']) <= res['confidence'] <= (0.5 + engine.config['uncertainty_threshold']),
                'active_lock': 1 if res['action'] == 'SWITCH_LEFT' else (0 if res['action'] == 'SWITCH_RIGHT' else None),
                'candidate': engine.last_candidate,
                'consecutive_count': engine.consecutive_agreement_count,
                'active_consecutive': res.get('consecutive_used', engine.config['minimum_consecutive_windows']),
                'switch_gap_ok': (engine.window_index - engine.last_switch_time) >= engine.config['minimum_switch_gap']
            })
            
        trace_df = pd.DataFrame(trace)
        trace_df['active_lock'] = trace_df['active_lock'].ffill()
        
        time_delta = trace_df['timestamp_sec'].diff().median() if len(trace_df) > 1 else 0.0625
        
        for sp in splices:
            ts = sp['timestamp_sec']
            tgt = sp['new_gt']
            
            # Find the peak evidence just before the switch
            pre_splice = trace_df[trace_df['timestamp_sec'] < ts]
            if not pre_splice.empty:
                peak_ev = pre_splice.iloc[-1]['cumulative_evidence']
                peak_evidences.append(abs(peak_ev)) # Absolute value for peak confidence magnitude
                
            post_splice = trace_df[trace_df['timestamp_sec'] >= ts]
            lock_row = post_splice[post_splice['active_lock'] == tgt]
            if lock_row.empty: continue
            
            lock_ts = lock_row.iloc[0]['timestamp_sec']
            delay_frames = post_splice[post_splice['timestamp_sec'] < lock_ts]
            
            for _, frame in delay_frames.iterrows():
                st = frame['state']
                
                # Check if it was in a hold state (Release Delay)
                is_releasing = (st in ['LOCKED', 'STABILIZING'] and frame['decision'] == (1 - tgt)) or st == 'COOLDOWN'
                
                if not is_releasing:
                    continue # Not a release delay block
                    
                cat = 'Unknown Blockage'
                
                if st == 'COOLDOWN':
                    cat = 'Cooldown Constraint'
                elif st == 'LOCKED' and frame['time_in_state'] < engine.config['minimum_lock_duration']:
                    cat = 'Minimum Lock Constraint'
                else:
                    if frame['is_uncertain'] == False and (frame['candidate'] is None or frame['candidate'] == frame['decision']):
                        # Engine is not uncertain, and candidate is either None or still the old decision
                        # because evidence hasn't decayed yet.
                        cat = 'Insufficient Negative Evidence'
                    elif frame['candidate'] is not None and frame['candidate'] != frame['decision']:
                        if frame['consecutive_count'] < frame['active_consecutive']:
                            cat = 'Consecutive Confirmation Constraint'
                        elif not frame['switch_gap_ok']:
                            cat = 'Switch Gap Constraint'
                        else:
                            # It should have switched!
                            cat = 'Unknown Blockage'
                    else:
                        cat = 'Insufficient Negative Evidence'
                        
                budget_counts[cat] += 1
                budget_time[cat] += time_delta

    total_budget_time = sum(budget_time.values())
    budget_df = pd.DataFrame([{
        'Condition Preventing Release': k,
        'Frames': budget_counts[k],
        'Time (s)': budget_time[k],
        'Percentage': (budget_time[k] / total_budget_time * 100) if total_budget_time > 0 else 0
    } for k in budget_counts.keys()])
    
    budget_df = budget_df.sort_values('Percentage', ascending=False)
    budget_df.to_csv(out_dir / "release_dissection_budget.csv", index=False)
    
    return budget_df, np.mean(peak_evidences) if peak_evidences else 0.0

def main():
    print("====================================================")
    print("PHASE 21: RELEASE LOGIC DISSECTION")
    print("====================================================")
    
    files = find_prediction_files()
    if not files: return
    
    out_dir = REPO_ROOT / "results" / "phase21"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print("Running Forensic Dissection...")
    budget_df, mean_peak_ev = run_release_dissection(files, out_dir)
    
    print("Generating Final Report...")
    with open(out_dir / "phase21_report.md", "w") as f:
        f.write("# Phase 21: Release Logic Dissection\n\n")
        
        f.write(f"**Mean Peak Evidence (LLR) before switch:** {mean_peak_ev:.1f}\n\n")
        f.write("*(Note: A peak evidence of >100 means it takes mathematically hundreds of consecutive negative windows to decay back to uncertainty)*\n\n")
        
        f.write("## The Release Blockage Budget\n")
        f.write("When the controller delayed releasing the old speaker, exactly which `if` statement blocked it?\n\n")
        f.write(budget_df.to_markdown(index=False))
        f.write("\n\n")
        
        largest = budget_df.iloc[0]['Condition Preventing Release']
        
        f.write(f"## Engineering Conclusion\n")
        if largest == "Insufficient Negative Evidence":
            f.write("The controller wanted to release, but the `confidence` never dropped to the `is_uncertain` band. ")
            f.write("Because we use unbounded accumulation (SPRT), the evidence climbs to astronomical numbers during the speaker's turn. ")
            f.write("When the ground truth switches, the weak negative margins subtract from a massive mountain of old evidence, taking tens of seconds just to cross 0.\n\n")
            f.write("**Mandatory Fix:** Implement Asymmetric Evidence Decay. For example, cap the maximum evidence, or apply an exponential decay factor when evidence contradicts the current state.\n")
        elif largest == "Minimum Lock Constraint":
            f.write("The hardcoded `minimum_lock_duration` is artificially forcing the system to hold incorrect locks for too long.\n")
        elif largest == "Cooldown Constraint":
            f.write("The `COOLDOWN` heuristic is blinding the system to valid counter-evidence.\n")
        else:
            f.write("The release logic is hindered by secondary heuristic constraints.\n")

    print(f"Done! Phase 21 artifacts generated in {out_dir}")

if __name__ == '__main__':
    main()
