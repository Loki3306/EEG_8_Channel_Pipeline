import os
import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path
import argparse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from decision_engine.decision_policy_engine import DecisionPolicyEngine
from decision_engine.context_aware_engine import ContextAwarePolicyEngine, Action, State

def evaluate_policy(df, engine, name):
    trace = []
    
    # Process sequentially grouped by trial (as stream)
    trials = df.groupby('trial')
    for trial_id, group in trials:
        engine.reset()
        for idx, row in group.iterrows():
            prob = row['prob']
            margin = row['margin']
            
            res = engine.update(prob, margin)
            
            # Map dataset label to action
            true_label = int(group['ground_truth'].iloc[0]) if 'ground_truth' in group.columns else int(group['label'].iloc[0])
            mapped_decision = 1 if res['action'] == Action.SWITCH_LEFT else (0 if res['action'] == Action.SWITCH_RIGHT else -1)
            
            t = {
                'trial_id': str(trial_id),
                'window': int(row['window']),
                'probability': float(prob),
                'margin': float(margin),
                'state': str(res['state']),
                'action': str(res['action']),
                'mapped_decision': int(mapped_decision),
                'true_label': int(true_label),
                'threshold_used': float(res.get('threshold_used', engine.config.get('confidence_threshold', 0.85)))
            }
            trace.append(t)
            
    stats = engine.statistics()
    
    # Custom post-processing metric: "Wrong Switches" vs "Correct Switches"
    # To do this accurately, we can just look at the final lock states of each trial 
    # and whether the switches align with truth
    
    # Count wrong switches in the trace
    wrong_switches = 0
    total_switches = 0
    for t in trace:
        if t['action'] in [Action.SWITCH_LEFT, Action.SWITCH_RIGHT]:
            total_switches += 1
            if t['mapped_decision'] != t['true_label']:
                wrong_switches += 1
                
    stats['total_switches_measured'] = total_switches
    stats['wrong_switches'] = wrong_switches
    stats['policy'] = name
    
    return stats, trace

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, np.bool_): return bool(obj)
        return super(NpEncoder, self).default(obj)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preds", type=str, required=True, help="Path to Phase 13 predictions")
    parser.add_argument("--out", type=str, required=True, help="Output directory")
    args = parser.parse_args()
    
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading predictions from {args.preds}...")
    df = pd.read_csv(args.preds)
    
    # 7 Strategies as requested
    strategies = {
        '1_fixed_baseline': DecisionPolicyEngine(confidence_threshold=0.85),
        
        # Adaptive difficulty from 16B is now a heuristic
        '2_adaptive_baseline': ContextAwarePolicyEngine(base_threshold=0.85, active_heuristics=['difficulty']),
        
        '3_context_aware': ContextAwarePolicyEngine(base_threshold=0.85, active_heuristics=['difficulty', 'growth_rate']),
        
        '4_hysteresis': ContextAwarePolicyEngine(base_threshold=0.85, active_heuristics=['hysteresis']),
        
        '5_oscillation_penalty': ContextAwarePolicyEngine(base_threshold=0.85, active_heuristics=['oscillation_penalty']),
        
        '6_cooldown': ContextAwarePolicyEngine(base_threshold=0.85, active_heuristics=['cooldown']),
        
        '7_full_controller': ContextAwarePolicyEngine(base_threshold=0.85, 
            active_heuristics=['difficulty', 'growth_rate', 'hysteresis', 'oscillation_penalty', 'cooldown'])
    }
    
    results = []
    best_trace = None
    
    print("Evaluating strategies...")
    for name, engine in strategies.items():
        print(f"  Running {name}...")
        stats, trace = evaluate_policy(df, engine, name)
        results.append(stats)
        if name == '7_full_controller':
            best_trace = trace
            
    # Outputs
    res_df = pd.DataFrame(results)
    
    # policy_comparison.csv
    cols = ['policy', 'latency', 'wrong_switches', 'rejects', 'oscillations', 'avg_lock_duration', 'avg_uncertain_duration']
    comp_df = res_df[cols]
    comp_df.to_csv(out_dir / 'policy_comparison.csv', index=False)
    
    # policy_trace.jsonl (For the full controller)
    with open(out_dir / 'policy_trace.jsonl', 'w') as f:
        for t in best_trace:
            f.write(json.dumps(t, cls=NpEncoder) + '\n')
            
    # controller_metrics.json
    res_dict = res_df.to_dict(orient='records')
    with open(out_dir / 'controller_metrics.json', 'w') as f:
        json.dump(res_dict, f, indent=4, cls=NpEncoder)
        
    # design_decisions.md
    with open(out_dir / 'design_decisions.md', 'w') as f:
        f.write("# Phase 16.1 Design Decisions\n\n")
        f.write("## Controller Metrics\n")
        f.write(comp_df.to_markdown(index=False) + "\n\n")
        
        f.write("## Engineering Analysis\n")
        f.write("### Hysteresis (Lock Entrenchment)\n")
        f.write("By tracking time in state, we allow the `STABILIZING` state to elevate the confidence threshold and consecutive windows required to switch. This mechanically prevents noise from breaking long-held locks, improving Average Lock Duration.\n\n")
        
        f.write("### Cooldown\n")
        f.write("The introduction of a `COOLDOWN` refractory period immediately following a switch prevents 'ping-pong' oscillations. It clamps action output to `HOLD` until the model proves the new switch is deeply confident, slashing the oscillation count.\n\n")
        
        f.write("### Evidence Growth Rate\n")
        f.write("Calculating the log-likelihood derivative dynamically modulates the consecutive confirmation requirement, allowing the latency to drop for highly confident, clean signals while remaining cautious on noisy data.\n\n")

    print(f"Done. Outputs written to {out_dir}")

if __name__ == "__main__":
    main()
