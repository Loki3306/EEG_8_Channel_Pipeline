import os
import sys
import json
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm
from scipy import stats

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from decision_policy_engine import DecisionPolicyEngine, State

class ContinuousSimulator:
    def __init__(self, mode='fixed', early_threshold_map=None):
        self.mode = mode
        # These heuristic thresholds were strictly predefined from Phase 15.4.1 prior knowledge,
        # they are NOT fit on the evaluation dataset.
        self.early_threshold_map = early_threshold_map or {
            'EASY': 0.70,
            'MEDIUM': 0.85,
            'HARD': 0.95
        }
    
    def run_simulation(self, df):
        rng = np.random.RandomState(42)
        
        global_metrics = {
            'total_trials': 0,
            'total_windows': 0,
            'switches': 0,
            'rejects': 0,
            'oscillations': 0,
            'false_locks': 0,
            'wrong_switches': 0,
            'lock_latencies': [],
            'lock_durations': [],
            'uncertainty_durations': [],
            'state_occupancy': {s: 0 for s in [State.INITIALIZING, State.WAITING, State.LOCKED, State.SWITCHING, State.UNCERTAIN]}
        }
        
        state_trace = []
        decision_log = []
        
        required_cols = ['subject', 'trial', 'window', 'margin']
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Dataset missing required column: {col}")
                
        if 'prob_platt' not in df.columns and 'calibrated_prob' not in df.columns:
            raise ValueError("Dataset missing probability column (prob_platt or calibrated_prob)")
            
        if 'ground_truth' not in df.columns and 'label' not in df.columns:
            raise ValueError("Dataset missing label column (ground_truth or label)")
        
        for (subj, trial), group in tqdm(df.groupby(['subject', 'trial'])):
            group = group.sort_values('window')
            
            if 'ground_truth' in group:
                true_label = int(group['ground_truth'].iloc[0])
            else:
                true_label = int(group['label'].iloc[0])
                
            # Verify dataset label semantics (0=Right, 1=Left)
            if true_label not in [0, 1]:
                raise ValueError("Ground truth labels must be binary 0 or 1.")
                
            engine = DecisionPolicyEngine(confidence_threshold=self.early_threshold_map['MEDIUM'])
            
            trial_id = f"{subj}_{trial}"
            global_metrics['total_trials'] += 1
            
            prob_buffer = []
            difficulty_locked = False
            pre_random_diff = rng.choice(['EASY', 'MEDIUM', 'HARD'])
            
            for _, row in group.iterrows():
                global_metrics['total_windows'] += 1
                
                prob = float(row['prob_platt'] if 'prob_platt' in row else row['calibrated_prob'])
                margin = float(row['margin'])
                win = int(row['window'])
                
                if self.mode == 'adaptive':
                    if not difficulty_locked:
                        prob_buffer.append(prob)
                        if len(prob_buffer) >= 5: 
                            # Pre-calibrated heuristics applied online without look-ahead
                            prob_mean = np.mean(prob_buffer)
                            if prob_mean > 0.65: diff = 'EASY'
                            elif prob_mean < 0.55: diff = 'HARD'
                            else: diff = 'MEDIUM'
                            engine.config['confidence_threshold'] = self.early_threshold_map[diff]
                            difficulty_locked = True
                elif self.mode == 'random':
                    engine.config['confidence_threshold'] = self.early_threshold_map[pre_random_diff]
                elif self.mode == 'time_decay':
                    decay_conf = max(0.70, 0.95 - (win / 30.0) * 0.25)
                    engine.config['confidence_threshold'] = decay_conf
                elif self.mode == 'aggressive':
                    engine.config['confidence_threshold'] = self.early_threshold_map['EASY']
                elif self.mode == 'conservative':
                    engine.config['confidence_threshold'] = self.early_threshold_map['HARD']
                elif self.mode == 'fixed':
                    pass 
                
                res = engine.update(prob, margin)
                
                # Directly track absolute state occupancy to avoid percentage reconstruction assumptions
                global_metrics['state_occupancy'][res['state']] += 1
                
                state_trace.append({
                    'trial_id': trial_id,
                    'window': win,
                    'probability': prob,
                    'margin': margin,
                    'evidence': res['evidence'],
                    'confidence': res['confidence'],
                    'state': res['state'],
                    'decision': res['decision'],
                    'threshold_used': engine.config['confidence_threshold']
                })
                
                if res['action'] in ["SWITCH_LEFT", "SWITCH_RIGHT"] and res['decision'] is not None:
                    dec_val = int(res['decision'])
                    # Verify DecisionPolicyEngine semantics (Decision 1 = LEFT, Decision 0 = RIGHT)
                    if dec_val not in [0, 1]:
                        raise RuntimeError(f"Engine emitted non-binary decision: {dec_val}")
                        
                    is_corr = (dec_val == true_label)
                    
                    decision_log.append({
                        'trial_id': trial_id,
                        'window': win,
                        'decision': dec_val,
                        'true_label': true_label,
                        'is_correct': is_corr,
                        'confidence_threshold': engine.config['confidence_threshold']
                    })
                    
                    if not is_corr:
                        global_metrics['wrong_switches'] += 1
                        global_metrics['false_locks'] += 1
            
            stat = engine.statistics()
            global_metrics['switches'] += stat['switches']
            global_metrics['rejects'] += stat['rejects']
            global_metrics['oscillations'] += stat['oscillations']
            
            adaptation_penalty = 5 if self.mode == 'adaptive' else 0
            
            if stat['latency'] is not None:
                global_metrics['lock_latencies'].append(stat['latency'] + adaptation_penalty)
            if stat['avg_lock_duration'] > 0:
                global_metrics['lock_durations'].append(stat['avg_lock_duration'])
            if stat['avg_uncertain_duration'] > 0:
                global_metrics['uncertainty_durations'].append(stat['avg_uncertain_duration'])
                
        results = {
            'mode': self.mode,
            'Total Trials': global_metrics['total_trials'],
            'Total Windows': global_metrics['total_windows'],
            'Average Lock Latency': np.mean(global_metrics['lock_latencies']) if global_metrics['lock_latencies'] else 0,
            'Average Lock Duration': np.mean(global_metrics['lock_durations']) if global_metrics['lock_durations'] else 0,
            'Average Uncertainty Duration': np.mean(global_metrics['uncertainty_durations']) if global_metrics['uncertainty_durations'] else 0,
            'Total Switches': global_metrics['switches'],
            'Wrong Switches': global_metrics['wrong_switches'],
            'Total Rejects': global_metrics['rejects'],
            'Total Oscillations': global_metrics['oscillations'],
            'raw_latencies': global_metrics['lock_latencies'],
            'state_occupancy': global_metrics['state_occupancy']
        }
        
        return results, state_trace, decision_log

def write_phase16A_outputs(out_dir, results, trace, dec):
    p16a_dir = out_dir / 'phase16A'
    p16a_dir.mkdir(parents=True, exist_ok=True)
    
    with open(p16a_dir / 'state_trace.jsonl', 'w') as f:
        for t in trace: f.write(json.dumps(t) + '\n')
        
    with open(p16a_dir / 'decision_log.jsonl', 'w') as f:
        for d in dec: f.write(json.dumps(d) + '\n')
        
    pd.DataFrame(trace).to_csv(p16a_dir / 'timeline.csv', index=False)
    
    res_dump = {k:v for k,v in results.items() if k not in ['raw_latencies', 'state_occupancy']}
    with open(p16a_dir / 'metrics.json', 'w') as f:
        json.dump(res_dump, f, indent=4)
        
    total_steps = sum(results['state_occupancy'].values())
    state_df = pd.DataFrame([{
        'State': k, 
        'Avg Occupancy Pct': v / max(1, total_steps)
    } for k, v in results['state_occupancy'].items()])
    state_df.to_csv(p16a_dir / 'state_statistics.csv', index=False)
    
    with open(p16a_dir / 'phase16A_report.md', 'w') as f:
        f.write("# Phase 16A: Continuous Hearing Aid Simulator (Baseline)\n\n")
        f.write("## Overview\n")
        f.write("Simulated continuous EEG stream processing through the fixed-threshold SPRT engine.\n\n")
        f.write("## Baseline Metrics\n")
        f.write(f"- Average Lock Latency: {results['Average Lock Latency']:.2f} windows\n")
        f.write(f"- Average Lock Duration: {results['Average Lock Duration']:.2f} windows\n")
        f.write(f"- Total Switches: {results['Total Switches']}\n")
        f.write(f"- Wrong Switches (False Locks): {results['Wrong Switches']}\n")
        f.write(f"- Total Rejects: {results['Total Rejects']}\n")
        f.write(f"- Oscillations: {results['Total Oscillations']}\n")

def write_phase16B_outputs(out_dir, res_16a, res_16b, trace_16b, dec_16b, ab_df):
    p16b_dir = out_dir / 'phase16B'
    p16b_dir.mkdir(parents=True, exist_ok=True)
    
    with open(p16b_dir / 'adaptive_state_trace.jsonl', 'w') as f:
        for t in trace_16b: f.write(json.dumps(t) + '\n')
        
    with open(p16b_dir / 'adaptive_decision_log.jsonl', 'w') as f:
        for d in dec_16b: f.write(json.dumps(d) + '\n')
        
    res_a_clean = {k:v for k,v in res_16a.items() if k not in ['raw_latencies', 'state_occupancy']}
    res_b_clean = {k:v for k,v in res_16b.items() if k not in ['raw_latencies', 'state_occupancy']}
    comp_df = pd.DataFrame([res_a_clean, res_b_clean])
    comp_df.to_csv(p16b_dir / 'comparison_metrics.csv', index=False)
    
    ab_clean = []
    for r in ab_df.to_dict('records'):
        ab_clean.append({k:v for k,v in r.items() if k not in ['raw_latencies', 'state_occupancy']})
    pd.DataFrame(ab_clean).to_csv(p16b_dir / 'ablation_results.csv', index=False)
    
    lat_a = res_16a['Average Lock Latency']
    lat_b = res_16b['Average Lock Latency']
    err_a = res_16a['Wrong Switches']
    err_b = res_16b['Wrong Switches']
    
    # Non-parametric Wilcoxon signed-rank test for paired latency differences
    lat_a_arr = np.array(res_16a['raw_latencies'])
    lat_b_arr = np.array(res_16b['raw_latencies'])
    min_len = min(len(lat_a_arr), len(lat_b_arr))
    
    if min_len > 1:
        diff = lat_a_arr[:min_len] - lat_b_arr[:min_len]
        if np.all(diff == 0):
            p_val = 1.0
            stat_sig = False
        else:
            res_w = stats.wilcoxon(lat_a_arr[:min_len], lat_b_arr[:min_len])
            p_val = res_w.pvalue
            stat_sig = p_val < 0.05
    else:
        p_val = 1.0
        stat_sig = False
    
    with open(p16b_dir / 'phase16B_report.md', 'w') as f:
        f.write("# Phase 16B: Adaptive Decision Controller\n\n")
        f.write("## Architecture & Finite State Machine\n")
        f.write("The controller uses the `DecisionPolicyEngine` SPRT engine with states: INITIALIZING, WAITING, LOCKED, SWITCHING, UNCERTAIN.\n")
        f.write("In Phase 16B, the Early Difficulty Predictor maps trials to EASY, MEDIUM, or HARD based on the first 5 windows of `prob_mean`. During these first 5 windows, the threshold remains fixed (no look-ahead leakage).\n")
        f.write("The threshold is dynamically adjusted: 0.70 (EASY), 0.85 (MEDIUM), 0.95 (HARD). These thresholds are prior domain heuristics derived from Phase 15.4.1.\n\n")
        
        f.write("## A/B Comparison\n")
        f.write(comp_df[['mode', 'Average Lock Latency', 'Wrong Switches', 'Total Rejects', 'Total Oscillations']].to_markdown(index=False) + "\n\n")
        
        f.write("## Ablation Study\n")
        f.write(pd.DataFrame(ab_clean)[['mode', 'Average Lock Latency', 'Wrong Switches', 'Total Rejects']].to_markdown(index=False) + "\n\n")
        
        f.write("## Engineering Discussion\n")
        f.write("**DISCLAIMER: The heuristic difficulty predictor was configured using Phase 15 priors. While applied causally online, performance claims remain exploratory until fully validated on a held-out dataset.**\n\n")
        
        if lat_b < lat_a:
            f.write(f"The adaptive controller successfully lowered latency from {lat_a:.2f} to {lat_b:.2f} windows. ")
            if stat_sig:
                f.write(f"This difference is statistically significant (Wilcoxon signed-rank test, p={p_val:.4f}). ")
            else:
                f.write(f"However, this difference was not statistically significant (p={p_val:.4f}). ")
        else:
            f.write(f"The adaptive controller did NOT improve latency (Base: {lat_a:.2f}, Adapt: {lat_b:.2f}, p={p_val:.4f}). ")
            
        if err_b <= err_a:
            f.write(f"This was achieved without increasing false locks (Base errors: {err_a}, Adapt errors: {err_b}).\n")
        else:
            f.write(f"However, this came at the cost of increased false locks (Base errors: {err_a}, Adapt errors: {err_b}).\n")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--preds', type=str, required=True)
    parser.add_argument('--out', type=str, required=True)
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.preds)
    
    print("---------------------------------------")
    
    sim_16a = ContinuousSimulator(mode='fixed')
    res_16a, trace_16a, dec_16a = sim_16a.run_simulation(df)
    write_phase16A_outputs(out_dir, res_16a, trace_16a, dec_16a)
    print("Phase 16A ........ COMPLETE")
    
    sim_16b = ContinuousSimulator(mode='adaptive')
    res_16b, trace_16b, dec_16b = sim_16b.run_simulation(df)
    print("Phase 16B ........ COMPLETE")
    
    ablations = ['fixed', 'adaptive', 'time_decay', 'random', 'conservative', 'aggressive']
    ablation_res = []
    
    for mode in ablations:
        if mode == 'fixed':
            ablation_res.append(res_16a)
        elif mode == 'adaptive':
            ablation_res.append(res_16b)
        else:
            sim = ContinuousSimulator(mode=mode)
            res, _, _ = sim.run_simulation(df)
            ablation_res.append(res)
            
    ab_df = pd.DataFrame(ablation_res)
    write_phase16B_outputs(out_dir, res_16a, res_16b, trace_16b, dec_16b, ab_df)
    
    print("Baseline Metrics")
    print("Adaptive Metrics")
    print(f"Files Written")
    print("Done")
    print("---------------------------------------")

if __name__ == '__main__':
    main()
