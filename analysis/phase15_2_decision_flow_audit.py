import os
import sys
import json
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from decision_policy_engine import DecisionPolicyEngine, State

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--preds', type=str, required=True, help='Path to calibration predictions CSV')
    parser.add_argument('--out', type=str, required=True, help='Output directory')
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    df = pd.read_csv(args.preds)
    
    trajectories = []
    trial_summaries = []
    
    categories = {'EASY': 0, 'SLOW': 0, 'HARD': 0, 'WRONG': 0}
    
    for (subj, trial), group in df.groupby(['subject', 'trial']):
        subj_clean = int(subj) if isinstance(subj, (int, np.integer)) else str(subj)
        trial_clean = int(trial) if isinstance(trial, (int, np.integer)) else str(trial)
        
        group = group.sort_values('window')
        
        engine = DecisionPolicyEngine()
        
        trial_trajectory = []
        
        for _, row in group.iterrows():
            prob = row['prob_platt'] if 'prob_platt' in row else row.get('calibrated_prob', 0.5)
            margin = row.get('margin', 0.0)
            win = int(row['window'])
            
            result = engine.update(prob, margin)
            
            trial_trajectory.append({
                'subject': subj_clean,
                'trial': trial_clean,
                'window': win,
                'time_sec': win,
                'probability': float(prob),
                'margin': float(margin),
                'calibrated_probability': float(prob),
                'sprt_evidence': float(result['evidence']),
                'state': result['state'].name if hasattr(result['state'], 'name') else str(result['state']),
                'decision': result['decision'],
                'locked': 1 if result['state'] == State.LOCKED else 0,
                'accepted': 1 if result['state'] == State.LOCKED else 0,
                'confidence': float(result['confidence'] if 'confidence' in result else 0.5)
            })
            
        true_label = group['ground_truth'].iloc[0] if 'ground_truth' in group else group.get('label', 1).iloc[0]
        
        reached_lock = False
        lock_window = -1
        final_decision = None
        
        for t in trial_trajectory:
            if t['state'] == 'LOCKED' and not reached_lock:
                reached_lock = True
                lock_window = t['window']
                final_decision = t['decision']
        
        if not reached_lock:
            category = 'HARD'
        else:
            if final_decision != true_label:
                category = 'WRONG'
            else:
                if lock_window <= 30:
                    category = 'EASY'
                else:
                    category = 'SLOW'
                    
        categories[category] += 1
        
        for t in trial_trajectory:
            t['category'] = category
            trajectories.append(t)
            
        trial_summaries.append({
            'subject': subj_clean,
            'trial': trial_clean,
            'category': category,
            'latency': lock_window if reached_lock else len(trial_trajectory),
            'reached_lock': reached_lock,
            'correct': 1 if reached_lock and final_decision == true_label else 0,
            'mean_probability': float(group['prob_platt'].mean()) if 'prob_platt' in group else 0.5,
            'median_probability': float(group['prob_platt'].median()) if 'prob_platt' in group else 0.5,
            'mean_margin': float(group['margin'].mean()),
            'median_margin': float(group['margin'].median()),
            'variance_margin': float(group['margin'].var()) if len(group) > 1 else 0.0,
            'final_evidence': float(trial_trajectory[-1]['sprt_evidence']) if len(trial_trajectory) > 0 else 0.0
        })
        
    with open(out_dir / 'decision_flow.jsonl', 'w') as f:
        for t in trajectories:
            f.write(json.dumps(t) + '\n')
            
    sum_df = pd.DataFrame(trial_summaries)
    sum_df.to_csv(out_dir / 'trial_summary.csv', index=False)
    
    traj_df = pd.DataFrame(trajectories)
    
    traj_stats = {}
    temporal_shape = {}
    
    for cat in ['EASY', 'SLOW', 'HARD', 'WRONG']:
        cat_df = traj_df[traj_df['category'] == cat]
        cat_sum = sum_df[sum_df['category'] == cat]
        
        if len(cat_sum) > 0:
            traj_stats[cat] = {
                'mean_probability': float(cat_sum['mean_probability'].mean()),
                'median_probability': float(cat_sum['median_probability'].median()),
                'mean_margin': float(cat_sum['mean_margin'].mean()),
                'median_margin': float(cat_sum['median_margin'].median()),
                'mean_calibrated_confidence': float(cat_df['confidence'].mean()),
                'mean_accumulated_evidence': float(cat_df['sprt_evidence'].mean()),
                'variance': float(cat_df['sprt_evidence'].var()),
                'decision_latency': float(cat_sum['latency'].mean()),
                'average_uncertainty_duration': float(len(cat_df[cat_df['state'] == 'UNCERTAIN']) / len(cat_sum)),
                'average_lock_duration': float(len(cat_df[cat_df['state'] == 'LOCKED']) / len(cat_sum))
            }
            
            slopes = []
            for _, trial_group in cat_df.groupby(['subject', 'trial']):
                y = trial_group['sprt_evidence'].values
                if len(y) > 1:
                    x = np.arange(len(y))
                    slope, _ = np.polyfit(x, y, 1)
                    slopes.append(slope)
            mean_slope = np.mean(slopes) if slopes else 0
            
            shape_var = cat_df.groupby(['subject', 'trial'])['sprt_evidence'].var().mean()
            
            if mean_slope > 0.1:
                shape = "Growing"
            elif mean_slope < -0.1:
                shape = "Decaying"
            elif shape_var > 5.0:
                shape = "Oscillating"
            else:
                shape = "Flat"
                
            temporal_shape[cat] = {
                'average_slope': float(mean_slope),
                'variance': float(shape_var),
                'maximum_evidence': float(cat_df.groupby(['subject', 'trial'])['sprt_evidence'].max().mean()),
                'final_evidence': float(cat_sum['final_evidence'].mean()),
                'shape': shape
            }
            
    with open(out_dir / 'trajectory_statistics.json', 'w') as f:
        json.dump({'statistics': traj_stats, 'temporal_shape': temporal_shape}, f, indent=4)
        
    early_pred = {}
    easy_hard_df = sum_df[sum_df['category'].isin(['EASY', 'HARD'])].copy()
    if len(easy_hard_df) > 0:
        easy_hard_df['target'] = (easy_hard_df['category'] == 'EASY').astype(int)
        
        for time_pt in [10, 20, 30]:
            evidence_at_t = []
            prob_at_t = []
            valid_indices = []
            for idx, row in easy_hard_df.iterrows():
                trial_data = traj_df[(traj_df['subject'] == row['subject']) & (traj_df['trial'] == row['trial'])]
                trial_data = trial_data[trial_data['window'] <= time_pt]
                if len(trial_data) > 0:
                    evidence_at_t.append(trial_data['sprt_evidence'].iloc[-1])
                    prob_at_t.append(trial_data['probability'].mean())
                    valid_indices.append(idx)
                    
            if len(valid_indices) > 0 and len(np.unique(easy_hard_df.loc[valid_indices, 'target'])) > 1:
                targets = easy_hard_df.loc[valid_indices, 'target']
                auroc_ev = roc_auc_score(targets, evidence_at_t)
                auroc_prob = roc_auc_score(targets, prob_at_t)
                early_pred[f'{time_pt}s'] = {
                    'auroc_evidence': float(auroc_ev),
                    'auroc_probability': float(auroc_prob)
                }
                
    report = f"""# Phase 15.2 Decision Flow Audit Report

## 1. Executive Summary
The Decision Flow Audit categorized all trials into EASY, SLOW, HARD, and WRONG to determine why the decision policy succeeds or fails.

## 2. Category Counts
- EASY: {categories['EASY']}
- SLOW: {categories['SLOW']}
- HARD: {categories['HARD']}
- WRONG: {categories['WRONG']}

## 3. Trajectory Comparison
| Category | Mean Prob | Mean Margin | Latency | Avg Uncertainty Duration |
|----------|-----------|-------------|---------|--------------------------|
"""
    for cat, stats in traj_stats.items():
        report += f"| {cat} | {stats['mean_probability']:.4f} | {stats['mean_margin']:.4f} | {stats['decision_latency']:.1f} | {stats['average_uncertainty_duration']:.1f} |\n"

    report += "\n## 4. Evidence Growth Analysis\n"
    for cat, shape in temporal_shape.items():
        report += f"- **{cat}**: Shape={shape['shape']}, Avg Slope={shape['average_slope']:.4f}, Final Evidence={shape['final_evidence']:.4f}\n"
        
    report += "\n## 5. Early Predictability\n"
    report += "Predicting EASY vs HARD within first N seconds:\n"
    for t, vals in early_pred.items():
        report += f"- **{t}**: AUROC (Evidence) = {vals['auroc_evidence']:.4f}, AUROC (Probability) = {vals['auroc_probability']:.4f}\n"
        
    report += "\n## 6. Root Cause Analysis\n"
    if '30s' in early_pred and early_pred['30s']['auroc_evidence'] > 0.75:
        rc = "A. The decoder already shows strong separability early on. The issue lies in the Decision Policy Engine not capitalizing on early evidence."
    else:
        rc = "B. Early predictability is weak. The decoder struggles to separate EASY vs HARD early on, indicating an intrinsic ambiguity or decoder weakness."
        
    report += f"{rc}\n\n"
    
    report += "## 7. Recommended Next Phase\n"
    if 'A' in rc:
        report += "1. Redesign Decision Policy Engine (High Impact)\n2. Tune accumulation thresholds (Medium Impact)"
    else:
        report += "1. Decoder Retraining / Contrastive Loss (High Impact)\n2. Recalibrate Model (Medium Impact)"

    with open(out_dir / 'decision_flow_report.md', 'w') as f:
        f.write(report)
        
    print(f"Trials processed\t{sum(categories.values())}")
    print(f"Easy Trials\t{categories['EASY']}")
    print(f"Slow Trials\t{categories['SLOW']}")
    print(f"Hard Trials\t{categories['HARD']}")
    print(f"Wrong Trials\t{categories['WRONG']}")
    print(f"Output files\t{out_dir}")
    print("Done.")

if __name__ == '__main__':
    main()
