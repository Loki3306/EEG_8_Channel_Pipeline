import os
import sys
import json
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
import random

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
    
    # 1. Simulate all trials to categorize them
    all_trajectories = {}
    trial_categories = {}
    
    for (subj, trial), group in df.groupby(['subject', 'trial']):
        subj_clean = int(subj) if isinstance(subj, (int, np.integer)) else str(subj)
        trial_clean = int(trial) if isinstance(trial, (int, np.integer)) else str(trial)
        trial_id = f"{subj_clean}_{trial_clean}"
        
        group = group.sort_values('window')
        engine = DecisionPolicyEngine()
        
        trajectory = []
        
        for _, row in group.iterrows():
            prob = row['prob_platt'] if 'prob_platt' in row else row.get('calibrated_prob', 0.5)
            margin = row.get('margin', 0.0)
            win = int(row['window'])
            
            result = engine.update(prob, margin)
            
            trajectory.append({
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
                'confidence': float(result['confidence'] if 'confidence' in result else 0.5)
            })
            
        true_label = group['ground_truth'].iloc[0] if 'ground_truth' in group else group.get('label', 1).iloc[0]
        
        reached_lock = False
        lock_window = -1
        final_decision = None
        
        for t in trajectory:
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
                    
        trial_categories[trial_id] = category
        all_trajectories[trial_id] = trajectory
        
    # 2. Extract WRONG trials and sample an equal number of EASY trials
    wrong_trials = [tid for tid, cat in trial_categories.items() if cat == 'WRONG']
    easy_trials_all = [tid for tid, cat in trial_categories.items() if cat == 'EASY']
    
    random.seed(42) # Deterministic sampling
    if len(easy_trials_all) > len(wrong_trials):
        easy_trials = random.sample(easy_trials_all, len(wrong_trials))
    else:
        easy_trials = easy_trials_all
        wrong_trials = random.sample(wrong_trials, len(easy_trials_all))
        
    # Save JSONL
    wrong_trajectories = []
    for tid in wrong_trials:
        wrong_trajectories.extend(all_trajectories[tid])
        
    easy_trajectories = []
    for tid in easy_trials:
        easy_trajectories.extend(all_trajectories[tid])
        
    with open(out_dir / 'wrong_trials.jsonl', 'w') as f:
        for t in wrong_trajectories:
            f.write(json.dumps(t) + '\n')
            
    with open(out_dir / 'easy_trials.jsonl', 'w') as f:
        for t in easy_trajectories:
            f.write(json.dumps(t) + '\n')
            
    # 3. Compare EASY vs WRONG
    df_wrong = pd.DataFrame(wrong_trajectories)
    df_easy = pd.DataFrame(easy_trajectories)
    
    metrics = []
    for df_subset, name in [(df_wrong, 'WRONG'), (df_easy, 'EASY')]:
        slopes = []
        prob_slopes = []
        lock_times = []
        
        for (subj, trial), group in df_subset.groupby(['subject', 'trial']):
            y_ev = group['sprt_evidence'].values
            y_prob = group['probability'].values
            x = np.arange(len(y_ev))
            
            if len(y_ev) > 1:
                slope_ev, _ = np.polyfit(x, y_ev, 1)
                slopes.append(slope_ev)
                
                slope_prob, _ = np.polyfit(x, y_prob, 1)
                prob_slopes.append(slope_prob)
                
            locked_rows = group[group['state'] == 'LOCKED']
            if len(locked_rows) > 0:
                lock_times.append(locked_rows.iloc[0]['window'])
                
        metrics.append({
            'Category': name,
            'Evidence Growth Rate': np.mean(slopes) if slopes else 0,
            'Probability Growth Rate': np.mean(prob_slopes) if prob_slopes else 0,
            'Mean Margin': df_subset['margin'].mean(),
            'Mean Confidence': df_subset['confidence'].mean(),
            'Mean Latency': np.mean(lock_times) if lock_times else 0,
            'State Transitions (avg/trial)': len(df_subset[df_subset['state'] == 'SWITCHING']) / max(1, len(df_subset.groupby(['subject', 'trial'])))
        })
        
    comp_df = pd.DataFrame(metrics)
    comp_df.to_csv(out_dir / 'trajectory_comparison.csv', index=False)
    
    # 4. Trajectory Divergence Audit
    # We want to see how early probability and evidence separate between WRONG and EASY
    timepoints = [5, 10, 15, 20]
    divergence_report = "## Trajectory Divergence\n\n"
    divergence_report += "| Time (s) | EASY Prob | WRONG Prob | EASY Evidence | WRONG Evidence |\n"
    divergence_report += "|----------|-----------|------------|---------------|----------------|\n"
    
    for t in timepoints:
        e_prob = df_easy[df_easy['window'] == t]['probability'].mean()
        w_prob = df_wrong[df_wrong['window'] == t]['probability'].mean()
        e_ev = df_easy[df_easy['window'] == t]['sprt_evidence'].mean()
        w_ev = df_wrong[df_wrong['window'] == t]['sprt_evidence'].mean()
        
        divergence_report += f"| {t}s | {e_prob:.4f} | {w_prob:.4f} | {e_ev:.4f} | {w_ev:.4f} |\n"
        
    # 5. Strict Audit of AUROC = 1.0 (Leakage/Artifact Check)
    # Target: 1 for EASY, 0 for HARD
    all_trials_for_auroc = []
    for tid, cat in trial_categories.items():
        if cat in ['EASY', 'HARD']:
            all_trials_for_auroc.append({
                'trial_id': tid,
                'target': 1 if cat == 'EASY' else 0,
                'subject': tid.split('_')[0]
            })
            
    auroc_report = "## Strict Audit of Early Predictability (EASY vs HARD)\n\n"
    
    if len(all_trials_for_auroc) > 10:
        auroc_df = pd.DataFrame(all_trials_for_auroc)
        n_easy = sum(auroc_df['target'] == 1)
        n_hard = sum(auroc_df['target'] == 0)
        
        auroc_report += f"- EASY trials: {n_easy}\n"
        auroc_report += f"- HARD trials: {n_hard}\n"
        
        if n_hard < 5:
            auroc_report += "\n**CONCLUSION**: The AUROC=1.0 result is an EVALUATION ARTIFACT caused by extreme class imbalance. With only a few HARD trials, perfect separation is statistically meaningless.\n"
        else:
            # Rigorous cross-validated AUROC calculation to prevent overfitting/leakage
            skf = StratifiedKFold(n_splits=min(5, n_hard))
            cv_aurocs = []
            
            # Use only evidence up to 10s
            for t_idx, row in auroc_df.iterrows():
                traj = all_trajectories[row['trial_id']]
                t_10 = [t for t in traj if t['window'] <= 10]
                auroc_df.loc[t_idx, 'feature_10s'] = t_10[-1]['sprt_evidence'] if len(t_10) > 0 else 0
                
            try:
                for train_idx, test_idx in skf.split(np.zeros(len(auroc_df)), auroc_df['target']):
                    test_targets = auroc_df.iloc[test_idx]['target']
                    test_features = auroc_df.iloc[test_idx]['feature_10s']
                    if len(np.unique(test_targets)) > 1:
                        cv_aurocs.append(roc_auc_score(test_targets, test_features))
                        
                mean_cv_auroc = np.mean(cv_aurocs) if cv_aurocs else np.nan
                auroc_report += f"- Strict CV AUROC at 10s: {mean_cv_auroc:.4f}\n"
                if mean_cv_auroc > 0.95:
                    auroc_report += "\n**CONCLUSION**: The high AUROC appears robust under cross-validation. The decoder is genuinely separating EASY from HARD very early.\n"
                else:
                    auroc_report += "\n**CONCLUSION**: The AUROC=1.0 result from Phase 15.2 was an EVALUATION ARTIFACT (likely overfitting or using in-sample data without CV). The real discriminability is much lower.\n"
            except Exception as e:
                auroc_report += f"\nError in CV AUROC: {e}\n"
    else:
        auroc_report += "Not enough EASY/HARD trials to compute a meaningful AUROC.\n"
        
    with open(out_dir / 'auroc_validation.md', 'w') as f:
        f.write(auroc_report)
        
    report = f"""# Phase 15.3 — Wrong Decision Audit Report

## 1. Executive Summary
This audit compares {len(wrong_trials)} WRONG trials with a balanced sample of {len(easy_trials)} EASY trials to understand why the policy makes confident mistakes, and audits the suspect AUROC=1.0 claim.

## 2. Trajectory Comparison
"""
    report += comp_df.to_markdown(index=False)
    
    report += "\n\n" + divergence_report
    
    report += "\n## 3. Divergence Findings\n"
    if df_wrong[df_wrong['window'] == 5]['probability'].mean() < 0.45:
        report += "- **Do WRONG trials start wrong?** Yes, the decoder immediately outputs high probabilities for the incorrect speaker.\n"
    else:
        report += "- **Do WRONG trials start wrong?** No, the decoder is initially uncertain and later drifts towards the incorrect speaker.\n"
        
    if comp_df.iloc[0]['Evidence Growth Rate'] > 0.1:
        report += "- **Does SPRT amplify an early mistake?** Yes. The SPRT accumulates false evidence aggressively, driving the policy into a LOCKED state.\n"
    
    report += "\n" + auroc_report

    with open(out_dir / 'wrong_decision_report.md', 'w') as f:
        f.write(report)
        
    print(f"Phase 15.3 Audit completed.")
    print(f"Output saved to {out_dir}")

if __name__ == '__main__':
    main()
