import os
import sys
import json
import pandas as pd
import numpy as np
from collections import defaultdict
from pathlib import Path
import argparse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from decision_policy_engine import DecisionPolicyEngine, State, Action

class AuditedDecisionPolicyEngine(DecisionPolicyEngine):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def update(self, probability, margin):
        prev_state = self.state
        prev_evidence = self.evidence
        prev_confidence = 1.0 / (1.0 + np.exp(-self.evidence)) if self.evidence != 0 else 0.5
        
        # Call the core logic
        result = super().update(probability, margin)
        
        # Calculate trends
        ev_trend = self.evidence - prev_evidence
        conf_trend = result['confidence'] - prev_confidence
        
        exact_reason = None
        if self.state == State.UNCERTAIN:
            if prev_state == State.WAITING:
                exact_reason = "TIMEOUT_WAITING"
            elif prev_state == State.LOCKED:
                exact_reason = "CONFIDENCE_COLLAPSED"
            elif prev_state == State.UNCERTAIN:
                # Why did it stay in UNCERTAIN?
                # Does it have a candidate?
                if result['confidence'] >= self.config['confidence_threshold'] or result['confidence'] <= (1.0 - self.config['confidence_threshold']):
                    # Has a candidate, but hasn't reached consecutive windows
                    exact_reason = "AWAITING_CONSECUTIVE_CONFIRMATION"
                else:
                    # In the deadzone
                    exact_reason = "INSUFFICIENT_CONFIDENCE"
            else:
                exact_reason = "OTHER"
                
        result['exact_reason'] = exact_reason
        result['evidence_trend'] = ev_trend
        result['confidence_trend'] = conf_trend
        
        return result

def run_bottleneck_audit(preds_csv, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading predictions from {preds_csv}")
    df = pd.read_csv(preds_csv)
    
    df = df.sort_values(by=['subject', 'trial', 'window'])
    
    if 'correct' in df.columns and 'prob_platt' in df.columns:
        p_target = np.where(df['correct'] == 1, df['prob_platt'], 1 - df['prob_platt'])
        df['prob_platt'] = p_target
        df['ground_truth'] = 1

    transition_log_path = out_dir / "transition_log.jsonl"
    uncertainty_log_path = out_dir / "uncertainty_events.jsonl"
    
    # Remove existing logs to avoid appending indefinitely if run multiple times
    if transition_log_path.exists(): transition_log_path.unlink()
    if uncertainty_log_path.exists(): uncertainty_log_path.unlink()
    
    uncertainty_events = []
    reason_counts = defaultdict(int)
    uncertainty_streaks = []
    
    current_streak = 0
    total_windows = 0
    total_uncertain = 0
    
    print("Running Decision Policy Engine simulation for Bottleneck Audit...")
    
    with open(transition_log_path, 'w') as f_trans, open(uncertainty_log_path, 'w') as f_uncert:
        for (subj, trial), group in df.groupby(['subject', 'trial']):
            engine = AuditedDecisionPolicyEngine()
            
            prev_state_str = State.INITIALIZING
            
            for _, row in group.iterrows():
                total_windows += 1
                prob = row['prob_platt'] if 'prob_platt' in row else row.get('calibrated_prob', 0.5)
                margin = row.get('margin', 0.0)
                win = int(row['window'])
                
                result = engine.update(prob, margin)
                curr_state_str = result['state']
                
                # Log Transitions
                if curr_state_str != prev_state_str:
                    t_log = {
                        'subject': subj,
                        'trial': trial,
                        'window': win,
                        'previous_state': prev_state_str,
                        'new_state': curr_state_str,
                        'confidence': float(result['confidence']),
                        'margin': float(margin),
                        'accumulated_evidence': float(result['evidence']),
                        'decision': result['decision'],
                        'transition_reason': result['reason']
                    }
                    f_trans.write(json.dumps(t_log) + "\n")
                    
                # Track Uncertainty Events
                if curr_state_str == State.UNCERTAIN:
                    total_uncertain += 1
                    current_streak += 1
                    reason = result['exact_reason']
                    reason_counts[reason] += 1
                    
                    u_event = {
                        'subject': subj,
                        'trial': trial,
                        'window': win,
                        'probability': float(prob),
                        'margin': float(margin),
                        'evidence': float(result['evidence']),
                        'confidence': float(result['confidence']),
                        'evidence_trend': float(result['evidence_trend']),
                        'confidence_trend': float(result['confidence_trend']),
                        'exact_reason': reason
                    }
                    f_uncert.write(json.dumps(u_event) + "\n")
                    uncertainty_events.append(u_event)
                else:
                    if current_streak > 0:
                        uncertainty_streaks.append(current_streak)
                        current_streak = 0
                        
                prev_state_str = curr_state_str
                
            # End of trial
            if current_streak > 0:
                uncertainty_streaks.append(current_streak)
                current_streak = 0

    # Bottleneck Statistics
    uncert_df = pd.DataFrame(uncertainty_events)
    
    if len(uncert_df) == 0:
        print("No uncertainty events found.")
        return
        
    avg_streak = np.mean(uncertainty_streaks) if uncertainty_streaks else 0
    median_streak = np.median(uncertainty_streaks) if uncertainty_streaks else 0
    max_streak = np.max(uncertainty_streaks) if uncertainty_streaks else 0
    
    # Threshold Analysis
    threshold = 0.85 # Engine default
    conf = uncert_df['confidence']
    dist_to_th_high = threshold - conf
    dist_to_th_low = conf - (1.0 - threshold)
    
    # Distance is the minimum distance to either the upper or lower threshold
    min_dist = np.minimum(np.abs(dist_to_th_high), np.abs(dist_to_th_low))
    
    within_01 = (min_dist <= 0.01).mean() * 100
    within_02 = (min_dist <= 0.02).mean() * 100
    within_05 = (min_dist <= 0.05).mean() * 100
    
    # Evidence Analysis
    mean_prob = uncert_df['probability'].mean()
    median_prob = uncert_df['probability'].median()
    mean_margin = uncert_df['margin'].mean()
    median_margin = uncert_df['margin'].median()
    mean_ev = uncert_df['evidence'].mean()
    ev_var = uncert_df['evidence'].var()
    
    ev_trend_mean = uncert_df['evidence_trend'].mean()
    conf_trend_mean = uncert_df['confidence_trend'].mean()
    
    if ev_trend_mean > 0.01:
        trend_status = "Growing"
    elif ev_trend_mean < -0.01:
        trend_status = "Decaying"
    else:
        trend_status = "Flat"
        
    sorted_reasons = sorted(reason_counts.items(), key=lambda x: x[1], reverse=True)
    top_reason = sorted_reasons[0][0] if sorted_reasons else "N/A"
    top_reason_pct = (sorted_reasons[0][1] / total_uncertain) * 100 if sorted_reasons else 0
    second_reason = sorted_reasons[1][0] if len(sorted_reasons) > 1 else "N/A"
    
    summary = {
        "total_windows": total_windows,
        "uncertain_windows": total_uncertain,
        "uncertainty_percentage": (total_uncertain / total_windows) * 100 if total_windows > 0 else 0,
        "average_uncertainty_streak": float(avg_streak),
        "median_uncertainty_streak": float(median_streak),
        "max_uncertainty_streak": float(max_streak),
        "top_bottleneck": top_reason,
        "top_bottleneck_percentage": float(top_reason_pct),
        "second_bottleneck": second_reason,
        "mean_probability_uncertain": float(mean_prob),
        "median_probability_uncertain": float(median_prob),
        "mean_margin_uncertain": float(mean_margin),
        "median_margin_uncertain": float(median_margin),
        "mean_evidence_uncertain": float(mean_ev),
        "evidence_variance": float(ev_var),
        "evidence_trend_status": trend_status,
        "within_0.01_threshold_pct": float(within_01),
        "within_0.02_threshold_pct": float(within_02),
        "within_0.05_threshold_pct": float(within_05),
        "reason_counts": reason_counts
    }
    
    with open(out_dir / "policy_summary.json", "w") as f:
        json.dump(summary, f, indent=4)
        
    # Generate Markdown Report
    report = f"""# Phase 15.1: Decision Policy Bottleneck Audit Report

## 1. Executive Summary
The Decision Policy Engine spends `{summary['uncertainty_percentage']:.1f}%` of its time in the `UNCERTAIN` state. This audit diagnosed the underlying reasons without altering the model or thresholds.
The primary bottleneck is **{top_reason}** (accounting for {top_reason_pct:.1f}% of uncertain windows). 
The evidence trend while uncertain is **{trend_status}**.

## 2. Statistical Evidence
### Reason Counts
"""
    for r, count in sorted_reasons:
        pct = (count / total_uncertain) * 100
        report += f"- `{r}`: {pct:.1f}% ({count} windows)\n"
        
    report += f"""
### Uncertainty Durations
- Average Streak: {avg_streak:.1f} windows
- Median Streak: {median_streak:.1f} windows
- Longest Streak: {max_streak:.1f} windows

### Threshold Proximity (Are we barely missing?)
- Within ±0.01 of threshold: {within_01:.1f}%
- Within ±0.02 of threshold: {within_02:.1f}%
- Within ±0.05 of threshold: {within_05:.1f}%

### Evidence & Probability Metrics
- Mean Probability: {mean_prob:.4f}
- Mean Margin: {mean_margin:.4f}
- Mean Evidence (LLR): {mean_ev:.2f} (Variance: {ev_var:.2f})
- Evidence Trend: {trend_status} (Avg change per window: {ev_trend_mean:.4f})

## 3. Findings & Root Cause Ranking

"""
    
    if "INSUFFICIENT_CONFIDENCE" in top_reason:
        root_cause = "Weak Decoder Outputs"
        explanation = "The neural network is outputting probabilities that hover around 0.5, failing to accumulate enough evidence to break out of the uncertainty deadzone."
    elif "AWAITING_CONSECUTIVE_CONFIRMATION" in top_reason:
        root_cause = "Overly Conservative Policy (Hysteresis)"
        explanation = "The evidence has actually crossed the threshold, but the policy is suppressing the decision because it hasn't sustained for the required `minimum_consecutive_windows`."
    elif "TIMEOUT_WAITING" in top_reason:
        root_cause = "Timeout Logic"
        explanation = "The system is timing out during the initial collection phase before reaching a decision."
    elif "CONFIDENCE_COLLAPSED" in top_reason:
        root_cause = "Volatility in Evidence"
        explanation = "The system locks onto a target, but the confidence collapses back into the uncertainty zone, triggering an abort."
    else:
        root_cause = "Unknown"
        explanation = ""
        
    report += f"**Primary Root Cause**: {root_cause}\n{explanation}\n\n"
    
    report += "## 4. Recommended Fixes\n"
    report += "*(Ranked by expected impact based on data)*\n\n"
    
    if root_cause == "Weak Decoder Outputs":
        report += "1. **Improve Classifier Margin**: The policy is doing its job; the underlying probabilities are too weak. Retraining with a larger margin or better representation is required.\n"
        report += "2. **Decrease Confidence Threshold**: If we accept more risk, lowering the threshold from 0.85 will reduce uncertainty time, but increase flips.\n"
    elif root_cause == "Overly Conservative Policy (Hysteresis)":
        report += "1. **Reduce `minimum_consecutive_windows`**: The system is crossing the threshold but waiting too long to pull the trigger. Reducing this parameter will immediately unblock these states.\n"
        report += "2. **Smooth Probabilities Earlier**: Apply a moving average *before* the policy engine to reduce the micro-fluctuations that reset the consecutive counter.\n"
    elif root_cause == "Volatility in Evidence":
        report += "1. **Widen Uncertainty Deadzone**: Reduce the `uncertainty_threshold` parameter. Right now, it aborts if it drops below 0.85 to 0.65. If we only abort below 0.55, it might hold the lock longer.\n"
    
    report += "\n*(No changes were implemented during this audit.)*\n"
    
    with open(out_dir / "bottleneck_report.md", "w") as f:
        f.write(report)
        
    # Console Output (Strictly limited as requested)
    print("\n------------------------------------------")
    print("Phase 15.1 Summary")
    print(f"Trials processed: {df['trial'].nunique() * df['subject'].nunique()}")
    print(f"Average uncertainty: {(total_uncertain / total_windows) * 100:.1f}%")
    print(f"Top bottleneck: {top_reason}")
    print(f"Second bottleneck: {second_reason}")
    print(f"Average latency: N/A (Not tracked in this exact summary loop, see previous phase)")
    print("Files written:")
    print(f"  - {transition_log_path}")
    print(f"  - {uncertainty_log_path}")
    print(f"  - {out_dir / 'policy_summary.json'}")
    print(f"  - {out_dir / 'bottleneck_report.md'}")
    print("\nDone.")
    print("------------------------------------------")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--preds", type=str, default="results/phase13_margin_calibration/calibration_predictions.csv")
    parser.add_argument("--out", type=str, default="results/phase15_1")
    args = parser.parse_args()
    
    run_bottleneck_audit(args.preds, args.out)
