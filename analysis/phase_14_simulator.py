import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
import argparse

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(REPO_ROOT))

from analysis.phase_14_accumulators import (
    BayesianAccumulator, 
    SPRTAccumulator, 
    EMAAccumulator, 
    SlidingWindowAccumulator,
    Decision
)

def run_simulation(df: pd.DataFrame, accumulator_type: str = 'bayesian', 
                  threshold: float = 0.80, verbose: bool = True):
    """
    Runs a strict online streaming simulation across all trials in the dataset.
    """
    # Initialize the correct accumulator
    if accumulator_type == 'bayesian':
        acc = BayesianAccumulator(prior_prob=0.5)
    elif accumulator_type == 'sprt':
        alpha = 1.0 - threshold
        acc = SPRTAccumulator(alpha=alpha, beta=alpha)
    elif accumulator_type == 'ema':
        acc = EMAAccumulator(alpha=0.1)
    elif accumulator_type == 'sliding':
        acc = SlidingWindowAccumulator(window_size=5)
    else:
        raise ValueError(f"Unknown accumulator: {accumulator_type}")

    # Track metrics
    total_trials = 0
    total_accepted = 0
    total_correct = 0
    latencies = []
    
    # We assume df is already sorted by (subject, trial, window)
    grouped = df.groupby(['subject', 'trial'], sort=False)
    
    for (subject, trial_idx), trial_df in grouped:
        acc.reset()
        
        # Trial level tracking
        trial_windows = len(trial_df)
        accepted_window = -1
        final_decision = None
        max_conf = 0.0
        confs = []
        flips = 0
        last_decision_str = "CONTINUE"
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"Starting Trial: {subject} - Trial {trial_idx}")
            print(f"{'='*60}")
            
        for _, row in trial_df.iterrows():
            w = row['window']
            margin = row['margin']
            corr_a = row['corrA']
            corr_b = row['corrB']
            prob_raw = row['prob_raw']
            # We use platt by default as the calibrated probability
            prob_cal = row['prob_platt'] 
            
            # Step the accumulator
            if accumulator_type == 'sprt':
                posterior, sprt_dec = acc.update(prob_cal)
                acc_state = f"LLR: {acc.get_llr():.4f}"
                if sprt_dec == Decision.ACCEPT_STREAM_1:
                    running_dec = "STREAM_1"
                elif sprt_dec == Decision.ACCEPT_STREAM_2:
                    running_dec = "STREAM_2"
                else:
                    running_dec = "CONTINUE"
            else:
                posterior = acc.update(prob_cal)
                acc_state = f"Prob: {posterior:.4f}"
                if posterior >= threshold:
                    running_dec = "STREAM_1"
                elif posterior <= (1.0 - threshold):
                    running_dec = "STREAM_2"
                else:
                    running_dec = "CONTINUE"
                    
            max_conf = max(max_conf, posterior if posterior >= 0.5 else 1.0 - posterior)
            confs.append(posterior)
            
            decision_changed = (running_dec != last_decision_str and last_decision_str != "CONTINUE")
            if decision_changed:
                flips += 1
                
            last_decision_str = running_dec
            stopping_trigger = (running_dec != "CONTINUE")
            
            if verbose:
                print("-" * 40)
                print(f"Window #               : {int(w)}")
                print(f"Margin                 : {margin:.4f}")
                print(f"Pearson A              : {corr_a:.4f}")
                print(f"Pearson B              : {corr_b:.4f}")
                print(f"Calibrated Probability : {prob_cal:.4f}")
                print(f"Accumulator State      : {acc_state}")
                print(f"Running Confidence     : {posterior:.4f}")
                print(f"Running Decision       : {running_dec}")
                print(f"Decision Changed?      : {'YES' if decision_changed else 'NO'}")
                print(f"Stopping Trigger?      : {'YES' if stopping_trigger else 'NO'}")
                
            if stopping_trigger and accepted_window == -1:
                accepted_window = int(w)
                final_decision = 1 if running_dec == "STREAM_1" else 0
                # In strict online, we might stop processing here, but we continue to show flips
        
        # Trial Summary
        total_trials += 1
        truth = trial_df['ground_truth'].iloc[0]
        
        if accepted_window != -1:
            total_accepted += 1
            is_correct = (final_decision == truth)
            if is_correct:
                total_correct += 1
            latencies.append(accepted_window)
        else:
            is_correct = False
            
        if verbose:
            print("=" * 60)
            print("TRIAL SUMMARY")
            print(f"Total Windows          : {trial_windows}")
            print(f"Accepted Window        : {accepted_window if accepted_window != -1 else 'NEVER'}")
            print(f"Rejected Windows       : {trial_windows - accepted_window if accepted_window != -1 else trial_windows}")
            print(f"Decision               : {final_decision if final_decision is not None else 'NONE'} (Truth: {truth})")
            print(f"Decision Latency       : {accepted_window if accepted_window != -1 else 'N/A'}")
            print(f"Maximum Confidence     : {max_conf:.4f}")
            print(f"Average Confidence     : {np.mean(confs):.4f}")
            print(f"Decision Flips         : {flips}")
            print(f"Final Accuracy         : {'CORRECT' if is_correct else 'INCORRECT'}")
            print("=" * 60)
            
    print("\n" + "#" * 60)
    print("GLOBAL SIMULATION RESULTS")
    print(f"Accumulator Type       : {accumulator_type.upper()}")
    print(f"Threshold              : {threshold}")
    print(f"Total Trials           : {total_trials}")
    print(f"Accepted Trials        : {total_accepted} ({total_accepted/total_trials*100:.1f}%)")
    if total_accepted > 0:
        print(f"Accepted Accuracy      : {total_correct/total_accepted*100:.1f}%")
        print(f"Average Latency        : {np.mean(latencies):.2f} windows")
    print("#" * 60 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 14: Streaming Accumulator Simulator")
    parser.add_argument('--acc', type=str, default='bayesian', choices=['bayesian', 'sprt', 'ema', 'sliding'])
    parser.add_argument('--threshold', type=float, default=0.80)
    parser.add_argument('--quiet', action='store_true', help="Disable per-window debug printing")
    
    args = parser.parse_args()
    
    csv_path = REPO_ROOT / "results" / "phase13_margin_calibration" / "calibration_predictions.csv"
    if not csv_path.exists():
        print(f"Error: Could not find {csv_path}. Please run Phase 13 first.")
        sys.exit(1)
        
    df = pd.read_csv(csv_path)
    # Use the absolute_margin ablation logic for best probabilities
    if 'ablation' in df.columns:
        df = df[df['ablation'] == 'per_subject_absolute_margin']
        
    run_simulation(df, accumulator_type=args.acc, threshold=args.threshold, verbose=not args.quiet)
