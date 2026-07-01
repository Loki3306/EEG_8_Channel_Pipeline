import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(REPO_ROOT))

from analysis.phase_14_accumulators import (
    BayesianAccumulator, 
    SPRTAccumulator, 
    EMAAccumulator, 
    SlidingWindowAccumulator,
    Decision
)
from sklearn.metrics import brier_score_loss

def expected_calibration_error(y_true, y_prob, n_bins=10):
    bins = np.linspace(0., 1., n_bins + 1)
    binids = np.digitize(y_prob, bins) - 1
    ece = 0.0
    for i in range(n_bins):
        bin_idx = (binids == i)
        if np.sum(bin_idx) > 0:
            bin_acc = np.mean(y_true[bin_idx])
            bin_conf = np.mean(y_prob[bin_idx])
            ece += np.abs(bin_acc - bin_conf) * np.sum(bin_idx)
    return ece / len(y_true)

def run_validation_sweep(df: pd.DataFrame, out_dir: Path):
    results = []
    
    # Input Types (Phase 14.10 Ablations)
    # prob_platt: Calibrated
    # prob_raw: Uncalibrated
    # random: Random confidence
    # constant: Constant 0.51 confidence
    
    # Pre-generate random/constant for stability
    df = df.copy()
    np.random.seed(42)
    df['prob_random'] = np.random.uniform(0.0, 1.0, size=len(df))
    df['prob_constant'] = 0.51
    
    inputs = {
        'calibrated': 'prob_platt',
        'uncalibrated': 'prob_raw',
        'random': 'prob_random',
        'constant': 'prob_constant'
    }
    
    accumulators = ['bayesian', 'sprt', 'ema', 'sliding']
    thresholds = [0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
    
    grouped = list(df.groupby(['subject', 'trial'], sort=False))
    
    print("Running Phase 14 Validation Sweeps...")
    
    # Total iterations
    total_iters = len(inputs) * len(accumulators) * len(thresholds)
    pbar = tqdm(total=total_iters)
    
    for input_name, prob_col in inputs.items():
        for acc_type in accumulators:
            for thresh in thresholds:
                
                total_trials = 0
                accepted_trials = 0
                correct_accepted = 0
                latencies = []
                flips_list = []
                max_confs = []
                avg_confs = []
                
                all_trial_probs = []
                all_trial_truths = []
                
                for (subject, trial_idx), trial_df in grouped:
                    # Init accumulator
                    if acc_type == 'bayesian':
                        acc = BayesianAccumulator()
                    elif acc_type == 'sprt':
                        alpha = 1.0 - thresh
                        acc = SPRTAccumulator(alpha=alpha, beta=alpha)
                    elif acc_type == 'ema':
                        acc = EMAAccumulator(alpha=0.1)
                    elif acc_type == 'sliding':
                        acc = SlidingWindowAccumulator(window_size=5)
                        
                    accepted_window = -1
                    final_decision = None
                    max_conf = 0.0
                    confs = []
                    flips = 0
                    last_dec = "CONTINUE"
                    
                    for _, row in trial_df.iterrows():
                        p_t = row[prob_col]
                        w = row['window']
                        
                        if acc_type == 'sprt':
                            posterior, sprt_dec = acc.update(p_t)
                            if sprt_dec == Decision.ACCEPT_STREAM_1:
                                run_dec = "S1"
                            elif sprt_dec == Decision.ACCEPT_STREAM_2:
                                run_dec = "S2"
                            else:
                                run_dec = "CONTINUE"
                        else:
                            posterior = acc.update(p_t)
                            if posterior >= thresh:
                                run_dec = "S1"
                            elif posterior <= (1.0 - thresh):
                                run_dec = "S2"
                            else:
                                run_dec = "CONTINUE"
                                
                        confs.append(posterior)
                        max_conf = max(max_conf, posterior if posterior >= 0.5 else 1.0 - posterior)
                        
                        if run_dec != last_dec and last_dec != "CONTINUE":
                            flips += 1
                        last_dec = run_dec
                        
                        if run_dec != "CONTINUE" and accepted_window == -1:
                            accepted_window = w
                            final_decision = 1 if run_dec == "S1" else 0
                            
                    # Trial end
                    total_trials += 1
                    truth = trial_df['ground_truth'].iloc[0]
                    
                    if accepted_window != -1:
                        accepted_trials += 1
                        if final_decision == truth:
                            correct_accepted += 1
                        latencies.append(accepted_window)
                    
                    # For ECE/Brier on the final probability (if forced to guess at the end)
                    final_prob = confs[-1] if confs else 0.5
                    all_trial_probs.append(final_prob)
                    all_trial_truths.append(truth)
                    
                    flips_list.append(flips)
                    max_confs.append(max_conf)
                    avg_confs.append(np.mean(confs) if confs else 0.5)

                # Aggregate metrics
                coverage = accepted_trials / total_trials if total_trials > 0 else 0
                acc_accepted = correct_accepted / accepted_trials if accepted_trials > 0 else 0
                risk = 1.0 - acc_accepted
                avg_latency = np.mean(latencies) if latencies else -1
                
                ece = expected_calibration_error(np.array(all_trial_truths), np.array(all_trial_probs))
                brier = brier_score_loss(all_trial_truths, all_trial_probs)
                
                results.append({
                    'input_type': input_name,
                    'accumulator': acc_type,
                    'threshold': thresh,
                    'coverage': coverage,
                    'accepted_accuracy': acc_accepted,
                    'risk': risk,
                    'avg_latency': avg_latency,
                    'avg_flips': np.mean(flips_list),
                    'avg_max_conf': np.mean(max_confs),
                    'avg_conf': np.mean(avg_confs),
                    'ece': ece,
                    'brier': brier
                })
                
                pbar.update(1)
                
    pbar.close()
    
    results_df = pd.DataFrame(results)
    out_dir.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(out_dir / "validation_metrics.csv", index=False)
    print(f"Saved validation metrics to {out_dir / 'validation_metrics.csv'}")

if __name__ == "__main__":
    csv_path = REPO_ROOT / "results" / "phase13_margin_calibration" / "calibration_predictions.csv"
    if not csv_path.exists():
        print(f"Error: Could not find {csv_path}. Please run Phase 13 first.")
        sys.exit(1)
        
    df = pd.read_csv(csv_path)
    if 'ablation' in df.columns:
        df = df[df['ablation'] == 'per_subject_absolute_margin']
        
    out_dir = REPO_ROOT / "results" / "phase14_temporal_accumulation"
    run_validation_sweep(df, out_dir)
