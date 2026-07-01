import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse

def compute_cumulative_metrics(trial_df):
    """
    Computes expanding accumulation metrics from the start of the trial.
    """
    trial_df = trial_df.sort_values('window').copy()
    
    # Use Platt calibrated probabilities, clip to avoid log(0)
    p = np.clip(trial_df['prob_platt'].values, 1e-7, 1 - 1e-7)
    y = trial_df['ground_truth'].values
    
    # 1. Single window accuracy
    acc_single = (p > 0.5) == y
    
    # 2. SMA (expanding mean)
    sma_p = pd.Series(p).expanding().mean().values
    acc_sma = (sma_p > 0.5) == y
    
    # 3. EMA (expanding EMA)
    # alpha=0.3 corresponds to a center of mass of ~2.3 windows
    ema_p = pd.Series(p).ewm(alpha=0.3, adjust=False).mean().values
    acc_ema = (ema_p > 0.5) == y
    
    # 4. Bayesian (Cumulative LLR)
    llr = np.log(p / (1 - p))
    cum_llr = np.cumsum(llr)
    acc_bayes = (cum_llr > 0) == y
    
    trial_df['acc_single'] = acc_single
    trial_df['acc_sma'] = acc_sma
    trial_df['acc_ema'] = acc_ema
    trial_df['acc_bayes'] = acc_bayes
    trial_df['cum_llr'] = cum_llr
    
    return trial_df

def compute_sprt(trial_df, conf_threshold=0.95):
    """
    Simulates Sequential Probability Ratio Test (SPRT).
    Returns the latency to reach a decision and whether it was correct.
    """
    trial_df = trial_df.sort_values('window')
    p = np.clip(trial_df['prob_platt'].values, 1e-7, 1 - 1e-7)
    y = trial_df['ground_truth'].values[0] 
    
    llr = np.log(p / (1 - p))
    cum_llr = np.cumsum(llr)
    
    theta_u = np.log(conf_threshold / (1 - conf_threshold))
    theta_l = np.log((1 - conf_threshold) / conf_threshold)
    
    for i, L in enumerate(cum_llr):
        if L >= theta_u:
            return pd.Series({'decision': 1, 'correct': 1 == y, 'latency_windows': i + 1, 'hit': True})
        elif L <= theta_l:
            return pd.Series({'decision': 0, 'correct': 0 == y, 'latency_windows': i + 1, 'hit': True})
            
    return pd.Series({'decision': -1, 'correct': False, 'latency_windows': len(cum_llr), 'hit': False})

def run_accumulation_analysis(preds_path, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading predictions from {preds_path}...")
    try:
        df = pd.read_csv(preds_path)
    except FileNotFoundError:
        print(f"Error: Could not find {preds_path}.")
        return

    # Filter to per_subject platt calibration if multiple ablations exist
    if 'ablation' in df.columns:
        df = df[df['ablation'] == 'per_subject_absolute_margin'].copy()
        if len(df) == 0:
            # Fallback if the name is slightly different
            df = pd.read_csv(preds_path)
            if 'ablation' in df.columns:
                df = df[df['ablation'].str.contains('per_subject')].copy()

    print(f"Loaded {len(df)} windows.")
    
    # -----------------------------------------------------------------------------
    # CRITICAL FIX: In Phase 13, the stream assignment (1 vs 0) was randomized
    # PER WINDOW instead of PER TRIAL. This causes the ground truth to oscillate
    # wildly within a single trial, breaking temporal accumulation (random walk).
    #
    # To fix this without re-running Phase 13, we map all probabilities to the 
    # probability of the TRULY ATTENDED stream. 
    # Since prob_platt is P(model is correct), the probability assigned to the 
    # truly attended stream is exactly prob_platt when the model is correct, 
    # and 1 - prob_platt when the model is wrong.
    # 
    # We then set ground_truth = 1 for the whole trial, meaning our target class 
    # is always the truly attended stream.
    # -----------------------------------------------------------------------------
    p_target = np.where(df['correct'] == 1, df['prob_platt'], 1 - df['prob_platt'])
    df['prob_platt'] = p_target
    df['ground_truth'] = 1
    
    # Process expanding metrics
    print("Computing expanding accumulation metrics...")
    acc_df = df.groupby(['subject', 'trial']).apply(compute_cumulative_metrics, include_groups=False).reset_index(drop=True)
    
    # Calculate accuracy vs latency (window index)
    # Window 0 is 2s, window 1 is 4s, etc. (assuming 2s hop)
    latency_stats = acc_df.groupby('window')[['acc_single', 'acc_sma', 'acc_ema', 'acc_bayes']].mean().reset_index()
    latency_stats['time_sec'] = (latency_stats['window'] + 1) * 2
    
    # Save latency stats
    latency_stats.to_csv(out_dir / "accuracy_vs_latency.csv", index=False)
    
    # Plot Accuracy vs Latency
    plt.figure(figsize=(10, 6))
    plt.plot(latency_stats['time_sec'], latency_stats['acc_single'], label='Single Window (Baseline)', color='gray', linestyle='--')
    plt.plot(latency_stats['time_sec'], latency_stats['acc_sma'], label='SMA (Moving Average)', linewidth=2)
    plt.plot(latency_stats['time_sec'], latency_stats['acc_ema'], label='EMA (Exponential MA)', linewidth=2)
    plt.plot(latency_stats['time_sec'], latency_stats['acc_bayes'], label='Bayesian (Log-Odds)', linewidth=2)
    
    plt.axhline(0.90, color='r', linestyle=':', label='90% Target')
    plt.axhline(0.95, color='g', linestyle=':', label='95% Target')
    
    plt.xlim(0, 30) # Show first 30 seconds
    plt.ylim(0.5, 1.0)
    plt.title('Accuracy vs Latency (Temporal Evidence Accumulation)')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "accuracy_vs_latency.png", dpi=300)
    plt.close()
    print(f"Saved Accuracy vs Latency plot to {out_dir / 'accuracy_vs_latency.png'}")
    
    # Compute SPRT stats
    print("Computing SPRT metrics for different confidence thresholds...")
    thresholds = [0.80, 0.90, 0.95, 0.99]
    sprt_results = []
    
    for th in thresholds:
        sprt_df = df.groupby(['subject', 'trial']).apply(compute_sprt, conf_threshold=th, include_groups=False).reset_index()
        
        # Detailed audit breakdown
        total_trials = len(sprt_df)
        accepted_correct = len(sprt_df[(sprt_df['hit'] == True) & (sprt_df['correct'] == True)])
        accepted_incorrect = len(sprt_df[(sprt_df['hit'] == True) & (sprt_df['correct'] == False)])
        no_decision = len(sprt_df[sprt_df['hit'] == False])
        
        hit_rate = (accepted_correct + accepted_incorrect) / total_trials if total_trials > 0 else 0
        
        hits = sprt_df[sprt_df['hit'] == True]
        if len(hits) > 0:
            mean_latency_sec = hits['latency_windows'].mean() * 2
            accuracy = accepted_correct / len(hits)
        else:
            mean_latency_sec = np.nan
            accuracy = np.nan
            
        sprt_results.append({
            'Threshold': th,
            'Decision_Rate': hit_rate,
            'Mean_Latency_sec': mean_latency_sec,
            'Accuracy': accuracy,
            'Accepted_Correct': accepted_correct,
            'Accepted_Incorrect': accepted_incorrect,
            'No_Decision': no_decision,
            'Total_Trials': total_trials
        })
        
    sprt_res_df = pd.DataFrame(sprt_results)
    sprt_res_df.to_csv(out_dir / "sprt_metrics.csv", index=False)
    
    print("\n--- SPRT Results ---")
    print(sprt_res_df.to_string(index=False))
    
    # Generate Markdown Report
    report_path = out_dir / "phase_14_accumulation_report.md"
    with open(report_path, "w") as f:
        f.write("# Phase 14: Temporal Evidence Accumulation Report\n\n")
        
        f.write("## 1. Accuracy vs Latency (Fixed Time)\n")
        f.write("How accuracy improves as we accumulate evidence over a fixed duration.\n\n")
        
        # Extract accuracy at 2s, 10s, 20s
        t_points = [2, 10, 20]
        f.write("| Time | Single Window | SMA | EMA | Bayesian |\n")
        f.write("|------|---------------|-----|-----|----------|\n")
        for t in t_points:
            row = latency_stats[latency_stats['time_sec'] == t]
            if len(row) > 0:
                row = row.iloc[0]
                f.write(f"| {t}s | {row['acc_single']:.3f} | {row['acc_sma']:.3f} | {row['acc_ema']:.3f} | {row['acc_bayes']:.3f} |\n")
                
        f.write("\n## 2. SPRT (Sequential Probability Ratio Test)\n")
        f.write("Time required to reach a dynamic confidence threshold.\n\n")
        f.write("### Audit Breakdown\n")
        f.write(sprt_res_df[['Threshold', 'Total_Trials', 'Accepted_Correct', 'Accepted_Incorrect', 'No_Decision']].to_markdown(index=False))
        f.write("\n\n### Performance Metrics\n")
        f.write(sprt_res_df[['Threshold', 'Decision_Rate', 'Mean_Latency_sec', 'Accuracy']].to_markdown(index=False))
        f.write("\n")
        
    print(f"\nGenerated report at {report_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Temporal Accumulation")
    parser.add_argument("--preds", type=str, default="results/phase13_margin_calibration/calibration_predictions.csv")
    parser.add_argument("--out", type=str, default="results/phase14_accumulation")
    args = parser.parse_args()
    
    run_accumulation_analysis(args.preds, args.out)
