import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.metrics import brier_score_loss

def softmax(x):
    e_x = np.exp(x)
    return e_x / e_x.sum(axis=1, keepdims=True)

def step_2_3_calibration(df, out_dir):
    print("\n===========================================")
    print("STEP 2.3: CALIBRATION ANALYSIS")
    print("===========================================\n")
    
    # 1. Compute Softmax Confidence
    # Treat similarities as logits to get a proper probability in [0.5, 1.0]
    logits = df[['sim_A', 'sim_B']].values
    probs = softmax(logits)
    df['confidence'] = np.max(probs, axis=1)
    
    correct = df['correct'].values
    conf = df['confidence'].values
    
    # 2. Brier Score
    brier = brier_score_loss(correct, conf)
    print(f"Brier Score : {brier:.4f}")
    
    # 3. Calibration Table & ECE (10 Bins)
    n_bins = 10
    bins = np.linspace(0.5, 1.0, n_bins + 1)
    df['conf_bin'] = pd.cut(df['confidence'], bins=bins, include_lowest=True)
    
    calib_stats = df.groupby('conf_bin', observed=True).agg(
        count=('correct', 'size'),
        mean_conf=('confidence', 'mean'),
        accuracy=('correct', 'mean')
    ).reset_index()
    
    ece = 0.0
    total_samples = len(df)
    
    print("\n--- Calibration Table ---")
    print(f"{'Confidence Bin':<20} | {'Count':<8} | {'Mean Conf':<10} | {'Actual Acc':<10}")
    print("-" * 55)
    
    for _, row in calib_stats.iterrows():
        bin_str = str(row['conf_bin'])
        count = row['count']
        if count == 0:
            continue
            
        mean_conf = row['mean_conf']
        acc = row['accuracy']
        
        ece += (count / total_samples) * np.abs(acc - mean_conf)
        
        print(f"{bin_str:<20} | {count:<8} | {mean_conf*100:>7.2f}%   | {acc*100:>7.2f}%")
        
    print(f"\nExpected Calibration Error (ECE) : {ece:.4f}")
    
    # 4. Reliability Diagram
    os.makedirs(out_dir, exist_ok=True)
    sns.set_theme(style="whitegrid")
    
    plt.figure(figsize=(8, 8))
    
    # Filter out empty bins for plotting
    plot_stats = calib_stats[calib_stats['count'] > 0]
    
    # Plot perfect calibration line
    plt.plot([0.5, 1.0], [0.5, 1.0], 'k--', label="Perfect Calibration")
    
    # Plot empirical calibration
    plt.plot(plot_stats['mean_conf'], plot_stats['accuracy'], marker='o', linewidth=2, color='blue', label="MatchNet Softmax Conf")
    
    # Add a bar chart showing counts (histogram) at the bottom
    plt.bar(plot_stats['mean_conf'], plot_stats['count'] / total_samples, width=0.04, alpha=0.3, color='gray', label="% of Samples")
    
    plt.title("Reliability Diagram")
    plt.xlabel("Mean Predicted Confidence")
    plt.ylabel("Observed Accuracy / Sample Density")
    plt.xlim(0.5, 1.0)
    plt.ylim(0.0, 1.0)
    plt.legend()
    plt.grid(True)
    
    fig_path = os.path.join(out_dir, "fig5_reliability_diagram.png")
    plt.savefig(fig_path, dpi=300)
    plt.close()
    
    print(f"\nReliability Diagram saved to {fig_path}")
    
    print("\n--- Interpretation ---")
    if ece < 0.05:
        print("Model is well-calibrated (ECE < 5%). The confidence probabilities closely match actual correctness.")
    else:
        print("Model is poorly calibrated. There is a large gap between confidence probabilities and actual correctness.")
        print("If Softmax Confidence is clustered near 0.5 (Under-confident), this proves raw similarities need temperature scaling or a dedicated confidence head.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="matchnet_predictions.csv")
    parser.add_argument("--plot_dir", type=str, default="selective_aad_plots")
    args = parser.parse_args()
    
    try:
        df = pd.read_csv(args.csv)
    except FileNotFoundError:
        print(f"Error: Could not find {args.csv}")
        return
        
    step_2_3_calibration(df, args.plot_dir)

if __name__ == "__main__":
    main()
