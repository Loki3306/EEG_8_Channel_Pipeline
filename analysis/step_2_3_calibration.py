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

def compute_metrics_for_confidence(df, conf_col, name, out_dir):
    correct = df['correct'].values
    conf = df[conf_col].values
    
    # Brier Score
    brier = brier_score_loss(correct, conf)
    
    # ECE
    # Determine bins based on the range of confidence
    min_c, max_c = conf.min(), conf.max()
    # Use 10 bins between min and max
    bins = np.linspace(min_c, max_c, 11)
    # To handle potential identical bins, add a tiny jitter or use unique
    bins = np.unique(bins)
    if len(bins) < 2:
        bins = np.linspace(0, 1, 11) # fallback
        
    df['conf_bin'] = pd.cut(df[conf_col], bins=bins, include_lowest=True)
    
    calib_stats = df.groupby('conf_bin', observed=True).agg(
        count=('correct', 'size'),
        mean_conf=(conf_col, 'mean'),
        accuracy=('correct', 'mean')
    ).reset_index()
    
    ece = 0.0
    total_samples = len(df)
    
    for _, row in calib_stats.iterrows():
        count = row['count']
        if count == 0:
            continue
        mean_conf = row['mean_conf']
        acc = row['accuracy']
        ece += (count / total_samples) * np.abs(acc - mean_conf)
        
    # Reliability Diagram
    plt.figure(figsize=(8, 8))
    plot_stats = calib_stats[calib_stats['count'] > 0]
    
    plt.plot([min_c, max_c], [min_c, max_c], 'k--', label="Perfect Calibration")
    plt.plot(plot_stats['mean_conf'], plot_stats['accuracy'], marker='o', linewidth=2, color='blue', label=name)
    plt.bar(plot_stats['mean_conf'], plot_stats['count'] / total_samples * (max_c - min_c), 
            width=(max_c - min_c)/len(bins)*0.8, alpha=0.3, color='gray', label="% of Samples")
            
    plt.title(f"Reliability Diagram: {name}")
    plt.xlabel("Mean Predicted Confidence")
    plt.ylabel("Observed Accuracy / Sample Density")
    plt.legend()
    plt.grid(True)
    
    clean_name = name.lower().replace(" ", "_")
    fig_path = os.path.join(out_dir, f"fig_{clean_name}_reliability.png")
    plt.savefig(fig_path, dpi=300)
    plt.close()
    
    return brier, ece, calib_stats

def step_2_3_calibration(df, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    sns.set_theme(style="whitegrid")
    
    print("\n===========================================")
    print("STEP 2.3: CALIBRATION DISTRIBUTION AUDIT")
    print("===========================================\n")
    
    # 1. Compute Confidence Representations
    # A. Softmax Confidence
    logits = df[['sim_A', 'sim_B']].values
    probs = softmax(logits)
    df['softmax_conf'] = np.max(probs, axis=1)
    
    # B. Normalized Margin Confidence
    df['margin'] = np.abs(df['sim_A'] - df['sim_B'])
    
    # User formula: margin / margin.max() (Ranges 0 to 1)
    # Mathematical note: Binary confidence usually ranges 0.5 to 1.0. We map it to 0.5-1.0 so it is a valid probability.
    df['margin_conf_0_1'] = df['margin'] / df['margin'].max()
    df['margin_conf_scaled'] = 0.5 + 0.5 * (df['margin'] / df['margin'].max())
    
    # Let's audit BOTH Softmax and the 0.5-1.0 scaled Margin
    
    print("--- Distribution Audit: Softmax Confidence ---")
    print("Definition: max(softmax([sim_A, sim_B]))")
    print(df['softmax_conf'].describe())
    
    print("\n--- Distribution Audit: Margin Confidence (Scaled 0.5 to 1.0) ---")
    print("Definition: 0.5 + 0.5 * (margin / margin.max())")
    print(df['margin_conf_scaled'].describe())
    
    # Plot Histograms
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.hist(df['softmax_conf'], bins=50, color='purple', alpha=0.7)
    plt.title("Softmax Confidence Histogram")
    plt.xlabel("Confidence")
    
    plt.subplot(1, 2, 2)
    plt.hist(df['margin_conf_scaled'], bins=50, color='green', alpha=0.7)
    plt.title("Scaled Margin Confidence Histogram")
    plt.xlabel("Confidence")
    
    plt.tight_layout()
    hist_path = os.path.join(out_dir, "fig_confidence_histograms.png")
    plt.savefig(hist_path, dpi=300)
    plt.close()
    
    # 2. Side-by-Side Calibration Comparison
    print("\n===========================================")
    print("CALIBRATION METRICS COMPARISON")
    print("===========================================\n")
    
    brier_sm, ece_sm, calib_sm = compute_metrics_for_confidence(df, 'softmax_conf', "Softmax Confidence", out_dir)
    brier_mg, ece_mg, calib_mg = compute_metrics_for_confidence(df, 'margin_conf_scaled', "Margin Confidence", out_dir)
    
    print(f"{'Metric':<20} | {'Softmax Conf':<15} | {'Scaled Margin Conf':<15}")
    print("-" * 55)
    print(f"{'ECE':<20} | {ece_sm:<15.4f} | {ece_mg:<15.4f}")
    print(f"{'Brier Score':<20} | {brier_sm:<15.4f} | {brier_mg:<15.4f}")
    
    print("\n--- Softmax Calibration Table ---")
    for _, row in calib_sm.iterrows():
        if row['count'] > 0:
            print(f"Bin {str(row['conf_bin']):<15}: Count={row['count']:<4} | Mean Conf={row['mean_conf']:.4f} | Acc={row['accuracy']:.4f}")
            
    print("\n--- Margin Calibration Table ---")
    for _, row in calib_mg.iterrows():
        if row['count'] > 0:
            print(f"Bin {str(row['conf_bin']):<15}: Count={row['count']:<4} | Mean Conf={row['mean_conf']:.4f} | Acc={row['accuracy']:.4f}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="matchnet_predictions.csv")
    parser.add_argument("--plot_dir", type=str, default="calibration_plots")
    args = parser.parse_args()
    
    try:
        df = pd.read_csv(args.csv)
    except FileNotFoundError:
        print(f"Error: Could not find {args.csv}")
        return
        
    step_2_3_calibration(df, args.plot_dir)

if __name__ == "__main__":
    main()
