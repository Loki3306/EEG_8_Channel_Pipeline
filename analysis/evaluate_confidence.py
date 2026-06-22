import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr, pointbiserialr
import os

def softmax(x):
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / e_x.sum(axis=-1, keepdims=True)

def compute_confidence_metrics(df):
    print("Computing confidence metrics...")
    # Method A: Raw Margin
    df['margin'] = np.abs(df['sim_A'] - df['sim_B'])
    
    # Method B: Normalized Margin
    # Handle division by zero
    denom = np.abs(df['sim_A']) + np.abs(df['sim_B'])
    df['norm_margin'] = np.where(denom == 0, 0, df['margin'] / denom)
    
    # Method C: Softmax Confidence
    logits = df[['sim_A', 'sim_B']].values
    probs = softmax(logits)
    df['softmax_conf'] = np.max(probs, axis=1)
    
    return df

def generate_plots(df, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    sns.set_theme(style="whitegrid")
    
    # 1. Correct vs Incorrect Histogram
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x='margin', hue='correct', stat='density', common_norm=False, 
                 kde=True, palette={0: 'red', 1: 'blue'}, bins=30)
    plt.title("Figure 1: Margin Distribution (Correct vs Incorrect)")
    plt.xlabel("Similarity Margin |sim_A - sim_B|")
    plt.savefig(os.path.join(out_dir, "fig1_correct_vs_incorrect_hist.png"), dpi=300)
    plt.close()
    
    # 2. Subject-wise Confidence vs Accuracy
    subj_stats = df.groupby('subject_id').agg(
        accuracy=('correct', 'mean'),
        mean_margin=('margin', 'mean')
    ).reset_index()
    
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=subj_stats, x='mean_margin', y='accuracy', s=100)
    for i, row in subj_stats.iterrows():
        plt.annotate(row['subject_id'], (row['mean_margin'], row['accuracy']), 
                     xytext=(5, 5), textcoords='offset points')
    plt.title("Figure 2: Subject-wise Accuracy vs Mean Margin")
    plt.xlabel("Mean Similarity Margin")
    plt.ylabel("Accuracy")
    plt.savefig(os.path.join(out_dir, "fig2_subject_wise_confidence.png"), dpi=300)
    plt.close()
    
    # 3. Confidence Bins vs Accuracy
    # Create quantiles or fixed bins. Let's use 10 quantiles to ensure equal samples per bin
    df['margin_bin'] = pd.qcut(df['margin'], q=10, duplicates='drop')
    bin_stats = df.groupby('margin_bin', observed=True)['correct'].mean().reset_index()
    # Convert interval to string for plotting
    bin_stats['margin_bin_str'] = bin_stats['margin_bin'].astype(str)
    
    plt.figure(figsize=(12, 6))
    sns.barplot(data=bin_stats, x='margin_bin_str', y='correct', color='steelblue')
    plt.axhline(y=0.5, color='r', linestyle='--', label='Chance (50%)')
    plt.title("Figure 3: Accuracy across Confidence Margin Deciles")
    plt.xlabel("Margin Bin")
    plt.ylabel("Accuracy")
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fig3_confidence_bins_accuracy.png"), dpi=300)
    plt.close()
    
    # 4. Confidence vs Correctness Scatter/Violin
    # A scatter plot of continuous vs binary is hard to read, so a violin plot is better.
    plt.figure(figsize=(8, 6))
    sns.violinplot(data=df, x='correct', y='margin', palette={0: 'lightcoral', 1: 'lightblue'})
    plt.title("Figure 4: Confidence Margin vs Correctness")
    plt.xlabel("Correct (0=No, 1=Yes)")
    plt.ylabel("Similarity Margin")
    plt.savefig(os.path.join(out_dir, "fig4_confidence_vs_correctness.png"), dpi=300)
    plt.close()
    
    return subj_stats, bin_stats

def print_statistics(df, subj_stats, bin_stats):
    print("\n--- Phase 1: Confidence Benchmarking Statistics ---\n")
    
    # Correlations
    margin = df['margin'].values
    correct = df['correct'].values
    
    # Spearman
    rho, p_spearman = spearmanr(margin, correct)
    print(f"Spearman Correlation (Margin vs Correctness): rho = {rho:.4f}, p = {p_spearman:.4e}")
    
    # Point-Biserial
    r_pb, p_pb = pointbiserialr(correct, margin)
    print(f"Point-Biserial Correlation: r = {r_pb:.4f}, p = {p_pb:.4e}")
    
    print("\n--- Subject-wise Stats ---")
    subj_stats['accuracy'] = subj_stats['accuracy'] * 100
    print(subj_stats.sort_values(by='accuracy', ascending=False).to_string(index=False))
    
    print("\n--- Bin-wise Accuracy (Deciles) ---")
    bin_stats['correct'] = bin_stats['correct'] * 100
    for _, row in bin_stats.iterrows():
        print(f"Bin {row['margin_bin_str']:<20} : {row['correct']:.2f}% Accuracy")
        
    print("\n--- Criteria Check ---")
    print(f"Criterion 1 (Distributions Separate): Compare mean margin -> Correct: {df[df['correct']==1]['margin'].mean():.4f}, Incorrect: {df[df['correct']==0]['margin'].mean():.4f}")
    print(f"Criterion 2 (Monotonic Growth): Check Bin-wise Accuracy output above.")
    print(f"Criterion 3 (Significant Corr): p-value = {p_spearman:.4e} < 0.05 ? {'YES' if p_spearman < 0.05 else 'NO'}")
    print(f"Criterion 4 (Weak Subjects): Check Subject-wise Stats plot.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="matchnet_predictions.csv", help="Path to exported predictions CSV")
    parser.add_argument("--out_csv", type=str, default="matchnet_confidence.csv", help="Path to save confidence scores")
    parser.add_argument("--plot_dir", type=str, default="confidence_plots", help="Directory to save plots")
    args = parser.parse_args()
    
    if not os.path.exists(args.csv):
        print(f"Error: {args.csv} not found. Please run export_matchnet_predictions.py first.")
        return
        
    df = pd.read_csv(args.csv)
    print(f"Loaded {len(df)} predictions from {args.csv}")
    
    df = compute_confidence_metrics(df)
    
    df.to_csv(args.out_csv, index=False)
    print(f"Saved confidence metrics to {args.out_csv}")
    
    subj_stats, bin_stats = generate_plots(df, args.plot_dir)
    print(f"Saved plots to {args.plot_dir}/")
    
    print_statistics(df, subj_stats, bin_stats)

if __name__ == "__main__":
    main()
