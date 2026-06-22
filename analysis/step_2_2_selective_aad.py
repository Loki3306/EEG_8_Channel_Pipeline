import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def run_selective_aad(df, out_dir):
    print("\n===========================================")
    print("STEP 2.2: SELECTIVE AAD PILOT")
    print("===========================================\n")
    
    # Ensure margin is computed
    if 'margin' not in df.columns:
        df['margin'] = np.abs(df['sim_A'] - df['sim_B'])
        
    # Sort predictions by confidence (margin) in descending order
    df_sorted = df.sort_values(by='margin', ascending=False).reset_index(drop=True)
    
    total_predictions = len(df_sorted)
    
    # Coverages to evaluate
    coverages = [1.0, 0.95, 0.90, 0.80, 0.70, 0.60, 0.50, 0.40, 0.30]
    
    results = []
    
    for cov in coverages:
        k = int(cov * total_predictions)
        if k == 0:
            continue
            
        accepted = df_sorted.iloc[:k]
        accuracy = accepted['correct'].mean()
        risk = 1.0 - accuracy
        
        results.append({
            'coverage': cov,
            'accuracy': accuracy,
            'risk': risk
        })
        
    res_df = pd.DataFrame(results)
    
    # Print Table 1 & 2 combined
    print(f"{'Coverage':<10} | {'Accuracy':<10} | {'Risk':<10}")
    print("-" * 35)
    for _, row in res_df.iterrows():
        print(f"{row['coverage']*100:>7.1f}% | {row['accuracy']*100:>8.2f}% | {row['risk']*100:>8.2f}%")
        
    # Plotting
    os.makedirs(out_dir, exist_ok=True)
    sns.set_theme(style="whitegrid")
    
    # Figure 1: Coverage vs Accuracy
    plt.figure(figsize=(8, 6))
    plt.plot(res_df['coverage'] * 100, res_df['accuracy'] * 100, marker='o', linewidth=2, color='blue')
    plt.gca().invert_xaxis() # 100% to 0%
    plt.title("Figure 1: Coverage vs Accuracy")
    plt.xlabel("Coverage (%)")
    plt.ylabel("Accuracy (%)")
    plt.grid(True)
    plt.savefig(os.path.join(out_dir, "fig1_coverage_vs_accuracy.png"), dpi=300)
    plt.close()
    
    # Figure 2: Risk vs Coverage
    plt.figure(figsize=(8, 6))
    plt.plot(res_df['coverage'] * 100, res_df['risk'] * 100, marker='o', linewidth=2, color='red')
    plt.gca().invert_xaxis()
    plt.title("Figure 2: Risk vs Coverage")
    plt.xlabel("Coverage (%)")
    plt.ylabel("Risk (Error Rate) (%)")
    plt.grid(True)
    plt.savefig(os.path.join(out_dir, "fig2_risk_vs_coverage.png"), dpi=300)
    plt.close()
    
    print(f"\nPlots saved to {out_dir}/")
    print("Success Criteria: Check if rejecting low-confidence predictions (moving UP the table) significantly improves accuracy.")

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
        
    run_selective_aad(df, args.plot_dir)

if __name__ == "__main__":
    main()
