import argparse
import pandas as pd
import numpy as np

def analyze_subject_variability(df):
    print("\n===========================================")
    print("STEP 3.0: SUBJECT VARIABILITY BASELINE")
    print("===========================================\n")
    
    if 'margin' not in df.columns:
        df['margin'] = np.abs(df['sim_A'] - df['sim_B'])
        
    subject_stats = df.groupby('subject_id', observed=True).agg(
        n_trials=('correct', 'size'),
        accuracy=('correct', 'mean'),
        mean_margin=('margin', 'mean')
    ).reset_index()
    
    # Sort by accuracy
    subject_stats = subject_stats.sort_values('accuracy', ascending=False).reset_index(drop=True)
    
    print(f"{'Subject':<10} | {'Accuracy':<10} | {'Mean Margin':<15}")
    print("-" * 42)
    for _, row in subject_stats.iterrows():
        print(f"{row['subject_id']:<10} | {row['accuracy']*100:>7.2f}%   | {row['mean_margin']:.4f}")
        
    # Compute correlation
    acc = subject_stats['accuracy'].values
    margin = subject_stats['mean_margin'].values
    corr = np.corrcoef(acc, margin)[0, 1]
    
    print("\n--- Summary ---")
    print(f"Number of subjects : {len(subject_stats)}")
    print(f"Max Accuracy       : {acc.max()*100:.2f}%")
    print(f"Min Accuracy       : {acc.min()*100:.2f}%")
    print(f"Acc-Margin Pearson R: {corr:.4f}")
    
    print("\n--- Interpretation ---")
    if corr > 0.70:
        print("High Correlation: Mean Margin already strongly explains subject-level failures.")
        print("Subject-distance models might be redundant unless they explain the variance that margin misses.")
    else:
        print("Low/Moderate Correlation: Mean Margin does NOT fully explain subject-level failures.")
        print("This justifies Phase 3: Subject Distance might capture orthogonal information about failure modes.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="matchnet_predictions.csv")
    args = parser.parse_args()
    
    try:
        df = pd.read_csv(args.csv)
    except FileNotFoundError:
        print(f"Error: Could not find {args.csv}")
        return
        
    analyze_subject_variability(df)

if __name__ == "__main__":
    main()
