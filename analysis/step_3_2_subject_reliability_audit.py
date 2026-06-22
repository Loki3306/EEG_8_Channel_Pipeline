import argparse
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score

def compute_ece(correct, conf, n_bins=10):
    min_c, max_c = conf.min(), conf.max()
    bins = np.linspace(min_c, max_c, n_bins + 1)
    # Ensure unique bins in case of degenerate confidence
    bins = np.unique(bins)
    if len(bins) < 2:
        bins = np.linspace(0, 1, n_bins + 1)
        
    df = pd.DataFrame({'correct': correct, 'conf': conf})
    df['bin'] = pd.cut(df['conf'], bins=bins, include_lowest=True)
    
    stats = df.groupby('bin', observed=True).agg(
        count=('correct', 'size'),
        mean_conf=('conf', 'mean'),
        accuracy=('correct', 'mean')
    )
    
    ece = 0.0
    total = len(df)
    
    for _, row in stats.iterrows():
        if row['count'] > 0:
            ece += (row['count'] / total) * np.abs(row['accuracy'] - row['mean_conf'])
            
    return ece

def analyze_reliability_audit(df):
    print("\n===========================================")
    print("STEP 3.2: SUBJECT RELIABILITY AUDIT")
    print("===========================================\n")
    
    if 'margin' not in df.columns:
        df['margin'] = np.abs(df['sim_A'] - df['sim_B'])
        
    # Scale margin to [0.5, 1.0] for ECE calculation globally
    global_max_margin = df['margin'].max()
    df['margin_conf'] = 0.5 + 0.5 * (df['margin'] / global_max_margin)
    
    subject_stats = []
    
    for subj_id, group in df.groupby('subject_id', observed=True):
        n_samples = len(group)
        acc = group['correct'].mean()
        
        try:
            auroc = roc_auc_score(group['correct'], group['margin'])
        except ValueError:
            auroc = np.nan
            
        ece = compute_ece(group['correct'].values, group['margin_conf'].values)
        
        subject_stats.append({
            'subject_id': subj_id,
            'accuracy': acc,
            'auroc': auroc,
            'ece': ece
        })
        
    stats_df = pd.DataFrame(subject_stats)
    stats_df = stats_df.sort_values('accuracy', ascending=False).reset_index(drop=True)
    
    print(f"{'Subject':<10} | {'Accuracy':<10} | {'AUROC':<10} | {'ECE':<10}")
    print("-" * 50)
    for _, row in stats_df.iterrows():
        auroc_str = f"{row['auroc']:.4f}" if not pd.isna(row['auroc']) else "N/A"
        print(f"{row['subject_id']:<10} | {row['accuracy']*100:>7.2f}%   | {auroc_str:<10} | {row['ece']:.4f}")
        
    print("\n--- Correlation Matrix ---")
    corr_df = stats_df[['accuracy', 'auroc', 'ece']].corr()
    print(corr_df.round(4))
    
    print("\n--- Interpretation ---")
    print("Look for the correlation between Accuracy and ECE, and AUROC and ECE.")
    print("If bad subjects (low accuracy) have high ECE and low AUROC,")
    print("it proves that MatchNet's confidence fundamentally fails on OOD/weak subjects.")
    print("This is Subject-Dependent Confidence Calibration Failure.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="matchnet_predictions.csv")
    args = parser.parse_args()
    
    try:
        df = pd.read_csv(args.csv)
    except FileNotFoundError:
        print(f"Error: Could not find {args.csv}")
        return
        
    analyze_reliability_audit(df)

if __name__ == "__main__":
    main()
