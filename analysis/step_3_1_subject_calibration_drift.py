import argparse
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score

def analyze_calibration_drift(df):
    print("\n===========================================")
    print("STEP 3.1: SUBJECT CALIBRATION DRIFT")
    print("===========================================\n")
    
    if 'margin' not in df.columns:
        df['margin'] = np.abs(df['sim_A'] - df['sim_B'])
        
    subject_stats = []
    
    for subj_id, group in df.groupby('subject_id', observed=True):
        n_samples = len(group)
        acc = group['correct'].mean()
        mean_margin = group['margin'].mean()
        
        try:
            auroc = roc_auc_score(group['correct'], group['margin'])
        except ValueError:
            auroc = np.nan # Handles case if subject is 100% correct or 0% correct
            
        subject_stats.append({
            'subject_id': subj_id,
            'accuracy': acc,
            'mean_margin': mean_margin,
            'auroc': auroc,
            'n_samples': n_samples
        })
        
    stats_df = pd.DataFrame(subject_stats)
    
    # Sort by accuracy descending
    stats_df = stats_df.sort_values('accuracy', ascending=False).reset_index(drop=True)
    
    print(f"{'Subject':<10} | {'Accuracy':<10} | {'Mean Margin':<15} | {'Margin AUROC':<15} | {'Samples':<10}")
    print("-" * 72)
    for _, row in stats_df.iterrows():
        auroc_str = f"{row['auroc']:.4f}" if not pd.isna(row['auroc']) else "N/A"
        print(f"{row['subject_id']:<10} | {row['accuracy']*100:>7.2f}%   | {row['mean_margin']:<15.4f} | {auroc_str:<15} | {row['n_samples']:<10}")
        
    print("\n--- Correlation Matrix ---")
    corr_df = stats_df[['accuracy', 'mean_margin', 'auroc']].corr()
    print(corr_df.round(4))
    
    print("\n--- Interpretation ---")
    print("Look for Subject Calibration Drift:")
    print("If subjects with similar Mean Margin have vastly different Accuracy,")
    print("it means the same raw confidence score means different things for different subjects.")
    print("If this occurs, Subject-Aware Calibration becomes the primary novelty.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="matchnet_predictions.csv")
    args = parser.parse_args()
    
    try:
        df = pd.read_csv(args.csv)
    except FileNotFoundError:
        print(f"Error: Could not find {args.csv}")
        return
        
    analyze_calibration_drift(df)

if __name__ == "__main__":
    main()
