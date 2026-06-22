import argparse
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

def step_2_1_reliability(df):
    print("\n===========================================")
    print("STEP 2.1: RELIABILITY METRICS (AUROC/AUPRC)")
    print("===========================================\n")
    
    # Ensure margin is computed
    if 'margin' not in df.columns:
        df['margin'] = np.abs(df['sim_A'] - df['sim_B'])
        
    correct = df['correct'].values
    margin = df['margin'].values
    
    # Compute AUROC
    try:
        auroc = roc_auc_score(correct, margin)
        print(f"AUROC (Margin -> Correctness) : {auroc:.4f}")
    except Exception as e:
        print(f"AUROC Calculation Failed: {e}")
        
    # Compute AUPRC (Average Precision Score)
    try:
        auprc = average_precision_score(correct, margin)
        print(f"AUPRC (Margin -> Correctness) : {auprc:.4f}")
    except Exception as e:
        print(f"AUPRC Calculation Failed: {e}")
        
    print("\n--- Interpretation Guide ---")
    print("AUROC > 0.70 : Useful discriminatory power")
    print("AUROC > 0.80 : Strong discriminatory power")
    print("AUROC > 0.85 : Very strong discriminatory power")
    
    # Baseline AUPRC is the proportion of positives
    baseline_auprc = correct.mean()
    print(f"\nAUPRC Baseline (Chance)     : {baseline_auprc:.4f}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="matchnet_predictions.csv", help="Path to predictions CSV")
    args = parser.parse_args()
    
    try:
        df = pd.read_csv(args.csv)
    except FileNotFoundError:
        print(f"Error: Could not find {args.csv}")
        return
        
    step_2_1_reliability(df)

if __name__ == "__main__":
    main()
