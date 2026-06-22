import argparse
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')

def step_4_2_latent_distance(predictions_csv):
    print("\n===========================================")
    print("STEP 4.2: LATENT SUBJECT DISTANCE EVALUATION")
    print("===========================================\n")
    
    df = pd.read_csv(predictions_csv)
    
    # Check for NaNs or Infs
    valid_mask = np.isfinite(df['euc_dist']) & np.isfinite(df['mah_dist'])
    n_invalid = len(df) - valid_mask.sum()
    if n_invalid > 0:
        print(f"Warning: Dropped {n_invalid} rows with NaN or Infinite distances.")
        df = df[valid_mask].reset_index(drop=True)
        
    y_true = df['correct'].values
    margin = df['margin'].values
    
    # Note: We use negative distance because higher distance = lower confidence (lower probability of being correct)
    neg_euc = -df['euc_dist'].values
    neg_mah = -df['mah_dist'].values
    
    print(f"Evaluating {len(df)} predictions...")
    
    # Calculate Global AUROCs
    margin_auroc = roc_auc_score(y_true, margin)
    euc_auroc = roc_auc_score(y_true, neg_euc)
    mah_auroc = roc_auc_score(y_true, neg_mah)
    
    print("\n--- Latent Distance Evaluation ---")
    print(f"{'Feature':<25} | {'Global AUROC'}")
    print("-" * 45)
    print(f"{'Baseline (Margin)':<25} | {margin_auroc:.4f}")
    print(f"{'- Euclidean Distance':<25} | {euc_auroc:.4f}")
    print(f"{'- Mahalanobis Distance':<25} | {mah_auroc:.4f}")
    
    print("\n--- Subject-Level Analysis ---")
    print(f"{'Subject':<10} | {'Acc':<6} | {'Margin AUROC':<15} | {'Euc AUROC':<12} | {'Mah AUROC':<12}")
    print("-" * 65)
    
    for subj_id, group in df.groupby('subject_id', observed=True):
        acc = group['correct'].mean()
        try:
            subj_margin = roc_auc_score(group['correct'], group['margin'])
            subj_euc = roc_auc_score(group['correct'], -group['euc_dist'])
            subj_mah = roc_auc_score(group['correct'], -group['mah_dist'])
        except ValueError:
            subj_margin = subj_euc = subj_mah = 0.5
            
        print(f"{subj_id:<10} | {acc:.3f} | {subj_margin:<15.4f} | {subj_euc:<12.4f} | {subj_mah:<12.4f}")
        
    print("\n--- Interpretation ---")
    if mah_auroc > margin_auroc:
        print("SUCCESS: Latent Distance outperforms Margin. The missing confidence information")
        print("is successfully captured by the distance from the training manifold!")
    else:
        print("NEGATIVE: Latent Distance does not outperform Margin on its own.")
        print("If Mahalanobis AUROC is around 0.5, it means the latent space geometry")
        print("is completely uncalibrated to correctness. A Confidence Head is the only way forward.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="subject_distance_predictions.csv")
    args = parser.parse_args()
    
    try:
        step_4_2_latent_distance(args.csv)
    except FileNotFoundError:
        print(f"Error: Could not find {args.csv}. Make sure you ran export_subject_distance.py first.")
        return

if __name__ == "__main__":
    main()
