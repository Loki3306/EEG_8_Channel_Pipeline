import argparse
import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')

def step_4_4_s11_ablation(csv_path):
    print("\n===========================================")
    print("STEP 4.4: S11 ABLATION (LATENT DISTANCE)")
    print("===========================================\n")
    
    df = pd.read_csv(csv_path)
    
    valid_mask = np.isfinite(df['mah_dist'])
    df = df[valid_mask].reset_index(drop=True)
    
    # 1. Compute Subject-Level Stats
    subj_stats = []
    for subj_id, group in df.groupby('subject_id', observed=True):
        acc = group['correct'].mean()
        mean_mah = group['mah_dist'].mean()
        try:
            auroc = roc_auc_score(group['correct'], group['margin'])
        except ValueError:
            auroc = 0.5
        subj_stats.append({
            'subject_id': subj_id,
            'acc': acc,
            'auroc': auroc,
            'mean_mah': mean_mah
        })
        
    stats_df = pd.DataFrame(subj_stats)
    
    # 2. Evaluate Function
    def evaluate_correlations(df_subset, label):
        r_acc, p_acc_r = pearsonr(df_subset['acc'], df_subset['mean_mah'])
        rho_acc, p_acc_rho = spearmanr(df_subset['acc'], df_subset['mean_mah'])
        
        r_auc, p_auc_r = pearsonr(df_subset['auroc'], df_subset['mean_mah'])
        rho_auc, p_auc_rho = spearmanr(df_subset['auroc'], df_subset['mean_mah'])
        
        print(f"\n--- {label} (N={len(df_subset)}) ---")
        print("Accuracy vs Mean Mahalanobis:")
        print(f"  Pearson r    = {r_acc:.3f} (p = {p_acc_r:.4f})")
        print(f"  Spearman rho = {rho_acc:.3f} (p = {p_acc_rho:.4f})")
        
        print("\nMargin AUROC vs Mean Mahalanobis:")
        print(f"  Pearson r    = {r_auc:.3f} (p = {p_auc_r:.4f})")
        print(f"  Spearman rho = {rho_auc:.3f} (p = {p_auc_rho:.4f})")
        
    # 3. WITH S11
    evaluate_correlations(stats_df, "WITH S11 (Original)")
    
    # 4. WITHOUT S11
    stats_no_s11 = stats_df[~stats_df['subject_id'].str.contains('S11')]
    evaluate_correlations(stats_no_s11, "WITHOUT S11 (Ablated)")
    
    print("\n===========================================")
    print("Interpretation Guide:")
    print("- Pearson r assumes linear relationships and is sensitive to extreme outliers (like S11).")
    print("- Spearman rho evaluates rank correlation and is robust to outliers.")
    print("- If 'WITHOUT S11' Pearson r drops to ~0, then S11 drove the entire linear effect.")
    print("- If 'WITHOUT S11' Spearman rho is also ~0, then rank ordering of distance is meaningless for the rest of the subjects.")
    print("===========================================\n")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="subject_distance_predictions.csv")
    args = parser.parse_args()
    
    try:
        step_4_4_s11_ablation(args.csv)
    except FileNotFoundError:
        print(f"Cannot find {args.csv}. Run export script first.")

if __name__ == "__main__":
    main()
