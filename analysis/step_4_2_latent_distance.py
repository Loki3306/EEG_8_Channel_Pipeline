import argparse
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

def fit_evaluate_combined(X, y):
    """Evaluate a Logistic Regression using 5-Fold Stratified CV to combine features safely."""
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_preds = np.zeros(len(y))
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    for train_idx, test_idx in skf.split(X_scaled, y):
        X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
        y_train = y[train_idx]
        
        lr = LogisticRegression(class_weight='balanced')
        lr.fit(X_train, y_train)
        oof_preds[test_idx] = lr.predict_proba(X_test)[:, 1]
        
    return roc_auc_score(y, oof_preds)

def step_4_2_latent_distance(predictions_csv):
    print("\n===========================================")
    print("STEP 4.2: LATENT SUBJECT DISTANCE EVALUATION")
    print("===========================================\n")
    
    df = pd.read_csv(predictions_csv)
    
    valid_mask = np.isfinite(df['euc_dist']) & np.isfinite(df['mah_dist'])
    n_invalid = len(df) - valid_mask.sum()
    if n_invalid > 0:
        print(f"Warning: Dropped {n_invalid} rows with NaN or Infinite distances.")
        df = df[valid_mask].reset_index(drop=True)
        
    print(f"Evaluating {len(df)} predictions...\n")
    
    print("--- Distribution Diagnostics ---")
    print(df[['euc_dist', 'mah_dist']].describe().round(4))
    print("\n")
    
    print("--- Subject-Level Summaries ---")
    print(f"{'Subject':<10} | {'Accuracy':<10} | {'Mean EucDist':<15} | {'Mean MahDist':<15}")
    print("-" * 60)
    
    for subj_id, group in df.groupby('subject_id', observed=True):
        acc = group['correct'].mean()
        mean_euc = group['euc_dist'].mean()
        mean_mah = group['mah_dist'].mean()
        print(f"{subj_id:<10} | {acc:<10.3f} | {mean_euc:<15.4f} | {mean_mah:<15.4f}")
        
    print("\n--- Global Confidence Modeling ---")
    
    y_true = df['correct'].values
    margin = df['margin'].values
    neg_euc = -df['euc_dist'].values
    neg_mah = -df['mah_dist'].values
    
    # 1. Standalone Models (Direct AUROC)
    auroc_margin = roc_auc_score(y_true, margin)
    auroc_euc = roc_auc_score(y_true, neg_euc)
    auroc_mah = roc_auc_score(y_true, neg_mah)
    
    # 2. Combined Models (Logistic Regression CV)
    X_margin_euc = df[['margin', 'euc_dist']].values
    auroc_comb_euc = fit_evaluate_combined(X_margin_euc, y_true)
    
    X_margin_mah = df[['margin', 'mah_dist']].values
    auroc_comb_mah = fit_evaluate_combined(X_margin_mah, y_true)
    
    print(f"{'Method':<25} | {'AUROC':<10}")
    print("-" * 40)
    print(f"{'Margin':<25} | {auroc_margin:.4f}")
    print(f"{'Euclidean':<25} | {auroc_euc:.4f}")
    print(f"{'Mahalanobis':<25} | {auroc_mah:.4f}")
    print(f"{'Margin + Euclidean':<25} | {auroc_comb_euc:.4f}")
    print(f"{'Margin + Mahalanobis':<25} | {auroc_comb_mah:.4f}")
    
    print("\n--- Interpretation ---")
    if auroc_comb_mah > auroc_margin + 0.02:
        print("SUCCESS: Distance is complementary to margin!")
        print("The latent space contains subject-familiarity information that margin missed.")
    elif auroc_mah > 0.60:
        print("PARTIAL: Distance captures some missing signal, but combining it with margin doesn't yield huge gains.")
    else:
        print("NEGATIVE: Latent Distance hypothesis rejected. Distance completely fails to capture the confidence reliability.")

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
