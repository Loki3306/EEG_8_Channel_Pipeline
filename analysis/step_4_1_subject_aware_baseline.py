import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import warnings
warnings.filterwarnings('ignore')

def compute_ece(correct, conf, n_bins=10):
    """Compute Expected Calibration Error"""
    min_c, max_c = conf.min(), conf.max()
    bins = np.linspace(min_c, max_c, n_bins + 1)
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

def step_4_1_subject_aware_baseline(predictions_csv):
    print("\n===========================================")
    print("STEP 4.1: SUBJECT-AWARE CALIBRATION BASELINE")
    print("===========================================\n")
    
    df = pd.read_csv(predictions_csv)
    if 'margin' not in df.columns:
        df['margin'] = np.abs(df['sim_A'] - df['sim_B'])
        
    # Prepare features and target
    X = df[['margin', 'subject_id']].copy()
    y = df['correct'].values
    
    # 5-Fold Cross Validation Stratified by Subject
    # This ensures we get ~20% of every subject's trials in the test set of each fold
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Initialize OOF prediction arrays
    oof_baseline = np.zeros(len(df))
    oof_subject_aware = np.zeros(len(df))
    
    # Baseline Model: Logistic Regression on margin only
    # Subject-Aware Model: Logistic Regression on margin + one-hot subject_id
    
    # Preprocessor for Subject-Aware model
    preprocessor = ColumnTransformer(
        transformers=[
            ('margin', 'passthrough', ['margin']),
            ('subject', OneHotEncoder(handle_unknown='ignore', sparse_output=False), ['subject_id'])
        ]
    )
    
    print("Evaluating models using 5-Fold Cross Validation over trials...")
    
    for train_idx, test_idx in skf.split(X, X['subject_id']):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # 1. Baseline Model (Margin Only)
        # We need to reshape margin to 2D for sklearn
        margin_train = X_train[['margin']]
        margin_test = X_test[['margin']]
        
        lr_base = LogisticRegression(class_weight='balanced')
        lr_base.fit(margin_train, y_train)
        oof_baseline[test_idx] = lr_base.predict_proba(margin_test)[:, 1]
        
        # 2. Subject-Aware Model
        X_train_transformed = preprocessor.fit_transform(X_train)
        X_test_transformed = preprocessor.transform(X_test)
        
        lr_subj = LogisticRegression(class_weight='balanced', max_iter=500)
        lr_subj.fit(X_train_transformed, y_train)
        oof_subject_aware[test_idx] = lr_subj.predict_proba(X_test_transformed)[:, 1]
        
    # Calculate Global Metrics
    print("\n--- Calibration Results ---")
    
    # Baseline Metrics
    base_auroc = roc_auc_score(y, oof_baseline)
    base_brier = brier_score_loss(y, oof_baseline)
    base_ece = compute_ece(y, oof_baseline)
    
    # Subject-Aware Metrics
    subj_auroc = roc_auc_score(y, oof_subject_aware)
    subj_brier = brier_score_loss(y, oof_subject_aware)
    subj_ece = compute_ece(y, oof_subject_aware)
    
    print(f"{'Model':<25} | {'Global AUROC':<15} | {'Brier Score':<15} | {'ECE':<10}")
    print("-" * 75)
    print(f"{'Baseline (Margin)':<25} | {base_auroc:<15.4f} | {base_brier:<15.4f} | {base_ece:<10.4f}")
    print(f"{'Subject-Aware (Margin+ID)':<25} | {subj_auroc:<15.4f} | {subj_brier:<15.4f} | {subj_ece:<10.4f}")
    
    print("\n--- Interpretation ---")
    if subj_auroc > base_auroc + 0.05:
        print("MASSIVE IMPROVEMENT: Subject identity provides critical calibration information.")
        print("This proves that 'Confidence Reliability is Subject-Dependent', forming a perfect publishable thesis.")
    elif subj_auroc > base_auroc + 0.01:
        print("MODERATE IMPROVEMENT: Subject identity helps, but isn't a silver bullet alone.")
    else:
        print("NO IMPROVEMENT: Subject identity does not improve global ranking.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="matchnet_predictions.csv")
    args = parser.parse_args()
    
    try:
        step_4_1_subject_aware_baseline(args.csv)
    except FileNotFoundError:
        print(f"Error: Could not find {args.csv}")
        return

if __name__ == "__main__":
    main()
