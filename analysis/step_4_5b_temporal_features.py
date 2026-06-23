import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import xgboost as xgb
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def add_temporal_features(df):
    print("Computing strict causal temporal features...")
    df = df.sort_values(['subject_id', 'trial_id', 'window_id']).reset_index(drop=True)
    
    # 1. Rolling STD of margin (window=5, min_periods=1)
    df['rolling_std_margin'] = df.groupby(['subject_id', 'trial_id'])['margin'].rolling(window=5, min_periods=1).std().reset_index(level=[0,1], drop=True)
    df['rolling_std_margin'] = df['rolling_std_margin'].fillna(0.0)
    
    # 2. Trial Consistency
    def compute_consistency(group):
        preds = group['prediction'].values
        consistencies = []
        for i in range(len(preds)):
            if i == 0:
                consistencies.append(1.0)
            else:
                consistencies.append(np.mean(preds[:i] == preds[i]))
        group['trial_consistency'] = consistencies
        return group
        
    df = df.groupby(['subject_id', 'trial_id'], group_keys=False).apply(compute_consistency)
    return df

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="subject_distance_predictions.csv")
    args = parser.parse_args()
    
    try:
        df = pd.read_csv(args.csv)
    except FileNotFoundError:
        print(f"Cannot find {args.csv}. Please specify the correct path.")
        return
        
    df = df[np.isfinite(df['margin'])].reset_index(drop=True)
    df = add_temporal_features(df)
    
    subjects = df['subject_id'].unique()
    features = ['margin', 'sim_A', 'sim_B', 'rolling_std_margin', 'trial_consistency']
    
    print("\n===========================================")
    print("STEP 4.5B: TEMPORAL CONFIDENCE FEATURES")
    print("===========================================")
    print(f"Sanitized feature set: {features}")
    
    # ----------------------------------------------------
    # PHASE 4.5B.1: FEATURE AUDIT
    # ----------------------------------------------------
    print("\n--- PHASE 4.5B.1: INDIVIDUAL FEATURE AUROC AUDIT ---")
    print(f"{'Feature':<25} | {'AUROC':<10} | {'Direction':<25}")
    print("-" * 65)
    for feat in features:
        auc = roc_auc_score(df['correct'], df[feat])
        if auc < 0.5:
            auc = 1.0 - auc
            direction = "Negative (Lower=More Confident)"
        else:
            direction = "Positive (Higher=More Confident)"
        print(f"{feat:<25} | {auc:<10.4f} | {direction:<25}")
    print("-" * 65)
    print("\nProceeding to Nested LOSO evaluation...")
    # ----------------------------------------------------
    
    print(f"\nRunning strict nested LOSO across {len(subjects)} subjects...\n")
    
    df['prob_lr_margin'] = 0.0
    df['prob_lr_features'] = 0.0
    df['prob_xgb_features'] = 0.0
    
    for test_subj in subjects:
        train_mask = df['subject_id'] != test_subj
        test_mask = df['subject_id'] == test_subj
        
        y_train = df.loc[train_mask, 'correct']
        
        # A) LR on Margin Only
        lr1 = LogisticRegression()
        lr1.fit(df.loc[train_mask, ['margin']], y_train)
        df.loc[test_mask, 'prob_lr_margin'] = lr1.predict_proba(df.loc[test_mask, ['margin']])[:, 1]
        
        # B) LR on All Features
        X_train = df.loc[train_mask, features]
        X_test = df.loc[test_mask, features]
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        lr2 = LogisticRegression(max_iter=1000)
        lr2.fit(X_train_scaled, y_train)
        df.loc[test_mask, 'prob_lr_features'] = lr2.predict_proba(X_test_scaled)[:, 1]
        
        # C) XGBoost on All Features
        xgb_model = xgb.XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.05, n_jobs=-1, eval_metric='logloss')
        xgb_model.fit(X_train, y_train)
        df.loc[test_mask, 'prob_xgb_features'] = xgb_model.predict_proba(X_test)[:, 1]
        
    print("--- GLOBAL AUROC RESULTS ---")
    
    auroc_margin = roc_auc_score(df['correct'], df['margin'])
    auroc_lr_margin = roc_auc_score(df['correct'], df['prob_lr_margin'])
    auroc_lr_features = roc_auc_score(df['correct'], df['prob_lr_features'])
    auroc_xgb_features = roc_auc_score(df['correct'], df['prob_xgb_features'])
    
    print(f"A) Margin Only (Raw)                : {auroc_margin:.4f}  [The Absolute Floor]")
    print(f"B) LR on Margin (Platt scaling)     : {auroc_lr_margin:.4f}  [The Cross-Validation Floor]")
    print(f"C) LR on Sanitized Features         : {auroc_lr_features:.4f}  [Linear Context Gain]")
    print(f"D) XGBoost on Sanitized Features    : {auroc_xgb_features:.4f}  [Non-linear Context Gain]")
    
    print("\nInterpretation:")
    if auroc_xgb_features > auroc_lr_margin + 0.02:
        print("XGBoost achieved a meaningful improvement (>0.02) over the margin baseline.")
        print("Temporal statistics and non-linear interactions carry significant confidence information.")
    else:
        print("XGBoost failed to meaningfully improve ranking over the baseline.")
        print("The 50% noise floor is too strong. The confidence head roadmap is officially dead.")
        
    print("===========================================\n")

if __name__ == "__main__":
    main()
