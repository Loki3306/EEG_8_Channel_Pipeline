import argparse
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

def add_temporal_features(df):
    df = df.sort_values(['subject_id', 'trial_id', 'window_id']).reset_index(drop=True)
    
    # 1. Rolling STD of margin
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
    
    # FIX LEAKAGE: sim_A and sim_B leak the ground truth because A is always attended.
    # At inference time, we only know sim_chosen and sim_unchosen.
    df['sim_chosen'] = df[['sim_A', 'sim_B']].max(axis=1)
    df['sim_unchosen'] = df[['sim_A', 'sim_B']].min(axis=1)
    
    return df

def run_ablation(df, features, model_name="Logistic Regression"):
    subjects = df['subject_id'].unique()
    probs = np.zeros(len(df))
    
    for test_subj in subjects:
        train_mask = df['subject_id'] != test_subj
        test_mask = df['subject_id'] == test_subj
        
        X_train = df.loc[train_mask, features]
        y_train = df.loc[train_mask, 'correct']
        X_test = df.loc[test_mask, features]
        
        if model_name == "Logistic Regression":
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            lr = LogisticRegression(max_iter=1000)
            lr.fit(X_train_scaled, y_train)
            probs[test_mask] = lr.predict_proba(X_test_scaled)[:, 1]
        else:
            xgb_model = xgb.XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.05, n_jobs=-1, eval_metric='logloss')
            xgb_model.fit(X_train, y_train)
            probs[test_mask] = xgb_model.predict_proba(X_test)[:, 1]
            
    auc = roc_auc_score(df['correct'], probs)
    print(f"\nModel: {model_name} with features {features}")
    print(f"Global AUROC: {auc:.4f}")
    
    print("Per-subject AUROCs:")
    subj_aucs = []
    for subj in subjects:
        mask = df['subject_id'] == subj
        subj_auc = roc_auc_score(df.loc[mask, 'correct'], probs[mask])
        subj_aucs.append(f"{subj}: {subj_auc:.3f}")
    print(" | ".join(subj_aucs))

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
    
    print("\n===========================================")
    print("STEP 4.5B LEAKAGE FIX & ABLATIONS")
    print("===========================================")
    
    print("\n--- SANITIZED INDIVIDUAL FEATURE AUROC AUDIT ---")
    eval_features = ['margin', 'sim_chosen', 'sim_unchosen', 'rolling_std_margin', 'trial_consistency']
    for feat in eval_features:
        auc = roc_auc_score(df['correct'], df[feat])
        if auc < 0.5:
            auc = 1.0 - auc
            direction = "Negative (Lower=More Confident)"
        else:
            direction = "Positive (Higher=More Confident)"
        print(f"{feat:<20} | {auc:<10.4f} | {direction}")
    
    print("\n--- ABLATION RUNS (NESTED LOSO) ---")
    run_ablation(df, ['margin'], "Logistic Regression")
    run_ablation(df, ['margin', 'sim_chosen', 'sim_unchosen'], "Logistic Regression")
    run_ablation(df, ['margin', 'rolling_std_margin'], "Logistic Regression")
    run_ablation(df, ['margin', 'trial_consistency'], "Logistic Regression")
    run_ablation(df, ['margin', 'sim_chosen', 'rolling_std_margin', 'trial_consistency'], "XGBoost")

if __name__ == "__main__":
    main()
