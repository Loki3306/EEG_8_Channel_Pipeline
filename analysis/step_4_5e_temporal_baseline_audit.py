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
    
    df['rolling_std_margin'] = df.groupby(['subject_id', 'trial_id'])['margin'].rolling(window=5, min_periods=1).std().reset_index(level=[0,1], drop=True)
    df['rolling_std_margin'] = df['rolling_std_margin'].fillna(0.0)
    
    # Moving-average margin (smoothed margin)
    df['rolling_mean_margin'] = df.groupby(['subject_id', 'trial_id'])['margin'].rolling(window=5, min_periods=1).mean().reset_index(level=[0,1], drop=True)
    df['rolling_mean_margin'] = df['rolling_mean_margin'].fillna(0.0)
    
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
    return auc

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
    print("STEP 4.5E: TEMPORAL BASELINE AUDIT")
    print("===========================================")
    
    print("\nRunning Strict Nested LOSO Evaluations...\n")
    
    # 1. Consistency only (Logistic Regression)
    auc_consistency = run_ablation(df, ['trial_consistency'], "Logistic Regression")
    print(f"{'Consistency Only (LR)':<35}: {auc_consistency:.4f}")
    
    # 2. Margin only (Logistic Regression)
    auc_margin = run_ablation(df, ['margin'], "Logistic Regression")
    print(f"{'Margin Only (LR)':<35}: {auc_margin:.4f}")
    
    # 3. Simple Linear Fusion (Margin + Consistency)
    auc_fusion = run_ablation(df, ['margin', 'trial_consistency'], "Logistic Regression")
    print(f"{'Linear Fusion (Margin+Consistency)':<35}: {auc_fusion:.4f}")
    
    # 4. Moving-average margin (smoothed margin)
    auc_smoothed = run_ablation(df, ['rolling_mean_margin'], "Logistic Regression")
    print(f"{'Smoothed Margin Only (LR)':<35}: {auc_smoothed:.4f}")
    
    # 5. XGBoost (all valid features)
    features_xgb = ['margin', 'sim_chosen', 'sim_unchosen', 'rolling_std_margin', 'trial_consistency']
    auc_xgb = run_ablation(df, features_xgb, "XGBoost")
    print(f"{'XGBoost (All sanitized features)':<35}: {auc_xgb:.4f}")
    
    print("\nInterpretation:")
    if auc_xgb - auc_fusion > 0.02:
        print("XGBoost significantly outperforms Linear Fusion. Non-linear interactions are necessary.")
    else:
        print("Linear Fusion is sufficient. The confidence framework is elegantly simple:")
        print("Confidence = a * Margin + b * Consistency")
        
    print("===========================================\n")

if __name__ == "__main__":
    main()
