import argparse
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
import xgboost as xgb
from sklearn.inspection import permutation_importance
import warnings
warnings.filterwarnings('ignore')

def add_temporal_features(df):
    df = df.sort_values(['subject_id', 'trial_id', 'window_id']).reset_index(drop=True)
    
    df['rolling_std_margin'] = df.groupby(['subject_id', 'trial_id'])['margin'].rolling(window=5, min_periods=1).std().reset_index(level=[0,1], drop=True)
    df['rolling_std_margin'] = df['rolling_std_margin'].fillna(0.0)
    
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
    
    features = ['margin', 'sim_chosen', 'sim_unchosen', 'rolling_std_margin', 'trial_consistency']
    subjects = df['subject_id'].unique()
    
    print("\n===========================================")
    print("STEP 4.5D: XGBOOST FEATURE IMPORTANCE AUDIT")
    print("===========================================")
    
    fold_gain_importances = {f: [] for f in features}
    fold_perm_importances = {f: [] for f in features}
    
    for test_subj in subjects:
        train_mask = df['subject_id'] != test_subj
        test_mask = df['subject_id'] == test_subj
        
        X_train = df.loc[train_mask, features]
        y_train = df.loc[train_mask, 'correct']
        X_test = df.loc[test_mask, features]
        y_test = df.loc[test_mask, 'correct']
        
        xgb_model = xgb.XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.05, n_jobs=-1, eval_metric='logloss')
        xgb_model.fit(X_train, y_train)
        
        # 1. Gain Importance
        booster = xgb_model.get_booster()
        gain_dict = booster.get_score(importance_type='gain')
        for f in features:
            fold_gain_importances[f].append(gain_dict.get(f, 0.0))
            
        # 2. Permutation Importance (on Held-out test set!)
        def auroc_scorer(estimator, X, y):
            probs = estimator.predict_proba(X)[:, 1]
            return roc_auc_score(y, probs)
            
        perm_result = permutation_importance(xgb_model, X_test, y_test, scoring=auroc_scorer, n_repeats=5, random_state=42)
        for i, f in enumerate(features):
            fold_perm_importances[f].append(perm_result.importances_mean[i])
            
    print("\n--- XGBoost Gain Importance (Mean across folds) ---")
    avg_gain = {f: np.mean(fold_gain_importances[f]) for f in features}
    total_gain = sum(avg_gain.values())
    
    for f in sorted(features, key=lambda x: avg_gain[x], reverse=True):
        pct = (avg_gain[f] / total_gain) * 100 if total_gain > 0 else 0
        print(f"{f:<20} | {avg_gain[f]:<10.4f} ({pct:.1f}%)")
        
    print("\n--- Permutation Importance (Mean AUROC drop across unseen test folds) ---")
    avg_perm = {f: np.mean(fold_perm_importances[f]) for f in features}
    
    for f in sorted(features, key=lambda x: avg_perm[x], reverse=True):
        print(f"{f:<20} | {avg_perm[f]:<10.4f} AUROC Drop")

    print("\n===========================================\n")

if __name__ == "__main__":
    main()
