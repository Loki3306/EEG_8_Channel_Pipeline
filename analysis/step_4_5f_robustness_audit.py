import argparse
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

def add_robustness_features(df):
    df = df.sort_values(['subject_id', 'trial_id', 'window_id']).reset_index(drop=True)
    
    df['sim_chosen'] = df[['sim_A', 'sim_B']].max(axis=1)
    df['sim_unchosen'] = df[['sim_A', 'sim_B']].min(axis=1)
    
    df['rolling_std_margin'] = df.groupby(['subject_id', 'trial_id'])['margin'].rolling(window=5, min_periods=1).std().reset_index(level=[0,1], drop=True)
    df['rolling_std_margin'] = df['rolling_std_margin'].fillna(0.0)
    
    def compute_consistencies(group):
        preds = group['prediction'].values
        n = len(preds)
        
        c_all, c_3, c_5, c_10 = np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n)
        
        for i in range(n):
            if i == 0:
                c_all[i], c_3[i], c_5[i], c_10[i] = 1.0, 1.0, 1.0, 1.0
            else:
                c_all[i] = np.mean(preds[:i] == preds[i])
                c_3[i] = np.mean(preds[max(0, i-3):i] == preds[i])
                c_5[i] = np.mean(preds[max(0, i-5):i] == preds[i])
                c_10[i] = np.mean(preds[max(0, i-10):i] == preds[i])
                
        group['consistency_all'] = c_all
        group['consistency_last_3'] = c_3
        group['consistency_last_5'] = c_5
        group['consistency_last_10'] = c_10
        
        preds_bin = np.where(preds == 'A', 1.0, -1.0)
        s = pd.Series(preds_bin)
        group['consistency_ewma_span5'] = s.ewm(span=5, min_periods=1).mean().abs().values
        group['consistency_ewma_span10'] = s.ewm(span=10, min_periods=1).mean().abs().values
        
        return group

    df = df.groupby(['subject_id', 'trial_id'], group_keys=False).apply(compute_consistencies)
    return df

def evaluate_nested_loso(df, features, model_type):
    subjects = df['subject_id'].unique()
    probs = np.zeros(len(df))
    
    for test_subj in subjects:
        train_mask = df['subject_id'] != test_subj
        test_mask = df['subject_id'] == test_subj
        
        X_train = df.loc[train_mask, features]
        y_train = df.loc[train_mask, 'correct']
        X_test = df.loc[test_mask, features]
        
        if model_type == "LR":
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
            
    return roc_auc_score(df['correct'], probs)

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
    df = add_robustness_features(df)
    
    print("\n===========================================")
    print("STEP 4.5F: CONSISTENCY ROBUSTNESS AUDIT")
    print("===========================================")
    print("Testing AUROC stability across 6 consistency definitions...\n")
    
    variations = [
        'consistency_all',
        'consistency_last_3',
        'consistency_last_5',
        'consistency_last_10',
        'consistency_ewma_span5',
        'consistency_ewma_span10'
    ]
    
    print(f"{'Definition':<25} | {'Linear Fusion':<15} | {'XGBoost (All feats)':<20}")
    print("-" * 65)
    
    for var in variations:
        fusion_features = ['margin', var]
        xgb_features = ['margin', 'sim_chosen', 'sim_unchosen', 'rolling_std_margin', var]
        
        auc_lr = evaluate_nested_loso(df, fusion_features, "LR")
        auc_xgb = evaluate_nested_loso(df, xgb_features, "XGBoost")
        
        print(f"{var:<25} | {auc_lr:<15.4f} | {auc_xgb:<20.4f}")
        
    print("-" * 65)
    print("\nInterpretation Criteria:")
    print("- If all AUROCs stay in the 0.74-0.78 band, the temporal confidence signal is highly robust.")
    print("- If only one variation works, the previous result was a feature-engineering artifact.")
    print("===========================================\n")

if __name__ == "__main__":
    main()
