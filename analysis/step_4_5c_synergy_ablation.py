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
    print("STEP 4.5C: SYNERGY ABLATION")
    print("===========================================")
    
    models = ["Logistic Regression", "XGBoost"]
    
    for model in models:
        print(f"\n--- {model.upper()} ---")
        auc_margin = run_ablation(df, ['margin'], model)
        auc_consistency = run_ablation(df, ['trial_consistency'], model)
        auc_combined = run_ablation(df, ['margin', 'trial_consistency'], model)
        
        print(f"{'Margin Only':<20}: {auc_margin:.4f}")
        print(f"{'Consistency Only':<20}: {auc_consistency:.4f}")
        print(f"{'Combined':<20}: {auc_combined:.4f}")
        
        gain_margin = auc_combined - auc_consistency
        gain_consistency = auc_combined - auc_margin
        
        print("\nSynergy Analysis:")
        print(f"Gain from adding Margin to Consistency : +{gain_margin:.4f}")
        print(f"Gain from adding Consistency to Margin : +{gain_consistency:.4f}")
        
        if gain_margin < 0.01:
            print("Conclusion: Consistency is carrying everything. Margin adds negligible value.")
        else:
            print("Conclusion: Features are synergistic. Margin provides independent confidence value.")
            
    print("\n===========================================\n")

if __name__ == "__main__":
    main()
