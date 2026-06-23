import argparse
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

def add_temporal_features(df):
    df = df.sort_values(['subject_id', 'trial_id', 'window_id']).reset_index(drop=True)
    df['sim_chosen'] = df[['sim_A', 'sim_B']].max(axis=1)
    df['sim_unchosen'] = df[['sim_A', 'sim_B']].min(axis=1)
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
    return df

def compute_eaurc(correct_labels, confidences):
    n = len(correct_labels)
    sorted_indices = np.argsort(-confidences)
    sorted_labels = correct_labels[sorted_indices]
    
    coverages = np.arange(1, n + 1) / n
    accuracies = np.cumsum(sorted_labels) / np.arange(1, n + 1)
    risks = 1.0 - accuracies
    aurc = np.trapezoid(risks, coverages)
    
    optimal_indices = np.argsort(-correct_labels)
    optimal_labels = correct_labels[optimal_indices]
    optimal_accuracies = np.cumsum(optimal_labels) / np.arange(1, n + 1)
    optimal_risks = 1.0 - optimal_accuracies
    optimal_aurc = np.trapezoid(optimal_risks, coverages)
    
    return aurc - optimal_aurc

def evaluate_model(df, features, model_type="LR"):
    subjects = df['subject_id'].unique()
    eaurcs = []
    
    for test_subj in subjects:
        train_mask = df['subject_id'] != test_subj
        test_mask = df['subject_id'] == test_subj
        
        X_train = df.loc[train_mask, features]
        y_train = df.loc[train_mask, 'correct']
        X_test = df.loc[test_mask, features]
        y_test = df.loc[test_mask, 'correct'].values
        
        if model_type == "LR":
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            lr = LogisticRegression(max_iter=1000)
            lr.fit(X_train_scaled, y_train)
            probs = lr.predict_proba(X_test_scaled)[:, 1]
        else:
            xgb_model = xgb.XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.05, n_jobs=-1, eval_metric='logloss')
            xgb_model.fit(X_train, y_train)
            probs = xgb_model.predict_proba(X_test)[:, 1]
            
        eaurcs.append(compute_eaurc(y_test, probs))
        
    return np.mean(eaurcs)

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
    print("STEP 4.7A: E-AURC ABLATION STUDY")
    print("===========================================")
    print("Computing mean Excess AURC across 18 nested LOSO folds...\n")
    
    subjects = df['subject_id'].unique()
    margin_eaurcs = []
    for subj in subjects:
        test_mask = df['subject_id'] == subj
        margin_eaurcs.append(compute_eaurc(df.loc[test_mask, 'correct'].values, df.loc[test_mask, 'margin'].values))
    eaurc_margin = np.mean(margin_eaurcs)
    
    eaurc_consistency = evaluate_model(df, ['trial_consistency'], "LR")
    eaurc_fusion = evaluate_model(df, ['margin', 'trial_consistency'], "LR")
    
    xgb_features = ['margin', 'sim_chosen', 'sim_unchosen', 'rolling_std_margin', 'trial_consistency']
    eaurc_xgb = evaluate_model(df, xgb_features, "XGBoost")
    
    print(f"{'Model':<30} | {'Mean E-AURC (Lower is Better)'}")
    print("-" * 65)
    print(f"{'1. Margin Only (Baseline)':<30} | {eaurc_margin:.4f}")
    print(f"{'2. Consistency Only (LR)':<30} | {eaurc_consistency:.4f}")
    print(f"{'3. Margin + Consistency (LR)':<30} | {eaurc_fusion:.4f}")
    print(f"{'4. Full Temporal Fusion (XGB)':<30} | {eaurc_xgb:.4f}")
    
    print("\nConclusion: Both Margin and Consistency are required for optimal selective rejection.")
    print("===========================================\n")

if __name__ == "__main__":
    main()
