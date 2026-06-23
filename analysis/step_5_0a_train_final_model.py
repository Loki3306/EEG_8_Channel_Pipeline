import pandas as pd
import numpy as np
import xgboost as xgb
import os
import argparse

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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="subject_distance_predictions.csv")
    args = parser.parse_args()
    
    print("Loading data...")
    try:
        df = pd.read_csv(args.csv)
    except FileNotFoundError:
        print(f"File {args.csv} not found.")
        return
        
    df = df[np.isfinite(df['margin'])].reset_index(drop=True)
    
    print("Computing offline features...")
    df = add_temporal_features(df)
    
    features = ['margin', 'sim_chosen', 'sim_unchosen', 'rolling_std_margin', 'trial_consistency']
    X = df[features]
    y = df['correct'].values
    
    print("Training final XGBoost model on ALL subjects...")
    model = xgb.XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.05, n_jobs=-1, eval_metric='logloss')
    model.fit(X, y)
    
    os.makedirs("models", exist_ok=True)
    model_path = "models/confidence_model.json"
    model.save_model(model_path)
    print(f"Model successfully saved to {model_path}")

if __name__ == "__main__":
    main()
