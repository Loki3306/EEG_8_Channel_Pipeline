import pandas as pd
import numpy as np
import xgboost as xgb
import os
import argparse
import warnings
from sklearn.metrics import roc_auc_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings('ignore')

def add_temporal_features(df):
    df = df.sort_values(['subject_id', 'trial_id', 'window_id']).reset_index(drop=True)
    df['sim_chosen'] = df[['sim_A', 'sim_B']].max(axis=1)
    df['sim_unchosen'] = df[['sim_A', 'sim_B']].min(axis=1)
    df['sim_sum'] = df['sim_A'] + df['sim_B']
    df['sim_ratio'] = df['sim_unchosen'] / (df['sim_chosen'] + 1e-6)
    
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
    
    # Drift without embeddings: change in similarity over time
    df['sim_chosen_drift'] = df.groupby(['subject_id', 'trial_id'])['sim_chosen'].diff().abs().fillna(0.0)
    df['sim_unchosen_drift'] = df.groupby(['subject_id', 'trial_id'])['sim_unchosen'].diff().abs().fillna(0.0)
    df['margin_drift'] = df.groupby(['subject_id', 'trial_id'])['margin'].diff().abs().fillna(0.0)
    
    return df

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="subject_distance_predictions.csv")
    args = parser.parse_args()
    
    print("\n" + "="*50)
    print("TASK 1: RECOVER MATCHNET OUTPUTS")
    print("="*50)
    print("Checking available columns in CSV...")
    try:
        df = pd.read_csv(args.csv)
        cols = df.columns.tolist()
        print(f"Available columns: {cols}")
        print("Note: Raw embeddings are NOT available in this CSV. Tasks requiring embeddings will use similarity-based proxies instead.")
    except FileNotFoundError:
        print(f"File {args.csv} not found.")
        return
        
    df = df[np.isfinite(df['margin'])].reset_index(drop=True)
    df = add_temporal_features(df)
    
    model_path = "models/confidence_model.json"
    if not os.path.exists(model_path):
        print(f"Error: {model_path} not found.")
        return
        
    model = xgb.XGBClassifier()
    model.load_model(model_path)
    
    # Get high conf windows
    conf_features = ['margin', 'sim_chosen', 'sim_unchosen', 'rolling_std_margin', 'trial_consistency']
    df['confidence'] = model.predict_proba(df[conf_features])[:, 1]
    
    high_conf = df[df['confidence'] >= 0.90].copy()
    failures = high_conf[high_conf['correct'] == 0]
    successes = high_conf[high_conf['correct'] == 1]
    
    if len(failures) == 0:
        print("No high confidence failures at 0.90. Lowering threshold to 0.85.")
        high_conf = df[df['confidence'] >= 0.85].copy()
        failures = high_conf[high_conf['correct'] == 0]
        successes = high_conf[high_conf['correct'] == 1]
    
    print(f"\nAnalyzing {len(failures)} High-Conf Failures vs {len(successes)} High-Conf Successes.")
    
    print("\n" + "="*50)
    print("TASK 2 & 4: FEATURE INVENTORY & DRIFT AUDIT")
    print("="*50)
    
    investigate_feats = [
        'margin', 'sim_A', 'sim_B', 'sim_chosen', 'sim_unchosen', 
        'sim_sum', 'sim_ratio', 'sim_chosen_drift', 'sim_unchosen_drift', 'margin_drift'
    ]
    
    print("Feature Distributions (Mean Values):")
    print(f"{'Feature':<20} | {'Successes':<10} | {'Failures':<10} | {'Ratio(F/S)':<10}")
    for f in investigate_feats:
        m_s = successes[f].mean()
        m_f = failures[f].mean()
        ratio = m_f / (m_s + 1e-6)
        print(f"{f:<20} | {m_s:<10.4f} | {m_f:<10.4f} | {ratio:<10.4f}")
        
    print("\n" + "="*50)
    print("TASK 6 & 7: FAILURE PREDICTABILITY TEST & RANKING")
    print("="*50)
    
    # Label: Success = 0, Failure = 1
    high_conf['is_failure'] = (high_conf['correct'] == 0).astype(int)
    y = high_conf['is_failure'].values
    
    # Test predictability of each feature independently
    results = []
    
    for f in investigate_feats:
        X = high_conf[[f]].values
        
        auroc_cv = []
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        for train_idx, test_idx in skf.split(X, y):
            clf = xgb.XGBClassifier(n_estimators=50, max_depth=2, learning_rate=0.1, random_state=42, eval_metric='logloss')
            clf.fit(X[train_idx], y[train_idx])
            prob = clf.predict_proba(X[test_idx])[:, 1]
            try:
                score = roc_auc_score(y[test_idx], prob)
                auroc_cv.append(score)
            except:
                pass
                
        mean_auc = np.mean(auroc_cv) if len(auroc_cv) > 0 else 0.5
        results.append({'Feature': f, 'Failure_AUROC': mean_auc})
        
    # Test all together
    X_all = high_conf[investigate_feats].values
    auroc_cv = []
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for train_idx, test_idx in skf.split(X_all, y):
        clf = xgb.XGBClassifier(n_estimators=50, max_depth=2, learning_rate=0.1, random_state=42, eval_metric='logloss')
        clf.fit(X_all[train_idx], y[train_idx])
        prob = clf.predict_proba(X_all[test_idx])[:, 1]
        try:
            score = roc_auc_score(y[test_idx], prob)
            auroc_cv.append(score)
        except:
            pass
            
    mean_auc_all = np.mean(auroc_cv) if len(auroc_cv) > 0 else 0.5
    results.append({'Feature': 'ALL_COMBINED', 'Failure_AUROC': mean_auc_all})
    
    res_df = pd.DataFrame(results).sort_values('Failure_AUROC', ascending=False)
    
    print("\nPredictability Ranking (Can we predict failures?):")
    print(res_df.to_string(index=False))
    
    print("\nInterpretation:")
    if mean_auc_all > 0.65:
        print("-> Hidden information EXISTS. The failures are predictable using combinations of similarities and drifts.")
    else:
        print("-> Failures are essentially UNPREDICTABLE. We have reached the information limit of the current similarity metrics.")
        
if __name__ == "__main__":
    main()
