import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import os
import argparse
from sklearn.neighbors import NearestNeighbors
import shap
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
    df = add_temporal_features(df)
    
    model_path = "models/confidence_model.json"
    if not os.path.exists(model_path):
        print(f"Error: {model_path} not found. Run step 5.0a first.")
        return
    model = xgb.XGBClassifier()
    model.load_model(model_path)
    
    features = ['margin', 'sim_chosen', 'sim_unchosen', 'rolling_std_margin', 'trial_consistency']
    X = df[features]
    df['confidence'] = model.predict_proba(X)[:, 1]
    
    failures_all = df[(df['correct'] == 0) & (df['confidence'] >= 0.90)].sort_values('confidence', ascending=False)
    if len(failures_all) == 0:
        failures_all = df[df['correct'] == 0].sort_values('confidence', ascending=False).head(36)
        
    n_fail = len(failures_all)
    successes_all = df[df['correct'] == 1].sort_values('confidence', ascending=False).head(n_fail)
    
    failures = failures_all.copy()
    successes = successes_all.copy()
    
    os.makedirs('analysis/figures/decision_path', exist_ok=True)
    
    print("\n" + "="*50)
    print(f"TASK 1: FAILURE VS SUCCESS COMPARISON (Top {n_fail} cases)")
    print("="*50)
    
    print("Mean statistics:")
    print("Feature             | Successes | Failures ")
    print(f"Margin              | {successes['margin'].mean():.4f}    | {failures['margin'].mean():.4f}")
    print(f"Consistency         | {successes['trial_consistency'].mean():.4f}    | {failures['trial_consistency'].mean():.4f}")
    print(f"Rolling Std         | {successes['rolling_std_margin'].mean():.4f}    | {failures['rolling_std_margin'].mean():.4f}")
    print(f"Confidence          | {successes['confidence'].mean():.4f}    | {failures['confidence'].mean():.4f}")
    
    fig, axs = plt.subplots(1, 4, figsize=(16, 4))
    axs[0].hist(successes['margin'], alpha=0.5, label='Success')
    axs[0].hist(failures['margin'], alpha=0.5, label='Failure')
    axs[0].set_title('Margin')
    axs[0].legend()
    
    axs[1].hist(successes['trial_consistency'], alpha=0.5, label='Success')
    axs[1].hist(failures['trial_consistency'], alpha=0.5, label='Failure')
    axs[1].set_title('Trial Consistency')
    
    axs[2].hist(successes['rolling_std_margin'], alpha=0.5, label='Success')
    axs[2].hist(failures['rolling_std_margin'], alpha=0.5, label='Failure')
    axs[2].set_title('Rolling Std')
    
    axs[3].hist(successes['confidence'], alpha=0.5, label='Success')
    axs[3].hist(failures['confidence'], alpha=0.5, label='Failure')
    axs[3].set_title('Confidence')
    
    plt.tight_layout()
    plt.savefig('analysis/figures/decision_path/task1_distributions.png', dpi=150)
    plt.close()
    
    print("\n" + "="*50)
    print("TASK 2 & 4: SHAP DECISION TRACE & COMPOSITION")
    print("="*50)
    
    explainer = shap.TreeExplainer(model)
    shap_vals_fail = explainer.shap_values(failures[features])
    
    type_A = 0
    type_B = 0
    type_C = 0
    
    print("Top 10 High-Confidence Failures SHAP Trace:")
    print("Subj | Trial | Win | Conf | Marg SHAP | Cons SHAP | Roll SHAP | Type")
    for i in range(len(failures)):
        sv = shap_vals_fail[i]
        m_shap = sv[0]
        c_shap = sv[4]
        r_shap = sv[3]
        
        conf = failures.iloc[i]['confidence']
        
        if c_shap > m_shap and c_shap > r_shap:
            t = "B (Consistency)"
            type_B += 1
        elif m_shap > c_shap and m_shap > r_shap:
            t = "A (Margin)"
            type_A += 1
        else:
            t = "C (Mixed/Roll)"
            type_C += 1
            
        if i < 10:
            row = failures.iloc[i]
            subj = row['subject_id'].replace('_data_preproc', '')
            print(f"{subj:<5} | {row['trial_id']:<5} | {row['window_id']:<3} | {conf:.3f} | {m_shap:+.3f}    | {c_shap:+.3f}    | {r_shap:+.3f}    | {t}")
            
    print("\nFailure Types (Based on max positive SHAP contribution):")
    print(f"Type A (Margin-dominated)      : {type_A}")
    print(f"Type B (Consistency-dominated) : {type_B}")
    print(f"Type C (Mixed/Roll-dominated)  : {type_C}")
    
    print("\n" + "="*50)
    print("TASK 3: SUBJECT ANALYSIS")
    print("="*50)
    
    weak_subjs = ['S6_data_preproc', 'S11_data_preproc', 'S14_data_preproc']
    df['group'] = np.where(df['subject_id'].isin(weak_subjs), 'Weak (S6, S11, S14)', 'Strong (Others)')
    
    subj_stats = df.groupby('group').agg(
        acc=('correct', 'mean'),
        conf=('confidence', 'mean'),
        margin=('margin', 'mean'),
        consist=('trial_consistency', 'mean'),
        roll=('rolling_std_margin', 'mean'),
        count=('correct', 'count')
    ).reset_index()
    
    print(subj_stats.to_string(index=False))
    
    print("\n" + "="*50)
    print("TASK 5: NEAREST NEIGHBOR ANALYSIS")
    print("="*50)
    
    feats_3d = ['margin', 'trial_consistency', 'rolling_std_margin']
    
    all_successes = df[(df['correct'] == 1) & (df['confidence'] >= 0.90)].copy()
    if len(all_successes) == 0:
        all_successes = successes
        
    X_success = all_successes[feats_3d].values
    X_fail = failures[feats_3d].values
    
    mean_vec = X_success.mean(axis=0)
    std_vec = X_success.std(axis=0)
    
    X_s_norm = (X_success - mean_vec) / std_vec
    X_f_norm = (X_fail - mean_vec) / std_vec
    
    nn = NearestNeighbors(n_neighbors=5, metric='euclidean')
    nn.fit(X_s_norm)
    
    dist_fail_to_success, _ = nn.kneighbors(X_f_norm)
    mean_dist_fail = np.mean(dist_fail_to_success)
    
    dist_succ_to_succ, _ = nn.kneighbors(X_s_norm)
    mean_dist_succ = np.mean(dist_succ_to_succ[:, 1:])
    
    print(f"Mean distance: Success -> Nearest Successes : {mean_dist_succ:.4f}")
    print(f"Mean distance: Failure -> Nearest Successes : {mean_dist_fail:.4f}")
    
    if mean_dist_fail < mean_dist_succ * 1.5:
        print("\nConclusion: Failures are NOT isolated outliers. They exist in the exact same feature space as highly successful predictions.")
    else:
        print("\nConclusion: Failures are outliers far away from the normal success regions.")
        
    print("\nAudit Complete. Artifacts saved.")

if __name__ == "__main__":
    main()
