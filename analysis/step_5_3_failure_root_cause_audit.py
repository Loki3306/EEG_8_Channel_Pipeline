import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import os
import argparse
from sklearn.cluster import KMeans
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
    
    # 1. Load Data
    try:
        df = pd.read_csv(args.csv)
    except FileNotFoundError:
        print(f"File {args.csv} not found.")
        return
        
    df = df[np.isfinite(df['margin'])].reset_index(drop=True)
    df = add_temporal_features(df)
    
    # 2. Load Model & Predict
    model_path = "models/confidence_model.json"
    if not os.path.exists(model_path):
        print(f"Error: {model_path} not found. Run step 5.0a first.")
        return
    model = xgb.XGBClassifier()
    model.load_model(model_path)
    
    features = ['margin', 'sim_chosen', 'sim_unchosen', 'rolling_std_margin', 'trial_consistency']
    X = df[features]
    df['confidence'] = model.predict_proba(X)[:, 1]
    
    # Define "High Confidence" threshold
    HIGH_CONF_THR = 0.90
    
    high_conf = df[df['confidence'] >= HIGH_CONF_THR]
    failures = high_conf[high_conf['correct'] == 0].copy()
    corrects = high_conf[high_conf['correct'] == 1].copy()
    
    print("\n" + "="*50)
    print(f"HIGH-CONFIDENCE FAILURES (Conf >= {HIGH_CONF_THR})")
    print(f"Total High-Conf Windows  : {len(high_conf)}")
    print(f"High-Conf Correct        : {len(corrects)}")
    print(f"High-Conf Failures       : {len(failures)}")
    print("="*50)
    
    if len(failures) == 0:
        print("No high confidence failures found at 0.90! Lowering threshold to 0.85 for analysis.")
        HIGH_CONF_THR = 0.85
        high_conf = df[df['confidence'] >= HIGH_CONF_THR]
        failures = high_conf[high_conf['correct'] == 0].copy()
        corrects = high_conf[high_conf['correct'] == 1].copy()
        print(f"New High-Conf Failures   : {len(failures)}")
        
    os.makedirs('analysis/figures/failure_root_cause', exist_ok=True)
    
    if len(failures) == 0:
        print("Still no failures. Exiting.")
        return
    
    # Q1. Subject Distribution
    print("\n--- Q1: Subject Failure Concentration ---")
    subj_counts = failures['subject_id'].value_counts()
    subj_rates = (subj_counts / df[df['correct']==0].groupby(df['subject_id']).size()).dropna().sort_values(ascending=False)
    print("Failures per subject (Count):")
    print(subj_counts)
    print("\nFailure Rate per subject (% of total subject errors that are high-conf):")
    print((subj_rates * 100).map("{:.2f}%".format))
    
    # Q2. Trial Distribution
    print("\n--- Q2: Trial Failure Concentration ---")
    trial_counts = failures.groupby(['subject_id', 'trial_id']).size().sort_values(ascending=False)
    print("Top 10 Trials with multiple high-conf failures:")
    print(trial_counts.head(10))
    
    # Q3. Window Position
    print("\n--- Q3: Window Position Distribution ---")
    win_counts = failures['window_id'].value_counts().sort_index()
    print(win_counts)
    
    plt.figure()
    win_counts.plot(kind='bar', color='salmon')
    plt.title('High-Conf Failures by Window Position')
    plt.xlabel('Window ID')
    plt.ylabel('Failure Count')
    plt.tight_layout()
    plt.savefig('analysis/figures/failure_root_cause/window_distribution.png')
    plt.close()
    
    # Q4. Compare Correct vs Incorrect Distributions
    print("\n--- Q4: Feature Distributions (Correct vs Incorrect @ High Conf) ---")
    print("Mean values:")
    print("Correct cases   -> Margin: {:.3f}, Consistency: {:.3f}, Roll_Std: {:.3f}".format(
        corrects['margin'].mean(), corrects['trial_consistency'].mean(), corrects['rolling_std_margin'].mean()
    ))
    print("Incorrect cases -> Margin: {:.3f}, Consistency: {:.3f}, Roll_Std: {:.3f}".format(
        failures['margin'].mean(), failures['trial_consistency'].mean(), failures['rolling_std_margin'].mean()
    ))
    
    fig, axs = plt.subplots(1, 3, figsize=(15, 4))
    axs[0].hist(corrects['margin'], bins=20, alpha=0.5, label='Correct', density=True)
    axs[0].hist(failures['margin'], bins=20, alpha=0.5, label='Incorrect', density=True)
    axs[0].set_title('Margin')
    axs[0].legend()
    
    axs[1].hist(corrects['trial_consistency'], bins=20, alpha=0.5, label='Correct', density=True)
    axs[1].hist(failures['trial_consistency'], bins=20, alpha=0.5, label='Incorrect', density=True)
    axs[1].set_title('Trial Consistency')
    
    axs[2].hist(corrects['rolling_std_margin'], bins=20, alpha=0.5, label='Correct', density=True)
    axs[2].hist(failures['rolling_std_margin'], bins=20, alpha=0.5, label='Incorrect', density=True)
    axs[2].set_title('Rolling Std Margin')
    plt.tight_layout()
    plt.savefig('analysis/figures/failure_root_cause/feature_distributions.png')
    plt.close()
    
    # Q5. Why did it fail?
    print("\n--- Q5: Why did it fail? (Top 10 High Conf Failures) ---")
    top_10 = failures.sort_values('confidence', ascending=False).head(10)
    print(top_10[['subject_id', 'trial_id', 'window_id', 'margin', 'trial_consistency', 'rolling_std_margin', 'confidence']].to_string(index=False))
    
    # Q6. Failure Archetypes (Clustering)
    print("\n--- Q6: Failure Archetypes ---")
    if len(failures) >= 3:
        cluster_feats = ['margin', 'trial_consistency', 'rolling_std_margin']
        X_fail = failures[cluster_feats].copy()
        X_fail_norm = (X_fail - X_fail.mean()) / X_fail.std()
        n_clust = min(3, len(failures))
        kmeans = KMeans(n_clusters=n_clust, random_state=42).fit(X_fail_norm)
        failures['archetype'] = kmeans.labels_
        
        for c in range(n_clust):
            c_df = failures[failures['archetype'] == c]
            print(f"\nArchetype {c} (N={len(c_df)}):")
            print("  Margin Mean : {:.3f}".format(c_df['margin'].mean()))
            print("  Consist Mean: {:.3f}".format(c_df['trial_consistency'].mean()))
            print("  RollStd Mean: {:.3f}".format(c_df['rolling_std_margin'].mean()))
    else:
        print("Not enough failures to cluster.")
        
    # Q7. Weakest Subjects
    print("\n--- Q7: Subject Pathology Audit ---")
    weak_subjs = ['S6_data_preproc', 'S11_data_preproc', 'S14_data_preproc']
    weak_fails = failures[failures['subject_id'].isin(weak_subjs)]
    strong_fails = failures[~failures['subject_id'].isin(weak_subjs)]
    print(f"Weak subjects generated {len(weak_fails)} high-conf failures.")
    print(f"Strong subjects generated {len(strong_fails)} high-conf failures.")
    if len(weak_fails) > 0 and len(strong_fails) > 0:
        print("Weak mean Margin: {:.3f} vs Strong mean Margin: {:.3f}".format(
            weak_fails['margin'].mean(), strong_fails['margin'].mean()))
            
    # Q8. Temporal Collapse Analysis
    print("\n--- Q8: Temporal Collapse Trajectories ---")
    lead_lag = {'t-2': [], 't-1': [], 't0': [], 't+1': [], 't+2': []}
    
    for _, row in failures.iterrows():
        subj, tr, win = row['subject_id'], row['trial_id'], row['window_id']
        trial_df = df[(df['subject_id'] == subj) & (df['trial_id'] == tr)].sort_values('window_id').reset_index(drop=True)
        idx = trial_df[trial_df['window_id'] == win].index[0]
        
        for offset, key in zip([-2, -1, 0, 1, 2], ['t-2', 't-1', 't0', 't+1', 't+2']):
            t_idx = idx + offset
            if 0 <= t_idx < len(trial_df):
                lead_lag[key].append(trial_df.loc[t_idx, 'confidence'])
            else:
                lead_lag[key].append(np.nan)

    if len(lead_lag['t0']) > 0:
        mean_lead_lag = {k: np.nanmean(v) for k, v in lead_lag.items()}
        print("Average Confidence around High-Conf Failure (t0):")
        for k, v in mean_lead_lag.items():
            print(f"  {k}: {v:.4f}")
            
        plt.figure(figsize=(6, 4))
        plt.plot(list(mean_lead_lag.keys()), list(mean_lead_lag.values()), marker='o', color='r')
        plt.title('Avg Confidence Around High-Conf Failure')
        plt.ylabel('Mean Confidence')
        plt.grid(True)
        plt.savefig('analysis/figures/failure_root_cause/temporal_collapse.png', dpi=150)
        plt.close()
    
    print("\nAudit Complete. Artifacts saved in analysis/figures/failure_root_cause/")

if __name__ == "__main__":
    main()
