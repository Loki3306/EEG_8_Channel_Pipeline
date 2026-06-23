import argparse
import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import os
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
    
    # 2. Load Final Model
    model_path = "models/confidence_model.json"
    if not os.path.exists(model_path):
        print(f"Error: {model_path} not found. Run step 5.0a first.")
        return
    model = xgb.XGBClassifier()
    model.load_model(model_path)
    
    # 3. Predict Confidence
    features = ['margin', 'sim_chosen', 'sim_unchosen', 'rolling_std_margin', 'trial_consistency']
    X = df[features]
    df['confidence'] = model.predict_proba(X)[:, 1]
    
    os.makedirs('analysis/figures/behavior_audit', exist_ok=True)
    
    print("\n" + "="*50)
    print("TASK 1: RANDOM TRIAL VISUALIZATION")
    print("="*50)
    # Pick 20 random trials (subject_id, trial_id combinations)
    trials = df[['subject_id', 'trial_id']].drop_duplicates().sample(n=min(20, df[['subject_id', 'trial_id']].drop_duplicates().shape[0]), random_state=42)
    
    for idx, (subj, tr) in enumerate(zip(trials['subject_id'], trials['trial_id'])):
        tdf = df[(df['subject_id'] == subj) & (df['trial_id'] == tr)]
        
        fig, axs = plt.subplots(4, 1, figsize=(10, 12), sharex=True)
        axs[0].plot(tdf['window_id'], tdf['margin'], marker='o', label='Margin')
        axs[0].axhline(0, color='k', linestyle='--')
        axs[0].set_ylabel('Margin')
        axs[0].set_title(f'Subject {subj} Trial {tr}')
        axs[0].grid(True)
        
        axs[1].plot(tdf['window_id'], tdf['confidence'], marker='s', color='g', label='Confidence')
        axs[1].axhline(0.80, color='r', linestyle='--', label='Threshold')
        axs[1].set_ylabel('Confidence')
        axs[1].set_ylim([-0.05, 1.05])
        axs[1].grid(True)
        
        axs[2].plot(tdf['window_id'], tdf['correct'], marker='^', color='m', label='Correctness')
        axs[2].set_ylabel('Correctness')
        axs[2].set_yticks([0, 1])
        axs[2].grid(True)
        
        accept = (tdf['confidence'] >= 0.80).astype(int)
        axs[3].step(tdf['window_id'], accept, where='post', color='orange')
        axs[3].set_ylabel('Accept/Reject')
        axs[3].set_xlabel('Window ID')
        axs[3].set_yticks([0, 1])
        axs[3].grid(True)
        
        plt.tight_layout()
        plt.savefig(f'analysis/figures/behavior_audit/trial_behavior_{idx+1}.png', dpi=150)
        plt.close()
    print("Saved 20 trial behavior plots to analysis/figures/behavior_audit/trial_behavior_*.png")
    
    print("\n" + "="*50)
    print("TASK 2: FAILURE ANALYSIS (High Conf, Incorrect)")
    print("="*50)
    failures = df[df['correct'] == 0].sort_values('confidence', ascending=False).head(20)
    print(failures[['subject_id', 'trial_id', 'window_id', 'margin', 'trial_consistency', 'confidence', 'prediction', 'correct']].to_string(index=False))
    
    print("\n" + "="*50)
    print("TASK 3: LOW-CONFIDENCE ANALYSIS (Low Conf, Correct)")
    print("="*50)
    low_conf = df[df['correct'] == 1].sort_values('confidence', ascending=True).head(20)
    print(low_conf[['subject_id', 'trial_id', 'window_id', 'margin', 'trial_consistency', 'confidence', 'prediction', 'correct']].to_string(index=False))
    
    print("\n" + "="*50)
    print("TASK 4: CONFIDENCE DYNAMICS (Correlations)")
    print("="*50)
    corr_margin = df['confidence'].corr(df['margin'].abs())
    corr_consist = df['confidence'].corr(df['trial_consistency'])
    corr_roll = df['confidence'].corr(df['rolling_std_margin'])
    print(f"Correlation(Confidence, abs(Margin))      : {corr_margin:.4f}")
    print(f"Correlation(Confidence, Trial Consistency): {corr_consist:.4f}")
    print(f"Correlation(Confidence, Rolling Std)      : {corr_roll:.4f}")
    
    print("\n" + "="*50)
    print("TASK 5: CONFIDENCE LEAD/LAG AUDIT")
    print("="*50)
    lead_lag = {'t-2': [], 't-1': [], 't0': [], 't+1': [], 't+2': []}
    
    for (subj, tr), group in df.groupby(['subject_id', 'trial_id']):
        group = group.sort_values('window_id').reset_index(drop=True)
        incorrect_idx = group.index[group['correct'] == 0].tolist()
        
        for idx in incorrect_idx:
            for offset, key in zip([-2, -1, 0, 1, 2], ['t-2', 't-1', 't0', 't+1', 't+2']):
                t_idx = idx + offset
                if 0 <= t_idx < len(group):
                    lead_lag[key].append(group.loc[t_idx, 'confidence'])
                else:
                    lead_lag[key].append(np.nan)

    if len(lead_lag['t0']) > 0:
        mean_lead_lag = {k: np.nanmean(v) for k, v in lead_lag.items()}
        print("Average Confidence around Failure (t0):")
        for k, v in mean_lead_lag.items():
            print(f"  {k}: {v:.4f}")
            
        plt.figure(figsize=(6, 4))
        plt.plot(list(mean_lead_lag.keys()), list(mean_lead_lag.values()), marker='o', color='r')
        plt.title('Average Confidence Around Failure')
        plt.ylabel('Mean Confidence')
        plt.grid(True)
        plt.savefig('analysis/figures/behavior_audit/confidence_before_failure.png', dpi=150)
        plt.close()
    else:
        print("No failures found to compute lead/lag.")

    print("\n" + "="*50)
    print("TASK 6: SUBJECT-SPECIFIC BEHAVIOR")
    print("="*50)
    subj_stats = df.groupby('subject_id').agg(
        accuracy=('correct', 'mean'),
        mean_confidence=('confidence', 'mean')
    ).reset_index()
    print(subj_stats.to_string(index=False))
    
    print("\n" + "="*50)
    print("TASK 7: PATHOLOGICAL CASE SEARCH")
    print("="*50)
    case_A = df[(df['confidence'] > 0.95) & (df['correct'] == 0)]
    case_B = df[(df['confidence'] < 0.20) & (df['correct'] == 1)]
    print(f"Found {len(case_A)} pathological Case A (High Conf, Incorrect).")
    print(f"Found {len(case_B)} pathological Case B (Low Conf, Correct).")
    
    case_A.to_csv('analysis/figures/behavior_audit/pathological_case_A.csv', index=False)
    case_B.to_csv('analysis/figures/behavior_audit/pathological_case_B.csv', index=False)
    print("Pathological cases exported to CSV in analysis/figures/behavior_audit/")

if __name__ == "__main__":
    main()
