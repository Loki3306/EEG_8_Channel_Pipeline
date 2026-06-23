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
    
    try:
        df = pd.read_csv(args.csv)
    except FileNotFoundError:
        print(f"Cannot find {args.csv}.")
        return
        
    df = df[np.isfinite(df['margin'])].reset_index(drop=True)
    df = add_temporal_features(df)
    
    subjects = df['subject_id'].unique()
    features = ['margin', 'sim_chosen', 'sim_unchosen', 'rolling_std_margin', 'trial_consistency']
    
    df['prob_xgb'] = 0.0
    
    print("\n===========================================")
    print("STEP 4.10: END-TO-END CONFIDENCE VALIDATION")
    print("===========================================")
    print("Computing nested LOSO confidence scores...\n")
    
    for test_subj in subjects:
        train_mask = df['subject_id'] != test_subj
        test_mask = df['subject_id'] == test_subj
        
        X_train_xgb = df.loc[train_mask, features]
        y_train = df.loc[train_mask, 'correct'].values
        X_test_xgb = df.loc[test_mask, features]
        
        xgb_model = xgb.XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.05, n_jobs=-1, eval_metric='logloss')
        xgb_model.fit(X_train_xgb, y_train)
        df.loc[test_mask, 'prob_xgb'] = xgb_model.predict_proba(X_test_xgb)[:, 1]

    # --- Threshold Sweep ---
    thresholds = [0.50, 0.60, 0.70, 0.80, 0.90, 0.95]
    
    print("\n--- OPERATIONAL PERFORMANCE ---")
    print(f"{'Threshold':<10} | {'Coverage':<10} | {'Acc (Accpt)':<12} | {'Acc (Rejct)':<12} | {'Accepted/Total'}")
    print("-" * 75)
    
    for thr in thresholds:
        accepted_mask = df['prob_xgb'] >= thr
        rejected_mask = ~accepted_mask
        
        total_windows = len(df)
        num_accepted = accepted_mask.sum()
        num_rejected = rejected_mask.sum()
        
        coverage = num_accepted / total_windows
        
        accepted_correct = df.loc[accepted_mask, 'correct'].sum()
        rejected_correct = df.loc[rejected_mask, 'correct'].sum()
        
        acc_accepted = accepted_correct / num_accepted if num_accepted > 0 else np.nan
        acc_rejected = rejected_correct / num_rejected if num_rejected > 0 else np.nan
        
        print(f"{thr:<10.2f} | {coverage*100:>8.1f}% | {acc_accepted*100:>10.1f}% | {acc_rejected*100:>10.1f}% | {num_accepted}/{total_windows}")

    # --- Reliability Binning ---
    bins = np.linspace(0, 1.0, 11)
    df['conf_bin'] = pd.cut(df['prob_xgb'], bins=bins, right=True)
    
    bin_stats = df.groupby('conf_bin', observed=False).agg(
        count=('correct', 'size'),
        actual_acc=('correct', 'mean'),
        mean_conf=('prob_xgb', 'mean')
    ).reset_index()
    
    print("\n--- CONFIDENCE RELIABILITY BINS ---")
    print(f"{'Bin Range':<20} | {'N Windows':<10} | {'Actual Acc':<12} | {'Mean Conf'}")
    print("-" * 65)
    for _, row in bin_stats.iterrows():
        bin_str = str(row['conf_bin'])
        acc = row['actual_acc'] * 100 if pd.notna(row['actual_acc']) else np.nan
        conf = row['mean_conf'] if pd.notna(row['mean_conf']) else np.nan
        count = row['count']
        if count > 0:
             print(f"{bin_str:<20} | {count:<10} | {acc:>9.1f}% | {conf:.3f}")

    # --- Plotting ---
    os.makedirs('analysis/figures', exist_ok=True)
    
    thresholds_fine = np.linspace(0.5, 0.99, 50)
    covs = []
    accs = []
    for thr in thresholds_fine:
        mask = df['prob_xgb'] >= thr
        n_acc = mask.sum()
        if n_acc > 0:
            covs.append(n_acc / len(df))
            accs.append(df.loc[mask, 'correct'].mean())
        else:
            covs.append(0)
            accs.append(np.nan)
            
    # Figure 1: Coverage vs Accuracy
    plt.figure(figsize=(8, 6))
    plt.plot([c*100 for c in covs], [a*100 for a in accs], 'b-', linewidth=2)
    plt.xlabel('Coverage (%)')
    plt.ylabel('Accepted Accuracy (%)')
    plt.title('Coverage vs Accuracy')
    plt.grid(True, linestyle='--')
    plt.gca().invert_xaxis()
    plt.savefig('analysis/figures/coverage_vs_accuracy_4_10.png', dpi=300)
    plt.close()
    
    # Figure 2: Threshold vs Accuracy
    plt.figure(figsize=(8, 6))
    plt.plot(thresholds_fine, [a*100 for a in accs], 'g-', linewidth=2)
    plt.xlabel('Confidence Threshold')
    plt.ylabel('Accepted Accuracy (%)')
    plt.title('Threshold vs Accuracy')
    plt.grid(True, linestyle='--')
    plt.savefig('analysis/figures/threshold_vs_accuracy.png', dpi=300)
    plt.close()
    
    # Figure 3: Threshold vs Coverage
    plt.figure(figsize=(8, 6))
    plt.plot(thresholds_fine, [c*100 for c in covs], 'r-', linewidth=2)
    plt.xlabel('Confidence Threshold')
    plt.ylabel('Coverage (%)')
    plt.title('Threshold vs Coverage')
    plt.grid(True, linestyle='--')
    plt.savefig('analysis/figures/threshold_vs_coverage.png', dpi=300)
    plt.close()
    
    # Figure 4: Reliability Diagram
    valid_bins = bin_stats[bin_stats['count'] > 0]
    plt.figure(figsize=(8, 8))
    plt.plot([0, 1], [0, 1], "k:", label="Perfect Calibration")
    plt.plot(valid_bins['mean_conf'], valid_bins['actual_acc'], 's-', color='purple', label="XGBoost Empirical")
    plt.xlabel('Predicted Confidence')
    plt.ylabel('Empirical Accuracy')
    plt.title('Confidence Reliability Diagram')
    plt.legend()
    plt.grid(True, linestyle='--')
    plt.savefig('analysis/figures/reliability_diagram_4_10.png', dpi=300)
    plt.close()
    
    print("\nGenerated end-to-end plots in analysis/figures/")

if __name__ == "__main__":
    main()
