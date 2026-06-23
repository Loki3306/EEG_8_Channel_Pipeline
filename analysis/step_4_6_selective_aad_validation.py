import argparse
import pandas as pd
import numpy as np
import xgboost as xgb
from scipy import stats
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

def compute_rc_metrics(correct_labels, confidences):
    n = len(correct_labels)
    sorted_indices = np.argsort(-confidences)
    sorted_labels = correct_labels.values[sorted_indices]
    
    coverages = np.arange(1, n + 1) / n
    accuracies = np.cumsum(sorted_labels) / np.arange(1, n + 1)
    risks = 1.0 - accuracies
    
    aurc = np.trapezoid(risks, coverages)
    
    optimal_indices = np.argsort(-correct_labels.values)
    optimal_labels = correct_labels.values[optimal_indices]
    optimal_accuracies = np.cumsum(optimal_labels) / np.arange(1, n + 1)
    optimal_risks = 1.0 - optimal_accuracies
    optimal_aurc = np.trapezoid(optimal_risks, coverages)
    
    e_aurc = aurc - optimal_aurc
    
    target_covs = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
    acc_at_cov = {}
    for c in target_covs:
        idx = max(0, int(c * n) - 1)
        acc_at_cov[c] = accuracies[idx]
        
    return aurc, e_aurc, acc_at_cov

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
    
    subjects = df['subject_id'].unique()
    features = ['margin', 'sim_chosen', 'sim_unchosen', 'rolling_std_margin', 'trial_consistency']
    
    print("\n===========================================")
    print("STEP 4.6: SELECTIVE AAD VALIDATION (AURC)")
    print("===========================================")
    print("Running strict nested LOSO across 18 subjects...\n")
    
    margin_aurcs = []
    xgb_aurcs = []
    margin_eaurcs = []
    xgb_eaurcs = []
    
    target_covs = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
    margin_accs = {c: [] for c in target_covs}
    xgb_accs = {c: [] for c in target_covs}
    
    for test_subj in subjects:
        train_mask = df['subject_id'] != test_subj
        test_mask = df['subject_id'] == test_subj
        
        X_train = df.loc[train_mask, features]
        y_train = df.loc[train_mask, 'correct']
        X_test = df.loc[test_mask, features]
        y_test = df.loc[test_mask, 'correct']
        
        # Method A: Margin Confidence
        margin_conf = df.loc[test_mask, 'margin']
        m_aurc, m_eaurc, m_accs = compute_rc_metrics(y_test, margin_conf)
        
        # Method B: XGBoost Confidence
        xgb_model = xgb.XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.05, n_jobs=-1, eval_metric='logloss')
        xgb_model.fit(X_train, y_train)
        xgb_conf = xgb_model.predict_proba(X_test)[:, 1]
        x_aurc, x_eaurc, x_accs = compute_rc_metrics(y_test, xgb_conf)
        
        margin_aurcs.append(m_aurc)
        xgb_aurcs.append(x_aurc)
        margin_eaurcs.append(m_eaurc)
        xgb_eaurcs.append(x_eaurc)
        
        for c in target_covs:
            margin_accs[c].append(m_accs[c])
            xgb_accs[c].append(x_accs[c])
            
    print("--- MEAN SELECTIVE AAD PERFORMANCE (COVERAGE vs ACCURACY) ---")
    print(f"{'Coverage':<10} | {'Margin Acc':<15} | {'XGBoost Acc':<15} | {'Lift'}")
    print("-" * 55)
    for c in target_covs:
        m_mean = np.mean(margin_accs[c])
        x_mean = np.mean(xgb_accs[c])
        diff = x_mean - m_mean
        print(f"{int(c*100):>8}% | {m_mean:<15.4f} | {x_mean:<15.4f} | {diff:+.4f}")
        
    print("\n--- EXCESS AURC (E-AURC) STATISTICS (LOWER IS BETTER) ---")
    m_eaurc_mean = np.mean(margin_eaurcs)
    x_eaurc_mean = np.mean(xgb_eaurcs)
    
    print(f"Mean Margin E-AURC  : {m_eaurc_mean:.4f}")
    print(f"Mean XGBoost E-AURC : {x_eaurc_mean:.4f}")
    
    improvements = np.array(margin_eaurcs) - np.array(xgb_eaurcs) # Positive is good (XGB has lower error)
    
    print(f"\nSubjects Improved (Lower E-AURC) : {np.sum(improvements > 0)} / {len(subjects)}")
    print(f"Subjects Worsened (Higher E-AURC): {np.sum(improvements < 0)} / {len(subjects)}")
    
    print("\n--- STATISTICAL TESTS ---")
    w_stat, p_wilcoxon = stats.wilcoxon(xgb_eaurcs, margin_eaurcs)
    print(f"Wilcoxon signed-rank (E-AURC): p={p_wilcoxon:.2e}")
    
    print("\n===========================================\n")

if __name__ == "__main__":
    main()
