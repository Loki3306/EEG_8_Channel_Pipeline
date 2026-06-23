import argparse
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
import xgboost as xgb
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

def add_robustness_features(df):
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

def cohen_d(x, y):
    nx = len(x)
    ny = len(y)
    dof = nx + ny - 2
    return (np.mean(x) - np.mean(y)) / np.sqrt(((nx-1)*np.std(x, ddof=1) ** 2 + (ny-1)*np.std(y, ddof=1) ** 2) / dof)

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
    
    subjects = df['subject_id'].unique()
    features = ['margin', 'sim_chosen', 'sim_unchosen', 'rolling_std_margin', 'trial_consistency']
    
    margin_aucs = []
    xgb_aucs = []
    
    print("\n===========================================")
    print("STEP 4.5G: STATISTICAL SIGNIFICANCE AUDIT")
    print("===========================================")
    print("Running strict nested LOSO across 18 subjects...\n")
    
    print(f"{'Subject':<20} | {'Margin AUROC':<15} | {'XGB AUROC':<15} | {'Improvement':<15}")
    print("-" * 75)
    
    for test_subj in subjects:
        train_mask = df['subject_id'] != test_subj
        test_mask = df['subject_id'] == test_subj
        
        X_train = df.loc[train_mask, features]
        y_train = df.loc[train_mask, 'correct']
        X_test = df.loc[test_mask, features]
        y_test = df.loc[test_mask, 'correct']
        
        auc_margin = roc_auc_score(y_test, df.loc[test_mask, 'margin'])
        margin_aucs.append(auc_margin)
        
        xgb_model = xgb.XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.05, n_jobs=-1, eval_metric='logloss')
        xgb_model.fit(X_train, y_train)
        probs = xgb_model.predict_proba(X_test)[:, 1]
        auc_xgb = roc_auc_score(y_test, probs)
        xgb_aucs.append(auc_xgb)
        
        diff = auc_xgb - auc_margin
        
        print(f"{test_subj:<20} | {auc_margin:<15.4f} | {auc_xgb:<15.4f} | {diff:+.4f}")

    margin_aucs = np.array(margin_aucs)
    xgb_aucs = np.array(xgb_aucs)
    improvements = xgb_aucs - margin_aucs
    
    print("-" * 75)
    print("\n--- AGGREGATE STATISTICS ---")
    print(f"Mean Margin AUROC  : {np.mean(margin_aucs):.4f} ± {np.std(margin_aucs):.4f}")
    print(f"Mean XGB AUROC     : {np.mean(xgb_aucs):.4f} ± {np.std(xgb_aucs):.4f}")
    print(f"Mean Improvement   : {np.mean(improvements):+.4f}")
    print(f"Median Improvement : {np.median(improvements):+.4f}")
    print(f"Min Improvement    : {np.min(improvements):+.4f}")
    print(f"Max Improvement    : {np.max(improvements):+.4f}")
    
    print(f"\nSubjects Improved  : {np.sum(improvements > 0)} / {len(subjects)}")
    print(f"Subjects Worsened  : {np.sum(improvements < 0)} / {len(subjects)}")
    
    print("\n--- STATISTICAL TESTS ---")
    t_stat, p_ttest = stats.ttest_rel(xgb_aucs, margin_aucs)
    print(f"Paired t-test      : t={t_stat:.4f}, p={p_ttest:.2e}")
    
    w_stat, p_wilcoxon = stats.wilcoxon(xgb_aucs, margin_aucs)
    print(f"Wilcoxon signed-rank: w={w_stat:.4f}, p={p_wilcoxon:.2e}")
    
    d = cohen_d(xgb_aucs, margin_aucs)
    print(f"\nEffect Size (Cohen's d): {d:.4f}")
    if abs(d) < 0.2:
        es = "Negligible"
    elif abs(d) < 0.5:
        es = "Small"
    elif abs(d) < 0.8:
        es = "Medium"
    else:
        es = "Large"
    print(f"Interpretation     : {es}")
    
    print("\n===========================================\n")

if __name__ == "__main__":
    main()
