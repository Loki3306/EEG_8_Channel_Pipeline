import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import roc_auc_score, brier_score_loss
import matplotlib.pyplot as plt
import os
import argparse
import warnings

warnings.filterwarnings('ignore')

def compute_ece(y_true, y_prob, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    binids = np.digitize(y_prob, bins) - 1
    ece = 0.0
    for i in range(n_bins):
        mask = binids == i
        if np.sum(mask) > 0:
            prob_mean = np.mean(y_prob[mask])
            acc_mean = np.mean(y_true[mask])
            ece += (np.sum(mask) / len(y_prob)) * np.abs(prob_mean - acc_mean)
    return ece

def compute_e_aurc(y_true, y_prob):
    df = pd.DataFrame({'true': y_true, 'prob': y_prob})
    df = df.sort_values('prob', ascending=False).reset_index(drop=True)
    
    coverages = []
    risks = []
    
    n = len(df)
    errors = (df['true'] == 0).values
    cum_errors = np.cumsum(errors)
    
    for i in range(1, n + 1):
        coverage = i / n
        risk = cum_errors[i-1] / i
        coverages.append(coverage)
        risks.append(risk)
        
    if hasattr(np, 'trapezoid'):
        e_aurc = np.trapezoid(risks, coverages)
    else:
        e_aurc = np.trapz(risks, coverages)
    return e_aurc

def evaluate_predictions(y_true, y_prob):
    auroc = roc_auc_score(y_true, y_prob)
    e_aurc = compute_e_aurc(y_true, y_prob)
    brier = brier_score_loss(y_true, y_prob)
    ece = compute_ece(y_true, y_prob)
    
    mask_90 = y_prob >= 0.90
    n_accepted = np.sum(mask_90)
    cov_90 = n_accepted / len(y_prob) if len(y_prob) > 0 else 0
    acc_90 = np.mean(y_true[mask_90]) if n_accepted > 0 else np.nan
    
    return {
        'AUROC': auroc,
        'E-AURC': e_aurc,
        'Brier': brier,
        'ECE': ece,
        'Acc@90': acc_90,
        'Cov@90': cov_90,
        'N_Acc@90': n_accepted
    }

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
    
    os.makedirs('analysis/figures/margin_audit', exist_ok=True)
    
    # ---------------------------------------------------------
    # EXPERIMENT 1: NESTED LOSO
    # ---------------------------------------------------------
    print("\n" + "="*50)
    print("EXPERIMENT 1: NESTED LOSO MODEL EVALUATION")
    print("="*50)
    
    models_def = {
        'Mod_A_Consist': ['trial_consistency'],
        'Mod_B_RollStd': ['rolling_std_margin'],
        'Mod_C_Consist_Roll': ['trial_consistency', 'rolling_std_margin'],
        'Mod_D_Marg_Consist': ['margin', 'trial_consistency'],
        'Mod_E_Marg_Roll': ['margin', 'rolling_std_margin'],
        'Mod_F_Marg_Consist_Roll': ['margin', 'trial_consistency', 'rolling_std_margin'],
        'Mod_G_Full': ['margin', 'sim_chosen', 'sim_unchosen', 'rolling_std_margin', 'trial_consistency']
    }
    
    subjects = df['subject_id'].unique()
    predictions = {m: np.zeros(len(df)) for m in models_def.keys()}
    
    for subj in subjects:
        train_mask = df['subject_id'] != subj
        test_mask = df['subject_id'] == subj
        y_train = df.loc[train_mask, 'correct'].values
        
        for m_name, m_feats in models_def.items():
            X_train = df.loc[train_mask, m_feats]
            X_test = df.loc[test_mask, m_feats]
            
            clf = xgb.XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.05, n_jobs=-1, eval_metric='logloss', random_state=42)
            clf.fit(X_train, y_train)
            predictions[m_name][test_mask] = clf.predict_proba(X_test)[:, 1]
            
    y_true = df['correct'].values
    results = {}
    for m_name in models_def.keys():
        results[m_name] = evaluate_predictions(y_true, predictions[m_name])
        
    res_df = pd.DataFrame.from_dict(results, orient='index')
    print(res_df.to_string())
    
    full_auroc = results['Mod_G_Full']['AUROC']
    print("\nRetained Performance % (AUROC):")
    for m_name in models_def.keys():
        pct = (results[m_name]['AUROC'] / full_auroc) * 100
        print(f"  {m_name:<25}: {pct:>6.2f}%")
        
    # ---------------------------------------------------------
    # EXPERIMENT 2: FEATURE CORRELATION AUDIT
    # ---------------------------------------------------------
    print("\n" + "="*50)
    print("EXPERIMENT 2: FEATURE CORRELATION AUDIT")
    print("="*50)
    
    c_m_r = df['margin'].abs().corr(df['rolling_std_margin'])
    c_m_c = df['margin'].abs().corr(df['trial_consistency'])
    c_r_c = df['rolling_std_margin'].corr(df['trial_consistency'])
    
    print(f"Corr(abs(margin), rolling_std_margin) : {c_m_r:.4f}")
    print(f"Corr(abs(margin), trial_consistency)  : {c_m_c:.4f}")
    print(f"Corr(rolling_std_margin, consistency) : {c_r_c:.4f}")
    
    # ---------------------------------------------------------
    # EXPERIMENT 3: SHAP AUDIT
    # ---------------------------------------------------------
    print("\n" + "="*50)
    print("EXPERIMENT 3: SHAP AUDIT")
    print("="*50)
    try:
        import shap
        # Train on ALL data for global SHAP explanation
        X_G = df[models_def['Mod_G_Full']]
        X_F = df[models_def['Mod_F_Marg_Consist_Roll']]
        y_all = df['correct'].values
        
        clf_G = xgb.XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.05, n_jobs=-1, random_state=42).fit(X_G, y_all)
        clf_F = xgb.XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.05, n_jobs=-1, random_state=42).fit(X_F, y_all)
        
        explainer_G = shap.TreeExplainer(clf_G)
        shap_values_G = explainer_G.shap_values(X_G)
        mean_abs_shap_G = np.abs(shap_values_G).mean(axis=0)
        
        print("Mean Absolute SHAP values (Model G - Full):")
        for feat, val in sorted(zip(models_def['Mod_G_Full'], mean_abs_shap_G), key=lambda x: x[1], reverse=True):
            print(f"  {feat:<20}: {val:.4f}")
            
        shap.summary_plot(shap_values_G, X_G, show=False)
        plt.tight_layout()
        plt.savefig('analysis/figures/margin_audit/shap_summary_Mod_G.png', dpi=300)
        plt.close()
        
        explainer_F = shap.TreeExplainer(clf_F)
        shap_values_F = explainer_F.shap_values(X_F)
        mean_abs_shap_F = np.abs(shap_values_F).mean(axis=0)
        
        print("\nMean Absolute SHAP values (Model F - Marg+Consist+Roll):")
        for feat, val in sorted(zip(models_def['Mod_F_Marg_Consist_Roll'], mean_abs_shap_F), key=lambda x: x[1], reverse=True):
            print(f"  {feat:<20}: {val:.4f}")
            
        shap.summary_plot(shap_values_F, X_F, show=False)
        plt.tight_layout()
        plt.savefig('analysis/figures/margin_audit/shap_summary_Mod_F.png', dpi=300)
        plt.close()
        print("SHAP summary plots saved.")
        
    except ImportError:
        print("SHAP library not found. Skipping SHAP analysis. Install via 'pip install shap' to run.")
        
    # ---------------------------------------------------------
    # EXPERIMENT 4: FAILURE CASE COMPARISON
    # ---------------------------------------------------------
    print("\n" + "="*50)
    print("EXPERIMENT 4: FAILURE CASE COMPARISON")
    print("="*50)
    
    # Store predictions in df
    df['prob_G'] = predictions['Mod_G_Full']
    df['prob_C'] = predictions['Mod_C_Consist_Roll']
    
    failures = df[df['correct'] == 0].copy()
    top_failures_G = failures.sort_values('prob_G', ascending=False).head(20)
    
    print("Top 20 Highest-Confidence Failures (Model G):")
    compare_cols = ['subject_id', 'trial_id', 'window_id', 'margin', 'trial_consistency', 'rolling_std_margin', 'prob_G', 'prob_C']
    print(top_failures_G[compare_cols].to_string(index=False))
    
    diff = np.mean(np.abs(top_failures_G['prob_G'] - top_failures_G['prob_C']))
    print(f"\nMean absolute difference in probability between Model G and Model C on these cases: {diff:.4f}")

if __name__ == "__main__":
    main()
