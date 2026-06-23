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
    
    # Pre-calculate errors to optimize
    errors = (df['true'] == 0).values
    cum_errors = np.cumsum(errors)
    
    for i in range(1, n + 1):
        coverage = i / n
        risk = cum_errors[i-1] / i
        coverages.append(coverage)
        risks.append(risk)
        
    # Area under risk-coverage curve
    e_aurc = np.trapz(risks, coverages)
    return e_aurc

def evaluate_predictions(y_true, y_prob):
    auroc = roc_auc_score(y_true, y_prob)
    e_aurc = compute_e_aurc(y_true, y_prob)
    brier = brier_score_loss(y_true, y_prob)
    ece = compute_ece(y_true, y_prob)
    
    # 90% threshold
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
    
    subjects = df['subject_id'].unique()
    
    all_features = ['margin', 'sim_chosen', 'sim_unchosen', 'rolling_std_margin', 'trial_consistency']
    
    models_def = {
        'M1_Margin': ['margin'],
        'M2_Consistency': ['trial_consistency'],
        'M3_Margin_Consist': ['margin', 'trial_consistency'],
        'M4_Marg_Consist_Roll': ['margin', 'trial_consistency', 'rolling_std_margin'],
        'M5_Full': all_features,
        'Abl_No_Margin': [f for f in all_features if f != 'margin'],
        'Abl_No_Consist': [f for f in all_features if f != 'trial_consistency'],
        'Abl_No_Roll': [f for f in all_features if f != 'rolling_std_margin'],
        'Abl_No_SimC': [f for f in all_features if f != 'sim_chosen'],
        'Abl_No_SimU': [f for f in all_features if f != 'sim_unchosen']
    }
    
    predictions = {m: np.zeros(len(df)) for m in models_def.keys()}
    
    print("Running Nested LOSO Evaluation for Minimal Audits...")
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
            
    print("\n" + "="*50)
    print("RESULTS SUMMARY")
    print("="*50)
    
    y_true = df['correct'].values
    results = {}
    for m_name in models_def.keys():
        res = evaluate_predictions(y_true, predictions[m_name])
        results[m_name] = res
        
    res_df = pd.DataFrame.from_dict(results, orient='index')
    print(res_df.to_string())
    
    full_res = results['M5_Full']
    
    print("\n" + "="*50)
    print("DELTA VS FULL MODEL")
    print("="*50)
    delta_df = res_df.copy()
    for col in delta_df.columns:
        if col in ['AUROC', 'Acc@90', 'Cov@90', 'N_Acc@90']:
            delta_df[col] = delta_df[col] - full_res[col]
        else:
            delta_df[col] = full_res[col] - delta_df[col]  # For error metrics, positive delta is better/closer to full
            
    print(delta_df.to_string())
    
    print("\n" + "="*50)
    print("RETAINED PERFORMANCE % (Higher is better)")
    print("="*50)
    print(f"M1 Margin Only      : AUROC = {results['M1_Margin']['AUROC']/full_res['AUROC']*100:.1f}%, E-AURC = {full_res['E-AURC']/results['M1_Margin']['E-AURC']*100:.1f}%")
    print(f"M2 Consistency Only : AUROC = {results['M2_Consistency']['AUROC']/full_res['AUROC']*100:.1f}%, E-AURC = {full_res['E-AURC']/results['M2_Consistency']['E-AURC']*100:.1f}%")
    print(f"M3 Margin+Consist   : AUROC = {results['M3_Margin_Consist']['AUROC']/full_res['AUROC']*100:.1f}%, E-AURC = {full_res['E-AURC']/results['M3_Margin_Consist']['E-AURC']*100:.1f}%")
    
    print("\n" + "="*50)
    print("FEATURE NECESSITY RANKING (Based on actual AUROC drop)")
    print("="*50)
    ablation_drops = {
        'margin': full_res['AUROC'] - results['Abl_No_Margin']['AUROC'],
        'trial_consistency': full_res['AUROC'] - results['Abl_No_Consist']['AUROC'],
        'rolling_std_margin': full_res['AUROC'] - results['Abl_No_Roll']['AUROC'],
        'sim_chosen': full_res['AUROC'] - results['Abl_No_SimC']['AUROC'],
        'sim_unchosen': full_res['AUROC'] - results['Abl_No_SimU']['AUROC']
    }
    
    sorted_drops = sorted(ablation_drops.items(), key=lambda x: x[1], reverse=True)
    for feat, drop in sorted_drops:
        print(f"  {feat:<20}: {-drop:+.4f} AUROC")
        
    os.makedirs('analysis/figures/behavior_audit', exist_ok=True)
    
    # Visualizations
    forward_models = ['M1_Margin', 'M2_Consistency', 'M3_Margin_Consist', 'M4_Marg_Consist_Roll', 'M5_Full']
    
    plt.figure(figsize=(10, 6))
    x = np.arange(len(forward_models))
    plt.bar(x, [results[m]['AUROC'] for m in forward_models], color='skyblue')
    plt.xticks(x, [m.replace('M', '').replace('_', '\n') for m in forward_models])
    plt.title('AUROC Across Forward Models')
    plt.ylabel('AUROC')
    plt.ylim([0.6, 0.85])
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig('analysis/figures/behavior_audit/model_comparison_barplot.png', dpi=300)
    plt.close()
    
    plt.figure(figsize=(10, 6))
    features_ordered = [x[0] for x in sorted_drops]
    drops_ordered = [x[1] for x in sorted_drops]
    plt.bar(features_ordered, drops_ordered, color='salmon')
    plt.title('AUROC Performance Drop When Feature is Removed')
    plt.ylabel('AUROC Loss')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('analysis/figures/behavior_audit/feature_ablation_drop.png', dpi=300)
    plt.close()
    
    plt.figure(figsize=(8, 6))
    retained = [
        results['M1_Margin']['AUROC']/full_res['AUROC']*100,
        results['M2_Consistency']['AUROC']/full_res['AUROC']*100,
        results['M3_Margin_Consist']['AUROC']/full_res['AUROC']*100,
        100.0
    ]
    labels = ['Margin', 'Consistency', 'Margin + Consist', 'Full']
    plt.bar(labels, retained, color='lightgreen')
    plt.axhline(100, color='r', linestyle='--')
    plt.title('% of Full Model AUROC Retained')
    plt.ylabel('Retained %')
    plt.ylim([80, 105])
    plt.tight_layout()
    plt.savefig('analysis/figures/behavior_audit/retained_performance.png', dpi=300)
    plt.close()
    
    print("Visualizations saved to analysis/figures/behavior_audit/")

if __name__ == "__main__":
    main()
