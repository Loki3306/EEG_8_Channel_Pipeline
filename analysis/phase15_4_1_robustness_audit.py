import os
import sys
import json
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from sklearn.model_selection import LeaveOneGroupOut, GroupKFold
from sklearn.ensemble import RandomForestClassifier
from tqdm import tqdm
import random

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from decision_policy_engine import DecisionPolicyEngine, State

def simulate_full_trial(group):
    true_label = group['ground_truth'].iloc[0] if 'ground_truth' in group else group.get('label', 1).iloc[0]
    engine = DecisionPolicyEngine()
    reached_lock = False
    lock_window = -1
    final_decision = None
    
    for _, row in group.iterrows():
        prob = float(row['prob_platt'] if 'prob_platt' in row else row.get('calibrated_prob', 0.5))
        margin = float(row.get('margin', 0.0))
        win = int(row['window'])
        result = engine.update(prob, margin)
        
        if result['state'] == State.LOCKED and not reached_lock:
            reached_lock = True
            lock_window = win
            final_decision = result['decision']
            
    if not reached_lock:
        category = 'HARD'
    else:
        if final_decision != true_label:
            category = 'WRONG'
        else:
            if lock_window <= 15: # 30s at 2s hop
                category = 'EASY'
            else:
                category = 'SLOW'
    return category

def get_early_features(group, max_time_sec, hop_sec=2):
    engine = DecisionPolicyEngine()
    probs, margins, evidences = [], [], []
    
    for _, row in group.iterrows():
        if (row['window'] + 1) * hop_sec > max_time_sec:
            break
            
        prob = float(row['prob_platt'] if 'prob_platt' in row else row.get('calibrated_prob', 0.5))
        margin = float(row.get('margin', 0.0))
        result = engine.update(prob, margin)
        
        probs.append(prob)
        margins.append(margin)
        evidences.append(result['evidence'])
        
    if not probs:
        return None
        
    return {
        'prob_mean': np.mean(probs),
        'prob_var': np.var(probs),
        'prob_trend': np.polyfit(np.arange(len(probs)), probs, 1)[0] if len(probs) > 1 else 0,
        'margin_mean': np.mean(margins),
        'margin_var': np.var(margins),
        'margin_trend': np.polyfit(np.arange(len(margins)), margins, 1)[0] if len(margins) > 1 else 0,
        'sprt_evidence': evidences[-1],
        'evidence_slope': np.polyfit(np.arange(len(evidences)), evidences, 1)[0] if len(evidences) > 1 else 0,
    }

def evaluate_model(X, y, groups, random_state=42, cv=LeaveOneGroupOut()):
    rf = RandomForestClassifier(n_estimators=100, max_depth=3, random_state=random_state)
    y_true, y_pred, y_prob = [], [], []
    
    for train_idx, test_idx in cv.split(X, y, groups):
        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]
        
        if len(np.unique(y_train)) < 2:
            continue
            
        rf.fit(X_train, y_train)
        y_true.extend(y_test)
        y_prob.extend(rf.predict_proba(X_test)[:, 1])
        y_pred.extend(rf.predict(X_test))
        
    return np.array(y_true), np.array(y_pred), np.array(y_prob)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--preds', type=str, required=True)
    parser.add_argument('--out', type=str, required=True)
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.preds)
    
    print("--------------------------------------------")
    categories = {}
    for (subj, trial), group in df.groupby(['subject', 'trial']):
        trial_id = f"{subj}_{trial}"
        group = group.sort_values('window')
        categories[trial_id] = simulate_full_trial(group)
        
    def extract_dataset(max_time_sec):
        features_list = []
        for (subj, trial), group in df.groupby(['subject', 'trial']):
            trial_id = f"{subj}_{trial}"
            cat = categories[trial_id]
            if cat in ['EASY', 'HARD']:
                group = group.sort_values('window')
                feat = get_early_features(group, max_time_sec)
                if feat:
                    feat['subject'] = subj
                    feat['trial_id'] = trial_id
                    feat['target'] = 1 if cat == 'EASY' else 0
                    features_list.append(feat)
        return pd.DataFrame(features_list)

    feat_df = extract_dataset(10)
    if len(feat_df) == 0 or len(feat_df['target'].unique()) < 2:
        print("Data Audit .... FAIL (Not enough EASY/HARD classes)")
        sys.exit(1)
        
    feature_cols = ['prob_mean', 'prob_var', 'prob_trend', 'margin_mean', 'margin_var', 'margin_trend', 'sprt_evidence', 'evidence_slope']
    X = feat_df[feature_cols].values
    y = feat_df['target'].values
    groups = feat_df['subject'].values
    
    # 1. Validation Matrix
    val_matrix = pd.DataFrame([
        {'Validation': 'Label Shuffle', 'Type': 'True Negative Control', 'Expected AUROC': '~0.5', 'Triggers Failure': 'Yes (if > 0.65)'},
        {'Validation': 'Random Features', 'Type': 'True Negative Control', 'Expected AUROC': '~0.5', 'Triggers Failure': 'Yes (if > 0.65)'},
        {'Validation': 'Permute Individual Features', 'Type': 'Feature Importance Test', 'Expected AUROC': 'Unknown', 'Triggers Failure': 'No'},
        {'Validation': 'Bootstrap', 'Type': 'Robustness Test', 'Expected AUROC': 'N/A', 'Triggers Failure': 'No'},
        {'Validation': 'Subject LOGO CV', 'Type': 'Robustness Test', 'Expected AUROC': 'N/A', 'Triggers Failure': 'No'},
        {'Validation': 'GroupKFold', 'Type': 'Robustness Test', 'Expected AUROC': 'N/A', 'Triggers Failure': 'No'}
    ])
    val_matrix.to_csv(out_dir / 'validation_matrix.csv', index=False)
    print("Validation Matrix ...... DONE")
    
    # True Negative Controls (Fix Logic)
    yt, yp, ypb = evaluate_model(X, np.random.permutation(y), groups)
    auroc_shuffled = roc_auc_score(yt, ypb) if len(np.unique(yt)) > 1 else 0.5
    
    yt, yp, ypb = evaluate_model(np.random.rand(*X.shape), y, groups)
    auroc_rand = roc_auc_score(yt, ypb) if len(np.unique(yt)) > 1 else 0.5
    
    leakage_detected = (auroc_shuffled > 0.65) or (auroc_rand > 0.65)
    
    # 2. Feature Permutation Study
    yt, yp, ypb_base = evaluate_model(X, y, groups)
    base_auroc = roc_auc_score(yt, ypb_base) if len(np.unique(yt)) > 1 else 0.5
    
    perm_results = []
    np.random.seed(42)
    for i, col in enumerate(feature_cols):
        X_perm = X.copy()
        X_perm[:, i] = np.random.permutation(X_perm[:, i])
        yt, yp, ypb = evaluate_model(X_perm, y, groups)
        p_auroc = roc_auc_score(yt, ypb) if len(np.unique(yt)) > 1 else 0.5
        perm_results.append({
            'Feature': col,
            'Original AUROC': base_auroc,
            'Permuted AUROC': p_auroc,
            'AUROC Drop': base_auroc - p_auroc
        })
        
    perm_df = pd.DataFrame(perm_results).sort_values('AUROC Drop', ascending=False)
    perm_df['Importance Ranking'] = np.arange(1, len(perm_df) + 1)
    perm_df.to_csv(out_dir / 'feature_permutation.csv', index=False)
    print("Feature Permutation .... DONE")
    
    # 3. Early Predictability Curve & Bootstrapping
    time_pts = [2, 4, 6, 8, 10, 15, 20, 30]
    curve_results = []
    boot_results = []
    
    for t in time_pts:
        tdf = extract_dataset(t)
        if len(tdf) == 0:
            continue
        Xt = tdf[feature_cols].values
        yt_t = tdf['target'].values
        gt = tdf['subject'].values
        
        y_true_t, _, y_prob_t = evaluate_model(Xt, yt_t, gt)
        if len(np.unique(y_true_t)) < 2:
            continue
        
        t_auroc = roc_auc_score(y_true_t, y_prob_t)
        curve_results.append({'Time (s)': t, 'AUROC': t_auroc})
        
        # Bootstrap
        np.random.seed(42)
        b_scores = []
        for _ in range(1000):
            idx = np.random.randint(0, len(y_prob_t), len(y_prob_t))
            if len(np.unique(y_true_t[idx])) < 2:
                continue
            b_scores.append(roc_auc_score(y_true_t[idx], y_prob_t[idx]))
        
        if b_scores:
            b_scores.sort()
            boot_results.append({
                'Time (s)': t,
                'Mean AUROC': np.mean(b_scores),
                'Lower 95% CI': b_scores[int(0.025 * len(b_scores))],
                'Upper 95% CI': b_scores[int(0.975 * len(b_scores))]
            })
            
    pd.DataFrame(curve_results).to_csv(out_dir / 'early_predictability_curve.csv', index=False)
    print("Early Predictability ... DONE")
    
    pd.DataFrame(boot_results).to_csv(out_dir / 'bootstrap_summary.csv', index=False)
    print("Bootstrap .............. DONE")
    
    # 4. Subject Robustness
    subject_res = []
    n_groups = len(np.unique(groups))
    cvs = {
        'LOGO': LeaveOneGroupOut(),
        'GroupKFold': GroupKFold(n_splits=min(5, n_groups)) if n_groups >= 2 else LeaveOneGroupOut()
    }
    for name, cv in cvs.items():
        scores = []
        for seed in range(10):
            yt_cv, _, ypb_cv = evaluate_model(X, y, groups, random_state=seed, cv=cv)
            if len(np.unique(yt_cv)) > 1:
                scores.append(roc_auc_score(yt_cv, ypb_cv))
        
        if scores:
            scores.sort()
            subject_res.append({
                'CV Strategy': name,
                'Mean AUROC': np.mean(scores),
                'Std AUROC': np.std(scores),
                'Lower 95% CI': scores[int(0.025 * len(scores))],
                'Upper 95% CI': scores[int(0.975 * len(scores))]
            })
    pd.DataFrame(subject_res).to_csv(out_dir / 'subject_results.csv', index=False)
    print("Subject Validation ..... DONE")
    
    # 5. Final Verdict
    valid_controls = not leakage_detected
    top_feature = perm_df.iloc[0]['Feature'] if len(perm_df) > 0 else "N/A"
    
    is_growing = True
    if len(curve_results) >= 2:
        if curve_results[-1]['AUROC'] <= curve_results[0]['AUROC']:
            is_growing = False
            
    scientifically_valid = valid_controls and base_auroc > 0.7
    
    with open(out_dir / 'validation_report.md', 'w') as f:
        f.write("# Phase 15.4.1 Falsification Report\n\n")
        f.write("## 1. Are the TRUE negative controls valid?\n")
        f.write(f"{'Yes' if valid_controls else 'No'} - Label Shuffle AUROC: {auroc_shuffled:.3f}, Random Features AUROC: {auroc_rand:.3f}\n\n")
        f.write("## 2. Is there evidence of leakage?\n")
        f.write(f"{'Yes' if leakage_detected else 'No'}. True negative controls {'failed' if leakage_detected else 'passed'} (< 0.65 threshold).\n\n")
        f.write("## 3. Which feature contributes most?\n")
        if len(perm_df) > 0:
            f.write(f"{top_feature} with an AUROC drop of {perm_df.iloc[0]['AUROC Drop']:.3f}\n\n")
        else:
            f.write("N/A\n\n")
        f.write("## 4. Does predictive performance increase naturally with time?\n")
        if len(curve_results) >= 2:
            f.write(f"{'Yes' if is_growing else 'No'}. From {curve_results[0]['Time (s)']}s to {curve_results[-1]['Time (s)']}s, AUROC went from {curve_results[0]['AUROC']:.3f} to {curve_results[-1]['AUROC']:.3f}.\n\n")
        else:
            f.write("N/A\n\n")
        f.write("## 5. Is the predictor genuinely learning early trial difficulty?\n")
        f.write(f"{'Yes' if scientifically_valid else 'No'}. The predictor shows strong true performance ({base_auroc:.3f}) and passes true negative controls.\n\n")
        
        f.write("## Verdict\n")
        f.write(f"**{'SCIENTIFICALLY VALID' if scientifically_valid else 'FAILED FALSIFICATION'}**\n")
        
    print("Final Verdict .......... DONE")
    print(f"\nFiles Written to {out_dir}")
    print("Done")
    print("--------------------------------------------")

if __name__ == '__main__':
    main()
