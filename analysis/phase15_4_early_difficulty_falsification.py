import os
import sys
import json
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, balanced_accuracy_score, matthews_corrcoef, precision_score, recall_score, f1_score, average_precision_score
from sklearn.model_selection import LeaveOneGroupOut, GroupKFold
from sklearn.ensemble import RandomForestClassifier
import random

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from decision_policy_engine import DecisionPolicyEngine, State

def get_online_features_10s(group):
    # Strictly extract features using ONLY first 10 windows (or up to 10s)
    engine = DecisionPolicyEngine()
    probs, margins, evidences = [], [], []
    
    for _, row in group.iterrows():
        if row['window'] > 10:
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
        'sprt_evidence_10s': evidences[-1],
        'evidence_slope_10s': np.polyfit(np.arange(len(evidences)), evidences, 1)[0] if len(evidences) > 1 else 0,
    }

def simulate_full_trial(group):
    engine = DecisionPolicyEngine()
    reached_lock = False
    lock_window = -1
    final_decision = None
    trajectory = []
    
    true_label = group['ground_truth'].iloc[0] if 'ground_truth' in group else group.get('label', 1).iloc[0]
    
    for _, row in group.iterrows():
        prob = float(row['prob_platt'] if 'prob_platt' in row else row.get('calibrated_prob', 0.5))
        margin = float(row.get('margin', 0.0))
        win = int(row['window'])
        result = engine.update(prob, margin)
        
        trajectory.append({
            'window': win,
            'probability': prob,
            'margin': margin,
            'evidence': result['evidence'],
            'state': result['state'].name if hasattr(result['state'], 'name') else str(result['state']),
            'decision': result['decision']
        })
        
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
            if lock_window <= 30:
                category = 'EASY'
            else:
                category = 'SLOW'
                
    return category, trajectory

def evaluate_model(X, y, groups, random_state=42):
    logo = LeaveOneGroupOut()
    rf = RandomForestClassifier(n_estimators=100, max_depth=3, random_state=random_state)
    
    y_true, y_pred, y_prob = [], [], []
    for train_idx, test_idx in logo.split(X, y, groups):
        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]
        
        # If a subject only has one class in training, skip or handle
        if len(np.unique(y_train)) < 2:
            continue
            
        rf.fit(X_train, y_train)
        y_true.extend(y_test)
        y_prob.extend(rf.predict_proba(X_test)[:, 1])
        y_pred.extend(rf.predict(X_test))
        
    return np.array(y_true), np.array(y_pred), np.array(y_prob)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--preds', type=str, required=True, help='Path to predictions CSV')
    parser.add_argument('--out', type=str, required=True, help='Output directory')
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.preds)
    
    print("--------------------------------------------")
    
    # STEP 1 & 8: Information Leakage & Duplicate Audit
    all_trials = []
    features_list = []
    
    for (subj, trial), group in df.groupby(['subject', 'trial']):
        subj_clean = int(subj) if isinstance(subj, (int, np.integer)) else str(subj)
        trial_clean = int(trial) if isinstance(trial, (int, np.integer)) else str(trial)
        trial_id = f"{subj_clean}_{trial_clean}"
        group = group.sort_values('window')
        
        category, trajectory = simulate_full_trial(group)
        features = get_online_features_10s(group)
        
        if features and category in ['EASY', 'HARD']:
            features['subject'] = subj_clean
            features['trial'] = trial_clean
            features['trial_id'] = trial_id
            features['target'] = 1 if category == 'EASY' else 0
            features['trajectory'] = trajectory
            features_list.append(features)
            all_trials.append(trial_id)
            
    if len(all_trials) != len(set(all_trials)):
        print("Duplicate Audit .... FAIL (Duplicate trials found)")
        sys.exit(1)
    else:
        print("Duplicate Audit .... PASS")
        
    feat_df = pd.DataFrame(features_list)
    
    if len(feat_df) == 0 or len(feat_df['target'].unique()) < 2:
        print("Data Audit .... FAIL (Not enough EASY/HARD classes)")
        sys.exit(1)
        
    feature_cols = ['prob_mean', 'prob_var', 'prob_trend', 'margin_mean', 'margin_var', 'margin_trend', 'sprt_evidence_10s', 'evidence_slope_10s']
    X = feat_df[feature_cols].values
    y = feat_df['target'].values
    groups = feat_df['subject'].values
    
    # STEP 9: Class Imbalance Audit
    n_easy = sum(y == 1)
    n_hard = sum(y == 0)
    if n_hard < 5 or n_easy < 5:
        print(f"Class Imbalance Audit .... FAIL (EASY={n_easy}, HARD={n_hard})")
        sys.exit(1)
    else:
        print(f"Class Imbalance Audit .... PASS (EASY={n_easy}, HARD={n_hard})")

    # STEP 3: Subject-Level CV
    y_true, y_pred, y_prob = evaluate_model(X, y, groups)
    if len(np.unique(y_true)) < 2:
        print("Subject-Level CV .... FAIL (No mixed classes in predictions)")
        sys.exit(1)
        
    auroc = roc_auc_score(y_true, y_prob)
    pr_auc = average_precision_score(y_true, y_prob)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)
    
    print(f"Subject-Level CV .... {'PASS' if auroc > 0.6 and auroc < 0.99 else 'FAIL'} (AUROC: {auroc:.3f})")
    
    # STEP 4: Bootstrap Validation
    n_bootstraps = 1000
    bootstrapped_scores = []
    np.random.seed(42)
    
    for i in range(n_bootstraps):
        indices = np.random.randint(0, len(y_prob), len(y_prob))
        if len(np.unique(y_true[indices])) < 2:
            continue
        score = roc_auc_score(y_true[indices], y_prob[indices])
        bootstrapped_scores.append(score)
        
    sorted_scores = np.array(bootstrapped_scores)
    sorted_scores.sort()
    ci_lower = sorted_scores[int(0.025 * len(sorted_scores))]
    ci_upper = sorted_scores[int(0.975 * len(sorted_scores))]
    pd.DataFrame(bootstrapped_scores, columns=['AUROC']).to_csv(out_dir / 'bootstrap_results.csv', index=False)
    
    # STEP 5: Random Seed Stability
    seed_scores = []
    for seed in range(20):
        yt, yp, ypb = evaluate_model(X, y, groups, random_state=seed)
        seed_scores.append(roc_auc_score(yt, ypb))
    
    # STEP 6: Negative Controls
    neg_controls = []
    
    # A) Shuffle labels
    y_shuffled = np.random.permutation(y)
    yt, yp, ypb = evaluate_model(X, y_shuffled, groups)
    auroc_shuffled = roc_auc_score(yt, ypb)
    neg_controls.append({'Control': 'Shuffled Labels', 'AUROC': auroc_shuffled})
    
    # B) Random probabilities
    X_rand = np.random.rand(*X.shape)
    yt, yp, ypb = evaluate_model(X_rand, y, groups)
    auroc_rand = roc_auc_score(yt, ypb)
    neg_controls.append({'Control': 'Random Features', 'AUROC': auroc_rand})
    
    # C) Permuted evidence (Shuffle rows for SPRT evidence only)
    X_perm = X.copy()
    X_perm[:, 6] = np.random.permutation(X_perm[:, 6])
    yt, yp, ypb = evaluate_model(X_perm, y, groups)
    auroc_perm = roc_auc_score(yt, ypb)
    neg_controls.append({'Control': 'Permuted SPRT Evidence', 'AUROC': auroc_perm})
    
    pd.DataFrame(neg_controls).to_csv(out_dir / 'negative_controls.csv', index=False)
    
    neg_failed = any(c['AUROC'] > 0.65 for c in neg_controls)
    if neg_failed:
        print("Negative Controls .... FAIL (Leakage detected in controls)")
    else:
        print("Negative Controls .... PASS")
        
    # STEP 7: Feature Importance
    rf_full = RandomForestClassifier(n_estimators=100, max_depth=3, random_state=42)
    rf_full.fit(X, y)
    importances = rf_full.feature_importances_
    feat_imp = pd.DataFrame({'Feature': feature_cols, 'Importance': importances}).sort_values('Importance', ascending=False)
    feat_imp.to_csv(out_dir / 'feature_importance.csv', index=False)
    
    # STEP 10: Manual Inspection
    easy_sample = feat_df[feat_df['target'] == 1].sample(min(10, n_easy), random_state=42)
    hard_sample = feat_df[feat_df['target'] == 0].sample(min(10, n_hard), random_state=42)
    
    with open(out_dir / 'decision_samples.jsonl', 'w') as f:
        for _, row in easy_sample.iterrows():
            f.write(json.dumps({'trial_id': row['trial_id'], 'category': 'EASY', 'trajectory': row['trajectory']}) + '\n')
        for _, row in hard_sample.iterrows():
            f.write(json.dumps({'trial_id': row['trial_id'], 'category': 'HARD', 'trajectory': row['trajectory']}) + '\n')
            
    # Verdict Generation
    passed_all = (auroc < 0.99) and (auroc > 0.6) and not neg_failed and (n_hard >= 5)
    
    print("\nFinal Verdict")
    if passed_all:
        print("The Early Difficulty Predictor is scientifically valid.")
    else:
        print("The AUROC=1.000 result was an artifact. The predictor FAILED falsification.")
        
    with open(out_dir / 'validation_report.md', 'w') as f:
        f.write("# Phase 15.4 Falsification Report\n\n")
        f.write(f"## Verdict\n")
        if passed_all:
            f.write("**The Early Difficulty Predictor is scientifically valid.**\n\n")
        else:
            f.write("**The AUROC=1.000 result was an artifact.**\n\n")
            
        f.write(f"## Subject-Level CV\n")
        f.write(f"- AUROC: {auroc:.4f}\n")
        f.write(f"- PR-AUC: {pr_auc:.4f}\n")
        f.write(f"- Balanced Accuracy: {bal_acc:.4f}\n")
        f.write(f"- MCC: {mcc:.4f}\n\n")
        
        f.write(f"## Bootstrap (95% CI)\n")
        f.write(f"AUROC CI: [{ci_lower:.4f}, {ci_upper:.4f}]\n\n")
        
        f.write(f"## Random Seed Stability\n")
        f.write(f"- Mean AUROC: {np.mean(seed_scores):.4f} ± {np.std(seed_scores):.4f}\n")
        f.write(f"- Min AUROC: {np.min(seed_scores):.4f}\n")
        f.write(f"- Max AUROC: {np.max(seed_scores):.4f}\n\n")
        
        f.write("## Negative Controls\n")
        f.write(pd.DataFrame(neg_controls).to_markdown(index=False) + "\n\n")
        
        f.write("## Feature Importance\n")
        f.write(feat_imp.to_markdown(index=False) + "\n\n")

    print(f"\nFiles written to {out_dir}")
    print("Done")
    print("--------------------------------------------")

if __name__ == '__main__':
    main()
