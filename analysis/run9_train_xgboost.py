import os
import sys
import pandas as pd
import numpy as np
import xgboost as xgb
from pathlib import Path
from sklearn.metrics import brier_score_loss, roc_auc_score, precision_recall_curve, auc

REPO_ROOT = Path(__file__).resolve().parents[1]

def compute_ece(correct, conf, bins=10):
    bin_boundaries = np.linspace(0, 1, bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece, mce = 0.0, 0.0
    for i, (bin_lower, bin_upper) in enumerate(zip(bin_lowers, bin_uppers)):
        if i == bins - 1:
            in_bin = (conf >= bin_lower) & (conf <= bin_upper)
        else:
            in_bin = (conf >= bin_lower) & (conf < bin_upper)
            
        prop_in_bin = in_bin.mean()
        if prop_in_bin > 0:
            accuracy_in_bin = correct[in_bin].mean()
            avg_confidence_in_bin = conf[in_bin].mean()
            error = np.abs(avg_confidence_in_bin - accuracy_in_bin)
            ece += prop_in_bin * error
            mce = max(mce, error)
    return ece, mce

def main():
    print("--- Phase 9: Train & Evaluate XGBoost Confidence Baseline ---")
    
    features_dir = REPO_ROOT / "results" / "run9_xgboost" / "features"
    out_dir = REPO_ROOT / "results" / "run9_xgboost" / "predictions"
    os.makedirs(out_dir, exist_ok=True)
    
    if not features_dir.exists():
        print(f"Features directory not found: {features_dir}")
        print("Please run run9_extract_xgboost_features.py first.")
        return
        
    modes = ["clean", "random", "zero", "gaussian", "audio_permute", "label_shuffle", "circular_shift"]
    
    # We will gather all predictions across folds
    all_results = {m: [] for m in modes}
    
    subjects = [f.split("_")[1] for f in os.listdir(features_dir) if f.endswith("_train_clean.csv")]
    subjects = sorted(list(set(subjects)))
    
    print(f"Found features for {len(subjects)} subjects.")
    
    feature_cols = [f'z_{i}' for i in range(64)] + ['ca', 'cb', 'margin', 'latent_norm', 'latent_std']
    
    feature_importances = []
    
    for subj in subjects:
        print(f"\nTraining Fold for Test Subject: {subj}")
        train_csv = features_dir / f"fold_{subj}_train_clean.csv"
        if not train_csv.exists(): continue
            
        df_train = pd.read_csv(train_csv)
        X_train = df_train[feature_cols]
        y_train = df_train['correct']
        
        # XGBoost parameters (Conservative)
        params = {
            'objective': 'binary:logistic',
            'max_depth': 4,
            'learning_rate': 0.05,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'eval_metric': 'logloss',
            'random_state': 42
        }
        
        dtrain = xgb.DMatrix(X_train, label=y_train)
        model = xgb.train(params, dtrain, num_boost_round=100)
        
        # Save feature importances
        importance = model.get_score(importance_type='gain')
        for k, v in importance.items():
            feature_importances.append({'fold': subj, 'feature': k, 'gain': v})
        
        # Evaluate on test sets for all modes
        for mode in modes:
            test_csv = features_dir / f"fold_{subj}_test_{mode}.csv"
            if not test_csv.exists(): continue
                
            df_test = pd.read_csv(test_csv)
            X_test = df_test[feature_cols]
            
            dtest = xgb.DMatrix(X_test)
            preds = model.predict(dtest)
            
            # Reconstruct the results dictionary
            for i in range(len(df_test)):
                all_results[mode].append({
                    'subject': subj,
                    'margin': df_test['margin'].iloc[i],
                    'correct': df_test['correct'].iloc[i],
                    'confidence': preds[i],
                    'latent_norm': df_test['latent_norm'].iloc[i],
                    'ca': df_test['ca'].iloc[i],
                    'cb': df_test['cb'].iloc[i]
                })
                
    print("\n--- XGBoost Evaluation Complete ---")
    
    df_clean = pd.DataFrame(all_results["clean"])
    df_clean.to_csv(out_dir / "clean_predictions.csv", index=False)
    
    print("\n[Stage 2 & 7] Confidence Robustness")
    robust_stats = []
    for mode in modes:
        if len(all_results[mode]) == 0: continue
        df_mode = pd.DataFrame(all_results[mode])
        robust_stats.append({
            'Mode': mode,
            'Mean_Conf': df_mode['confidence'].mean(),
            'Median_Conf': df_mode['confidence'].median(),
            'Var_Conf': df_mode['confidence'].var()
        })
    rob_df = pd.DataFrame(robust_stats)
    print(rob_df.to_string(index=False))
    rob_df.to_csv(out_dir / "robustness_stats.csv", index=False)
    
    print("\n[Stage 3] Selective Prediction")
    thresholds = np.arange(0.50, 0.96, 0.05)
    sel_res = []
    for th in thresholds:
        accepted = df_clean[df_clean['confidence'] >= th]
        rejected = df_clean[df_clean['confidence'] < th]
        cov = len(accepted) / len(df_clean) if len(df_clean) > 0 else 0
        acc_acc = accepted['correct'].mean() if len(accepted) > 0 else np.nan
        rej_acc = rejected['correct'].mean() if len(rejected) > 0 else np.nan
        sel_res.append({'Threshold': th, 'Coverage': cov, 'Accepted_Acc': acc_acc, 'Rejected_Acc': rej_acc})
    sel_df = pd.DataFrame(sel_res)
    print(sel_df.to_string(index=False))
    sel_df.to_csv(out_dir / "selective_prediction.csv", index=False)
    
    print("\n[Stage 4 & 5 & 6] Statistical & Calibration Verification")
    correct = df_clean['correct'].values.astype(float)
    conf = df_clean['confidence'].values
    
    ece, mce = compute_ece(correct, conf)
    brier = brier_score_loss(correct, conf)
    auroc = roc_auc_score(correct, conf)
    precision, recall, _ = precision_recall_curve(correct, conf)
    auprc = auc(recall, precision)
    
    print(f"Global ECE: {ece:.4f} | MCE: {mce:.4f} | Brier: {brier:.4f}")
    print(f"AUROC: {auroc:.4f} | AUPRC: {auprc:.4f}")
    
    n_bootstraps = 1000
    boot_aurocs = []
    rng = np.random.RandomState(42)
    for i in range(n_bootstraps):
        indices = rng.randint(0, len(conf), len(conf))
        if len(np.unique(correct[indices])) < 2: continue
        boot_aurocs.append(roc_auc_score(correct[indices], conf[indices]))
    
    ci_lower = np.percentile(boot_aurocs, 2.5)
    ci_upper = np.percentile(boot_aurocs, 97.5)
    print(f"AUROC 95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
    
    print("\n[Feature Importance (Top 10)]")
    df_imp = pd.DataFrame(feature_importances)
    mean_imp = df_imp.groupby('feature')['gain'].mean().sort_values(ascending=False).reset_index()
    print(mean_imp.head(10).to_string(index=False))
    mean_imp.to_csv(out_dir / "feature_importance.csv", index=False)
    
    print("\nDone. Results saved to", out_dir)

if __name__ == "__main__":
    main()
