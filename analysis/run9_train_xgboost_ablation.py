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
    print("--- Phase 9.1: XGBoost Confidence Ablation ---")
    
    features_dir = REPO_ROOT / "results" / "run9_xgboost" / "features"
    out_dir = REPO_ROOT / "results" / "run9_1_ablation"
    os.makedirs(out_dir, exist_ok=True)
    
    if not features_dir.exists():
        print(f"Features directory not found: {features_dir}")
        return
        
    modes = ["clean", "random", "zero", "gaussian", "audio_permute", "label_shuffle", "circular_shift"]
    
    subjects = [f.split("_")[1] for f in os.listdir(features_dir) if f.endswith("_train_clean.csv")]
    subjects = sorted(list(set(subjects)))
    print(f"Found features for {len(subjects)} subjects.")
    
    latent_cols = [f'z_{i}' for i in range(64)] + ['latent_norm', 'latent_std']
    
    experiments = {
        'A_Margin': ['margin'],
        'B_Pearson': ['margin', 'ca', 'cb'],
        'C_Latent': latent_cols,
        'D_All': latent_cols + ['margin', 'ca', 'cb']
    }
    
    all_results = {exp_name: {m: [] for m in modes} for exp_name in experiments.keys()}
    feature_importances = []
    
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
    
    for subj in subjects:
        print(f"Training Fold for Test Subject: {subj}")
        train_csv = features_dir / f"fold_{subj}_train_clean.csv"
        if not train_csv.exists(): continue
            
        df_train = pd.read_csv(train_csv)
        y_train = df_train['correct']
        
        # Train a model for each experiment
        models = {}
        for exp_name, cols in experiments.items():
            X_train = df_train[cols]
            dtrain = xgb.DMatrix(X_train, label=y_train)
            model = xgb.train(params, dtrain, num_boost_round=100)
            models[exp_name] = model
            
            # Save feature importance for D_All to see what dominates
            if exp_name == 'D_All':
                importance = model.get_score(importance_type='gain')
                for k, v in importance.items():
                    feature_importances.append({'fold': subj, 'feature': k, 'gain': v})
        
        # Evaluate on test sets for all modes
        for mode in modes:
            test_csv = features_dir / f"fold_{subj}_test_{mode}.csv"
            if not test_csv.exists(): continue
                
            df_test = pd.read_csv(test_csv)
            
            for exp_name, cols in experiments.items():
                X_test = df_test[cols]
                dtest = xgb.DMatrix(X_test)
                preds = models[exp_name].predict(dtest)
                
                # Reconstruct the results dictionary
                for i in range(len(df_test)):
                    all_results[exp_name][mode].append({
                        'subject': subj,
                        'margin': df_test['margin'].iloc[i],
                        'correct': df_test['correct'].iloc[i],
                        'confidence': preds[i],
                        'latent_norm': df_test['latent_norm'].iloc[i],
                        'ca': df_test['ca'].iloc[i],
                        'cb': df_test['cb'].iloc[i]
                    })
                    
    print("\n--- Ablation Evaluation Complete ---")
    
    # Save predictions and compute stats for each experiment
    final_stats = []
    
    for exp_name in experiments.keys():
        exp_dir = out_dir / exp_name
        os.makedirs(exp_dir, exist_ok=True)
        
        df_clean = pd.DataFrame(all_results[exp_name]["clean"])
        df_clean.to_csv(exp_dir / "clean_predictions.csv", index=False)
        
        correct = df_clean['correct'].values.astype(float)
        conf = df_clean['confidence'].values
        
        ece, mce = compute_ece(correct, conf)
        brier = brier_score_loss(correct, conf)
        auroc = roc_auc_score(correct, conf)
        precision, recall, _ = precision_recall_curve(correct, conf)
        auprc = auc(recall, precision)
        
        final_stats.append({
            'Experiment': exp_name,
            'AUROC': auroc,
            'AUPRC': auprc,
            'ECE': ece,
            'MCE': mce,
            'Brier': brier
        })
        
        # Robustness
        robust_stats = []
        for mode in modes:
            if len(all_results[exp_name][mode]) == 0: continue
            df_mode = pd.DataFrame(all_results[exp_name][mode])
            robust_stats.append({
                'Mode': mode,
                'Mean_Conf': df_mode['confidence'].mean(),
                'Median_Conf': df_mode['confidence'].median(),
                'Var_Conf': df_mode['confidence'].var(),
                'AUROC': roc_auc_score(df_mode['correct'].values, df_mode['confidence'].values) if len(np.unique(df_mode['correct'])) > 1 else np.nan
            })
        pd.DataFrame(robust_stats).to_csv(exp_dir / "robustness_stats.csv", index=False)
        
        # Selective Prediction
        thresholds = np.arange(0.50, 0.96, 0.05)
        sel_res = []
        for th in thresholds:
            accepted = df_clean[df_clean['confidence'] >= th]
            rejected = df_clean[df_clean['confidence'] < th]
            cov = len(accepted) / len(df_clean) if len(df_clean) > 0 else 0
            acc_acc = accepted['correct'].mean() if len(accepted) > 0 else np.nan
            rej_acc = rejected['correct'].mean() if len(rejected) > 0 else np.nan
            sel_res.append({'Threshold': th, 'Coverage': cov, 'Accepted_Acc': acc_acc, 'Rejected_Acc': rej_acc})
        pd.DataFrame(sel_res).to_csv(exp_dir / "selective_prediction.csv", index=False)
        
    stats_df = pd.DataFrame(final_stats)
    print("\n[Final Ablation Comparison]")
    print(stats_df.to_string(index=False))
    stats_df.to_csv(out_dir / "ablation_summary.csv", index=False)
    
    print("\n[Feature Importance (Top 10 in D_All)]")
    df_imp = pd.DataFrame(feature_importances)
    mean_imp = df_imp.groupby('feature')['gain'].mean().sort_values(ascending=False).reset_index()
    print(mean_imp.head(10).to_string(index=False))
    mean_imp.to_csv(out_dir / "feature_importance_D_All.csv", index=False)
    
    print("\nDone. Results saved to", out_dir)

if __name__ == "__main__":
    main()
