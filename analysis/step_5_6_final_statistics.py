import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, auc
import xgboost as xgb
import os

# Set seed for reproducibility
np.random.seed(42)

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

def bootstrap_ci(metric_fn, y_true, y_pred, n_resamples=10000, ci=0.95):
    """Calculate bootstrap confidence interval for a metric."""
    bootstrapped_scores = []
    n = len(y_true)
    for _ in range(n_resamples):
        # Sample with replacement
        indices = np.random.randint(0, n, n)
        if len(np.unique(y_true[indices])) < 2 and metric_fn == roc_auc_score:
            continue  # Need both classes for AUROC
        score = metric_fn(y_true[indices], y_pred[indices])
        bootstrapped_scores.append(score)
        
    sorted_scores = np.sort(bootstrapped_scores)
    lower_bound = np.percentile(sorted_scores, (1 - ci) / 2 * 100)
    upper_bound = np.percentile(sorted_scores, (1 + ci) / 2 * 100)
    mean_score = np.mean(bootstrapped_scores)
    return mean_score, lower_bound, upper_bound

def accuracy_score_fn(y_true, y_pred):
    return np.mean(y_true == y_pred)

import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_file', type=str, default='subject_distance_predictions.csv')
    parser.add_argument('--model_path', type=str, default='models/confidence_model.json')
    args = parser.parse_args()
    
    csv_file = args.csv_file
    if not os.path.exists(csv_file):
        print(f"Error: {csv_file} not found. Are you in the right Kaggle directory?")
        return
        
    df = pd.read_csv(csv_file)
    df = df[np.isfinite(df['margin'])].reset_index(drop=True)
    df = add_temporal_features(df)
    
    model_path = args.model_path
    if not os.path.exists(model_path):
        print(f"Error: {model_path} not found.")
        return
        
    model = xgb.XGBClassifier()
    model.load_model(model_path)
    
    features = ['margin', 'sim_chosen', 'sim_unchosen', 'rolling_std_margin', 'trial_consistency']
    X = df[features]
    df['confidence'] = model.predict_proba(X)[:, 1]
    
    print("="*50)
    print("BOOTSTRAP CONFIDENCE INTERVALS (10,000 resamples)")
    print("="*50)
    
    # 1. Base MatchNet Accuracy
    base_acc_mean, base_acc_low, base_acc_high = bootstrap_ci(
        accuracy_score_fn, 
        df['correct'].values, 
        np.ones_like(df['correct'].values) # correct==1 means prediction matched label
    )
    # The actual base accuracy is mean of 'correct' since prediction is already evaluated
    print(f"Base MatchNet Accuracy: {df['correct'].mean()*100:.2f}% (95% CI: [{base_acc_low*100:.2f}%, {base_acc_high*100:.2f}%])")
    
    # 2. Confidence AUROC
    auroc_mean, auroc_low, auroc_high = bootstrap_ci(
        roc_auc_score,
        df['correct'].values,
        df['confidence'].values
    )
    print(f"Confidence AUROC: {roc_auc_score(df['correct'], df['confidence']):.4f} (95% CI: [{auroc_low:.4f}, {auroc_high:.4f}])")
    
    # 3. Selective Accuracy at 70% coverage (top 70% confidence)
    coverage_target = 0.70
    threshold = df['confidence'].quantile(1 - coverage_target)
    
    # We bootstrap the selective accuracy specifically at this threshold
    def selective_accuracy_fn(y_true, confs):
        accepted = confs >= threshold
        if np.sum(accepted) == 0: return 0.0
        return np.mean(y_true[accepted])
        
    sel_acc_mean, sel_acc_low, sel_acc_high = bootstrap_ci(
        selective_accuracy_fn,
        df['correct'].values,
        df['confidence'].values
    )
    actual_sel_acc = df[df['confidence'] >= threshold]['correct'].mean()
    print(f"Selective Accuracy (@ 70% coverage): {actual_sel_acc*100:.2f}% (95% CI: [{sel_acc_low*100:.2f}%, {sel_acc_high*100:.2f}%])")
    
    
    print("\n" + "="*50)
    print("SELECTIVE PREDICTION METRICS")
    print("="*50)
    
    # Sort by confidence descending
    df_sorted = df.sort_values('confidence', ascending=False)
    y_true_sorted = df_sorted['correct'].values
    
    # Coverages from 1.0 down to 0.0
    coverages = np.arange(1, len(df_sorted) + 1) / len(df_sorted)
    
    # Accuracies at each coverage
    cumulative_correct = np.cumsum(y_true_sorted)
    accuracies = cumulative_correct / np.arange(1, len(df_sorted) + 1)
    
    # Risks (Error Rates) = 1 - Accuracy
    risks = 1.0 - accuracies
    
    # Area Under Risk-Coverage (AURC)
    # Using np.trapz to integrate Risk over Coverage
    aurc = auc(coverages, risks)
    print(f"AURC (Area Under Risk-Coverage): {aurc:.4f}")
    
    # Excess AURC (E-AURC)
    # E-AURC = AURC - Optimal AURC
    # Optimal AURC happens when all incorrect predictions have the lowest confidence.
    y_true_optimal = np.sort(df['correct'].values)[::-1] # 1s first, 0s last
    optimal_cumulative_correct = np.cumsum(y_true_optimal)
    optimal_accuracies = optimal_cumulative_correct / np.arange(1, len(df) + 1)
    optimal_risks = 1.0 - optimal_accuracies
    optimal_aurc = auc(coverages, optimal_risks)
    
    e_aurc = aurc - optimal_aurc
    print(f"Optimal AURC: {optimal_aurc:.4f}")
    print(f"E-AURC (Excess AURC): {e_aurc:.4f}")
    
    print(f"Selective Gain (@ 70% coverage): {(actual_sel_acc - df['correct'].mean())*100:.2f}%")
    
    print("\n" + "="*50)
    print("EVIDENCE LOCK EXPORT")
    print("="*50)
    with open("analysis/summaries/final_statistics_lock.txt", "w") as f:
        f.write(f"base_acc:{df['correct'].mean():.4f}\n")
        f.write(f"base_acc_ci:[{base_acc_low:.4f},{base_acc_high:.4f}]\n")
        f.write(f"auroc:{roc_auc_score(df['correct'], df['confidence']):.4f}\n")
        f.write(f"auroc_ci:[{auroc_low:.4f},{auroc_high:.4f}]\n")
        f.write(f"sel_acc_70:{actual_sel_acc:.4f}\n")
        f.write(f"sel_acc_70_ci:[{sel_acc_low:.4f},{sel_acc_high:.4f}]\n")
        f.write(f"aurc:{aurc:.4f}\n")
        f.write(f"e_aurc:{e_aurc:.4f}\n")
        f.write(f"opt_aurc:{optimal_aurc:.4f}\n")
        
if __name__ == "__main__":
    main()
