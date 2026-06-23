import argparse
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import calibration_curve
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

def get_rc_curve(correct_labels, confidences):
    n = len(correct_labels)
    sorted_indices = np.argsort(-confidences)
    sorted_labels = correct_labels[sorted_indices]
    
    coverages = np.arange(1, n + 1) / n
    accuracies = np.cumsum(sorted_labels) / np.arange(1, n + 1)
    risks = 1.0 - accuracies
    return coverages, accuracies, risks

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
    
    df['prob_margin_raw'] = df['margin'] 
    df['prob_iso'] = 0.0
    df['prob_xgb'] = 0.0
    
    print("Computing nested LOSO probabilities...")
    for test_subj in subjects:
        train_mask = df['subject_id'] != test_subj
        test_mask = df['subject_id'] == test_subj
        
        X_train_margin = df.loc[train_mask, 'margin'].values.reshape(-1, 1)
        X_test_margin = df.loc[test_mask, 'margin'].values.reshape(-1, 1)
        y_train = df.loc[train_mask, 'correct'].values
        
        min_m, max_m = X_train_margin.min(), X_train_margin.max()
        scaled_test_margin = (X_test_margin - min_m) / (max_m - min_m)
        df.loc[test_mask, 'prob_margin_raw'] = np.clip(scaled_test_margin, 0, 1).flatten()
        
        iso = IsotonicRegression(out_of_bounds='clip')
        iso.fit(X_train_margin.flatten(), y_train)
        df.loc[test_mask, 'prob_iso'] = iso.predict(X_test_margin.flatten())
        
        X_train_xgb = df.loc[train_mask, features]
        X_test_xgb = df.loc[test_mask, features]
        xgb_model = xgb.XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.05, n_jobs=-1, eval_metric='logloss')
        xgb_model.fit(X_train_xgb, y_train)
        df.loc[test_mask, 'prob_xgb'] = xgb_model.predict_proba(X_test_xgb)[:, 1]

    os.makedirs('analysis/figures', exist_ok=True)
    y_true = df['correct'].values
    
    # --- Figure 1: Coverage vs Accuracy ---
    print("Generating Figure 1: Coverage vs Accuracy")
    cov_margin, acc_margin, _ = get_rc_curve(y_true, df['prob_margin_raw'].values)
    cov_xgb, acc_xgb, _ = get_rc_curve(y_true, df['prob_xgb'].values)
    
    plt.figure(figsize=(8, 6))
    plt.plot(cov_margin * 100, acc_margin * 100, label='Margin Only (Baseline)', color='blue', linewidth=2)
    plt.plot(cov_xgb * 100, acc_xgb * 100, label='XGBoost (Temporal Fusion)', color='red', linewidth=2)
    plt.xlabel('Coverage (%)', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.title('Selective AAD: Coverage vs Accuracy', fontsize=14)
    plt.gca().invert_xaxis()  # 100% to 0% coverage
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('analysis/figures/coverage_vs_accuracy.png', dpi=300)
    plt.close()
    
    # --- Figure 2: Risk-Coverage Curve ---
    print("Generating Figure 2: Risk-Coverage Curve")
    _, _, risk_margin = get_rc_curve(y_true, df['prob_margin_raw'].values)
    _, _, risk_xgb = get_rc_curve(y_true, df['prob_xgb'].values)
    
    plt.figure(figsize=(8, 6))
    plt.plot(cov_margin * 100, risk_margin, label='Margin Only (Baseline)', color='blue', linewidth=2)
    plt.plot(cov_xgb * 100, risk_xgb, label='XGBoost (Temporal Fusion)', color='red', linewidth=2)
    plt.xlabel('Coverage (%)', fontsize=12)
    plt.ylabel('Risk (Error Rate)', fontsize=12)
    plt.title('Selective AAD: Risk-Coverage Curve', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('analysis/figures/risk_coverage_curve.png', dpi=300)
    plt.close()
    
    # --- Figure 3: Calibration Diagram ---
    print("Generating Figure 3: Calibration Diagram")
    plt.figure(figsize=(8, 8))
    
    plt.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    
    prob_true_iso, prob_pred_iso = calibration_curve(y_true, df['prob_iso'].values, n_bins=10)
    plt.plot(prob_pred_iso, prob_true_iso, "s-", label="Margin (Isotonic Scaled)", color='blue')
    
    prob_true_xgb, prob_pred_xgb = calibration_curve(y_true, df['prob_xgb'].values, n_bins=10)
    plt.plot(prob_pred_xgb, prob_true_xgb, "o-", label="XGBoost (Temporal Fusion)", color='red')
    
    plt.xlabel('Mean Predicted Probability', fontsize=12)
    plt.ylabel('Fraction of Positives', fontsize=12)
    plt.title('Reliability Diagram (Calibration)', fontsize=14)
    plt.legend(loc="lower right", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('analysis/figures/calibration_diagram.png', dpi=300)
    plt.close()

    print("All figures saved to analysis/figures/")
    print("===========================================\n")

if __name__ == "__main__":
    main()
