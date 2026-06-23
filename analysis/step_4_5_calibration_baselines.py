import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.calibration import calibration_curve
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def compute_ece(y_true, y_prob, n_bins=10):
    bins = np.linspace(0., 1., n_bins + 1)
    binids = np.digitize(y_prob, bins) - 1
    
    ece = 0.0
    for i in range(n_bins):
        bin_idx = binids == i
        if np.sum(bin_idx) > 0:
            bin_acc = np.mean(y_true[bin_idx])
            bin_conf = np.mean(y_prob[bin_idx])
            bin_weight = np.sum(bin_idx) / len(y_prob)
            ece += bin_weight * np.abs(bin_acc - bin_conf)
    return ece

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="subject_distance_predictions.csv")
    args = parser.parse_args()
    
    try:
        df = pd.read_csv(args.csv)
    except FileNotFoundError:
        print(f"Cannot find {args.csv}. Please specify the correct path.")
        return
        
    # Clean invalid rows
    df = df[np.isfinite(df['margin'])].reset_index(drop=True)
    
    subjects = df['subject_id'].unique()
    
    df['prob_platt'] = 0.0
    df['prob_iso'] = 0.0
    
    print("\n===========================================")
    print("STEP 4.5: CLASSICAL CALIBRATION BASELINES")
    print("===========================================")
    print(f"Running strict nested LOSO calibration across {len(subjects)} subjects...\n")
    
    for test_subj in subjects:
        train_mask = df['subject_id'] != test_subj
        test_mask = df['subject_id'] == test_subj
        
        X_train = df.loc[train_mask, ['margin']]
        y_train = df.loc[train_mask, 'correct']
        
        X_test = df.loc[test_mask, ['margin']]
        
        # 1. Platt Scaling (Logistic Regression)
        lr = LogisticRegression()
        lr.fit(X_train, y_train)
        df.loc[test_mask, 'prob_platt'] = lr.predict_proba(X_test)[:, 1]
        
        # 2. Isotonic Regression
        iso = IsotonicRegression(out_of_bounds='clip')
        iso.fit(X_train['margin'], y_train)
        df.loc[test_mask, 'prob_iso'] = iso.predict(X_test['margin'])
        
    print("--- GLOBAL METRICS ---")
    
    # Naive scaling just for ECE baseline comparison
    margin_min, margin_max = df['margin'].min(), df['margin'].max()
    df['prob_naive'] = (df['margin'] - margin_min) / (margin_max - margin_min)
    
    for name, col in [("Raw Margin (MinMax scaled proxy)", "prob_naive"), 
                      ("Platt Scaling (1D Logistic)", "prob_platt"), 
                      ("Isotonic Regression", "prob_iso")]:
        # AUROC uses raw margin for the naive baseline, as MinMax doesn't change rank
        metric_col = df['margin'] if "Raw" in name else df[col]
        
        auroc = roc_auc_score(df['correct'], metric_col)
        brier = brier_score_loss(df['correct'], df[col])
        ece = compute_ece(df['correct'].values, df[col].values)
        
        print(f"\n{name}:")
        print(f"  AUROC : {auroc:.4f}")
        print(f"  Brier : {brier:.4f}")
        print(f"  ECE   : {ece:.4f}")
        
    # Calibration Plot
    plt.figure(figsize=(10, 8))
    plt.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    
    for name, col in [("Raw Margin Proxy", "prob_naive"), 
                      ("Platt Scaling", "prob_platt"), 
                      ("Isotonic Regression", "prob_iso")]:
        prob_true, prob_pred = calibration_curve(df['correct'], df[col], n_bins=10, strategy='uniform')
        plt.plot(prob_pred, prob_true, "s-", label=f"{name} (ECE={compute_ece(df['correct'].values, df[col].values):.3f})")
        
    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Fraction of Correct Predictions")
    plt.title("Calibration Curves (Reliability Diagram)\nStrict Nested LOSO Evaluation")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    
    out_path = Path.cwd() / "calibration_diagram.png"
    plt.tight_layout()
    plt.savefig(out_path)
    print(f"\nCalibration diagram saved to {out_path}")
    print("===========================================\n")

if __name__ == "__main__":
    main()
