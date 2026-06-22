import argparse
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

def step_3_1_evaluate_distance(df):
    print("\n===========================================")
    print("STEP 3.1: SUBJECT DISTANCE BASELINE")
    print("===========================================\n")
    
    correct = df['correct'].values
    margin = df['margin'].values
    euc_dist = df['euc_dist'].values
    mah_dist = df['mah_dist'].values
    
    # Calculate AUROC
    # For margin, larger is more confident
    auroc_margin = roc_auc_score(correct, margin)
    
    # For distance, smaller is more confident (more familiar subject), so we use negative distance
    auroc_euc = roc_auc_score(correct, -euc_dist)
    auroc_mah = roc_auc_score(correct, -mah_dist)
    
    # Calculate AUPRC
    auprc_margin = average_precision_score(correct, margin)
    auprc_euc = average_precision_score(correct, -euc_dist)
    auprc_mah = average_precision_score(correct, -mah_dist)
    
    print(f"{'Confidence Method':<22} | {'AUROC':<8} | {'AUPRC':<8}")
    print("-" * 44)
    print(f"{'Margin':<22} | {auroc_margin:<8.4f} | {auprc_margin:<8.4f}")
    print(f"{'Euclidean Distance':<22} | {auroc_euc:<8.4f} | {auprc_euc:<8.4f}")
    print(f"{'Mahalanobis Distance':<22} | {auroc_mah:<8.4f} | {auprc_mah:<8.4f}")
    
    print("\n--- Interpretation ---")
    best_auroc = max(auroc_margin, auroc_euc, auroc_mah)
    if best_auroc == auroc_margin:
        print("Weak Result: Subject Distance AUROC < Margin AUROC.")
        print("Subject familiarity alone does not outperform simple decision margin as a confidence indicator.")
    elif best_auroc > 0.70:
        print("Strong Result: Distance AUROC > Margin AUROC and > 0.70.")
        print("Subject familiarity strongly predicts whether MatchNet will decode accurately! This validates the subject-variability hypothesis.")
    else:
        print("Interesting Result: Distance AUROC is comparable to Margin AUROC.")
        print("Subject familiarity contains similar confidence information to decision margin.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="subject_distance_predictions.csv")
    args = parser.parse_args()
    
    try:
        df = pd.read_csv(args.csv)
    except FileNotFoundError:
        print(f"Error: Could not find {args.csv}")
        return
        
    step_3_1_evaluate_distance(df)

if __name__ == "__main__":
    main()
