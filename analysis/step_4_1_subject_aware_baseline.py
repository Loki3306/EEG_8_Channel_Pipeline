import os
import sys
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import warnings
warnings.filterwarnings('ignore')

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from baselines.ridge_aad import subject_files, load_subject_examples
from training.train_matchnet_loso import prepare_dataset, get_mapping_data
from analysis.step_3_3_physiological_features import extract_physiological_features

def compute_ece(correct, conf, n_bins=10):
    """Compute Expected Calibration Error"""
    min_c, max_c = conf.min(), conf.max()
    bins = np.linspace(min_c, max_c, n_bins + 1)
    bins = np.unique(bins)
    if len(bins) < 2:
        bins = np.linspace(0, 1, n_bins + 1)
        
    df = pd.DataFrame({'correct': correct, 'conf': conf})
    df['bin'] = pd.cut(df['conf'], bins=bins, include_lowest=True)
    
    stats = df.groupby('bin', observed=True).agg(
        count=('correct', 'size'),
        mean_conf=('conf', 'mean'),
        accuracy=('correct', 'mean')
    )
    
    ece = 0.0
    total = len(df)
    
    for _, row in stats.iterrows():
        if row['count'] > 0:
            ece += (row['count'] / total) * np.abs(row['accuracy'] - row['mean_conf'])
            
    return ece

def step_4_1_subject_aware_baseline(predictions_csv):
    print("\n===========================================")
    print("STEP 4.1: INFERENCE-SAFE SUBJECT-AWARE CALIBRATION")
    print("===========================================\n")
    
    df = pd.read_csv(predictions_csv)
    if 'margin' not in df.columns:
        df['margin'] = np.abs(df['sim_A'] - df['sim_B'])
        
    print("Loading datasets to extract inference-safe physiological features...")
    all_paths = subject_files()
    mapping, envelopes = get_mapping_data()
    
    subject_features = {}
    lowcut, highcut = 1.0, 9.0
    for path in tqdm(all_paths, desc="Extracting Features"):
        subj_id = str(path.stem)
        exs = load_subject_examples(path)
        if not exs: continue
        
        tX, _, _ = prepare_dataset(exs, channels=[13, 46, 43, 23, 50, 0, 52, 14], 
                                   lowcut=lowcut, highcut=highcut, 
                                   subject_id=subj_id, mapping=mapping, envelopes=envelopes)
                                   
        feats = extract_physiological_features(subj_id, tX)
        subject_features[subj_id] = feats
        
    # Map features to df
    df['variance'] = df['subject_id'].map(lambda x: subject_features[str(x)]['variance'] if str(x) in subject_features else np.nan)
    df['covariance'] = df['subject_id'].map(lambda x: subject_features[str(x)]['covariance'] if str(x) in subject_features else np.nan)
    df['spectral_entropy'] = df['subject_id'].map(lambda x: subject_features[str(x)]['spectral_entropy'] if str(x) in subject_features else np.nan)
    
    # Drop rows with missing features (if any subjects were skipped)
    df = df.dropna(subset=['variance', 'covariance', 'spectral_entropy']).reset_index(drop=True)
    
    X_base = df[['margin']].copy()
    X_subj = df[['margin', 'variance', 'covariance', 'spectral_entropy']].copy()
    y = df['correct'].values
    groups = df['subject_id'].values
    
    logo = LeaveOneGroupOut()
    
    oof_baseline = np.zeros(len(df))
    oof_subject_aware = np.zeros(len(df))
    
    print("\nEvaluating models using Leave-One-Group-Out (LOGO) Cross Validation...")
    
    for train_idx, test_idx in logo.split(X_base, y, groups):
        # Baseline
        X_train_b, X_test_b = X_base.iloc[train_idx], X_base.iloc[test_idx]
        y_train = y[train_idx]
        
        lr_base = LogisticRegression(class_weight='balanced')
        lr_base.fit(X_train_b, y_train)
        oof_baseline[test_idx] = lr_base.predict_proba(X_test_b)[:, 1]
        
        # Subject-Aware
        X_train_s, X_test_s = X_subj.iloc[train_idx], X_subj.iloc[test_idx]
        
        # Scale features
        lr_subj = make_pipeline(StandardScaler(), LogisticRegression(class_weight='balanced', max_iter=500))
        lr_subj.fit(X_train_s, y_train)
        oof_subject_aware[test_idx] = lr_subj.predict_proba(X_test_s)[:, 1]
        
    # Calculate Global Metrics
    print("\n--- Calibration Results ---")
    
    base_auroc = roc_auc_score(y, oof_baseline)
    base_brier = brier_score_loss(y, oof_baseline)
    base_ece = compute_ece(y, oof_baseline)
    
    subj_auroc = roc_auc_score(y, oof_subject_aware)
    subj_brier = brier_score_loss(y, oof_subject_aware)
    subj_ece = compute_ece(y, oof_subject_aware)
    
    print(f"{'Model':<35} | {'Global AUROC':<15} | {'Brier Score':<15} | {'ECE':<10}")
    print("-" * 82)
    print(f"{'Baseline (Margin)':<35} | {base_auroc:<15.4f} | {base_brier:<15.4f} | {base_ece:<10.4f}")
    print(f"{'Subject-Aware (Margin + EEG Feats)':<35} | {subj_auroc:<15.4f} | {subj_brier:<15.4f} | {subj_ece:<10.4f}")
    
    print("\n--- Interpretation ---")
    if subj_auroc > base_auroc + 0.05:
        print("MASSIVE IMPROVEMENT: Physiological descriptors provide critical calibration information.")
        print("This proves that 'Confidence Reliability is Subject-Dependent', forming a perfect publishable thesis.")
    elif subj_auroc > base_auroc + 0.01:
        print("MODERATE IMPROVEMENT: Descriptive statistics help somewhat, but maybe they aren't expressive enough.")
    else:
        print("NO IMPROVEMENT: Simple physiological descriptors do not generalize confidence prediction to unseen subjects.")
        print("Conclusion: MatchNet requires an architectural Confidence Head to learn deep subject representations.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="matchnet_predictions.csv")
    args = parser.parse_args()
    
    try:
        step_4_1_subject_aware_baseline(args.csv)
    except FileNotFoundError:
        print(f"Error: Could not find {args.csv}")
        return

if __name__ == "__main__":
    main()
