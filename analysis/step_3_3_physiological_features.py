import os
import sys
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from scipy.stats import mannwhitneyu, entropy
from scipy.signal import welch
from sklearn.metrics import roc_auc_score

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from baselines.ridge_aad import subject_files, load_subject_examples
from training.train_matchnet_loso import prepare_dataset, get_mapping_data

def spectral_entropy(x, fs=64.0, nperseg=128):
    """Calculate spectral entropy of the signals."""
    _, psd = welch(x, fs=fs, nperseg=nperseg, axis=1)
    psd_norm = psd / np.sum(psd, axis=1, keepdims=True)
    se = entropy(psd_norm, axis=1)
    return np.mean(se)

def extract_physiological_features(subject_id, tX):
    """
    tX is a list of numpy arrays, each shape [channels, time]
    """
    variances = []
    covariances = []
    spectral_entropies = []
    
    for x in tX:
        # Variance
        var = np.mean(np.var(x, axis=1))
        variances.append(var)
        
        # Off-diagonal Covariance (Absolute)
        cov_matrix = np.cov(x)
        # Extract upper triangle without diagonal
        upper_tri_indices = np.triu_indices_from(cov_matrix, k=1)
        off_diag_cov = np.mean(np.abs(cov_matrix[upper_tri_indices]))
        covariances.append(off_diag_cov)
        
        # Spectral Entropy
        se = spectral_entropy(x, fs=64.0)
        spectral_entropies.append(se)
        
    return {
        'variance': np.mean(variances),
        'covariance': np.mean(covariances),
        'spectral_entropy': np.mean(spectral_entropies)
    }

def step_3_3_characterize(predictions_csv):
    print("\n===========================================")
    print("STEP 3.3: SUBJECT RELIABILITY CHARACTERIZATION")
    print("===========================================\n")
    
    df = pd.read_csv(predictions_csv)
    if 'margin' not in df.columns:
        df['margin'] = np.abs(df['sim_A'] - df['sim_B'])
        
    subject_stats = []
    for subj_id, group in df.groupby('subject_id', observed=True):
        acc = group['correct'].mean()
        try:
            auroc = roc_auc_score(group['correct'], group['margin'])
        except ValueError:
            auroc = 0.5
        subject_stats.append({'subject_id': str(subj_id), 'accuracy': acc, 'auroc': auroc})
        
    stats_df = pd.DataFrame(subject_stats)
    stats_df = stats_df.sort_values('auroc', ascending=False).reset_index(drop=True)
    
    # Split into Top 9 (Reliable) and Bottom 9 (Unreliable)
    n_total = len(stats_df)
    reliable_subjs = stats_df.iloc[:n_total//2]['subject_id'].tolist()
    unreliable_subjs = stats_df.iloc[n_total//2:]['subject_id'].tolist()
    
    print("Loading datasets to extract EEG features...")
    all_paths = subject_files()
    mapping, envelopes = get_mapping_data()
    
    lowcut, highcut = 1.0, 9.0 # Standard MatchNet config
    
    feature_rows = []
    
    for path in tqdm(all_paths, desc="Processing Subjects"):
        subj_id = str(path.stem)
        if subj_id not in stats_df['subject_id'].values:
            continue
            
        exs = load_subject_examples(path)
        if not exs:
            continue
            
        # load raw data, apply standard filtering
        tX, _, _ = prepare_dataset(exs, channels=[13, 46, 43, 23, 50, 0, 52, 14], 
                                   lowcut=lowcut, highcut=highcut, 
                                   subject_id=subj_id, mapping=mapping, envelopes=envelopes)
                                   
        feats = extract_physiological_features(subj_id, tX)
        
        # Merge with accuracy/auroc
        row_stats = stats_df[stats_df['subject_id'] == subj_id].iloc[0]
        feats['subject_id'] = subj_id
        feats['accuracy'] = row_stats['accuracy']
        feats['auroc'] = row_stats['auroc']
        feats['group'] = 'Reliable' if subj_id in reliable_subjs else 'Unreliable'
        
        feature_rows.append(feats)
        
    feat_df = pd.DataFrame(feature_rows)
    feat_df = feat_df.sort_values('auroc', ascending=False)
    
    print("\n--- Per-Subject Physiological Features ---")
    print(f"{'Subject':<10} | {'Group':<10} | {'AUROC':<7} | {'Variance':<10} | {'Covariance':<10} | {'Spec Entropy':<12}")
    print("-" * 75)
    for _, row in feat_df.iterrows():
        print(f"{row['subject_id']:<10} | {row['group']:<10} | {row['auroc']:.4f} | {row['variance']:<10.4f} | {row['covariance']:<10.4f} | {row['spectral_entropy']:<12.4f}")
        
    print("\n--- Group Comparisons (Reliable vs Unreliable) ---")
    rel_df = feat_df[feat_df['group'] == 'Reliable']
    unrel_df = feat_df[feat_df['group'] == 'Unreliable']
    
    metrics = ['variance', 'covariance', 'spectral_entropy']
    print(f"{'Metric':<18} | {'Reliable Mean':<15} | {'Unreliable Mean':<15} | {'P-Value (Mann-Whitney)'}")
    print("-" * 75)
    
    for m in metrics:
        rel_mean = rel_df[m].mean()
        unrel_mean = unrel_df[m].mean()
        _, p_val = mannwhitneyu(rel_df[m], unrel_df[m], alternative='two-sided')
        print(f"{m:<18} | {rel_mean:<15.4f} | {unrel_mean:<15.4f} | {p_val:.4f}")
        
    print("\n--- Interpretation ---")
    print("If p-value < 0.05 for any metric, we have discovered a physiological marker")
    print("for Confidence Reliability Collapse. For example, if Spectral Entropy is significantly")
    print("higher in the Unreliable group, their brains exhibit more noise/less structured rhythm,")
    print("destroying MatchNet's ability to measure decision boundary distance effectively.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="matchnet_predictions.csv")
    args = parser.parse_args()
    
    try:
        step_3_3_characterize(args.csv)
    except FileNotFoundError:
        print(f"Error: Could not find {args.csv}")
        return

if __name__ == "__main__":
    main()
