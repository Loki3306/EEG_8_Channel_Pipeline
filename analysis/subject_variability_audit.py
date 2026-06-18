"""
subject_variability_audit.py
Analyzes the preprocessed DTU dataset to quantify covariance shifts, 
PSD profiles, and linear separability across subjects.
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.signal import welch
from scipy.stats import kurtosis
import sys
import seaborn as sns

# Add repo root to path
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from baselines.ridge_aad import subject_files, load_subject_examples

def print_and_write(f_out, text):
    print(text)
    f_out.write(text + "\n")

def run_subject_audit(out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "subject_variability_report.md"
    
    paths = subject_files()
    if not paths:
        print("ERROR: Could not find any preprocessed DTU .mat files.")
        return
        
    # Known good subjects from MatchNet LOSO analysis
    good_subjects = ['S1', 'S8', 'S14']
    
    fs = 64.0 # Preprocessed sampling rate
    
    subject_metrics = []
    subject_covs = {}
    subject_psds = {}
    f_psd = None
    
    print(f"Starting Subject Variability Audit on {len(paths)} subjects...")
    
    for p in paths:
        sub_id = p.stem.split("_")[0]
        print(f"Processing {sub_id}...")
        
        examples = load_subject_examples(p)
        if not examples:
            print(f"WARNING: No examples found for {sub_id}")
            continue
            
        all_eeg = []
        all_var = []
        all_kurt = []
        
        # Centroid separation prep
        feat_class1 = []
        feat_class2 = []
        
        trial_covs = []
        trial_psds = []
        
        for ex in examples:
            # ex.eeg is typically (channels, time)
            eeg = ex.eeg
            if eeg.shape[0] > eeg.shape[1]:
                eeg = eeg.T # Ensure (channels, time)
                
            # Signal quality
            var_mean = np.mean(np.var(eeg, axis=1))
            kurt_mean = np.mean(kurtosis(eeg, axis=1))
            all_var.append(var_mean)
            all_kurt.append(kurt_mean)
            
            # Covariance
            # Subtract mean over time per channel
            eeg_c = eeg - np.mean(eeg, axis=1, keepdims=True)
            cov = np.dot(eeg_c, eeg_c.T) / (eeg_c.shape[1] - 1)
            trial_covs.append(cov)
            
            # PSD
            f, pxx = welch(eeg, fs=fs, nperseg=int(fs*2), axis=1)
            if f_psd is None:
                f_psd = f
            # Average PSD across channels for this trial
            trial_psds.append(np.mean(pxx, axis=0))
            
            # Simple spatial feature for centroid distance (mean absolute amplitude per channel)
            spatial_feat = np.mean(np.abs(eeg), axis=1)
            if ex.label == 1:
                feat_class1.append(spatial_feat)
            else:
                feat_class2.append(spatial_feat)
                
        # Aggregate per subject
        mean_cov = np.mean(trial_covs, axis=0)
        mean_psd = np.mean(trial_psds, axis=0)
        
        subject_covs[sub_id] = mean_cov
        subject_psds[sub_id] = mean_psd
        
        c1_centroid = np.mean(feat_class1, axis=0) if feat_class1 else np.zeros(mean_cov.shape[0])
        c2_centroid = np.mean(feat_class2, axis=0) if feat_class2 else np.zeros(mean_cov.shape[0])
        
        centroid_dist = np.linalg.norm(c1_centroid - c2_centroid)
        
        subject_metrics.append({
            'Subject': sub_id,
            'Global_Variance': np.mean(all_var),
            'Mean_Kurtosis': np.mean(all_kurt),
            'Centroid_Dist': centroid_dist
        })
        
    df = pd.DataFrame(subject_metrics)
    
    # Calculate Reference Covariance (from Good Subjects)
    ref_covs = [subject_covs[s] for s in good_subjects if s in subject_covs]
    if ref_covs:
        grand_ref_cov = np.mean(ref_covs, axis=0)
    else:
        # Fallback if specific good subjects are not found
        grand_ref_cov = np.mean(list(subject_covs.values()), axis=0)
        
    # Compute Covariance Shift (Frobenius norm distance from reference)
    cov_shifts = []
    for s in df['Subject']:
        dist = np.linalg.norm(subject_covs[s] - grand_ref_cov, ord='fro')
        cov_shifts.append(dist)
    df['Covariance_Shift'] = cov_shifts
    
    # Save Report
    with open(report_path, "w") as f_out:
        print_and_write(f_out, "# Subject Variability Audit Report\n")
        
        print_and_write(f_out, "## 1. Global Metrics across Preprocessed DTU\n")
        print_and_write(f_out, df.to_markdown(index=False))
        print_and_write(f_out, "\n")
        
        # Correlation Matrix of metrics
        print_and_write(f_out, "## 2. Metric Correlations\n")
        corr = df.drop(columns=['Subject']).corr()
        print_and_write(f_out, corr.to_markdown())
        print_and_write(f_out, "\n")
        
        # High Variance/Shift subjects
        outliers = df.sort_values(by='Covariance_Shift', ascending=False).head(3)
        print_and_write(f_out, "## 3. Highest Covariance Shifts (Potential Outliers)\n")
        print_and_write(f_out, outliers[['Subject', 'Covariance_Shift', 'Global_Variance']].to_markdown(index=False))
        print_and_write(f_out, "\n")
        
        print_and_write(f_out, "> **Note:** To complete the LOSO correlation analysis, merge this table with the MatchNet LOSO accuracy results CSV.")
        
    # Plotting PSDs
    plt.figure(figsize=(12, 6))
    for s, psd in subject_psds.items():
        alpha = 0.8 if s in good_subjects else 0.3
        lw = 2 if s in good_subjects else 1
        plt.plot(f_psd, 10 * np.log10(psd), label=f"{s}{' (Good)' if s in good_subjects else ''}", alpha=alpha, linewidth=lw)
    plt.title("Subject PSD Profiles (0-32 Hz)")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Power (dB)")
    plt.xlim(0, 32)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(out_dir / "subject_psd_profiles.png")
    plt.close()
    
    # Plotting Covariance Shift
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df.sort_values('Covariance_Shift'), x='Subject', y='Covariance_Shift')
    plt.title("Covariance Shift from 'Good Subject' Reference")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(out_dir / "subject_covariance_shifts.png")
    plt.close()

    print(f"\nAudit complete. Report saved to {report_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, default="reports", help="Directory to save reports")
    args = parser.parse_args()
    
    run_subject_audit(args.out_dir)
