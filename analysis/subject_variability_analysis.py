import os
import sys
import yaml
import json
import subprocess
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from scipy.signal import welch
from scipy.stats import skew, kurtosis, entropy, mannwhitneyu, pearsonr, ttest_ind
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from statsmodels.stats.multitest import multipletests

try:
    import umap
except ImportError:
    pass

import matplotlib.pyplot as plt
import seaborn as sns

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from baselines.ridge_aad import subject_files, load_subject_examples

def get_git_commit():
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()
    except Exception:
        raise RuntimeError("Failed to retrieve git commit hash. Refusing to generate artifacts without reproducibility metadata.")

def write_metadata_header(filepath, title):
    commit = get_git_commit()
    timestamp = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')
    dataset = "DTU/KUL Auditory Attention"
    script = os.path.basename(__file__)
    
    with open(filepath, 'w') as f:
        f.write(f"# {title}\n\n")
        f.write(f"**Status:** Confirmed\n")
        f.write(f"**Generated:** {timestamp}\n")
        f.write(f"**Commit:** `{commit}`\n")
        f.write(f"**Dataset:** {dataset}\n")
        f.write(f"**Script:** `{script}`\n\n")

def add_csv_metadata(df, dataset_name="DTU/KUL"):
    df['commit_hash'] = get_git_commit()
    df['timestamp'] = datetime.utcnow().isoformat()
    df['dataset'] = dataset_name
    return df

def frobenius_distance(cov1, cov2):
    return np.linalg.norm(cov1 - cov2, ord='fro')

def riemannian_distance(cov1, cov2):
    from scipy.linalg import logm, sqrtm, inv
    try:
        sq_inv = inv(sqrtm(cov1))
        mat = sq_inv @ cov2 @ sq_inv
        log_mat = logm(mat)
        return np.linalg.norm(log_mat, ord='fro')
    except Exception:
        return np.nan

def cosine_similarity_cov(cov1, cov2):
    c1 = cov1.flatten()
    c2 = cov2.flatten()
    return np.dot(c1, c2) / (np.linalg.norm(c1) * np.linalg.norm(c2) + 1e-12)

def hjorth_parameters(x):
    # x is [time]
    dx = np.diff(x)
    ddx = np.diff(dx)
    
    var_x = np.var(x)
    var_dx = np.var(dx)
    var_ddx = np.var(ddx)
    
    activity = var_x
    mobility = np.sqrt(var_dx / (var_x + 1e-12))
    complexity = np.sqrt(var_ddx / (var_dx + 1e-12)) / (mobility + 1e-12)
    
    return activity, mobility, complexity

def cohens_d(x, y):
    n1, n2 = len(x), len(y)
    s1, s2 = np.var(x, ddof=1), np.var(y, ddof=1)
    s = np.sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))
    return (np.mean(x) - np.mean(y)) / (s + 1e-12)

def main():
    config_path = REPO_ROOT / "configs" / "subject_variability.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file missing: {config_path}")
        
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    stat_dir = REPO_ROOT / "results" / "statistics"
    fig_dir = REPO_ROOT / "results" / "figures"
    report_dir = REPO_ROOT / "results" / "reports"
    
    stat_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load LOSO Results
    loso_csv = stat_dir / "loso_accuracy.csv"
    if not loso_csv.exists():
        raise FileNotFoundError(f"LOSO accuracy file missing at {loso_csv}. Execute model evaluation first.")
        
    loso_df = pd.read_csv(loso_csv)
    
    if "subject" not in loso_df.columns or "accuracy" not in loso_df.columns:
        raise ValueError("LOSO accuracy CSV must contain 'subject' and 'accuracy' columns.")
        
    loso_df['rank'] = loso_df['accuracy'].rank(ascending=False)
    loso_df['quartile'] = pd.qcut(loso_df['accuracy'], 4, labels=['Q4', 'Q3', 'Q2', 'Q1'])
    loso_df = loso_df.sort_values('accuracy', ascending=False)
    
    add_csv_metadata(loso_df).to_csv(stat_dir / "subject_accuracy.csv", index=False)
    
    # 2. Analyze Subjects
    paths = subject_files()
    if not paths:
        raise FileNotFoundError("No subject EEG files found in dataset path.")
        
    # Input validation
    dataset_subjects = [p.stem.split('_')[0] for p in paths]
    csv_subjects = loso_df['subject'].tolist()
    
    if len(dataset_subjects) != len(csv_subjects):
        raise ValueError(f"Subject count mismatch. Dataset: {len(dataset_subjects)}, CSV: {len(csv_subjects)}")
        
    if set(dataset_subjects) != set(csv_subjects):
        raise ValueError("Subject names mismatch between dataset and CSV.")

    print("Analyzing actual subject files...")
    
    bandpowers = []
    signal_stats = []
    covariances = {}
    
    for p in paths:
        sub = p.stem.split('_')[0]
        examples = load_subject_examples(p)
        
        # Aggregate EEG data for subject
        all_eeg = np.concatenate([ex.eeg for ex in examples], axis=0) # [time, channels]
        n_channels = all_eeg.shape[1]
        
        # Signal stats (Per Channel)
        c_mean = np.mean(all_eeg, axis=0)
        c_std = np.std(all_eeg, axis=0)
        c_var = np.var(all_eeg, axis=0)
        c_skew = skew(all_eeg, axis=0)
        c_kurt = kurtosis(all_eeg, axis=0)
        
        c_ent = []
        c_act, c_mob, c_comp = [], [], []
        for ch in range(n_channels):
            hist, _ = np.histogram(all_eeg[:, ch], bins=100, density=True)
            c_ent.append(entropy(hist + 1e-12))
            
            act, mob, comp = hjorth_parameters(all_eeg[:, ch])
            c_act.append(act)
            c_mob.append(mob)
            c_comp.append(comp)
            
        ss_dict = {"subject": sub}
        for ch in range(n_channels):
            ss_dict[f"ch_{ch}_var"] = c_var[ch]
            ss_dict[f"ch_{ch}_entropy"] = c_ent[ch]
            ss_dict[f"ch_{ch}_hjorth_activity"] = c_act[ch]
            ss_dict[f"ch_{ch}_hjorth_mobility"] = c_mob[ch]
            ss_dict[f"ch_{ch}_hjorth_complexity"] = c_comp[ch]
            
        ss_dict["avg_var"] = np.mean(c_var)
        ss_dict["avg_entropy"] = np.mean(c_ent)
        ss_dict["avg_hjorth_mobility"] = np.mean(c_mob)
        ss_dict["avg_hjorth_complexity"] = np.mean(c_comp)
        
        signal_stats.append(ss_dict)
        
        # Covariance
        cov_mat = np.cov(all_eeg, rowvar=False)
        covariances[sub] = cov_mat
        
        # PSD Analysis (Welch's method)
        fs = 64
        freqs, psd = welch(all_eeg, fs=fs, axis=0) # psd is [freqs, channels]
        
        bands = config['bands']
        bp = {"subject": sub}
        for band_name, (low, high) in bands.items():
            idx = np.logical_and(freqs >= low, freqs <= high)
            band_psd = np.mean(psd[idx, :], axis=0) # [channels]
            
            for ch in range(n_channels):
                bp[f"ch_{ch}_{band_name}_power"] = band_psd[ch]
            bp[f"avg_{band_name}_power"] = np.mean(band_psd)
            
        bandpowers.append(bp)

    bp_df = pd.DataFrame(bandpowers)
    add_csv_metadata(bp_df).to_csv(stat_dir / "subject_bandpower.csv", index=False)
    
    ss_df = pd.DataFrame(signal_stats)
    add_csv_metadata(ss_df).to_csv(stat_dir / "subject_signal_stats.csv", index=False)
    
    # 3. Covariance Features & Distance Matrix
    subs = list(covariances.keys())
    n_subs = len(subs)
    
    frob_dist = np.zeros((n_subs, n_subs))
    riem_dist = np.zeros((n_subs, n_subs))
    cos_sim = np.zeros((n_subs, n_subs))
    
    cov_flat = []
    cov_features = []
    
    for i, s1 in enumerate(subs):
        c1 = covariances[s1]
        cov_flat.append(c1.flatten())
        
        cf = {
            "subject": s1,
            "trace": np.trace(c1),
            "determinant": np.linalg.det(c1),
            "condition_number": np.linalg.cond(c1)
        }
        cov_features.append(cf)
        
        for j, s2 in enumerate(subs):
            c2 = covariances[s2]
            frob_dist[i, j] = frobenius_distance(c1, c2)
            riem_dist[i, j] = riemannian_distance(c1, c2)
            cos_sim[i, j] = cosine_similarity_cov(c1, c2)
            
    frob_df = pd.DataFrame(frob_dist, index=subs, columns=subs)
    add_csv_metadata(frob_df).to_csv(stat_dir / "subject_frobenius_distance.csv")
    
    riem_df = pd.DataFrame(riem_dist, index=subs, columns=subs)
    add_csv_metadata(riem_df).to_csv(stat_dir / "subject_riemannian_distance.csv")
    
    cos_df = pd.DataFrame(cos_sim, index=subs, columns=subs)
    add_csv_metadata(cos_df).to_csv(stat_dir / "subject_cosine_similarity.csv")
    
    cf_df = pd.DataFrame(cov_features)
    add_csv_metadata(cf_df).to_csv(stat_dir / "subject_covariance_features.csv", index=False)
    
    # 4. Dimensionality Reduction
    X = np.array(cov_flat)
    
    # PCA
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X)
    explained_var = pca.explained_variance_ratio_
    
    plt.figure()
    plt.scatter(X_pca[:, 0], X_pca[:, 1])
    for i, txt in enumerate(subs):
        plt.annotate(txt, (X_pca[i, 0], X_pca[i, 1]))
    plt.title(f"PCA of Subject Covariances (Status: Confirmed)\nExplained Var: {explained_var[0]:.2f}, {explained_var[1]:.2f}\nCommit: {get_git_commit()}")
    plt.savefig(fig_dir / "pca_subjects.png")
    plt.close()
    
    # t-SNE (Exploratory)
    tsne = TSNE(n_components=2, perplexity=config['tsne'].get('perplexity', 5), random_state=42)
    X_tsne = tsne.fit_transform(X)
    plt.figure()
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1])
    for i, txt in enumerate(subs):
        plt.annotate(txt, (X_tsne[i, 0], X_tsne[i, 1]))
    plt.title(f"t-SNE of Subject Covariances (Status: Exploratory)\nCommit: {get_git_commit()}")
    plt.savefig(fig_dir / "tsne_subjects.png")
    plt.close()
    
    try:
        reducer = umap.UMAP(n_neighbors=config['umap'].get('n_neighbors', 5), random_state=42)
        X_umap = reducer.fit_transform(X)
        plt.figure()
        plt.scatter(X_umap[:, 0], X_umap[:, 1])
        for i, txt in enumerate(subs):
            plt.annotate(txt, (X_umap[i, 0], X_umap[i, 1]))
        plt.title(f"UMAP of Subject Covariances (Status: Exploratory)\nCommit: {get_git_commit()}")
        plt.savefig(fig_dir / "umap_subjects.png")
        plt.close()
    except NameError:
        pass
        
    # 5. Feature Engineering and Correlation Analysis
    merged = pd.merge(loso_df, bp_df, on="subject")
    merged = pd.merge(merged, ss_df, on="subject")
    merged = pd.merge(merged, cf_df, on="subject")
    
    numeric_cols = merged.select_dtypes(include=[np.number])
    feature_cols = [c for c in numeric_cols.columns if c not in ['accuracy', 'rank']]
    
    correlations = []
    for f in feature_cols:
        r, p_val = pearsonr(merged[f], merged['accuracy'])
        correlations.append({
            "feature": f,
            "pearson_r": r,
            "p_value": p_val,
            "sample_size": len(merged)
        })
        
    corr_df = pd.DataFrame(correlations)
    # Multiple comparison correction
    reject, pvals_corrected, _, _ = multipletests(corr_df['p_value'], method='fdr_bh')
    corr_df['p_value_fdr'] = pvals_corrected
    corr_df['significant'] = reject
    
    corr_df = corr_df.sort_values('pearson_r', ascending=False)
    add_csv_metadata(corr_df).to_csv(stat_dir / "performance_correlations.csv", index=False)
    
    # 6. Good vs Bad Subject Analysis (Mann-Whitney U)
    q1_acc = np.percentile(merged['accuracy'], config.get('bottom_percentile', 25))
    q3_acc = np.percentile(merged['accuracy'], 100 - config.get('top_percentile', 25))
    
    good_subs = merged[merged['accuracy'] >= q3_acc]
    bad_subs = merged[merged['accuracy'] <= q1_acc]
    
    tests = []
    for f in feature_cols:
        good_vals = good_subs[f].values
        bad_vals = bad_subs[f].values
        
        if len(good_vals) < 2 or len(bad_vals) < 2:
            continue
            
        stat, p_val = mannwhitneyu(good_vals, bad_vals, alternative='two-sided')
        d = cohens_d(good_vals, bad_vals)
        
        tests.append({
            "feature": f,
            "mann_whitney_u": stat,
            "p_value": p_val,
            "cohens_d": d,
            "good_mean": np.mean(good_vals),
            "bad_mean": np.mean(bad_vals)
        })
        
    tests_df = pd.DataFrame(tests)
    tests_df = tests_df.sort_values('p_value')
    add_csv_metadata(tests_df).to_csv(stat_dir / "good_vs_bad_tests.csv", index=False)
    
    # 7. Generate Strict Report
    report_file = report_dir / "subject_variability_report.md"
    write_metadata_header(report_file, "Subject Variability Analysis")
    
    with open(report_file, 'a') as f:
        f.write("## 1. Dataset Summary\n")
        f.write(f"- Total Subjects Analyzed: {len(merged)}\n")
        f.write(f"- Mean Accuracy: {merged['accuracy'].mean():.4f}\n")
        f.write(f"- Median Accuracy: {merged['accuracy'].median():.4f}\n")
        f.write(f"- Accuracy StdDev: {merged['accuracy'].std():.4f}\n\n")
        
        f.write("## 2. LOSO Ranking\n")
        f.write(loso_df[['subject', 'accuracy']].to_markdown(index=False) + "\n\n")
        
        f.write("## 3. Best Subjects (Top Quartile)\n")
        f.write(good_subs[['subject', 'accuracy']].to_markdown(index=False) + "\n\n")
        
        f.write("## 4. Worst Subjects (Bottom Quartile)\n")
        f.write(bad_subs[['subject', 'accuracy']].to_markdown(index=False) + "\n\n")
        
        f.write("## 5. Statistical Tests (Good vs Bad)\n")
        f.write("Comparing Top 25% vs Bottom 25% subjects using Mann-Whitney U tests.\n\n")
        sig_tests = tests_df[tests_df['p_value'] < 0.05]
        if len(sig_tests) > 0:
            f.write(sig_tests[['feature', 'p_value', 'cohens_d', 'good_mean', 'bad_mean']].to_markdown(index=False) + "\n\n")
        else:
            f.write("*No statistically significant differences (p < 0.05) found between Good and Bad subjects for the computed features.*\n\n")
            
        f.write("## 6. Significant Correlations\n")
        f.write("Pearson correlations between subject features and LOSO accuracy (FDR corrected).\n\n")
        sig_corr = corr_df[corr_df['significant'] == True]
        if len(sig_corr) > 0:
            f.write(sig_corr[['feature', 'pearson_r', 'p_value_fdr', 'sample_size']].to_markdown(index=False) + "\n\n")
        else:
            f.write("*No statistically significant correlations found after FDR correction.*\n\n")
            
        f.write("## 7. Cluster Observations\n")
        f.write(f"- PCA Analysis completed. Explained Variance Ratio (PC1, PC2): {explained_var[0]:.2f}, {explained_var[1]:.2f}.\n")
        f.write("- Frobenius, Riemannian, and Cosine similarity matrices have been written to the statistics directory.\n\n")
        
        f.write("## 8. Actionable Hypotheses\n")
        if len(sig_corr) > 0:
            top_feature = sig_corr.iloc[0]['feature']
            r_val = sig_corr.iloc[0]['pearson_r']
            f.write(f"- Hypothesis derived from data: {top_feature} exhibits a strong correlation (r={r_val:.2f}) with LOSO accuracy. Calibration algorithms targeting {top_feature} distribution matching may improve cross-subject generalization.\n")
        else:
            f.write("- Hypothesis derived from data: Simple bandpower and standard covariance markers do not linearly separate Top from Bottom performers. Non-linear mapping or direct adaptation (e.g., CORAL/DANN) is strictly required to address the domain shift.\n")

if __name__ == "__main__":
    main()
