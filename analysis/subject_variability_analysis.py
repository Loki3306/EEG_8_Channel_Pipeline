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
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.covariance import MinCovDet
from statsmodels.stats.multitest import multipletests
import time
import traceback

class Timer:
    def __init__(self, name):
        self.name = name
    def __enter__(self):
        print(f"[{self.name}] Started...")
        self.start = time.time()
        return self
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.time() - self.start
        if exc_type is None:
            print(f"[{self.name}] Completed in {duration:.2f} sec")
        else:
            print(f"[FATAL ERROR] {self.name} failed after {duration:.2f} sec")
            print(f"Exception: {exc_type.__name__}: {exc_val}")
            traceback.print_exception(exc_type, exc_val, exc_tb)
        return False

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False

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
    from datetime import timezone
    timestamp = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')
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
    from datetime import timezone
    df['timestamp'] = datetime.now(timezone.utc).isoformat()
    df['dataset'] = dataset_name
    return df

def frobenius_distance(cov1, cov2):
    return np.linalg.norm(cov1 - cov2, ord='fro')

def riemannian_distance(cov1, cov2):
    from scipy.linalg import eigvals
    try:
        def ensure_spd(c):
            w, v = np.linalg.eigh(c)
            w = np.clip(w, 1e-6, None)
            return v @ np.diag(w) @ v.T
            
        c1 = ensure_spd(cov1)
        c2 = ensure_spd(cov2)
        
        gen_evals = eigvals(c2, c1)
        gen_evals = np.clip(np.real(gen_evals), 1e-8, None)
        return np.sqrt(np.sum(np.log(gen_evals)**2))
    except Exception as e:
        print(f"  [Warning] Riemannian distance failed: {e}")
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
    
    with Timer("Stage 1: Feature Extraction"):
        bandpowers = []
        signal_stats = []
        covariances = {}
        
        for i, p in enumerate(paths):
            sub = p.stem.split('_')[0]
            print(f"  [{i+1}/{len(paths)}] Processing subject {sub}...")
            try:
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
                
                # --- Signal Quality Metrics (Trial-Level Stability) ---
                trial_covs = []
                trial_psds = []
                for ex in examples:
                    ex_cov = np.cov(ex.eeg, rowvar=False)
                    trial_covs.append(ex_cov.flatten())
                    _, ex_psd = welch(ex.eeg, fs=fs, axis=0)
                    trial_psds.append(np.mean(ex_psd, axis=0)) # Mean PSD over freq per channel
                
                # Covariance stability: variance of flattened covariances across trials
                ss_dict["cov_stability"] = np.mean(np.var(trial_covs, axis=0))
                # PSD stability: variance of mean PSDs across trials
                ss_dict["psd_stability"] = np.mean(np.var(trial_psds, axis=0))
                
                # Spectral Power Ratio: Ratio of low-frequency to high-frequency power
                signal_bands = ['delta', 'theta', 'alpha']
                noise_bands = ['beta', 'gamma']
                
                signal_power = np.mean([bp[f"avg_{b}_power"] for b in signal_bands if f"avg_{b}_power" in bp])
                noise_power = np.mean([bp[f"avg_{b}_power"] for b in noise_bands if f"avg_{b}_power" in bp])
                ss_dict["spectral_power_ratio"] = signal_power / (noise_power + 1e-12)
                # ------------------------------------------------------
                
                bandpowers.append(bp)
            except Exception as e:
                print(f"  [FATAL ERROR] Failed processing {sub}: {e}")
                traceback.print_exc()
                continue
                
            # Intermediate Save
            bp_df = pd.DataFrame(bandpowers)
            add_csv_metadata(bp_df).to_csv(stat_dir / "subject_bandpower.csv", index=False)
            
            ss_df = pd.DataFrame(signal_stats)
            add_csv_metadata(ss_df).to_csv(stat_dir / "subject_signal_stats.csv", index=False)
    
    # 3. Covariance Features & Distance Matrix
    with Timer("Stage 2: Covariance Features & Distance Matrix"):
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
                "log_determinant": np.linalg.slogdet(c1)[1],
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
    with Timer("Stage 3: Dimensionality Reduction"):
        X = np.array(cov_flat)
        
        try:
            X_scaled = StandardScaler().fit_transform(X)
            
            # True Population Centroid Distance
            cov_list = list(covariances.values())
            population_centroid = np.mean(cov_list, axis=0)
            
            centroid_riem = [riemannian_distance(c, population_centroid) for c in cov_list]
            centroid_frob = [frobenius_distance(c, population_centroid) for c in cov_list]
            centroid_cos = [cosine_similarity_cov(c, population_centroid) for c in cov_list]
            
            # Mahalanobis Distance
            cov_estimator = MinCovDet(random_state=42).fit(X_scaled)
            mahalanobis_dist = cov_estimator.mahalanobis(X_scaled)
            
            # Isolation Forest
            iso_forest = IsolationForest(random_state=42).fit(X_scaled)
            iso_scores = iso_forest.decision_function(X_scaled)
            
            # Add to covariance features
            for i, cf in enumerate(cov_features):
                cf['mean_riemannian_dist'] = centroid_riem[i]
                cf['mean_frobenius_dist'] = centroid_frob[i]
                cf['mean_cosine_sim'] = centroid_cos[i]
                cf['mahalanobis_dist'] = mahalanobis_dist[i]
                cf['isolation_forest_score'] = iso_scores[i]
                
            # Re-save cf_df with new metrics
            cf_df = pd.DataFrame(cov_features)
            add_csv_metadata(cf_df).to_csv(stat_dir / "subject_covariance_features.csv", index=False)
        except Exception as e:
            print(f"  [FATAL ERROR] Outlier detection failed: {e}")
            X_scaled = X  # Fallback
            
        # PCA
        try:
            pca = PCA(n_components=2, random_state=42)
            X_pca = pca.fit_transform(X_scaled)
            explained_var = pca.explained_variance_ratio_
            
            # Print Top Loadings
            pc1_loadings = pca.components_[0]
            top_idx = np.argsort(np.abs(pc1_loadings))[::-1][:5]
            print(f"  [PCA Diagnostics] Explained Variance: PC1={explained_var[0]:.2f}, PC2={explained_var[1]:.2f}")
            print(f"  [PCA Diagnostics] Top PC1 Covariance Feature Indices: {top_idx}")
            
            plt.figure()
            plt.scatter(X_pca[:, 0], X_pca[:, 1])
            for i, txt in enumerate(subs):
                plt.annotate(txt, (X_pca[i, 0], X_pca[i, 1]))
            plt.title(f"PCA of Subject Covariances (Status: Confirmed)\nExplained Var: {explained_var[0]:.2f}, {explained_var[1]:.2f}\nCommit: {get_git_commit()}")
            plt.savefig(fig_dir / "pca_subjects.png")
            plt.close()
        except Exception as e:
            print(f"  [FATAL ERROR] PCA failed: {e}")
        
        # t-SNE (Exploratory)
        try:
            max_perp = max(1, min(config['tsne'].get('perplexity', 5), len(subs) - 1))
            tsne = TSNE(n_components=2, perplexity=max_perp, random_state=42)
            X_tsne = tsne.fit_transform(X_scaled)
            plt.figure()
            plt.scatter(X_tsne[:, 0], X_tsne[:, 1])
            for i, txt in enumerate(subs):
                plt.annotate(txt, (X_tsne[i, 0], X_tsne[i, 1]))
            plt.title(f"t-SNE of Subject Covariances (Status: Exploratory)\nCommit: {get_git_commit()}")
            plt.savefig(fig_dir / "tsne_subjects.png")
            plt.close()
        except Exception as e:
            print(f"  [FATAL ERROR] t-SNE failed: {e}")
        
        if UMAP_AVAILABLE:
            try:
                reducer = umap.UMAP(n_neighbors=config['umap'].get('n_neighbors', 5), random_state=42)
                X_umap = reducer.fit_transform(X_scaled)
                plt.figure()
                plt.scatter(X_umap[:, 0], X_umap[:, 1])
                for i, txt in enumerate(subs):
                    plt.annotate(txt, (X_umap[i, 0], X_umap[i, 1]))
                plt.title(f"UMAP of Subject Covariances (Status: Exploratory)\nCommit: {get_git_commit()}")
                plt.savefig(fig_dir / "umap_subjects.png")
                plt.close()
            except Exception as e:
                print(f"  [FATAL ERROR] UMAP failed: {e}")
            
    # 5. Feature Engineering and Correlation Analysis
    with Timer("Stage 4: Feature Correlation"):
        # Drop metadata columns before merging to prevent duplicate suffixes
        def _strip_meta(d):
            return d.drop(columns=['commit_hash', 'timestamp', 'dataset'], errors='ignore')
            
        merged = pd.merge(_strip_meta(loso_df), _strip_meta(bp_df), on="subject")
        merged = pd.merge(merged, _strip_meta(ss_df), on="subject")
        merged = pd.merge(merged, _strip_meta(cf_df), on="subject")
        
        numeric_cols = merged.select_dtypes(include=[np.number])
        # Exclude 'accuracy', 'rank', and all channel-specific features ('ch_X_...')
        feature_cols = [c for c in numeric_cols.columns if c not in ['accuracy', 'rank'] and not c.startswith('ch_')]
        
        correlations = []
        for f in feature_cols:
            if np.std(merged[f]) < 1e-12:
                continue
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
            if len(np.unique(good_vals)) <= 1 and len(np.unique(bad_vals)) <= 1:
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
            
            f.write("## 8. Dual-Hypothesis Decision Test\n")
            f.write("### A. Domain Shift Hypothesis\n")
            domain_metrics = ['mean_riemannian_dist', 'mean_frobenius_dist', 'mean_cosine_sim', 'mahalanobis_dist', 'isolation_forest_score']
            domain_corr = corr_df[corr_df['feature'].isin(domain_metrics)].copy()
            # Mark IsolationForest as exploratory
            domain_corr.loc[domain_corr['feature'] == 'isolation_forest_score', 'feature'] = 'isolation_forest_score (Exploratory)'
            if len(domain_corr) > 0:
                f.write(domain_corr[['feature', 'pearson_r', 'p_value_fdr']].to_markdown(index=False) + "\n\n")
            else:
                f.write("*Domain metrics not computed.*\n\n")
            
            f.write("### B. Signal Quality Hypothesis\n")
            signal_metrics = ['cov_stability', 'psd_stability', 'spectral_power_ratio', 'avg_entropy', 'avg_var']
            signal_corr = corr_df[corr_df['feature'].isin(signal_metrics)]
            if len(signal_corr) > 0:
                f.write(signal_corr[['feature', 'pearson_r', 'p_value_fdr']].to_markdown(index=False) + "\n\n")
            else:
                f.write("*Signal quality metrics not computed.*\n\n")
            
            f.write("### Final Assessment & Evidence Summary\n")
            f.write("The auto-decision engine has been disabled to prevent statistically unsafe conclusions. ")
            f.write("Instead, please review the correlations above and determine:\n")
            f.write("1. Are the correlations in Hypothesis A (Domain Shift) strong and statistically significant?\n")
            f.write("2. Are the correlations in Hypothesis B (Signal Quality) strong and statistically significant?\n")
            f.write("\n**Decision Matrix:**\n")
            f.write("- If **A** dominates: Domain Adaptation (CORAL, MMD, DANN) is strongly justified.\n")
            f.write("- If **B** dominates: Calibration, data cleaning, and trial-level SNR filtering are justified.\n")
            f.write("- If neither dominates: Consider latent-space analysis or checking for dataset leakage.\n")

if __name__ == "__main__":
        main()
