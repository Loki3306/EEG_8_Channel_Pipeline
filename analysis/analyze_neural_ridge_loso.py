import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats

def main():
    repo_root = Path(__file__).resolve().parents[1]
    
    # 1. Load Ridge Baseline
    ridge_summary_path = repo_root / "analysis" / "summaries" / "ridge_loso_summary.json"
    ridge_acc = {}
    if ridge_summary_path.exists():
        with open(ridge_summary_path, 'r') as f:
            ridge_data = json.load(f)
            best_run = ridge_data.get("best", ridge_data.get("window_runs", [{}])[0].get("best", {}))
            for fold in best_run.get("per_subject", []):
                subj = fold["held_out_subject"].replace("_data_preproc", "")
                ridge_acc[subj] = fold["trial_accuracy"]
                
    # 2. Load Neural Ridge Baseline (if it exists)
    # Using temporal CNN as proxy if neural ridge is missing, or just NA
    nr_acc = {}
    
    # 3. Load Discriminative Pearson (Our new LOSO results)
    loso_summary_path = repo_root / "results" / "neural_ridge_smoke" / "neural_ridge_loso_summary.json"
    dp_acc = {}
    dp_margins = {}
    
    if loso_summary_path.exists():
        with open(loso_summary_path, 'r') as f:
            dp_data = json.load(f)
            for subj, metrics in dp_data.items():
                dp_acc[subj] = metrics["trial_accuracy"]
                dp_margins[subj] = metrics.get("fold_trial_margins", [])
                if subj not in ridge_acc and "base_trial_accuracy" in metrics:
                    ridge_acc[subj] = metrics["base_trial_accuracy"]
    else:
        print(f"File not found: {loso_summary_path}")
        print("Waiting for training to complete...")
        return
        
    subjects = sorted(list(dp_acc.keys()))
    
    # Compile Table
    records = []
    improvements = []
    
    print("\n" + "="*80)
    print(f"{'Subject':<10} | {'Ridge Acc':<12} | {'NR Acc':<10} | {'Disc Pearson':<15} | {'Improv (vs Ridge)'}")
    print("-" * 80)
    
    for subj in subjects:
        r_a = ridge_acc.get(subj, np.nan)
        n_a = nr_acc.get(subj, np.nan)
        d_a = dp_acc.get(subj, np.nan)
        
        imp_r = d_a - r_a if not np.isnan(r_a) and not np.isnan(d_a) else np.nan
        
        if not np.isnan(imp_r):
            improvements.append(imp_r)
            
        print(f"{subj:<10} | {r_a*100:5.1f}%      | {n_a*100 if not np.isnan(n_a) else np.nan:5.1f}%  | {d_a*100:5.1f}%          | {imp_r*100:+5.1f}%")
        
        records.append({
            "Subject": subj,
            "Ridge_Acc": r_a,
            "NR_Acc": n_a,
            "DP_Acc": d_a,
            "Improv_Ridge": imp_r
        })
        
    df = pd.DataFrame(records)
    
    if len(improvements) > 0:
        imp = np.array(improvements)
        print("-" * 80)
        print(f"Mean Improvement   : {np.mean(imp)*100:+.2f}%")
        print(f"Median Improvement : {np.median(imp)*100:+.2f}%")
        print(f"Best Improvement   : {np.max(imp)*100:+.2f}%")
        print(f"Worst Degradation  : {np.min(imp)*100:+.2f}%")
        
        # Statistical Test
        print("\n===========================================")
        print("STATISTICAL SIGNIFICANCE")
        print("===========================================")
        t_stat, p_val = stats.ttest_rel(df['DP_Acc'].dropna(), df['Ridge_Acc'].dropna())
        print(f"Paired t-test vs Ridge: t={t_stat:.4f}, p={p_val:.4e}")
        
        d = np.mean(imp) / np.std(imp, ddof=1)
        print(f"Cohen's d: {d:.4f}")
        
    print("\n===========================================")
    print("RESEARCH CONCLUSIONS")
    print("===========================================")
    print("1. Does the new model consistently outperform Ridge?")
    pos = sum(i > 0 for i in imp)
    print(f"   Model improved {pos}/{len(imp)} subjects.")
    
    print("\n2. Which subjects improve the most?")
    best_subjs = df.nlargest(3, 'Improv_Ridge')['Subject'].tolist()
    print(f"   {best_subjs}")
    
    print("\n3. Which subjects still fail?")
    worst_subjs = df.nsmallest(3, 'DP_Acc')['Subject'].tolist()
    print(f"   {worst_subjs}")
    
    # Visualizations
    out_dir = repo_root / "results" / "neural_ridge_smoke"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Bar Chart
    plt.figure(figsize=(12, 6))
    x = np.arange(len(subjects))
    width = 0.35
    plt.bar(x - width/2, [ridge_acc.get(s, 0)*100 for s in subjects], width, label='Ridge')
    plt.bar(x + width/2, [dp_acc.get(s, 0)*100 for s in subjects], width, label='Discriminative Pearson')
    plt.xticks(x, subjects, rotation=45)
    plt.ylabel('Trial Accuracy (%)')
    plt.title('Subject-wise Trial Accuracy')
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "subject_accuracy_bar.png")
    
    # 2. Histogram
    plt.figure(figsize=(8, 6))
    sns.histplot(imp, bins=10, kde=True)
    plt.xlabel('Improvement over Ridge')
    plt.title('Distribution of Improvements')
    plt.axvline(0, color='r', linestyle='--')
    plt.tight_layout()
    plt.savefig(out_dir / "improvement_histogram.png")
    
    # 3. Scatter Plot
    plt.figure(figsize=(8, 8))
    r = [ridge_acc.get(s, 0)*100 for s in subjects]
    d = [dp_acc.get(s, 0)*100 for s in subjects]
    plt.scatter(r, d)
    plt.plot([min(r+d), max(r+d)], [min(r+d), max(r+d)], 'r--')
    plt.xlabel('Ridge Accuracy (%)')
    plt.ylabel('Discriminative Pearson Accuracy (%)')
    plt.title('Ridge vs Discriminative Pearson Accuracy')
    plt.tight_layout()
    plt.savefig(out_dir / "accuracy_scatter.png")
    
    # 4. Margin Distribution
    if dp_margins:
        plt.figure(figsize=(15, 8))
        plot_data = []
        for s in subjects:
            for m in dp_margins.get(s, []):
                plot_data.append({'Subject': s, 'Margin': m})
        margin_df = pd.DataFrame(plot_data)
        if not margin_df.empty:
            sns.boxplot(x='Subject', y='Margin', data=margin_df)
            plt.axhline(0, color='r', linestyle='--')
            plt.title('Margin Distribution per Subject')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(out_dir / "margin_distribution.png")

if __name__ == "__main__":
    main()
