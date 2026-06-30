import json
import numpy as np
from pathlib import Path
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "results" / "conformer_loso"
SUMMARY_FILE = RESULTS_DIR / "conformer_loso_multiseed_summary.json"
RIDGE_FILE = REPO_ROOT / "analysis" / "summaries" / "ridge_loso_summary.json"

def cohens_d(x, y):
    diff = x - y
    return np.mean(diff) / (np.std(diff, ddof=1) + 1e-8)

def main():
    if not SUMMARY_FILE.exists():
        print(f"Error: {SUMMARY_FILE} not found.")
        return
        
    with open(SUMMARY_FILE, "r") as f:
        conf_data = json.load(f)
        
    seeds = sorted(list(conf_data.keys()))
    subjects = sorted(list(conf_data[seeds[0]].keys()))
    
    # Compute mean conformer accuracy per subject across seeds
    conf_subject_accs = {s: [] for s in subjects}
    
    for seed in seeds:
        for s in subjects:
            conf_subject_accs[s].append(conf_data[seed][s]["trial_accuracy"])
            
    conf_means = {s: np.mean(accs) for s, accs in conf_subject_accs.items()}
    
    # Load Ridge baseline
    ridge_accs = {}
    if RIDGE_FILE.exists():
        with open(RIDGE_FILE, "r") as f:
            ridge_data = json.load(f)
            
        if "best" in ridge_data and "per_subject" in ridge_data["best"]:
            for item in ridge_data["best"]["per_subject"]:
                raw_subj = item["held_out_subject"]
                # Convert 'S1_data_preproc' to 'S1'
                clean_subj = raw_subj.replace("_data_preproc", "")
                ridge_accs[clean_subj] = item["trial_accuracy"]
    else:
        print("Ridge baseline not found. Cannot perform statistical comparison.")
        return
        
    # Align data
    x_conf = []
    y_ridge = []
    
    for s in subjects:
        if s in ridge_accs:
            x_conf.append(conf_means[s])
            y_ridge.append(ridge_accs[s])
            
    if not x_conf:
        print("No overlapping subjects found between Conformer and Ridge.")
        return
        
    x_conf = np.array(x_conf)
    y_ridge = np.array(y_ridge)
    
    diff = x_conf - y_ridge
    
    # Statistical tests
    t_stat, p_ttest = stats.ttest_rel(x_conf, y_ridge)
    w_stat, p_wilcoxon = stats.wilcoxon(x_conf, y_ridge)
    d = cohens_d(x_conf, y_ridge)
    
    # 95% Confidence Interval for the difference
    ci_lower, ci_upper = stats.t.interval(0.95, len(diff)-1, loc=np.mean(diff), scale=stats.sem(diff))
    
    report = [
        "=== STATISTICAL VALIDATION REPORT ===",
        f"N Subjects: {len(x_conf)}",
        f"Conformer Mean: {np.mean(x_conf)*100:.2f}% ± {np.std(x_conf)*100:.2f}%",
        f"Ridge Mean: {np.mean(y_ridge)*100:.2f}% ± {np.std(y_ridge)*100:.2f}%",
        f"Absolute Improvement: {np.mean(diff)*100:.2f}%",
        "",
        "--- Hypothesis Testing ---",
        f"Paired t-test: t = {t_stat:.4f}, p = {p_ttest:.4e}",
        f"Wilcoxon signed-rank: W = {w_stat:.1f}, p = {p_wilcoxon:.4e}",
        "",
        "--- Effect Size & Confidence Intervals ---",
        f"Cohen's d: {d:.4f}",
        f"95% CI of Difference: [{ci_lower*100:.2f}%, {ci_upper*100:.2f}%]"
    ]
    
    report_text = "\n".join(report)
    print(report_text)
    
    out_dir = RESULTS_DIR / "statistics"
    out_dir.mkdir(exist_ok=True, parents=True)
    
    with open(out_dir / "statistical_validation.txt", "w") as f:
        f.write(report_text)
        
    print(f"\nSaved statistical report to {out_dir}")

if __name__ == "__main__":
    main()
