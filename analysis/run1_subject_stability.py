import json
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "results" / "conformer_loso"
SUMMARY_FILE = RESULTS_DIR / "conformer_loso_multiseed_summary.json"

def main():
    if not SUMMARY_FILE.exists():
        print(f"Error: {SUMMARY_FILE} not found.")
        return
        
    with open(SUMMARY_FILE, "r") as f:
        data = json.load(f)
        
    seeds = sorted(list(data.keys()))
    subjects = sorted(list(data[seeds[0]].keys()))
    
    # Rows = Subjects, Cols = Seeds
    acc_matrix = np.zeros((len(subjects), len(seeds)))
    
    for j, seed in enumerate(seeds):
        for i, s in enumerate(subjects):
            acc_matrix[i, j] = data[seed][s]["trial_accuracy"] * 100
            
    # Compute Subject Stats
    subject_means = np.mean(acc_matrix, axis=1)
    subject_stds = np.std(acc_matrix, axis=1)
    
    # Sort subjects by mean accuracy
    sort_idx = np.argsort(subject_means)[::-1]
    sorted_subjects = [subjects[i] for i in sort_idx]
    sorted_matrix = acc_matrix[sort_idx, :]
    
    # Create Heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(sorted_matrix, annot=True, fmt=".1f", cmap="YlGnBu",
                xticklabels=[f"Seed {s}" for s in seeds],
                yticklabels=sorted_subjects, cbar_kws={'label': 'Accuracy (%)'})
    plt.title("Subject Accuracy Stability Across Seeds")
    plt.ylabel("Subject (Sorted by Mean Performance)")
    plt.xlabel("Random Seed")
    
    out_dir = RESULTS_DIR / "figures"
    out_dir.mkdir(exist_ok=True, parents=True)
    plt.savefig(out_dir / "subject_stability_heatmap.png", dpi=300, bbox_inches="tight")
    plt.close()
    
    # Create Variance Bar Chart
    plt.figure(figsize=(10, 6))
    plt.bar(sorted_subjects, subject_stds[sort_idx], color='coral', edgecolor='black')
    plt.axhline(np.mean(subject_stds), color='r', linestyle='--', label=f'Mean Std ({np.mean(subject_stds):.2f}%)')
    plt.ylabel("Standard Deviation Across Seeds (%)")
    plt.title("Inter-Seed Variance by Subject")
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    plt.savefig(out_dir / "subject_variance_bar.png", dpi=300, bbox_inches="tight")
    plt.close()
    
    # Save CSV
    df = pd.DataFrame({
        "Subject": sorted_subjects,
        "Mean Accuracy (%)": subject_means[sort_idx],
        "Std Dev (%)": subject_stds[sort_idx]
    })
    
    for j, seed in enumerate(seeds):
        df[f"Seed {seed} (%)"] = sorted_matrix[:, j]
        
    df.to_csv(RESULTS_DIR / "subject_stability.csv", index=False)
    print(f"Saved subject stability plots and tables to {RESULTS_DIR}")

if __name__ == "__main__":
    main()
