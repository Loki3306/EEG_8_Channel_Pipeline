import json
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "results" / "conformer_loso"
SUMMARY_FILE = RESULTS_DIR / "conformer_loso_multiseed_summary.json"

def main():
    if not SUMMARY_FILE.exists():
        print(f"Error: {SUMMARY_FILE} not found. Please run the multi-seed training script first.")
        return
        
    with open(SUMMARY_FILE, "r") as f:
        data = json.load(f)
        
    seeds = sorted(list(data.keys()))
    subjects = sorted(list(data[seeds[0]].keys()))
    
    seed_accuracies = []
    seed_margins = []
    
    print("=== SEED LEVEL STATISTICS ===")
    
    records = []
    
    for seed in seeds:
        seed_data = data[seed]
        accs = [seed_data[s]["trial_accuracy"] for s in subjects]
        margins = [seed_data[s]["median_margin"] for s in subjects]
        
        mean_acc = np.mean(accs)
        mean_margin = np.mean(margins)
        
        seed_accuracies.append(mean_acc)
        seed_margins.append(mean_margin)
        
        print(f"Seed {seed}: Accuracy = {mean_acc*100:.1f}%, Median Margin = {mean_margin:.4f}")
        records.append({"Seed": seed, "Mean Accuracy": mean_acc, "Mean Median Margin": mean_margin})
        
    print("\n=== OVERALL REPRODUCIBILITY ===")
    acc_mean = np.mean(seed_accuracies)
    acc_std = np.std(seed_accuracies)
    acc_cv = (acc_std / acc_mean) * 100 if acc_mean > 0 else 0
    
    margin_mean = np.mean(seed_margins)
    margin_std = np.std(seed_margins)
    margin_cv = (margin_std / margin_mean) * 100 if margin_mean > 0 else 0
    
    print(f"Global Accuracy: {acc_mean*100:.2f}% ± {acc_std*100:.2f}% (CV: {acc_cv:.1f}%)")
    print(f"Global Margin: {margin_mean:.4f} ± {margin_std:.4f} (CV: {margin_cv:.1f}%)")
    
    # Generate Boxplot
    plt.figure(figsize=(8, 6))
    plt.boxplot([np.array(seed_accuracies)*100], tick_labels=["Conformer"])
    plt.ylabel("LOSO Accuracy (%)")
    plt.title("Reproducibility Across 5 Random Seeds")
    plt.grid(True, linestyle="--", alpha=0.7)
    
    out_dir = RESULTS_DIR / "figures"
    out_dir.mkdir(exist_ok=True, parents=True)
    
    plt.savefig(out_dir / "reproducibility_boxplot.png", dpi=300)
    plt.close()
    
    # Save table
    df = pd.DataFrame(records)
    df.to_csv(RESULTS_DIR / "reproducibility_table.csv", index=False)
    print(f"Saved plots and tables to {RESULTS_DIR}")

if __name__ == "__main__":
    main()
