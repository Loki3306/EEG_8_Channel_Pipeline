import json
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "results" / "conformer_loso"
SUMMARY_FILE = RESULTS_DIR / "conformer_loso_multiseed_summary.json"

def calculate_ece(confidences, accuracies, num_bins=10):
    """
    Computes Expected Calibration Error.
    confidences: array of uncalibrated confidence proxies scaled to [0, 1]
    accuracies: array of 0s and 1s
    """
    bins = np.linspace(0, 1, num_bins + 1)
    ece = 0.0
    total_samples = len(confidences)
    
    bin_accs = []
    bin_confs = []
    
    for i in range(num_bins):
        bin_lower = bins[i]
        bin_upper = bins[i+1]
        
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        if i == 0:
            in_bin = (confidences >= bin_lower) & (confidences <= bin_upper)
            
        bin_count = np.sum(in_bin)
        if bin_count > 0:
            bin_acc = np.mean(accuracies[in_bin])
            bin_conf = np.mean(confidences[in_bin])
            bin_accs.append(bin_acc)
            bin_confs.append(bin_conf)
            ece += (bin_count / total_samples) * np.abs(bin_acc - bin_conf)
        else:
            bin_accs.append(np.nan)
            bin_confs.append(np.nan)
            
    return ece, bin_accs, bin_confs, bins

def main():
    if not SUMMARY_FILE.exists():
        print(f"Error: {SUMMARY_FILE} not found.")
        return
        
    with open(SUMMARY_FILE, "r") as f:
        data = json.load(f)
        
    all_margins = []
    
    for seed, seed_data in data.items():
        for subject, metrics in seed_data.items():
            all_margins.extend(metrics["fold_trial_margins"])
            
    all_margins = np.array(all_margins)
    
    # Correctness is 1 if margin > 0 (Attended > Unattended)
    accuracies = (all_margins > 0).astype(int)
    
    # Confidence proxy is the absolute margin
    confidences_raw = np.abs(all_margins)
    
    # Scale to [0, 1] using min-max scaling for ECE
    if np.max(confidences_raw) > 0:
        confidences_scaled = (confidences_raw - np.min(confidences_raw)) / (np.max(confidences_raw) - np.min(confidences_raw))
    else:
        confidences_scaled = confidences_raw
        
    ece, bin_accs, bin_confs, bins = calculate_ece(confidences_scaled, accuracies, num_bins=10)
    
    print("=== CONFIDENCE CALIBRATION ===")
    print(f"Expected Calibration Error (ECE): {ece:.4f}")
    
    # Reliability Diagram
    plt.figure(figsize=(8, 8))
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfect Calibration')
    
    valid_bins = ~np.isnan(bin_accs)
    centers = (bins[:-1] + bins[1:]) / 2
    
    plt.bar(centers[valid_bins], bin_accs[valid_bins], width=0.1, alpha=0.7, edgecolor='k', label='Model')
    plt.plot(centers[valid_bins], bin_confs[valid_bins], marker='o', color='red', label='Mean Confidence')
    
    plt.xlabel("Scaled Confidence Proxy")
    plt.ylabel("Empirical Accuracy")
    plt.title(f"Reliability Diagram (ECE = {ece:.4f})")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    
    out_dir = RESULTS_DIR / "figures"
    out_dir.mkdir(exist_ok=True, parents=True)
    plt.savefig(out_dir / "reliability_diagram.png", dpi=300)
    plt.close()
    
    # Margin Histogram
    plt.figure(figsize=(8, 6))
    plt.hist(all_margins[accuracies == 1], bins=30, alpha=0.6, color='green', label='Correct')
    plt.hist(all_margins[accuracies == 0], bins=30, alpha=0.6, color='red', label='Incorrect')
    plt.axvline(x=0, color='k', linestyle='--')
    plt.xlabel("Pearson Correlation Margin (Att - Unatt)")
    plt.ylabel("Number of Trials")
    plt.title("Distribution of Decision Margins")
    plt.legend()
    
    plt.savefig(out_dir / "margin_histogram.png", dpi=300)
    plt.close()
    
    print(f"Saved calibration figures to {out_dir}")

if __name__ == "__main__":
    main()
