import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Adjust path if needed
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.confidence.selective_metrics import get_risk_coverage_curve, calculate_aurc

def simulate_predictions(n_samples=1000):
    """
    Simulates predictions and confidence scores for testing the sweep logic.
    In production, this would load the exported predictions from the pipeline.
    """
    np.random.seed(42)
    # Ground truth
    y_true = np.random.randint(0, 2, n_samples)
    
    # Simulate a model that is 70% accurate overall
    # and more confident when correct
    y_pred = []
    y_conf = []
    
    for t in y_true:
        is_correct = np.random.rand() < 0.70
        pred = t if is_correct else (1 - t)
        
        # Conf is higher if correct
        if is_correct:
            conf = np.clip(np.random.normal(0.8, 0.15), 0.5, 1.0)
        else:
            conf = np.clip(np.random.normal(0.6, 0.15), 0.5, 1.0)
            
        y_pred.append(pred)
        y_conf.append(conf)
        
    return y_true, y_pred, y_conf

def run_threshold_sweep():
    print("=" * 80)
    print("PHASE 11 — THRESHOLD SWEEP & RISK-COVERAGE ANALYSIS")
    print("=" * 80)
    
    # Load or simulate data
    print("Loading predictions...")
    y_true, y_pred, y_conf = simulate_predictions(n_samples=2000)
    
    # Calculate curves
    thresholds, coverages, risks, accuracies = get_risk_coverage_curve(y_true, y_pred, y_conf)
    aurc = calculate_aurc(coverages, risks)
    
    print(f"Computed Risk-Coverage curve across {len(thresholds)} unique thresholds.")
    print(f"Area Under Risk-Coverage (AURC): {aurc:.4f}")
    
    # Print key operating points
    print("\nKey Operating Points:")
    print(f"{'Threshold':<12} | {'Coverage':<10} | {'Accuracy':<10} | {'Selective Risk':<15}")
    print("-" * 55)
    
    # For learned confidence, we care about the [0.5, 0.9] range
    for target_cov in [1.0, 0.9, 0.8, 0.5, 0.2]:
        # Find closest coverage
        idx = (np.abs(coverages - target_cov)).argmin()
        t = thresholds[idx]
        c = coverages[idx]
        a = accuracies[idx]
        r = risks[idx]
        print(f"{t:<12.3f} | {c*100:<9.1f}% | {a*100:<9.1f}% | {r:<15.4f}")
        
    # Generate Plots
    os.makedirs("analysis/plots", exist_ok=True)
    
    # 1. Accuracy vs Coverage
    plt.figure(figsize=(8, 6))
    plt.plot(coverages, accuracies, 'b-', linewidth=2)
    plt.title("Selective AAD: Accuracy vs Coverage")
    plt.xlabel("Coverage (Fraction of Accepted Trials)")
    plt.ylabel("Accepted Accuracy")
    plt.grid(True)
    plt.savefig("analysis/plots/accuracy_vs_coverage.png")
    plt.close()
    
    # 2. Risk vs Coverage
    plt.figure(figsize=(8, 6))
    plt.plot(coverages, risks, 'r-', linewidth=2)
    plt.title(f"Risk-Coverage Curve (AURC: {aurc:.4f})")
    plt.xlabel("Coverage")
    plt.ylabel("Selective Risk (Error Rate)")
    plt.grid(True)
    plt.savefig("analysis/plots/risk_coverage.png")
    plt.close()
    
    print("\nSaved plots to analysis/plots/")

if __name__ == "__main__":
    run_threshold_sweep()
