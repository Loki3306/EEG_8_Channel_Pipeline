import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_curve, auc, precision_recall_curve, brier_score_loss, roc_auc_score
import json

REPO_ROOT = Path(__file__).resolve().parent.parent

def expected_calibration_error(y_true, y_prob, n_bins=10):
    bins = np.linspace(0., 1., n_bins + 1)
    binids = np.digitize(y_prob, bins) - 1
    ece = 0.0
    for i in range(n_bins):
        bin_idx = (binids == i)
        if np.sum(bin_idx) > 0:
            bin_acc = np.mean(y_true[bin_idx])
            bin_conf = np.mean(y_prob[bin_idx])
            ece += np.abs(bin_acc - bin_conf) * np.sum(bin_idx)
    return ece / len(y_true)

def plot_margin_histogram(df, out_dir):
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x='margin', hue='correct', bins=50, kde=True, palette='Set1', common_norm=False)
    plt.title('Absolute Margin Distribution: Correct vs Incorrect')
    plt.xlabel('Absolute Pearson Margin |corrA - corrB|')
    plt.ylabel('Density')
    plt.savefig(out_dir / 'margin_histogram.png')
    plt.close()

def plot_reliability_diagrams(df, models, out_dir):
    plt.figure(figsize=(12, 10))
    for i, model in enumerate(models):
        prob_col = f'prob_{model}'
        prob_true, prob_pred = calibration_curve(df['correct'], df[prob_col], n_bins=10)
        
        plt.subplot(2, 2, i + 1)
        plt.plot(prob_pred, prob_true, marker='o', label=model, color='blue')
        plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfect Calibration')
        plt.title(f'Reliability Diagram: {model.capitalize()}')
        plt.xlabel('Mean Predicted Probability')
        plt.ylabel('Fraction of Positives')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
    plt.tight_layout()
    plt.savefig(out_dir / 'reliability_diagrams.png')
    plt.close()

def plot_risk_coverage(df, models, out_dir):
    plt.figure(figsize=(10, 6))
    for model in models:
        prob_col = f'prob_{model}'
        
        # Sort by confidence
        sorted_df = df.sort_values(by=prob_col, ascending=False)
        corrects = sorted_df['correct'].values
        
        coverages = []
        risks = []
        
        for p in np.linspace(0.1, 1.0, 50):
            cutoff = int(p * len(corrects))
            if cutoff > 0:
                top_k = corrects[:cutoff]
                risk = 1.0 - np.mean(top_k)  # Error rate on accepted samples
                coverages.append(p)
                risks.append(risk)
                
        plt.plot(coverages, risks, label=model, linewidth=2)
        
    plt.title('Selective Risk: Error Rate vs Coverage')
    plt.xlabel('Coverage (Fraction of data accepted)')
    plt.ylabel('Selective Risk (Error Rate)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(out_dir / 'risk_coverage.png')
    plt.close()

def plot_roc_pr(df, models, out_dir):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    for model in models:
        prob_col = f'prob_{model}'
        
        # ROC
        fpr, tpr, _ = roc_curve(df['correct'], df[prob_col])
        roc_auc = auc(fpr, tpr)
        ax1.plot(fpr, tpr, label=f'{model} (AUC = {roc_auc:.3f})')
        
        # PR
        precision, recall, _ = precision_recall_curve(df['correct'], df[prob_col])
        pr_auc = auc(recall, precision)
        ax2.plot(recall, precision, label=f'{model} (AUC = {pr_auc:.3f})')
        
    ax1.plot([0, 1], [0, 1], 'k--')
    ax1.set_title('ROC Curve (Confidence -> Correctness)')
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.legend()
    
    ax2.set_title('Precision-Recall Curve')
    ax2.set_xlabel('Recall')
    ax2.set_ylabel('Precision')
    ax2.legend()
    
    plt.savefig(out_dir / 'roc_pr_curves.png')
    plt.close()

def plot_confidence_distributions(df, models, out_dir):
    plt.figure(figsize=(12, 10))
    for i, model in enumerate(models):
        prob_col = f'prob_{model}'
        
        plt.subplot(2, 2, i + 1)
        sns.histplot(df[prob_col], bins=50, kde=True, color='purple')
        plt.title(f'Confidence Distribution: {model.capitalize()}')
        plt.xlabel('Predicted Probability')
        plt.ylabel('Count')
        plt.xlim(0, 1)
        
    plt.tight_layout()
    plt.savefig(out_dir / 'confidence_distributions.png')
    plt.close()

def generate_report(df, out_dir):
    models = ['raw', 'platt', 'temp', 'iso']
    
    print("Generating visualizations...")
    plot_margin_histogram(df, out_dir)
    plot_reliability_diagrams(df, models, out_dir)
    plot_risk_coverage(df, models, out_dir)
    plot_roc_pr(df, models, out_dir)
    plot_confidence_distributions(df, models, out_dir)
    
    print("Calculating final metrics...")
    metrics = []
    
    for model in models:
        prob_col = f'prob_{model}'
        probs = df[prob_col].values
        labels = df['correct'].values
        
        ece = expected_calibration_error(labels, probs)
        brier = brier_score_loss(labels, probs)
        
        # Log Loss / NLL (clip to avoid inf)
        eps = 1e-15
        p_clipped = np.clip(probs, eps, 1 - eps)
        nll = -np.mean(labels * np.log(p_clipped) + (1 - labels) * np.log(1 - p_clipped))
        
        try:
            auroc = roc_auc_score(labels, probs)
        except:
            auroc = 0.5
            
        precision, recall, _ = precision_recall_curve(labels, probs)
        auprc = auc(recall, precision)
        
        # MCE (Maximum Calibration Error)
        bins = np.linspace(0., 1., 11)
        binids = np.digitize(probs, bins) - 1
        mce = 0.0
        for i in range(10):
            bin_idx = (binids == i)
            if np.sum(bin_idx) > 0:
                bin_acc = np.mean(labels[bin_idx])
                bin_conf = np.mean(probs[bin_idx])
                mce = max(mce, np.abs(bin_acc - bin_conf))
                
        metrics.append({
            'Model': model.capitalize(),
            'ECE': ece,
            'MCE': mce,
            'Brier': brier,
            'NLL': nll,
            'AUROC': auroc,
            'AUPRC': auprc
        })
        
    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(out_dir / 'calibration_metrics.csv', index=False)
    
    # Generate Markdown Report
    report_path = out_dir / 'phase_13_calibration_report.md'
    with open(report_path, 'w') as f:
        f.write("# Phase 13 Margin Calibration Report\n\n")
        
        f.write("## 1. Core Metrics\n\n")
        f.write(metrics_df.to_markdown(index=False))
        f.write("\n\n")
        
        f.write("## 2. Analysis\n")
        f.write("If Platt or Isotonic scaling successfully restored the dynamic range, we should see significant drops in ECE and Brier score compared to the Raw method.\n")
        f.write("AUROC and AUPRC should remain largely identical, proving that calibration strictly refines probabilities without altering the model's discriminative rank order.\n")
        
    print(f"Report saved to {report_path}")

def main():
    print("--- Phase 13 Calibration Report Generation ---")
    data_dir = REPO_ROOT / "results" / "phase13_margin_calibration"
    csv_path = data_dir / "calibration_predictions.csv"
    
    if not csv_path.exists():
        print(f"Error: Could not find {csv_path}. Please run phase_13_margin_calibration.py first.")
        return
        
    df = pd.read_csv(csv_path)
    # We filter to just the 'per_subject_absolute_margin' ablation for the main report
    if 'ablation' in df.columns:
        df = df[df['ablation'] == 'per_subject_absolute_margin']
        
    generate_report(df, data_dir)
    print("Done!")

if __name__ == "__main__":
    main()
