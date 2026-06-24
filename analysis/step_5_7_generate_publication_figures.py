import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
import shap
import os

# Set global aesthetics for publication-quality figures
plt.style.use('seaborn-v0_8-paper')
sns.set_theme(style="whitegrid", context="paper", font_scale=1.5)
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']

def add_temporal_features(df):
    df = df.sort_values(['subject_id', 'trial_id', 'window_id']).reset_index(drop=True)
    df['sim_chosen'] = df[['sim_A', 'sim_B']].max(axis=1)
    df['sim_unchosen'] = df[['sim_A', 'sim_B']].min(axis=1)
    df['rolling_std_margin'] = df.groupby(['subject_id', 'trial_id'])['margin'].rolling(window=5, min_periods=1).std().reset_index(level=[0,1], drop=True)
    df['rolling_std_margin'] = df['rolling_std_margin'].fillna(0.0)
    
    def compute_consistency(group):
        preds = group['prediction'].values
        consistencies = []
        for i in range(len(preds)):
            if i == 0:
                consistencies.append(1.0)
            else:
                consistencies.append(np.mean(preds[:i] == preds[i]))
        group['trial_consistency'] = consistencies
        return group

    df = df.groupby(['subject_id', 'trial_id'], group_keys=False).apply(compute_consistency)
    return df

import argparse

def generate_figures():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_file', type=str, default='subject_distance_predictions.csv')
    parser.add_argument('--model_path', type=str, default='models/confidence_model.json')
    args = parser.parse_args()

    os.makedirs('analysis/figures/publication', exist_ok=True)
    
    # 1. Load Data
    csv_file = args.csv_file
    if not os.path.exists(csv_file):
        print(f"Error: {csv_file} not found.")
        return
        
    df = pd.read_csv(csv_file)
    df = df[np.isfinite(df['margin'])].reset_index(drop=True)
    df = add_temporal_features(df)
    
    # Load Model
    model_path = args.model_path
    model = xgb.XGBClassifier()
    model.load_model(model_path)
    
    features = ['margin', 'sim_chosen', 'sim_unchosen', 'rolling_std_margin', 'trial_consistency']
    X = df[features]
    df['confidence'] = model.predict_proba(X)[:, 1]
    
    print("Generating Figure 1: Calibration (Margin vs Accuracy)...")
    # Bin margin and calculate accuracy
    df['margin_bin'] = pd.cut(df['margin'], bins=np.arange(0, 0.35, 0.05))
    margin_acc = df.groupby('margin_bin', observed=False)['correct'].mean().reset_index()
    margin_acc['bin_center'] = [i.mid for i in margin_acc['margin_bin']]
    
    plt.figure(figsize=(8, 6))
    plt.plot(margin_acc['bin_center'], margin_acc['correct'], marker='o', linewidth=2, markersize=8, color='#1f77b4')
    plt.xlabel('Similarity Margin', fontweight='bold')
    plt.ylabel('Empirical Accuracy', fontweight='bold')
    plt.title('Accuracy vs. Similarity Margin', fontweight='bold')
    plt.ylim(0.4, 1.05)
    plt.axhline(0.5, color='gray', linestyle='--', label='Chance (50%)')
    plt.legend()
    plt.tight_layout()
    plt.savefig('analysis/figures/publication/fig1_margin_accuracy.png', dpi=300)
    plt.close()
    
    print("Generating Figure 2: Reliability Diagram...")
    from sklearn.calibration import calibration_curve
    prob_true, prob_pred = calibration_curve(df['correct'], df['confidence'], n_bins=10)
    
    plt.figure(figsize=(8, 6))
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfect Calibration')
    plt.plot(prob_pred, prob_true, marker='o', linewidth=2, markersize=8, color='#d62728', label='XGBoost Confidence')
    plt.xlabel('Predicted Confidence Probability', fontweight='bold')
    plt.ylabel('Empirical Accuracy (Fraction of Positives)', fontweight='bold')
    plt.title('Confidence Reliability Diagram', fontweight='bold')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig('analysis/figures/publication/fig2_reliability_diagram.png', dpi=300)
    plt.close()
    
    print("Generating Figure 3: Coverage vs Selective Accuracy...")
    df_sorted = df.sort_values('confidence', ascending=False)
    y_true_sorted = df_sorted['correct'].values
    coverages = np.arange(1, len(df_sorted) + 1) / len(df_sorted)
    cumulative_correct = np.cumsum(y_true_sorted)
    accuracies = cumulative_correct / np.arange(1, len(df_sorted) + 1)
    
    plt.figure(figsize=(8, 6))
    plt.plot(coverages * 100, accuracies * 100, linewidth=3, color='#2ca02c')
    plt.gca().invert_xaxis()  # 100% down to 0%
    plt.xlabel('Coverage (%)', fontweight='bold')
    plt.ylabel('Selective Accuracy (%)', fontweight='bold')
    plt.title('Accuracy-Coverage Tradeoff', fontweight='bold')
    
    # Highlight specific operating points
    plt.axvline(70, color='gray', linestyle='--', alpha=0.7)
    plt.axvline(50, color='gray', linestyle='--', alpha=0.7)
    
    idx_70 = np.abs(coverages - 0.70).argmin()
    idx_50 = np.abs(coverages - 0.50).argmin()
    
    plt.plot(70, accuracies[idx_70]*100, marker='*', markersize=15, color='#ff7f0e')
    plt.plot(50, accuracies[idx_50]*100, marker='*', markersize=15, color='#ff7f0e')
    
    plt.text(72, accuracies[idx_70]*100 + 1, f'{accuracies[idx_70]*100:.1f}% @ 70%', fontweight='bold')
    plt.text(52, accuracies[idx_50]*100 + 1, f'{accuracies[idx_50]*100:.1f}% @ 50%', fontweight='bold')
    
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.tight_layout()
    plt.savefig('analysis/figures/publication/fig3_coverage_accuracy.png', dpi=300)
    plt.close()
    
    print("Generating Figure 4: Risk-Coverage Curve...")
    risks = 1.0 - accuracies
    
    plt.figure(figsize=(8, 6))
    plt.plot(coverages * 100, risks * 100, linewidth=3, color='#9467bd')
    plt.fill_between(coverages * 100, risks * 100, 0, alpha=0.2, color='#9467bd')
    plt.gca().invert_xaxis()
    plt.xlabel('Coverage (%)', fontweight='bold')
    plt.ylabel('Risk (Error Rate %)', fontweight='bold')
    plt.title('Risk-Coverage Curve', fontweight='bold')
    plt.tight_layout()
    plt.savefig('analysis/figures/publication/fig4_risk_coverage.png', dpi=300)
    plt.close()
    
    print("Generating Figure 5: SHAP Summary Plot...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X, show=False, plot_type="bar")
    plt.title("Feature Importance (Mean |SHAP|)", fontweight='bold')
    plt.tight_layout()
    plt.savefig('analysis/figures/publication/fig5_shap_bar.png', dpi=300)
    plt.close()
    
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X, show=False)
    plt.title("SHAP Value Distribution", fontweight='bold')
    plt.tight_layout()
    plt.savefig('analysis/figures/publication/fig5_shap_beeswarm.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Generating Figure 6: Feature Distributions (Correct vs Incorrect)...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    sns.kdeplot(data=df, x='margin', hue='correct', fill=True, ax=axes[0], 
                palette={0: '#d62728', 1: '#1f77b4'}, alpha=0.4, linewidth=2)
    axes[0].set_title('Margin Distribution', fontweight='bold')
    axes[0].set_xlabel('Margin')
    
    sns.kdeplot(data=df, x='rolling_std_margin', hue='correct', fill=True, ax=axes[1],
                palette={0: '#d62728', 1: '#1f77b4'}, alpha=0.4, linewidth=2)
    axes[1].set_title('Rolling Std Distribution', fontweight='bold')
    axes[1].set_xlabel('Rolling Std (Margin)')
    
    # Custom legend
    handles = [plt.Line2D([0], [0], color='#1f77b4', lw=4, label='Correct (1)'),
               plt.Line2D([0], [0], color='#d62728', lw=4, label='Incorrect (0)')]
    for ax in axes:
        ax.legend(handles=handles)
        
    plt.tight_layout()
    plt.savefig('analysis/figures/publication/fig6_distributions.png', dpi=300)
    plt.close()
    
    print("All figures generated successfully in analysis/figures/publication/")

if __name__ == "__main__":
    generate_figures()
