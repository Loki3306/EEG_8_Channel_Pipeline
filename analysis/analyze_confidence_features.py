import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

def main():
    print("--- Phase 7: Analyzing Confidence Features ---")
    
    in_csv = REPO_ROOT / "results" / "run7_learned_confidence" / "confidence_features.csv"
    if not in_csv.exists():
        print(f"[Error] Dataset not found: {in_csv}")
        return
        
    out_dir = REPO_ROOT / "results" / "run7_feature_analysis"
    os.makedirs(out_dir, exist_ok=True)
    
    df = pd.read_csv(in_csv)
    print(f"Loaded {len(df)} samples.")
    
    correct_mask = df['correct'] == 1
    
    # 1. Feature Importance (Correlation with correctness)
    # We correlate latent norms and margins with the target
    print("\n[Computing Correlations]")
    features_to_check = ['embedding_norm', 'margin', 'corr_a', 'corr_b']
    latent_cols = [c for c in df.columns if c.startswith('z_')]
    
    corrs = {}
    for f in features_to_check + latent_cols:
        corrs[f] = df[f].corr(df['correct'])
        
    corr_df = pd.DataFrame(list(corrs.items()), columns=['Feature', 'Correlation']).sort_values('Correlation', key=abs, ascending=False)
    print("Top 10 features correlated with correctness:")
    print(corr_df.head(10).to_string(index=False))
    
    corr_df.to_csv(out_dir / "feature_correlations.csv", index=False)
    
    # 2. Histograms: Correct vs Incorrect for key metrics
    print("\n[Generating Plots]")
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.histplot(data=df, x='margin', hue='correct', bins=50, kde=True, ax=axes[0])
    axes[0].set_title('Margin Distribution (Correct vs Incorrect)')
    
    sns.histplot(data=df, x='embedding_norm', hue='correct', bins=50, kde=True, ax=axes[1])
    axes[1].set_title('Latent Embedding Norm (Correct vs Incorrect)')
    
    plt.tight_layout()
    plt.savefig(out_dir / "fig_distributions.png", dpi=300)
    plt.close()
    
    # 3. Boxplots of top 4 latent features
    top_latents = [f for f in corr_df['Feature'].head(10) if f.startswith('z_')][:4]
    if top_latents:
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        for i, col in enumerate(top_latents):
            sns.boxplot(data=df, x='correct', y=col, ax=axes[i])
            axes[i].set_title(f'{col} (r={corrs[col]:.3f})')
        plt.tight_layout()
        plt.savefig(out_dir / "fig_latent_boxplots.png", dpi=300)
        plt.close()
        
    # 4. Correlation Matrix of top features
    top_features = features_to_check + top_latents
    corr_mat = df[top_features + ['correct']].corr()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_mat, annot=True, cmap='coolwarm', fmt=".2f", vmin=-1, vmax=1)
    plt.title('Correlation Matrix')
    plt.tight_layout()
    plt.savefig(out_dir / "fig_correlation_matrix.png", dpi=300)
    plt.close()
    
    print(f"\nAnalysis complete. Results saved to {out_dir}")

if __name__ == "__main__":
    main()
