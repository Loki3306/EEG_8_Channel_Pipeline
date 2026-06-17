"""
Phase 3: Channel Importance Analysis
Ranks all channels empirically based on variance, spectral entropy, and consistency.
"""

import sys
import numpy as np
import pandas as pd
from scipy.stats import entropy
from scipy.signal import welch
from pathlib import Path

sys.path.append(str(Path(__file__).parent))
from raw_eeg_utils import load_raw_eeg

def compute_spectral_entropy(tr, fs):
    # tr: (channels, time)
    nperseg = min(int(fs * 4), tr.shape[1])
    f, Pxx = welch(tr, fs=fs, nperseg=nperseg)
    Pxx = Pxx / np.sum(Pxx, axis=1, keepdims=True)
    se = entropy(Pxx, axis=1)
    return se

def analyze_subject_channels(mat_path):
    print(f"Processing {mat_path.name}...")
    try:
        eeg_data = load_raw_eeg(mat_path)
    except Exception as e:
        print(f"Failed to load {mat_path.name}: {e}")
        return None
        
    fs = eeg_data['fsample']
    labels = eeg_data['labels']
    trials = eeg_data['trials']
    
    if not trials: return None
    
    n_channels = len(labels) if labels else trials[0].shape[0]
    
    # Metrics per channel
    variances = []
    spectral_entropies = []
    
    for tr in trials:
        variances.append(np.var(tr, axis=1))
        spectral_entropies.append(compute_spectral_entropy(tr, fs))
        
    mean_var = np.mean(variances, axis=0)
    mean_se = np.mean(spectral_entropies, axis=0)
    
    res = []
    for ch in range(n_channels):
        label = labels[ch] if labels else f"Ch_{ch}"
        res.append({
            "subject": mat_path.stem,
            "channel_idx": ch,
            "channel_name": label,
            "variance": mean_var[ch],
            "spectral_entropy": mean_se[ch]
        })
    return pd.DataFrame(res)

def main(raw_dir, out_dir):
    raw_dir = Path(raw_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    mats = list(raw_dir.glob("*.mat"))
    if not mats:
        print(f"No .mat files found in {raw_dir}")
        return
        
    dfs = []
    for m in mats:
        df = analyze_subject_channels(m)
        if df is not None: dfs.append(df)
            
    if not dfs:
        print("No valid results computed.")
        return
        
    full_df = pd.concat(dfs)
    
    # Average across subjects
    agg = full_df.groupby(["channel_idx", "channel_name"]).mean().reset_index()
    agg = agg.drop(columns=["subject"], errors='ignore')
    
    # Rank channels based on a composite score
    # High variance and high spectral entropy generally indicate good signal richness, but eye blinks can cause high variance.
    # For now, we normalize both metrics to 0-1 and average them.
    norm_var = (agg["variance"] - agg["variance"].min()) / (agg["variance"].max() - agg["variance"].min())
    norm_se = (agg["spectral_entropy"] - agg["spectral_entropy"].min()) / (agg["spectral_entropy"].max() - agg["spectral_entropy"].min())
    
    agg["score"] = norm_var + norm_se
    agg = agg.sort_values("score", ascending=False)
    
    csv_path = out_dir / "channel_ranking.csv"
    agg.to_csv(csv_path, index=False)
    
    # Markdown output
    md_path = out_dir / "channel_ranking_summary.md"
    with open(md_path, 'w') as f:
        f.write("# Channel Importance Ranking\n\n")
        f.write("Top 8 Channels:\n")
        for i, row in agg.head(8).iterrows():
            f.write(f"- {row['channel_name']} (Score: {row['score']:.2f})\n")
            
        f.write("\nTop 16 Channels:\n")
        for i, row in agg.head(16).iterrows():
            f.write(f"- {row['channel_name']}\n")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir", type=str, default="/kaggle/input/datasets/lokeshgile/raw-eeg")
    parser.add_argument("--out_dir", type=str, default="/kaggle/working/reports")
    args = parser.parse_args()
    main(args.raw_dir, args.out_dir)
