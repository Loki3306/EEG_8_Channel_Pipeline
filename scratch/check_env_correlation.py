import torch
from pathlib import Path
import numpy as np

def check_env_correlation():
    cache_path = Path('/kaggle/working/eeg_cache/S1_processed.pt')
    if not cache_path.exists():
        print("Cache not found.")
        return
        
    cached = torch.load(cache_path, map_location='cpu', weights_only=False)
    trials = cached['raw']
    
    corrs = []
    for i, tr in enumerate(trials):
        env_l = tr['env_l'].numpy()
        env_r = tr['env_r'].numpy()
        
        c = np.corrcoef(env_l, env_r)[0, 1]
        corrs.append(c)
        print(f"Trial {i} Env L vs Env R correlation: {c:.4f}")
        
    print(f"\nAverage Correlation: {np.mean(corrs):.4f}")

if __name__ == "__main__":
    check_env_correlation()
