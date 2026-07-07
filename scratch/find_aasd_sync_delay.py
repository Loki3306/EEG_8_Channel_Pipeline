import torch
import numpy as np
import scipy.signal
from pathlib import Path
import matplotlib.pyplot as plt

def fast_ccf(eeg, env, max_lag_samples):
    """
    Computes Pearson correlation across multiple lags using FFT.
    Returns: CCF array, Lags array
    """
    # Z-score to normalize
    eeg = (eeg - np.mean(eeg)) / (np.std(eeg) + 1e-8)
    env = (env - np.mean(env)) / (np.std(env) + 1e-8)
    
    # Cross correlation using FFT
    ccf = scipy.signal.correlate(eeg, env, mode='full', method='fft')
    ccf = ccf / len(eeg) # Scale to Pearson correlation [-1, 1]
    
    # Extract the requested lag window
    center = len(env) - 1
    start = center - max_lag_samples
    end = center + max_lag_samples + 1
    
    lags = np.arange(-max_lag_samples, max_lag_samples + 1)
    return ccf[start:end], lags

def main():
    cache_path = Path('/kaggle/working/eeg_cache/S1_processed.pt')
    if not cache_path.exists():
        print("Error: Cache not found. Please run generate_aasd_cache.py first.")
        return
        
    print("Loading S1 Cache...")
    cached = torch.load(cache_path, map_location='cpu', weights_only=False)
    trials = cached['raw']
    
    SR = 128
    MAX_LAG_SEC = 5.0
    MAX_LAG_SAMPLES = int(MAX_LAG_SEC * SR)
    
    avg_ccf_l = np.zeros(2 * MAX_LAG_SAMPLES + 1)
    avg_ccf_r = np.zeros(2 * MAX_LAG_SAMPLES + 1)
    
    print(f"Sweeping Temporal Response Function from -{MAX_LAG_SEC}s to +{MAX_LAG_SEC}s...")
    
    for i, tr in enumerate(trials):
        eeg = tr['eeg'].numpy() # (60, Time)
        env_l = tr['env_l'].numpy()
        env_r = tr['env_r'].numpy()
        
        trial_ccf_l = np.zeros(2 * MAX_LAG_SAMPLES + 1)
        trial_ccf_r = np.zeros(2 * MAX_LAG_SAMPLES + 1)
        
        for ch in range(eeg.shape[0]):
            c_l, lags = fast_ccf(eeg[ch], env_l, MAX_LAG_SAMPLES)
            c_r, _  = fast_ccf(eeg[ch], env_r, MAX_LAG_SAMPLES)
            
            # Absolute correlation since dipole polarities vary by channel
            trial_ccf_l += np.abs(c_l)
            trial_ccf_r += np.abs(c_r)
            
        avg_ccf_l += (trial_ccf_l / eeg.shape[0])
        avg_ccf_r += (trial_ccf_r / eeg.shape[0])
        
    avg_ccf_l /= len(trials)
    avg_ccf_r /= len(trials)
    
    best_lag_l = lags[np.argmax(avg_ccf_l)]
    best_lag_r = lags[np.argmax(avg_ccf_r)]
    
    print("\n" + "="*60)
    print(" CROSS-CORRELATION DELAY ANALYSIS")
    print("="*60)
    print("If dataset is properly synchronized, peak should be near +0.150s")
    print("-" * 60)
    print(f"Maximum correlation with Left Audio at lag:  {best_lag_l/SR:+.3f} seconds")
    print(f"Maximum correlation with Right Audio at lag: {best_lag_r/SR:+.3f} seconds")
    
    print("\nTop 5 Largest Peaks (Left Audio):")
    top_l = np.argsort(avg_ccf_l)[-5:][::-1]
    for idx in top_l:
        print(f"  {lags[idx]/SR:+.3f}s (Correlation: {avg_ccf_l[idx]:.4f})")
        
    print("\nTop 5 Largest Peaks (Right Audio):")
    top_r = np.argsort(avg_ccf_r)[-5:][::-1]
    for idx in top_r:
        print(f"  {lags[idx]/SR:+.3f}s (Correlation: {avg_ccf_r[idx]:.4f})")

if __name__ == "__main__":
    main()
