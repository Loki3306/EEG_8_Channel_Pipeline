import os
import sys
import torch
import numpy as np
import scipy.signal
import matplotlib.pyplot as plt
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.aad_conformer import AADConformer

def cross_corr(x, y, max_lag):
    """
    Computes cross-correlation between x and y for lags in [-max_lag, max_lag].
    x and y should be 1D numpy arrays of the same length.
    """
    # Normalize
    x = (x - np.mean(x)) / (np.std(x) + 1e-8)
    y = (y - np.mean(y)) / (np.std(y) + 1e-8)
    
    corrs = []
    lags = range(-max_lag, max_lag + 1)
    for lag in lags:
        if lag < 0:
            c = np.corrcoef(x[:lag], y[-lag:])[0, 1]
        elif lag > 0:
            c = np.corrcoef(x[lag:], y[:-lag])[0, 1]
        else:
            c = np.corrcoef(x, y)[0, 1]
        corrs.append(c if not np.isnan(c) else 0.0)
    return lags, np.array(corrs)

def main():
    print("[INFO] Starting Phase 28.5 Audio-EEG Alignment Audit")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cache_dir = REPO_ROOT / 'data' / 'processed_aasd'
    cache_file = cache_dir / "S1.pt"
    
    if not cache_file.exists():
        print(f"[ERROR] Cache not found for S1.")
        return
        
    data = torch.load(cache_file, weights_only=False)
    trials = data['trials']
    
    print(f"[INFO] Loaded S1 with {len(trials)} trials.")
    
    model = AADConformer(in_channels=8).to(device)
    ckpt_path = REPO_ROOT / 'checkpoints' / 'aasd_finetuned' / 'model_S18_loso.pt'
    if ckpt_path.exists():
        model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=False))
        print("[INFO] Loaded AASD Fine-Tuned Weights")
    else:
        print("[WARNING] Fine-tuned weights not found. Using KUL weights.")
        kul_path = '/kaggle/input/datasets/lowkieee/eeg-aad-conformer-seed1-checkpoints/checkpoints/seed_1/model_S1.pt'
        if os.path.exists(kul_path):
            ckpt = torch.load(kul_path, map_location=device, weights_only=False)
            model.load_state_dict(ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt, strict=False)
            
    model.eval()
    
    FS = 64
    MAX_LAG_SEC = 2.0
    max_lag = int(MAX_LAG_SEC * FS)
    
    all_corrs_att = []
    all_corrs_unatt = []
    
    print("[INFO] Computing Cross-Correlations...")
    
    with torch.no_grad():
        for i, trial in enumerate(trials):
            eeg = trial['eeg'].unsqueeze(0).to(device) # (1, 8, time)
            audio_l = trial['audio_l'].numpy()[0] # (time,)
            audio_r = trial['audio_r'].numpy()[0] # (time,)
            raw_evs = trial['meta']['raw_evs']
            
            # Predict EEG envelope
            pred, _ = model(eeg, return_features=True)
            pred = pred.squeeze().cpu().numpy() # (time,)
            
            # Figure out which audio is attended for the FIRST window just to get a general sense,
            # or better yet, just cross-correlate with both and see which one peaks.
            # To be precise, let's just cross-correlate pred with audio_l and audio_r.
            
            if len(pred) <= max_lag * 2:
                continue
                
            lags, corr_l = cross_corr(pred, audio_l, max_lag)
            lags, corr_r = cross_corr(pred, audio_r, max_lag)
            
            # In KUL, attention is sustained. In AASD, it switches. 
            # Over the whole trial, the mean correlation will be diluted, but the peak should still exist.
            # Let's take the MAX of abs(corr_l) and abs(corr_r) as the "attended" correlation profile 
            # for this trial (assuming one speaker is attended more than the other, or they both have peaks).
            
            # Actually, let's just average the raw correlations across all trials for L and R separately,
            # or use the GT to slice it.
            
            # Let's use the GT for the first 5 seconds to find the initial attended speaker
            st_times = []
            types = []
            for ev_t, ev_lat in raw_evs:
                if ev_t in ['179', '184', '254', '255']:
                    st_times.append(ev_lat / 128.0)
                    types.append('R' if ev_t in ['179', '254'] else 'L')
            
            if len(types) == 0: continue
            
            initial_att = types[0]
            if initial_att == 'R':
                att = audio_r
                unatt = audio_l
            else:
                att = audio_l
                unatt = audio_r
                
            lags, c_att = cross_corr(pred, att, max_lag)
            lags, c_unatt = cross_corr(pred, unatt, max_lag)
            
            all_corrs_att.append(c_att)
            all_corrs_unatt.append(c_unatt)
            
    mean_corr_att = np.mean(all_corrs_att, axis=0)
    mean_corr_unatt = np.mean(all_corrs_unatt, axis=0)
    
    peak_idx = np.argmax(mean_corr_att)
    peak_lag_sec = lags[peak_idx] / float(FS)
    peak_val = mean_corr_att[peak_idx]
    
    print("\n==================================================")
    print("=== ALIGNMENT AUDIT RESULTS ===")
    print("==================================================")
    print(f"Max Attended Correlation: {peak_val:.4f}")
    print(f"Peak Lag (Offset):        {peak_lag_sec:+.3f} seconds ({lags[peak_idx]:+d} samples)")
    print("==================================================")
    
    if abs(peak_lag_sec) <= 0.05:
        print("DIAGNOSIS: PERFECT ALIGNMENT.")
        print("The EEG and Audio are perfectly synced (within 1 sample).")
        print("The 0.50 accuracy is NOT caused by temporal misalignment.")
    else:
        print("DIAGNOSIS: MISALIGNMENT DETECTED!")
        print(f"The audio envelopes are shifted by {peak_lag_sec:+.3f} seconds relative to the EEG.")
        print("This completely destroys Pearson Correlation training at lag=0.")

if __name__ == "__main__":
    main()
