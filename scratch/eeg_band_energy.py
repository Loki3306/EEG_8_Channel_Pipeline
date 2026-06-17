"""
Phase 2: Frequency Content Analysis
Computes PSD using Welch for all subjects on RAW EEG.
Measures energy in standard frequency bands.
"""

import sys
import numpy as np
import pandas as pd
from scipy.signal import welch
from pathlib import Path

# Add scratch to path to import raw_eeg_utils
sys.path.append(str(Path(__file__).parent))
from raw_eeg_utils import load_raw_eeg

BANDS = {
    "Delta": (1, 4),
    "Theta": (4, 8),
    "Alpha": (8, 12),
    "Beta": (12, 30),
    "Gamma": (30, 70),
    "HighGamma": (70, 150)
}

def analyze_subject(mat_path):
    print(f"Processing {mat_path.name}...")
    try:
        eeg_data = load_raw_eeg(mat_path)
    except Exception as e:
        print(f"Failed to load {mat_path.name}: {e}")
        return None
        
    fs = eeg_data['fsample']
    trials = eeg_data['trials']
    
    if not trials:
        return None
        
    # We will average PSD across all trials and all channels
    all_psds = []
    freqs = None
    
    for tr in trials:
        # tr is (channels, time)
        # welch computes along last axis by default
        nperseg = min(int(fs * 4), tr.shape[1]) # 4 second windows for 0.25 Hz resolution
        f, Pxx = welch(tr, fs=fs, nperseg=nperseg)
        if freqs is None: freqs = f
        all_psds.append(np.mean(Pxx, axis=0)) # Average across channels for this trial
        
    mean_psd = np.mean(all_psds, axis=0) # Average across trials
    
    total_power = np.trapz(mean_psd, freqs)
    
    res = {"subject": mat_path.stem, "fsample": fs}
    for band_name, (low, high) in BANDS.items():
        idx = np.logical_and(freqs >= low, freqs <= high)
        if not np.any(idx):
            res[f"{band_name}_pct"] = 0.0
            continue
        power = np.trapz(mean_psd[idx], freqs[idx])
        pct = (power / total_power) * 100
        res[f"{band_name}_pct"] = pct
        
    return res

def main(raw_dir, out_dir):
    raw_dir = Path(raw_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    mats = list(raw_dir.glob("*.mat"))
    if not mats:
        print(f"No .mat files found in {raw_dir}")
        return
        
    results = []
    for m in mats:
        res = analyze_subject(m)
        if res:
            results.append(res)
            
    if not results:
        print("No valid results computed.")
        return
        
    df = pd.DataFrame(results)
    csv_path = out_dir / "eeg_band_energy.csv"
    df.to_csv(csv_path, index=False)
    
    # Generate markdown summary
    md_path = out_dir / "eeg_band_energy_summary.md"
    with open(md_path, 'w') as f:
        f.write("# EEG Frequency Band Energy (Raw Data)\n\n")
        f.write("This report summarizes the percentage of total signal energy in each frequency band across all subjects.\n\n")
        
        # Calculate means
        means = df[[f"{b}_pct" for b in BANDS]].mean()
        f.write("## Average Energy Across Subjects\n")
        for b in BANDS:
            f.write(f"- **{b}**: {means[f'{b}_pct']:.2f}%\n")
            
        f.write("\n## Scientific Questions Addressed\n")
        f.write("1. **Does raw EEG contain significant beta activity?**\n")
        f.write(f"   > Beta energy is {means['Beta_pct']:.2f}%. (If >5%, it's significant).\n\n")
        f.write("2. **Does raw EEG contain gamma activity?**\n")
        f.write(f"   > Gamma energy is {means['Gamma_pct']:.2f}%. (If >2%, it's significant).\n\n")
        f.write("3. **Is our current 1-8 Hz pipeline discarding useful signal?**\n")
        f.write(f"   > The >8 Hz bands contain {means['Alpha_pct'] + means['Beta_pct'] + means['Gamma_pct'] + means['HighGamma_pct']:.2f}% of the energy.\n")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir", type=str, default="/kaggle/input/datasets/lokeshgile/raw-eeg")
    parser.add_argument("--out_dir", type=str, default="/kaggle/working/reports")
    args = parser.parse_args()
    main(args.raw_dir, args.out_dir)
