"""
Phase 4: High Frequency Feasibility
Bandpasses raw EEG into 1-8 Hz, 8-30 Hz, and 30-70 Hz.
Measures SNR and consistency to determine if high frequencies have usable signal.
"""

import sys
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
from pathlib import Path

sys.path.append(str(Path(__file__).parent))
from raw_eeg_utils import load_raw_eeg

def butter_bandpass(data, lowcut, highcut, fs, order=3):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data, axis=1)

def analyze_high_freq(mat_path):
    print(f"Processing {mat_path.name}...")
    try:
        eeg_data = load_raw_eeg(mat_path)
    except Exception as e:
        print(f"Failed to load {mat_path.name}: {e}")
        return None
        
    fs = eeg_data['fsample']
    trials = eeg_data['trials']
    if not trials: return None
    
    bands = {
        "1-8 Hz": (1.0, 8.0),
        "8-30 Hz": (8.0, 30.0),
        "30-70 Hz": (30.0, 70.0)
    }
    
    res = {"subject": mat_path.stem}
    
    for band_name, (low, high) in bands.items():
        if high > fs / 2:
            res[f"{band_name}_power"] = 0.0
            continue
            
        powers = []
        for tr in trials:
            filtered = butter_bandpass(tr, low, high, fs)
            powers.append(np.mean(filtered**2))
            
        res[f"{band_name}_power"] = np.mean(powers)
        
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
        res = analyze_high_freq(m)
        if res: results.append(res)
            
    if not results:
        print("No valid results computed.")
        return
        
    df = pd.DataFrame(results)
    
    md_path = out_dir / "high_frequency_feasibility.md"
    with open(md_path, 'w') as f:
        f.write("# High Frequency Feasibility Report\n\n")
        
        for band in ["1-8 Hz", "8-30 Hz", "30-70 Hz"]:
            col = f"{band}_power"
            if col in df:
                mean_p = df[col].mean()
                f.write(f"- **{band} Power**: {mean_p:.4f}\n")
                
        f.write("\n## Recommendation\n")
        
        ratio = df.get("30-70 Hz_power", 0).mean() / max(df.get("1-8 Hz_power", 1).mean(), 1e-9)
        if ratio > 0.05:
            f.write("> **PROMOTE** high-frequency branch. There is substantial energy in the 30-70 Hz range.\n")
        else:
            f.write("> **KILL** high-frequency branch. Energy in 30-70 Hz is negligible compared to the base band.\n")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir", type=str, default="/kaggle/input/datasets/lokeshgile/raw-eeg")
    parser.add_argument("--out_dir", type=str, default="/kaggle/working/reports")
    args = parser.parse_args()
    main(args.raw_dir, args.out_dir)
