"""
Phase 6: EOG / Eye Movement Analysis
Identifies eye channels (EXG or EOG), measures blink activity.
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.append(str(Path(__file__).parent))
from raw_eeg_utils import load_raw_eeg

def analyze_eog(mat_path):
    print(f"Processing {mat_path.name}...")
    try:
        eeg_data = load_raw_eeg(mat_path)
    except Exception as e:
        print(f"Failed to load {mat_path.name}: {e}")
        return None
        
    labels = eeg_data['labels']
    trials = eeg_data['trials']
    if not labels or not trials: return None
    
    # Identify EOG channels (common names: EXG1, EXG2, VEOG, HEOG, EOG)
    eog_idx = [i for i, l in enumerate(labels) if 'EXG' in l.upper() or 'EOG' in l.upper()]
    
    if not eog_idx:
        return {"subject": mat_path.stem, "eog_found": False}
        
    res = {"subject": mat_path.stem, "eog_found": True, "eog_channels": [labels[i] for i in eog_idx]}
    
    # Measure blink activity (variance in EOG channels)
    vars_eog = []
    for tr in trials:
        eog_data_tr = tr[eog_idx, :]
        vars_eog.append(np.var(eog_data_tr, axis=1))
        
    mean_var = np.mean(vars_eog, axis=0)
    for i, idx in enumerate(eog_idx):
        res[f"var_{labels[idx]}"] = mean_var[i]
        
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
        res = analyze_eog(m)
        if res: results.append(res)
            
    if not results:
        print("No valid results computed.")
        return
        
    md_path = out_dir / "eog_analysis.md"
    with open(md_path, 'w') as f:
        f.write("# EOG / Eye Movement Analysis\n\n")
        
        found_count = sum(1 for r in results if r["eog_found"])
        f.write(f"EOG channels found in {found_count} out of {len(results)} subjects.\n\n")
        
        if found_count > 0:
            f.write("## EOG Channels Detected\n")
            eog_chans = set()
            for r in results:
                if r["eog_found"]: eog_chans.update(r["eog_channels"])
            f.write(f"- {', '.join(eog_chans)}\n\n")
            
            f.write("## Blink/Eye Movement Variance\n")
            f.write("High variance in these channels indicates presence of eye movements.\n")
            for c in eog_chans:
                vals = [r[f"var_{c}"] for r in results if f"var_{c}" in r]
                if vals:
                    f.write(f"- **{c}**: {np.mean(vals):.4f}\n")
                    
            f.write("\n## Conclusion\n")
            f.write("If EOG contains strong signal, we can evaluate whether excluding them reduces noise, or if artifact removal algorithms (like ICA) are necessary.\n")
        else:
            f.write("No EOG channels found in the raw data. The eye channels might have been separated or discarded before this stage.\n")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir", type=str, default="/kaggle/input/datasets/lokeshgile/raw-eeg")
    parser.add_argument("--out_dir", type=str, default="/kaggle/working/reports")
    args = parser.parse_args()
    main(args.raw_dir, args.out_dir)
