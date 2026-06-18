"""
analyze_s1_raw_stage2.py
Stage-2 Forensic Analysis for S1.mat
Goal: Decode channels, events, and compute PSD across the entire continuous
recording to evaluate discarded information (Alpha/Beta/Gamma & EOG).
"""

import os
import sys
import argparse
import scipy.io as sio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import welch
from pathlib import Path

def print_and_write(f_out, text):
    print(text)
    f_out.write(text + "\n")

def run_stage2_analysis(mat_path, out_dir):
    mat_path = Path(mat_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    report_path = out_dir / "s1_stage2_report.md"
    
    if not mat_path.exists():
        print(f"ERROR: The file {mat_path} does not exist.")
        return
        
    print(f"Loading {mat_path.name} into RAM (this may take a moment)...")
    try:
        mat = sio.loadmat(mat_path, squeeze_me=True, struct_as_record=False)
    except Exception as e:
        print(f"Failed to load MAT file: {e}")
        return
        
    data_struct = getattr(mat, 'data', None)
    if not data_struct:
        print("ERROR: Could not find 'data' struct in MAT file.")
        return

    with open(report_path, "w") as f_out:
        print_and_write(f_out, "# Stage-2 Forensic Audit: Frequency Content & EOG\n")
        
        # 1. Extract Sampling Rate
        fs = None
        try:
            fs = float(data_struct.fsample.eeg)
        except Exception:
            pass
        if not fs:
            print_and_write(f_out, "WARNING: Could not extract fsample. Defaulting to 512.0 Hz")
            fs = 512.0
            
        print_and_write(f_out, f"**Sampling Rate:** {fs} Hz\n")
        
        # 2. Decode Channel Names
        labels = []
        try:
            chan_obj = data_struct.dim.chan.eeg
            labels = [str(x) for x in chan_obj]
        except Exception as e:
            print_and_write(f_out, f"WARNING: Failed to decode labels: {e}")
            
        print_and_write(f_out, f"**Total Channels:** {len(labels)}")
        
        eog_chs = [(i, l) for i, l in enumerate(labels) if 'EOG' in l.upper() or 'EXG' in l.upper()]
        
        # Target EEG channels based on user prompt
        target_eeg_names = ['T7', 'T8', 'CZ', 'FCZ', 'TP7', 'TP8']
        eeg_chs = []
        for l in target_eeg_names:
            matches = [i for i, name in enumerate(labels) if name.upper() == l]
            if matches:
                eeg_chs.append((matches[0], labels[matches[0]]))
                
        # If specific targets missing, just grab first 6 non-EOG
        if not eeg_chs and labels:
            eeg_chs = [(i, l) for i, l in enumerate(labels) if 'EOG' not in l.upper()][:6]
            
        print_and_write(f_out, f"**EOG Channels Found ({len(eog_chs)}):** {', '.join([l for i, l in eog_chs])}")
        print_and_write(f_out, f"**Target EEG Channels for PSD:** {', '.join([l for i, l in eeg_chs])}\n")
        
        # 3. Decode Event Values
        try:
            events_val = data_struct.event.eeg.value
            events_sample = data_struct.event.eeg.sample
            
            # Count unique trigger values
            val_counts = pd.Series(events_val).value_counts().to_dict()
            print_and_write(f_out, "### Event Markers (Triggers)")
            print_and_write(f_out, f"Total Events: {len(events_val)}")
            print_and_write(f_out, "Trigger Value Counts:")
            for val, count in val_counts.items():
                print_and_write(f_out, f"- Value **{val}**: {count} occurrences")
            print_and_write(f_out, "")
        except Exception as e:
            print_and_write(f_out, f"WARNING: Failed to decode events: {e}\n")

        # 4 & 5. Continuous EEG Data & PSD Computation
        try:
            # Shape is expected to be (time, channels) based on Stage 1 (3141632, 73)
            eeg_data = data_struct.eeg
            print_and_write(f_out, f"### Continuous Data")
            print_and_write(f_out, f"Shape: {eeg_data.shape} (Time x Channels)")
            print_and_write(f_out, f"Total Duration: {eeg_data.shape[0] / fs / 60:.2f} minutes\n")
            
            # Helper to compute band percentages
            bands = {
                "Delta (1-4 Hz)": (1, 4),
                "Theta (4-8 Hz)": (4, 8),
                "Alpha (8-12 Hz)": (8, 12),
                "Beta (12-30 Hz)": (12, 30),
                "Low Gamma (30-50 Hz)": (30, 50),
                "High Gamma (50-80 Hz)": (50, 80)
            }
            
            def compute_psd_and_bands(ch_list, title, filename):
                print_and_write(f_out, f"#### {title}")
                plt.figure(figsize=(12, 6))
                
                band_results = {b: [] for b in bands}
                
                for idx, name in ch_list:
                    # eeg_data is (time, chan). Extract continuous 1D array
                    sig = eeg_data[:, idx]
                    
                    # Welch PSD. 10 second windows for 0.1 Hz resolution
                    f_psd, Pxx = welch(sig, fs=fs, nperseg=int(fs*10))
                    
                    plt.plot(f_psd, 10 * np.log10(Pxx), label=name)
                    
                    # Compute band energy
                    valid_idx = f_psd <= 80
                    total_power = np.trapz(Pxx[valid_idx], f_psd[valid_idx])
                    
                    if total_power > 0:
                        for b_name, (low, high) in bands.items():
                            b_idx = np.logical_and(f_psd >= low, f_psd <= high)
                            b_power = np.trapz(Pxx[b_idx], f_psd[b_idx])
                            band_results[b_name].append((b_power / total_power) * 100)
                
                plt.title(f"{title} - PSD (0-100 Hz)")
                plt.xlabel("Frequency (Hz)")
                plt.ylabel("Power (dB)")
                plt.xlim(0, 100)
                plt.legend()
                plt.tight_layout()
                plt.savefig(out_dir / filename)
                plt.close()
                
                # Print average band percentages across these channels
                print_and_write(f_out, "Average Band Energy (0-80Hz = 100%):")
                for b_name in bands:
                    mean_pct = np.mean(band_results[b_name]) if band_results[b_name] else 0
                    print_and_write(f_out, f"- **{b_name}**: {mean_pct:.2f}%")
                print_and_write(f_out, "")
            
            if eeg_chs:
                compute_psd_and_bands(eeg_chs, "Representative EEG Channels", "s1_psd_eeg.png")
            
            if eog_chs:
                compute_psd_and_bands(eog_chs, "EOG Channels", "s1_psd_eog.png")
                
        except Exception as e:
            print_and_write(f_out, f"ERROR processing continuous data: {e}")

    print(f"\nStage 2 Audit complete. Report saved to {report_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--s1_path", type=str, default="/kaggle/input/datasets/lowk1ee/raw-eeh/S1.mat", help="Path to raw S1.mat")
    parser.add_argument("--out_dir", type=str, default="/kaggle/working/reports", help="Directory to save reports")
    args = parser.parse_args()
    
    run_stage2_analysis(args.s1_path, args.out_dir)
