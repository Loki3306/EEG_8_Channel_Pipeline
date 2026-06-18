"""
analyze_s1_raw_stage2.py
Stage-2 Forensic Analysis for S1.mat
Goal: Decode channels, events, and compute PSD across sampled continuous
recording segments to evaluate discarded information (Alpha/Beta/Gamma & EOG).
Simulates 1-8Hz preprocessing to highlight the exact information loss.
"""

import os
import sys
import argparse
import scipy.io as sio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import welch, butter, filtfilt, decimate
from pathlib import Path

def print_and_write(f_out, text):
    print(text)
    f_out.write(text + "\n")

def butter_bandpass(data, lowcut, highcut, fs, order=3):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)

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
        
    data_struct = mat.get('data')
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
            
        if fs is None:
            raise RuntimeError("Could not extract sampling rate from data.fsample.eeg")
            
        print_and_write(f_out, f"**Sampling Rate:** {fs} Hz\n")
        
        # 2. Decode Channel Names Safely
        print_and_write(f_out, "### Channel Decoding")
        labels = []
        try:
            chan_obj = data_struct.dim.chan.eeg
            for x in chan_obj:
                if isinstance(x, np.ndarray) and x.size > 0:
                    labels.append(str(x.flat[0]))
                else:
                    labels.append(str(x))
        except Exception as e:
            print_and_write(f_out, f"WARNING: Failed to decode labels: {e}")
            
        print_and_write(f_out, f"**Total Channels:** {len(labels)}")
        print_and_write(f_out, f"**First 20 labels:** {labels[:20]}")
        print_and_write(f_out, f"**Last 20 labels:** {labels[-20:]}\n")
        
        eog_chs = [(i, l) for i, l in enumerate(labels) if 'EOG' in l.upper() or 'EXG' in l.upper()]
        
        # Target EEG channels
        target_eeg_names = ['T7', 'T8', 'CZ', 'FCZ', 'TP7', 'TP8']
        eeg_chs = []
        for l in target_eeg_names:
            matches = [i for i, name in enumerate(labels) if name.upper() == l]
            if matches:
                eeg_chs.append((matches[0], labels[matches[0]]))
                
        if not eeg_chs and labels:
            eeg_chs = [(i, l) for i, l in enumerate(labels) if 'EOG' not in l.upper()][:6]
            
        print_and_write(f_out, f"**EOG Channels Found ({len(eog_chs)}):** {', '.join([l for i, l in eog_chs])}")
        print_and_write(f_out, f"**Target EEG Channels for PSD:** {', '.join([l for i, l in eeg_chs])}\n")
        
        # 3. Decode Event Values Detail
        try:
            events_val = data_struct.event.eeg.value
            events_sample = data_struct.event.eeg.sample
            
            val_counts = pd.Series(events_val).value_counts().to_dict()
            print_and_write(f_out, "### Event Markers (Triggers)")
            print_and_write(f_out, f"Total Events: {len(events_val)}")
            print_and_write(f_out, "Trigger Value Counts:")
            for val, count in val_counts.items():
                print_and_write(f_out, f"- Value **{val}**: {count} occurrences")
            
            print_and_write(f_out, f"\nFirst 20 Trigger Values: {events_val[:20].tolist() if hasattr(events_val, 'tolist') else events_val[:20]}")
            print_and_write(f_out, f"First 20 Sample Indices: {events_sample[:20].tolist() if hasattr(events_sample, 'tolist') else events_sample[:20]}\n")
        except Exception as e:
            print_and_write(f_out, f"WARNING: Failed to decode events: {e}\n")

        # 4 & 5. Sampled PSD Computation
        try:
            eeg_data = data_struct.eeg
            print_and_write(f_out, f"### Continuous Data Sampled PSD")
            print_and_write(f_out, f"Shape: {eeg_data.shape} (Time x Channels)")
            
            # Extract 10 random 60-second segments
            n_segments = 10
            segment_len = int(fs * 60)
            rng = np.random.default_rng(42)
            max_start = eeg_data.shape[0] - segment_len
            
            if max_start <= 0:
                print_and_write(f_out, "WARNING: Recording is shorter than 60 seconds. Using entire recording.")
                starts = [0]
                n_segments = 1
                segment_len = eeg_data.shape[0]
            else:
                starts = rng.integers(0, max_start, size=n_segments)
                
            print_and_write(f_out, f"Extracting {n_segments} segments of 60 seconds for PSD estimation...\n")
            
            bands = {
                "Drift (0-1 Hz)": (0, 1),
                "Delta (1-4 Hz)": (1, 4),
                "Theta (4-8 Hz)": (4, 8),
                "Alpha (8-12 Hz)": (8, 12),
                "Beta (12-30 Hz)": (12, 30),
                "Low Gamma (30-50 Hz)": (30, 50),
                "High Gamma (50-80 Hz)": (50, 80),
                "Ultra Gamma 1 (80-120 Hz)": (80, 120),
                "Ultra Gamma 2 (120-200 Hz)": (120, 200),
                "High Freq Noise (200+ Hz)": (200, 256)
            }
            
            def compute_psd_sampled(ch_list, title, filename):
                print_and_write(f_out, f"#### {title}")
                plt.figure(figsize=(12, 6))
                
                band_results = {b: [] for b in bands}
                
                for idx, name in ch_list:
                    all_Pxx = []
                    f_psd = None
                    for start in starts:
                        sig = eeg_data[start:start+segment_len, idx]
                        f, Pxx = welch(sig, fs=fs, nperseg=int(fs*4)) # 4-sec windows for better variance
                        if f_psd is None: f_psd = f
                        all_Pxx.append(Pxx)
                        
                    avg_Pxx = np.mean(all_Pxx, axis=0)
                    
                    plt.plot(f_psd, 10 * np.log10(avg_Pxx), label=name)
                    
                    valid_idx = f_psd <= 256
                    total_power = np.trapezoid(avg_Pxx[valid_idx], f_psd[valid_idx])
                    
                    if total_power > 0:
                        for b_name, (low, high) in bands.items():
                            b_idx = np.logical_and(f_psd >= low, f_psd <= high)
                            b_power = np.trapezoid(avg_Pxx[b_idx], f_psd[b_idx])
                            band_results[b_name].append((b_power / total_power) * 100)
                
                plt.title(f"{title} - Sampled PSD (0-256 Hz)")
                plt.xlabel("Frequency (Hz)")
                plt.ylabel("Power (dB)")
                plt.xlim(0, 256)
                plt.legend()
                plt.tight_layout()
                plt.savefig(out_dir / filename)
                plt.close()
                
                print_and_write(f_out, "Average Band Energy (0-256Hz = 100%):")
                for b_name in bands:
                    mean_pct = np.mean(band_results[b_name]) if band_results[b_name] else 0
                    print_and_write(f_out, f"- **{b_name}**: {mean_pct:.2f}%")
                print_and_write(f_out, "")
            
            if eeg_chs:
                compute_psd_sampled(eeg_chs, "Representative EEG Channels", "s1_psd_eeg.png")
            
            if eog_chs:
                compute_psd_sampled(eog_chs, "EOG Channels", "s1_psd_eog.png")
                
            # 6. Raw vs Preprocessed Simulation
            if eeg_chs:
                print_and_write(f_out, "### Raw vs Preprocessed Simulation")
                target_idx, target_name = eeg_chs[0]
                print_and_write(f_out, f"Simulating 1-8 Hz pipeline on channel {target_name}...\n")
                
                # Take 1 segment
                start = starts[0]
                sig_raw = eeg_data[start:start+segment_len, target_idx]
                
                # Preprocess: Bandpass 1-8 Hz, Resample to 64 Hz
                sig_bp = butter_bandpass(sig_raw, 1.0, 8.0, fs, order=3)
                q = int(fs / 64)
                if q > 1:
                    sig_preproc = decimate(sig_bp, q)
                else:
                    sig_preproc = sig_bp
                    
                # PSD
                f_raw, Pxx_raw = welch(sig_raw, fs=fs, nperseg=int(fs*4))
                f_pre, Pxx_pre = welch(sig_preproc, fs=64.0, nperseg=int(64.0*4))
                
                plt.figure(figsize=(12, 6))
                plt.plot(f_raw, 10 * np.log10(Pxx_raw), label=f"Raw 512Hz", alpha=0.7)
                plt.plot(f_pre, 10 * np.log10(Pxx_pre), label=f"Preprocessed 1-8Hz @ 64Hz", alpha=0.9, linewidth=2)
                plt.title(f"{target_name} - Raw vs Preprocessed Pipeline")
                plt.xlabel("Frequency (Hz)")
                plt.ylabel("Power (dB)")
                plt.xlim(0, 100)
                plt.legend()
                plt.tight_layout()
                plt.savefig(out_dir / "s1_psd_raw_vs_preproc.png")
                plt.close()

        except Exception as e:
            print_and_write(f_out, f"ERROR processing continuous data: {e}")

    print(f"\nStage 2 Audit complete. Report saved to {report_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--s1_path", type=str, default="/kaggle/input/datasets/lowk1ee/raw-eeh/S1.mat", help="Path to raw S1.mat")
    parser.add_argument("--out_dir", type=str, default="/kaggle/working/reports", help="Directory to save reports")
    args = parser.parse_args()
    
    run_stage2_analysis(args.s1_path, args.out_dir)
