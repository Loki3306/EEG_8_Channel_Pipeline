"""
analyze_s1_raw.py
Comprehensive Forensic Analysis of Raw EEG data for S1.mat

Phases 1-9 as requested.
Robust to MATLAB v7.2 and v7.3 (HDF5). Does not assume FieldTrip structure blindly.
"""

import os
import sys
import argparse
import numpy as np
import scipy.io as sio
import h5py
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import welch
from pathlib import Path

def print_and_write(f_out, text):
    print(text)
    f_out.write(text + "\n")

# ---------------------------------------------------------
# PHASE 1 & 2: Structure Discovery & Dataset Inventory
# ---------------------------------------------------------

def inspect_scipy_mat(mat, out_dir):
    struct_report_path = out_dir / "s1_structure.txt"
    inventory = {}
    
    with open(struct_report_path, "w") as f:
        print_and_write(f, "=== Phase 1: File Structure Discovery (scipy.io, MATLAB <= v7.2) ===")
        
        def traverse(obj, prefix=""):
            if isinstance(obj, dict):
                for k, v in obj.items():
                    if k.startswith('__'): continue
                    
                    shape = getattr(v, 'shape', 'scalar')
                    dtype = getattr(v, 'dtype', type(v).__name__)
                    
                    # Store potential key variables
                    name = k.lower()
                    if name in ['trial', 'time', 'label', 'fsample', 'sampleinfo', 'trialinfo', 'event', 'cfg', 'hdr', 'data', 'eeg']:
                        inventory[name] = v
                        
                    print_and_write(f, f"{prefix}- Key: {k} | Type: {type(v).__name__} | Shape: {shape} | Dtype: {dtype}")
                    traverse(v, prefix + "  ")
                    
            elif hasattr(obj, '_fieldnames'):  # mat_struct
                for k in obj._fieldnames:
                    v = getattr(obj, k)
                    shape = getattr(v, 'shape', 'scalar')
                    dtype = getattr(v, 'dtype', type(v).__name__)
                    
                    name = k.lower()
                    if name in ['trial', 'time', 'label', 'fsample', 'sampleinfo', 'trialinfo', 'event', 'cfg', 'hdr']:
                        inventory[name] = v
                    elif name in ['data', 'eeg'] and hasattr(v, '_fieldnames'):
                         # It's a top-level struct, traverse it
                         traverse(v, prefix + "  ")
                         
                    print_and_write(f, f"{prefix}- Field: {k} | Type: {type(v).__name__} | Shape: {shape} | Dtype: {dtype}")
                    # Don't recurse infinitely into trials, just show the top level fields
                    if name not in ['trial', 'time']:
                        traverse(v, prefix + "  ")
                        
        traverse(mat)
    
    return inventory

def inspect_h5py_mat(f_mat, out_dir):
    struct_report_path = out_dir / "s1_structure.txt"
    inventory = {}
    
    with open(struct_report_path, "w") as f:
        print_and_write(f, "=== Phase 1: File Structure Discovery (h5py, MATLAB v7.3) ===")
        
        def traverse(name, node):
            shape = getattr(node, 'shape', 'scalar')
            dtype = getattr(node, 'dtype', type(node).__name__)
            size_bytes = getattr(node, 'nbytes', 0)
            
            print_and_write(f, f"- {name} | Type: {type(node).__name__} | Shape: {shape} | Dtype: {dtype} | Size: {size_bytes} bytes")
            
            # Simple keyword matching
            base_name = name.split('/')[-1].lower()
            if base_name in ['trial', 'time', 'label', 'fsample', 'sampleinfo', 'trialinfo', 'event', 'cfg', 'hdr', 'data', 'eeg']:
                inventory[name] = node
                
        f_mat.visititems(traverse)
        
    return inventory


# ---------------------------------------------------------
# HELPER: Extract Data from Inventory
# ---------------------------------------------------------
def extract_dataset(inventory, f_h5=None):
    """
    Tries to intelligently find fsample, channels, and trials based on the inventory.
    """
    extracted = {
        'fsample': None,
        'labels': [],
        'trials': [] # list of (channels, time) numpy arrays
    }
    
    # 1. Sampling Rate
    fs_keys = [k for k in inventory.keys() if 'fsample' in k.lower()]
    if fs_keys:
        k = fs_keys[0]
        v = inventory[k]
        if f_h5 is not None:
            # h5py dataset
            val = v[()]
            if isinstance(val, np.ndarray):
                extracted['fsample'] = float(val.flat[0])
            else:
                extracted['fsample'] = float(val)
        else:
            extracted['fsample'] = float(v)
            
    # 2. Labels
    label_keys = [k for k in inventory.keys() if 'label' in k.lower()]
    if label_keys:
        k = label_keys[0]
        v = inventory[k]
        if f_h5 is not None:
            refs = v[:, 0] if len(v.shape) > 1 else v[:]
            labels = []
            for ref in refs:
                try:
                    obj = f_h5[ref]
                    lbl = ''.join(chr(c[0]) for c in obj[:])
                    labels.append(lbl)
                except Exception:
                    pass
            extracted['labels'] = labels
        else:
            if isinstance(v, np.ndarray) and v.dtype == object:
                extracted['labels'] = [str(x) for x in v]
            else:
                extracted['labels'] = [str(v)]
                
    # 3. Trials
    trial_keys = [k for k in inventory.keys() if 'trial' in k.lower() and 'info' not in k.lower()]
    if trial_keys:
        k = trial_keys[0] # Try the first one
        v = inventory[k]
        
        if f_h5 is not None:
            if isinstance(v, h5py.Dataset):
                # Is it an array of object references (cell array)?
                if v.dtype.kind == 'O': 
                    refs = v[:, 0] if len(v.shape) > 1 else v[:]
                    for ref in refs:
                        try:
                            tr = f_h5[ref][()]
                            extracted['trials'].append(tr)
                        except Exception:
                            pass
                else:
                    # It's a contiguous matrix, e.g. (channels, time, trials) or (trials, channels, time)
                    extracted['trials'] = [v[()]]
        else:
            if isinstance(v, np.ndarray) and v.dtype == object:
                extracted['trials'] = [tr for tr in v]
            elif isinstance(v, np.ndarray):
                extracted['trials'] = [v]
                
    return extracted

# ---------------------------------------------------------
# MAIN ANALYSIS PHASES
# ---------------------------------------------------------
def run_analysis(mat_path, out_dir):
    mat_path = Path(mat_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Starting forensic analysis on {mat_path}")
    
    inventory = {}
    extracted = None
    f_h5 = None
    
    try:
        mat = sio.loadmat(mat_path, squeeze_me=True, struct_as_record=False)
        print("Loaded with scipy.io")
        inventory = inspect_scipy_mat(mat, out_dir)
        extracted = extract_dataset(inventory)
    except NotImplementedError:
        print("scipy.io failed. Attempting h5py (MATLAB v7.3)...")
        f_h5 = h5py.File(mat_path, 'r')
        inventory = inspect_h5py_mat(f_h5, out_dir)
        extracted = extract_dataset(inventory, f_h5)
    except Exception as e:
        print(f"FATAL ERROR loading MAT file: {e}")
        return

    print("Phase 1 & 2 Complete. Structure report generated.")

    # Phase 3 - Channel Information
    if extracted['labels']:
        df_ch = pd.DataFrame({
            "index": range(len(extracted['labels'])),
            "channel_name": extracted['labels']
        })
        
        eog_chs = [c for c in extracted['labels'] if 'EOG' in c.upper() or 'EXG' in c.upper()]
        trig_chs = [c for c in extracted['labels'] if 'TRIG' in c.upper() or 'STATUS' in c.upper()]
        
        df_ch['type'] = 'EEG'
        df_ch.loc[df_ch['channel_name'].isin(eog_chs), 'type'] = 'EOG/EXG'
        df_ch.loc[df_ch['channel_name'].isin(trig_chs), 'type'] = 'TRIGGER'
        
        df_ch.to_csv(out_dir / "s1_channels.csv", index=False)
        print("Phase 3 Complete: Channels extracted.")
    else:
        print("Phase 3: No channel labels found.")

    # Phase 4 & 5 - Sampling & Trial Statistics
    fs = extracted.get('fsample')
    trials = extracted.get('trials', [])
    
    if fs is None and len(trials) > 0:
        print("Warning: fsample not found, assuming 64 Hz for downstream analysis.")
        fs = 64.0
        
    with open(out_dir / "s1_sampling.txt", "w") as f:
        f.write("=== Phase 4 & 5: Sampling and Trial Statistics ===\n")
        f.write(f"Sampling Rate: {fs} Hz\n")
        f.write(f"Total Trials Found: {len(trials)}\n\n")
        
        if trials:
            lengths = [tr.shape[-1] for tr in trials]
            durations = [l / fs for l in lengths] if fs else lengths
            
            f.write(f"Shortest Trial: {min(durations):.2f} sec ({min(lengths)} samples)\n")
            f.write(f"Longest Trial: {max(durations):.2f} sec ({max(lengths)} samples)\n")
            f.write(f"Average Trial: {np.mean(durations):.2f} sec ({np.mean(lengths):.1f} samples)\n")
            
            # Global min/max/mean/std over all trials
            all_means = [np.mean(tr) for tr in trials]
            all_stds = [np.std(tr) for tr in trials]
            all_maxs = [np.max(tr) for tr in trials]
            all_mins = [np.min(tr) for tr in trials]
            
            f.write(f"Global Data Min: {np.min(all_mins):.4f}\n")
            f.write(f"Global Data Max: {np.max(all_maxs):.4f}\n")
            f.write(f"Average Mean across trials: {np.mean(all_means):.4f}\n")
            f.write(f"Average Std across trials: {np.mean(all_stds):.4f}\n")
    print("Phase 4 & 5 Complete.")

    # Select specific channels if available
    labels_upper = [str(l).upper() for l in extracted.get('labels', [])]
    target_chs = ['T7', 'T8', 'CZ', 'FCZ']
    ch_indices = []
    
    for tc in target_chs:
        if tc in labels_upper:
            ch_indices.append((tc, labels_upper.index(tc)))
    
    if not ch_indices and trials:
        # Fallback to first 4 channels
        ch_indices = [(f"CH_{i}", i) for i in range(min(4, trials[0].shape[0]))]

    # Phase 6 & 7 - Raw Signal Inspection & Frequency Audit
    if trials and ch_indices:
        print("Running Phase 6 & 7: Signal plotting and PSD computation...")
        
        trial_idxs = [0, 9, 29, 49] # 1, 10, 30, 50 (0-indexed)
        trial_idxs = [i for i in trial_idxs if i < len(trials)]
        
        for t_idx in trial_idxs:
            tr = trials[t_idx]
            
            # Phase 6: Plot 10 seconds of raw EEG
            n_samples = min(int(fs * 10), tr.shape[-1]) if fs else min(1000, tr.shape[-1])
            time_vec = np.arange(n_samples) / (fs if fs else 1)
            
            plt.figure(figsize=(15, 8))
            for name, idx in ch_indices:
                if idx < tr.shape[0]:
                    plt.plot(time_vec, tr[idx, :n_samples] + (ch_indices.index((name,idx)) * 50), label=name)
            plt.title(f"Trial {t_idx+1}: 10s Raw Signal")
            plt.xlabel("Time (s)")
            plt.legend()
            plt.tight_layout()
            plt.savefig(out_dir / f"s1_raw_signal_trial_{t_idx+1}.png")
            plt.close()
            
            # Phase 7: PSD Plotting for this trial
            plt.figure(figsize=(15, 6))
            for name, idx in ch_indices:
                if idx < tr.shape[0]:
                    # Welch
                    f_psd, Pxx = welch(tr[idx], fs=fs if fs else 1.0, nperseg=min(int((fs if fs else 1)*2), tr.shape[-1]))
                    plt.plot(f_psd, 10 * np.log10(Pxx), label=name)
            plt.title(f"Trial {t_idx+1}: PSD (0-100 Hz)")
            plt.xlabel("Frequency (Hz)")
            plt.ylabel("Power (dB)")
            plt.xlim(0, 100)
            plt.legend()
            plt.tight_layout()
            plt.savefig(out_dir / f"s1_psd_0_100_trial_{t_idx+1}.png")
            plt.close()
            
            plt.figure(figsize=(15, 6))
            for name, idx in ch_indices:
                if idx < tr.shape[0]:
                    f_psd, Pxx = welch(tr[idx], fs=fs if fs else 1.0, nperseg=min(int((fs if fs else 1)*2), tr.shape[-1]))
                    plt.plot(f_psd, 10 * np.log10(Pxx), label=name)
            plt.title(f"Trial {t_idx+1}: PSD (0-40 Hz)")
            plt.xlabel("Frequency (Hz)")
            plt.ylabel("Power (dB)")
            plt.xlim(0, 40)
            plt.legend()
            plt.tight_layout()
            plt.savefig(out_dir / f"s1_psd_0_40_trial_{t_idx+1}.png")
            plt.close()

        # Compute global band energy percentages on all trials for the selected channels
        bands = {
            "Delta": (1, 4),
            "Theta": (4, 8),
            "Alpha": (8, 12),
            "Beta": (12, 30),
            "Gamma": (30, 80)
        }
        
        all_powers = {b: [] for b in bands}
        for tr in trials:
            for _, idx in ch_indices:
                if idx < tr.shape[0]:
                    f_psd, Pxx = welch(tr[idx], fs=fs if fs else 1.0, nperseg=min(int((fs if fs else 1)*2), tr.shape[-1]))
                    
                    for b, (low, high) in bands.items():
                        b_idx = np.logical_and(f_psd >= low, f_psd <= high)
                        if np.any(b_idx):
                            all_powers[b].append(np.trapz(Pxx[b_idx], f_psd[b_idx]))
                            
        total_p = sum(np.mean(p) for p in all_powers.values() if p)
        band_pcts = {b: (np.mean(all_powers[b]) / total_p * 100) if total_p > 0 and all_powers[b] else 0.0 for b in bands}
        print("Phase 6 & 7 Complete.")

    # Phase 8 - EOG
    eog_corr = None
    if trials and extracted.get('labels'):
        eog_chs = [(l, i) for i, l in enumerate(extracted['labels']) if 'EOG' in l.upper() or 'EXG' in l.upper()]
        eeg_chs = [(l, i) for i, l in enumerate(extracted['labels']) if 'EOG' not in l.upper() and 'EXG' not in l.upper() and 'TRIG' not in l.upper() and 'STATUS' not in l.upper()]
        
        if eog_chs and eeg_chs:
            print("Phase 8: Plotting EOG vs EEG...")
            tr = trials[0]
            
            eog_name, eog_idx = eog_chs[0]
            eeg_name, eeg_idx = eeg_chs[0]
            
            if eog_idx < tr.shape[0] and eeg_idx < tr.shape[0]:
                n_samples = min(int(fs * 10), tr.shape[-1]) if fs else 1000
                time_vec = np.arange(n_samples) / (fs if fs else 1)
                
                plt.figure(figsize=(15, 6))
                plt.plot(time_vec, tr[eeg_idx, :n_samples], label=f"EEG: {eeg_name}")
                plt.plot(time_vec, tr[eog_idx, :n_samples], label=f"EOG: {eog_name}", alpha=0.7)
                plt.title(f"Trial 1: {eeg_name} vs {eog_name}")
                plt.xlabel("Time (s)")
                plt.legend()
                plt.tight_layout()
                plt.savefig(out_dir / "s1_eog_vs_eeg.png")
                plt.close()
                
                # Correlation
                r = np.corrcoef(tr[eeg_idx], tr[eog_idx])[0, 1]
                eog_corr = r
                print(f"EOG Correlation: {r:.4f}")
    
    # Phase 9 - Report
    print("Phase 9: Generating opportunity report...")
    with open(out_dir / "s1_opportunities.md", "w") as f:
        f.write("# Phase 9: Attention-Relevant Opportunity Report\n\n")
        
        f.write(f"Sampling rate = {fs if fs else 'UNKNOWN'} Hz\n\n")
        
        if trials and 'band_pcts' in locals():
            has_beta = band_pcts['Beta'] > 2.0
            has_gamma = band_pcts['Gamma'] > 1.0
            f.write(f"Useful energy above 12 Hz:\n{'YES' if has_beta else 'NO'} ({band_pcts['Beta']:.2f}% Beta)\n\n")
            f.write(f"Useful energy above 30 Hz:\n{'YES' if has_gamma else 'NO'} ({band_pcts['Gamma']:.2f}% Gamma)\n\n")
        else:
            f.write("Useful energy above 12 Hz:\nUNKNOWN (PSD calculation failed)\n\n")
            f.write("Useful energy above 30 Hz:\nUNKNOWN (PSD calculation failed)\n\n")
            
        has_eog = any('EOG' in l.upper() or 'EXG' in l.upper() for l in extracted.get('labels', []))
        f.write(f"EOG channels present:\n{'YES' if has_eog else 'NO'}\n")
        if eog_corr is not None:
            f.write(f"EOG/EEG Global Correlation (Trial 1): {eog_corr:.4f}\n")
        f.write("\n")
        
        f.write("Raw data contains substantially more information than current 1-8 Hz preprocessing:\n")
        if trials and 'band_pcts' in locals():
            if band_pcts['Beta'] > 2.0 or band_pcts['Gamma'] > 1.0:
                f.write("YES. High frequencies are intact.\n\n")
            else:
                f.write("NO. The data appears to already lack high frequencies, mirroring the preprocessing.\n\n")
        else:
             f.write("UNKNOWN.\n\n")
             
        f.write("Potential opportunities:\n")
        if trials and 'band_pcts' in locals() and (band_pcts['Beta'] > 2.0 or band_pcts['Gamma'] > 1.0):
            f.write("- Higher frequency bands\n")
        if has_eog:
            f.write("- Eye movement features\n")
        if fs and fs > 64:
            f.write("- Better temporal resolution\n")
        
    print(f"\nAnalysis complete. All reports saved to {out_dir}")
    
    if f_h5:
        f_h5.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--s1_path", type=str, default="/kaggle/input/raw-eeh/S1.mat", help="Path to raw S1.mat")
    parser.add_argument("--out_dir", type=str, default="/kaggle/working/reports", help="Directory to save reports")
    args = parser.parse_args()
    
    run_analysis(args.s1_path, args.out_dir)
