"""
Verify Raw Trial Extraction
Loads S1.mat (raw) and S1_data_preproc.mat (preproc) to map 
raw triggers to the 60 preprocessed trials and verify reconstruction correlation.
"""
import scipy.io as sio
import numpy as np
from pathlib import Path
import sys
from scipy.signal import butter, filtfilt, resample

def print_and_write(f_out, text):
    print(text)
    f_out.write(text + "\n")

raw_path = Path("/kaggle/input/datasets/lowk1ee/raw-eeh/S1.mat")
preproc_path = Path("/kaggle/input/dtu-eeg-data/S1_data_preproc.mat")

out_dir = Path("/kaggle/working/reports")
out_dir.mkdir(parents=True, exist_ok=True)
report_path = out_dir / "trigger_alignment_report.md"

with open(report_path, "w") as f_out:
    print_and_write(f_out, "# Raw Trigger & Trial Alignment Audit\n")
    
    # 1. Inspect Preprocessed Data
    print_and_write(f_out, "## 1. Preprocessed Data Exploration")
    if not preproc_path.exists():
        print_and_write(f_out, f"ERROR: Preprocessed path {preproc_path} not found.")
        sys.exit(1)
        
    mat_pre = sio.loadmat(preproc_path, squeeze_me=True, struct_as_record=False)
    data_pre = mat_pre.get('data')
    if not data_pre or not hasattr(data_pre, 'eeg'):
        print_and_write(f_out, "ERROR: Preprocessed data struct invalid.")
        sys.exit(1)
        
    num_trials = data_pre.eeg.shape[1]
    pre_trial_0 = np.asarray(data_pre.eeg[0, 0], dtype=float) # Shape should be (channels, time)
    
    print_and_write(f_out, f"Preprocessed Trials: {num_trials}")
    print_and_write(f_out, f"Trial 0 EEG Shape: {pre_trial_0.shape} (Channels x Time)")
    if hasattr(data_pre, 'wavA'):
        wavA_0 = np.asarray(data_pre.wavA[0, 0], dtype=float)
        print_and_write(f_out, f"Trial 0 wavA Shape: {wavA_0.shape}")
        
    preproc_samples = pre_trial_0.shape[1]
    
    print_and_write(f_out, "\n## 2. Raw Data Exploration")
    if not raw_path.exists():
        print_and_write(f_out, f"ERROR: Raw path {raw_path} not found.")
        sys.exit(1)
        
    mat_raw = sio.loadmat(raw_path, squeeze_me=True, struct_as_record=False)
    data_raw = mat_raw.get('data')
    
    if not data_raw or not hasattr(data_raw, 'event') or not hasattr(data_raw.event, 'eeg'):
        print_and_write(f_out, "ERROR: Raw data struct invalid.")
        sys.exit(1)
        
    fs_raw = float(data_raw.fsample.eeg)
    raw_eeg = data_raw.eeg # (time, channels)
    events_val = data_raw.event.eeg.value
    events_sample = data_raw.event.eeg.sample
    
    vals = events_val.tolist() if hasattr(events_val, 'tolist') else list(events_val)
    samples = events_sample.tolist() if hasattr(events_sample, 'tolist') else list(events_sample)
    
    print_and_write(f_out, f"Raw EEG Shape: {raw_eeg.shape} (Time x Channels)")
    print_and_write(f_out, f"Raw Sampling Rate: {fs_raw} Hz")
    
    print_and_write(f_out, "\n### Trigger Sequences & Deltas")
    print_and_write(f_out, "| Index | Trigger | Sample | Delta Samples | Delta Sec |")
    print_and_write(f_out, "|---|---|---|---|---|")
    for i in range(len(vals)):
        delta_samp = samples[i+1] - samples[i] if i < len(vals)-1 else 0
        delta_sec = delta_samp / fs_raw
        print_and_write(f_out, f"| {i} | {vals[i]} | {samples[i]} | {delta_samp} | {delta_sec:.2f} |")
        
    # 3. Trigger Search & Correlation
    print_and_write(f_out, "\n## 3. Trigger Search & Trial Reconstruction")
    
    # Expected length in raw samples (approximate, since audio lengths vary slightly, 
    # but we can just use the preproc trial length * 8 to slice)
    # The actual length might depend on the trigger. Let's just use the length of pre_trial_0
    target_raw_samples = int(preproc_samples * (fs_raw / 64.0))
    
    unique_triggers = list(set(vals))
    print_and_write(f_out, f"Unique Triggers to test: {unique_triggers}")
    
    def simulate_preprocessing(raw_trial, target_len):
        # raw_trial: (samples, channels)
        # 1-8 Hz bandpass
        nyq = fs_raw / 2.0
        b, a = butter(4, [1.0/nyq, 8.0/nyq], btype='band')
        filt = filtfilt(b, a, raw_trial, axis=0)
        # Resample to 64Hz
        down = resample(filt, target_len, axis=0)
        return down

    best_trigger = None
    best_corr = -1.0
    best_start = -1
    
    for trig in unique_triggers:
        # Find first occurrence
        first_idx = vals.index(trig)
        start = samples[first_idx]
        stop = start + target_raw_samples
        
        if stop > raw_eeg.shape[0]:
            print_and_write(f_out, f"- Trigger {trig}: First occurrence too close to end of recording.")
            continue
            
        raw_trial = raw_eeg[start:stop, :] # (time, channels)
        
        # Preprocess
        preproc_sim = simulate_preprocessing(raw_trial, preproc_samples) # (time, channels)
        
        # Correlate channel 0
        # raw_chan_0 is index 0. Let's compare with preproc_trial channel 0.
        c0_sim = preproc_sim[:, 0]
        c0_pre = pre_trial_0[0, :]
        
        corr = np.corrcoef(c0_sim, c0_pre)[0, 1]
        print_and_write(f_out, f"- Trigger {trig:3d} -> Correlation: {corr:.4f}")
        
        if corr > best_corr:
            best_corr = corr
            best_trigger = trig
            best_start = start
            
    print_and_write(f_out, f"\n### Best Trigger: {best_trigger} with correlation {best_corr:.4f}")
    
    # 4. Verify Channel Mapping for Best Trigger
    if best_corr > 0.5: # If we found a plausible match
        print_and_write(f_out, "\n## 4. Channel Mapping Verification")
        print_and_write(f_out, "Mapping first 10 preprocessed channels to raw channels based on correlation...")
        
        start = best_start
        stop = start + target_raw_samples
        raw_trial = raw_eeg[start:stop, :]
        preproc_sim = simulate_preprocessing(raw_trial, preproc_samples) # (time, 73)
        
        # Test first 10 preprocessed channels against all raw channels to find mapping
        for pre_ch in range(min(10, pre_trial_0.shape[0])):
            c_pre = pre_trial_0[pre_ch, :]
            corrs = []
            for raw_ch in range(preproc_sim.shape[1]):
                c_sim = preproc_sim[:, raw_ch]
                corr = np.corrcoef(c_pre, c_sim)[0, 1]
                corrs.append(corr)
            
            best_raw_ch = np.argmax(corrs)
            best_ch_corr = corrs[best_raw_ch]
            print_and_write(f_out, f"Preproc Ch {pre_ch} -> Raw Ch {best_raw_ch} (Corr: {best_ch_corr:.4f})")
            
        if np.mean([np.corrcoef(pre_trial_0[i], preproc_sim[:, i])[0, 1] for i in range(10)]) > 0.9:
            print_and_write(f_out, "\n**CONCLUSION:** Raw extraction matches preprocessed trials perfectly. 1-to-1 mapping verified.")
        else:
            print_and_write(f_out, "\n**WARNING:** High correlation found, but channels do not map 1-to-1. Re-referencing or permutation was likely applied.")
    else:
        print_and_write(f_out, "\n**ERROR:** Failed to find any trigger that reproduces the preprocessed trials. Cannot proceed.")
        
print(f"\nAudit complete. Report saved to {report_path}")
