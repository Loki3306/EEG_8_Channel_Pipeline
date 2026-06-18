"""
raw_reconstruction_validator.py
Forensic script to rigorously validate raw trial extraction by aligning 
continuous raw EEG to preprocessed trials using cross-correlation.
Tests multiple occurrences, references, and sliding window lags.
"""

import scipy.io as sio
import numpy as np
from scipy.signal import butter, filtfilt, resample_poly, correlate
from pathlib import Path
import sys

def print_and_write(f_out, text):
    print(text)
    f_out.write(text + "\n")

def preprocess_chunk(chunk, fs_in=512.0, fs_out=64.0):
    nyq = fs_in / 2.0
    # 4th order butterworth bandpass, applied zero-phase
    b, a = butter(4, [1.0/nyq, 8.0/nyq], btype='band')
    filt = filtfilt(b, a, chunk, axis=0)
    
    # Resample using polyphase filtering (linear phase, avoids IIR shifting)
    down = resample_poly(filt, up=1, down=int(fs_in/fs_out), axis=0)
    return down

def get_references(chunk, labels):
    refs = {}
    refs['Raw'] = chunk.copy()
    
    eeg_chans = [i for i, l in enumerate(labels) if 'EXG' not in l.upper() and 'STATUS' not in l.upper()]
    
    # Common Average Reference (CAR)
    if eeg_chans:
        car = chunk.copy()
        eeg_only = chunk[:, eeg_chans]
        car[:, eeg_chans] = eeg_only - np.mean(eeg_only, axis=1, keepdims=True)
        refs['CAR'] = car
        
    # Mastoid Reference
    mastoid_idx = [i for i, l in enumerate(labels) if l.upper() in ['EXG1', 'EXG2', 'M1', 'M2']]
    if mastoid_idx and eeg_chans:
        mast = chunk.copy()
        mast_mean = np.mean(chunk[:, mastoid_idx], axis=1, keepdims=True)
        mast[:, eeg_chans] = chunk[:, eeg_chans] - mast_mean
        refs['Mastoid'] = mast
        
    return refs

def run_validator():
    raw_path = Path("/kaggle/input/datasets/lowk1ee/raw-eeh/S1.mat")
    preproc_path = Path("/kaggle/input/dtu-eeg-data/S1_data_preproc.mat")

    out_dir = Path("/kaggle/working/reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "raw_reconstruction_report.md"

    with open(report_path, "w") as f_out:
        print_and_write(f_out, "# Raw ↔ Preprocessed Reconstruction Validator\n")
        
        # 1. Load Preprocessed Data
        if not preproc_path.exists():
            print_and_write(f_out, f"ERROR: {preproc_path} not found.")
            return
            
        print("Loading preprocessed data...")
        mat_pre = sio.loadmat(preproc_path, squeeze_me=True, struct_as_record=False)
        data_pre = mat_pre.get('data')
        
        if not data_pre or not hasattr(data_pre, 'eeg'):
            print_and_write(f_out, "ERROR: Invalid preprocessed struct.")
            return
            
        pre_trial_0 = np.asarray(data_pre.eeg[0, 0], dtype=float) # (channels, time)
        fs_pre = 64.0
        n_pre_samples = pre_trial_0.shape[1]
        
        print_and_write(f_out, f"**Preprocessed Target**: Trial 0")
        print_and_write(f_out, f"**Shape**: {pre_trial_0.shape} (Channels x Time)")
        print_and_write(f_out, f"**Duration**: {n_pre_samples / fs_pre:.2f} sec\n")
        
        # 2. Load Raw Data
        if not raw_path.exists():
            print_and_write(f_out, f"ERROR: {raw_path} not found.")
            return
            
        print("Loading raw data...")
        mat_raw = sio.loadmat(raw_path, squeeze_me=True, struct_as_record=False)
        data_raw = mat_raw.get('data')
        
        fs_raw = float(data_raw.fsample.eeg)
        raw_eeg = data_raw.eeg # (time, channels)
        
        events_val = data_raw.event.eeg.value
        events_sample = data_raw.event.eeg.sample
        
        vals = events_val.tolist() if hasattr(events_val, 'tolist') else list(events_val)
        samples = events_sample.tolist() if hasattr(events_sample, 'tolist') else list(events_sample)
        
        # Decode labels
        labels = []
        chan_obj = data_raw.dim.chan.eeg
        for x in chan_obj:
            if isinstance(x, np.ndarray) and x.size > 0: labels.append(str(x.flat[0]))
            else: labels.append(str(x))
            
        print_and_write(f_out, f"**Raw Data**: {raw_eeg.shape} @ {fs_raw} Hz\n")
        
        # 3. Search All Candidates
        print_and_write(f_out, "## Exhaustive Trigger Search\n")
        print_and_write(f_out, "Extracting continuous window around EVERY trigger occurrence, simulating preprocessing, and sliding cross-correlation against Preprocessed Trial 0...\n")
        
        best_overall = {
            'corr_mean': -1.0,
            'trigger': None,
            'occurrence': None,
            'reference': None,
            'lag_sec': None,
            'corr_median': None,
            'corr_max': None
        }
        
        unique_triggers = set(vals)
        
        for trig in unique_triggers:
            occurrences = [i for i, v in enumerate(vals) if v == trig]
            
            for occ_idx, ev_idx in enumerate(occurrences):
                start_samp = samples[ev_idx]
                
                # Extract chunk: -2 seconds to + (trial_length + 5) seconds
                pad_before = int(2.0 * fs_raw)
                pad_after = int((n_pre_samples / fs_pre + 5.0) * fs_raw)
                
                chunk_start = max(0, start_samp - pad_before)
                chunk_stop = min(raw_eeg.shape[0], start_samp + pad_after)
                
                if chunk_stop - chunk_start < int(n_pre_samples * (fs_raw / fs_pre)):
                    continue # Too short
                    
                chunk = raw_eeg[chunk_start:chunk_stop, :]
                
                # Apply references
                refs = get_references(chunk, labels)
                
                for ref_name, ref_chunk in refs.items():
                    # Simulate pipeline
                    chunk_64 = preprocess_chunk(ref_chunk, fs_in=fs_raw, fs_out=fs_pre)
                    
                    # Cross-correlate using Channel 0 to find lag
                    sig1 = chunk_64[:, 0]
                    sig2 = pre_trial_0[0, :]
                    
                    # Valid mode xcorr
                    if len(sig1) < len(sig2):
                        continue
                        
                    xcorr = correlate(sig1 - np.mean(sig1), sig2 - np.mean(sig2), mode='valid')
                    best_lag = np.argmax(xcorr)
                    
                    # Extract aligned 64Hz chunk
                    aligned = chunk_64[best_lag : best_lag + n_pre_samples, :]
                    
                    # Calculate multi-channel correlation
                    n_chans_to_test = min(64, aligned.shape[1], pre_trial_0.shape[0])
                    corrs = []
                    for ch in range(n_chans_to_test):
                        c_raw = aligned[:, ch]
                        c_pre = pre_trial_0[ch, :]
                        if np.std(c_raw) > 0 and np.std(c_pre) > 0:
                            corrs.append(np.corrcoef(c_raw, c_pre)[0, 1])
                            
                    if not corrs: continue
                    
                    c_mean = np.mean(corrs)
                    c_med = np.median(corrs)
                    c_max = np.max(corrs)
                    
                    if c_mean > best_overall['corr_mean']:
                        best_overall.update({
                            'corr_mean': c_mean,
                            'corr_median': c_med,
                            'corr_max': c_max,
                            'trigger': trig,
                            'occurrence': occ_idx,
                            'reference': ref_name,
                            'lag_sec': (best_lag - (pad_before * (fs_pre/fs_raw))) / fs_pre
                        })
                        
        print_and_write(f_out, "### **BEST MATCH FOUND**")
        print_and_write(f_out, f"- **Trigger**: {best_overall['trigger']}")
        print_and_write(f_out, f"- **Occurrence**: {best_overall['occurrence']}")
        print_and_write(f_out, f"- **Reference Strategy**: {best_overall['reference']}")
        print_and_write(f_out, f"- **Time Offset (from trigger)**: {best_overall['lag_sec']:.3f} seconds")
        print_and_write(f_out, f"- **Mean Channel Correlation**: {best_overall['corr_mean']:.4f}")
        print_and_write(f_out, f"- **Median Channel Correlation**: {best_overall['corr_median']:.4f}")
        print_and_write(f_out, f"- **Max Channel Correlation**: {best_overall['corr_max']:.4f}\n")
        
        if best_overall['corr_mean'] > 0.95:
            print_and_write(f_out, "✅ **SUCCESS:** Mathematical proof achieved. Raw reconstruction accurately mimics the released preprocessing pipeline. Ready for Raw Expansion Study.")
        elif best_overall['corr_mean'] > 0.80:
            print_and_write(f_out, "⚠️ **WARNING:** High correlation found, but not identical (>0.95). A different filter order, referencing scheme, or ICA artifact rejection was likely used by DTU.")
        else:
            print_and_write(f_out, "❌ **FAILURE:** Could not align raw data with preprocessed data. Do not proceed to experimental studies until this is resolved.")

if __name__ == "__main__":
    run_validator()
