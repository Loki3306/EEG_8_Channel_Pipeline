import os
import glob
import numpy as np
import scipy.io
import scipy.signal

def get_ev_attr(e, attr_name, array_idx=0):
    try:
        if hasattr(e, attr_name): return getattr(e, attr_name)
        if hasattr(e.flat[0], attr_name): return getattr(e.flat[0], attr_name)
        return e[array_idx]
    except: return ''

def extract_erp(mat_path):
    print(f"\n--- Checking ERPs in {os.path.basename(mat_path)} ---")
    
    mat = scipy.io.loadmat(mat_path, squeeze_me=True, struct_as_record=False)
    eeg_var = [k for k in mat.keys() if not k.startswith('__')][0]
    eeg_struct = mat[eeg_var]
    
    fs = 128
    if hasattr(eeg_struct, 'srate'):
        fs = float(eeg_struct.srate)
        
    eeg_data = eeg_struct.data # (channels, time)
    events = eeg_struct.event
    
    # Identify Audio Start Markers (11 to 70)
    audio_latencies = []
    for ev in events:
        ev_type = str(get_ev_attr(ev, 'type', 0)).strip()
        if ev_type.isdigit():
            val = int(ev_type)
            if 11 <= val <= 70:
                lat = int(get_ev_attr(ev, 'latency', 1))
                audio_latencies.append(lat)
                
    if not audio_latencies:
        print("ERROR: No audio start markers found in this file.")
        return
        
    print(f"Found {len(audio_latencies)} Audio Start Markers.")
    
    # Bandpass filter the EEG for ERPs (1-15 Hz is standard for AEPs)
    nyq = 0.5 * fs
    b, a = scipy.signal.butter(4, [1.0/nyq, 15.0/nyq], btype='band')
    eeg_filt = scipy.signal.filtfilt(b, a, eeg_data, axis=1)
    
    # Epoching
    pre_sec = 0.2
    post_sec = 0.8
    pre_samples = int(pre_sec * fs)
    post_samples = int(post_sec * fs)
    
    epochs = []
    for lat in audio_latencies:
        start = lat - pre_samples
        end = lat + post_samples
        if start >= 0 and end < eeg_filt.shape[1]:
            # Extract 8 specific channels or all? Let's use the 8 AAD channels if available
            # If not, just average all channels to maximize SNR for the ERP
            epoch = eeg_filt[:8, start:end]
            
            # Baseline correction
            baseline = np.mean(epoch[:, :pre_samples], axis=1, keepdims=True)
            epoch = epoch - baseline
            epochs.append(epoch)
            
    if not epochs:
        print("ERROR: All epochs out of bounds.")
        return
        
    epochs = np.array(epochs) # (trials, channels, time)
    grand_avg = np.mean(epochs, axis=(0, 1)) # (time,)
    
    time_axis = np.linspace(-pre_sec, post_sec, len(grand_avg))
    
    # Measure SNR (Peak amplitude in 50-250ms vs standard deviation of baseline)
    baseline_std = np.std(grand_avg[:pre_samples])
    
    t_start_idx = pre_samples + int(0.05 * fs)
    t_end_idx = pre_samples + int(0.25 * fs)
    
    if t_end_idx > len(grand_avg):
        t_end_idx = len(grand_avg)
        
    erp_window = grand_avg[t_start_idx:t_end_idx]
    max_amp = np.max(np.abs(erp_window))
    
    snr = max_amp / (baseline_std + 1e-9)
    
    print(f"Baseline STD: {baseline_std:.4f} uV")
    print(f"Max ERP Amplitude (50-250ms): {max_amp:.4f} uV")
    print(f"ERP Signal-to-Noise Ratio: {snr:.2f}")
    
    if snr > 3.0:
        print("DIAGNOSIS: STRONG ERP DETECTED.")
        print("The EEG markers are perfectly synchronized with the actual audio delivery.")
    elif snr > 1.5:
        print("DIAGNOSIS: WEAK ERP DETECTED.")
        print("The EEG markers are loosely synchronized, but the signal is very noisy.")
    else:
        print("DIAGNOSIS: NO ERP DETECTED (SNR < 1.5).")
        print("The EEG triggers are completely detached from the physical reality of the audio.")
        
    # Print a few points around 100ms and 200ms
    t_100_idx = pre_samples + int(0.10 * fs)
    t_200_idx = pre_samples + int(0.20 * fs)
    print(f"Amplitude at 100ms: {grand_avg[t_100_idx]:.4f} uV")
    print(f"Amplitude at 200ms: {grand_avg[t_200_idx]:.4f} uV")

def main():
    print("[INFO] Starting Phase 28.7 Neurophysiological ERP Audit")
    
    mat_files = glob.glob('/kaggle/input/datasets/lokeshgile/aasd-processed-eeg/Processed EEG/*/*.mat')
    if not mat_files:
        print("No .mat files found.")
        return
        
    mat_files = sorted(mat_files)
    for f in mat_files[:2]:
        extract_erp(f)

if __name__ == "__main__":
    main()
