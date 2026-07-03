import os
import glob
import numpy as np
import scipy.io

def check_audio_file(filepath):
    print(f"\n--- Checking {os.path.basename(filepath)} ---")
    
    try:
        data = np.load(filepath)
    except Exception as e:
        print(f"Failed to load: {e}")
        return
        
    if 'env_l' not in data or 'env_r' not in data:
        print("MISSING KEYS: Expected 'env_l' and 'env_r'")
        print("Found keys:", list(data.keys()))
        return
        
    env_l = data['env_l']
    env_r = data['env_r']
    
    print(f"Shape L: {env_l.shape}, Shape R: {env_r.shape}")
    
    if len(env_l) == 0 or len(env_r) == 0:
        print("ERROR: Empty arrays")
        return
        
    # Check RMS Power
    rms_l = np.sqrt(np.mean(env_l**2))
    rms_r = np.sqrt(np.mean(env_r**2))
    print(f"RMS L: {rms_l:.4f} | RMS R: {rms_r:.4f}")
    
    if rms_l == 0 or rms_r == 0:
        print("WARNING: Zero RMS power detected. Audio is silent.")
        
    # Check Correlation between L and R (Separation check)
    if rms_l > 0 and rms_r > 0:
        min_len = min(len(env_l), len(env_r))
        corr = np.corrcoef(env_l[:min_len], env_r[:min_len])[0, 1]
        print(f"L/R Correlation: {corr:.4f}")
        
        if corr > 0.5:
            print("  -> DIAGNOSIS: STEREO MIX DETECTED! (L and R are highly correlated)")
            print("  -> These are NOT isolated speakers. They are mixed audio tracks.")
        elif corr < 0.2:
            print("  -> DIAGNOSIS: ISOLATED SPEAKERS. (L and R are independent)")
            
    # Print stats of first 5 seconds (assuming 64Hz = 320 samples)
    s_l = env_l[:320]
    s_r = env_r[:320]
    print(f"First 5s L - Mean: {np.mean(s_l):.4f}, Std: {np.std(s_l):.4f}, Min: {np.min(s_l):.4f}, Max: {np.max(s_l):.4f}")
    print(f"First 5s R - Mean: {np.mean(s_r):.4f}, Std: {np.std(s_r):.4f}, Min: {np.min(s_r):.4f}, Max: {np.max(s_r):.4f}")

def check_metadata():
    print("\n--- Checking Dataset Metadata (S1) ---")
    mat_files = glob.glob('/kaggle/input/datasets/lokeshgile/aasd-processed-eeg/Processed EEG/*/*.mat')
    if not mat_files:
        print("No .mat files found.")
        return
        
    S1_path = next((f for f in mat_files if 'S1.mat' in f or 'S01' in f), mat_files[0])
    print(f"Loading {os.path.basename(S1_path)}")
    
    mat = scipy.io.loadmat(S1_path, squeeze_me=True, struct_as_record=False)
    eeg_var = [k for k in mat.keys() if not k.startswith('__')][0]
    events = mat[eeg_var].event
    
    def get_ev_attr(e, attr_name, array_idx=0):
        try:
            if hasattr(e, attr_name): return getattr(e, attr_name)
            if hasattr(e.flat[0], attr_name): return getattr(e.flat[0], attr_name)
            return e[array_idx]
        except: return ''
        
    types = [str(get_ev_attr(ev, 'type', 0)).strip() for ev in events]
    from collections import Counter
    counts = Counter(types)
    print("Most common event markers:")
    for k, v in counts.most_common():
        print(f"  Marker {k}: {v} times")
        
    # Check what the audio markers are
    audio_markers = [t for t in types if t.isdigit() and int(t) < 100]
    print(f"\nPotential Audio Stimulus Markers found: {set(audio_markers)}")

def main():
    print("[INFO] Starting Phase 28.6 Audio Integrity Audit")
    
    # 1. Check Audio Files
    audio_dir = '/kaggle/input/datasets/lokeshgile/aasd-audio-gammatones'
    if not os.path.exists(audio_dir):
        print(f"Directory not found: {audio_dir}")
    else:
        npz_files = glob.glob(os.path.join(audio_dir, '*.npz'))
        print(f"Found {len(npz_files)} .npz files.")
        
        # Test first 3 files
        for f in sorted(npz_files)[:3]:
            check_audio_file(f)
            
    # 2. Check Metadata Events
    check_metadata()

if __name__ == "__main__":
    main()
