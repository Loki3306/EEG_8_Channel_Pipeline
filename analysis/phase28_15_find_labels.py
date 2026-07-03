import os
import sys
import numpy as np
import scipy.io
import glob

def get_ev_attr(e, attr_name, array_idx=0):
    try:
        if hasattr(e, attr_name): return getattr(e, attr_name)
        if hasattr(e.flat[0], attr_name): return getattr(e.flat[0], attr_name)
        return e[array_idx]
    except: return ''

def main():
    print("==================================================")
    print("=== PHASE 28.15 LABEL DISCOVERY =================")
    print("==================================================\n")
    
    S1_mat = '/kaggle/input/datasets/lokeshgile/aasd-processed-eeg/Processed EEG/S01/S1.mat'
    if not os.path.exists(S1_mat):
        matches = glob.glob('/kaggle/input/datasets/lokeshgile/aasd-processed-eeg/Processed EEG/*/*.mat')
        if not matches:
            print("ERROR: S1.mat not found. Please run on Kaggle.")
            return
        S1_mat = matches[0]
        
    mat = scipy.io.loadmat(S1_mat, squeeze_me=True, struct_as_record=False)
    eeg_var = [k for k in mat.keys() if not k.startswith('__')][0]
    events = mat[eeg_var].event
    
    all_types = set()
    for ev in events:
        t_str = str(get_ev_attr(ev, 'type', 0)).strip()
        all_types.add(t_str)
        
    print(f"Unique event types in entire dataset:\n{sorted(list(all_types))}\n")
    
    # Find latencies of audio markers 11-70
    audio_markers = []
    for ev in events:
        t_str = str(get_ev_attr(ev, 'type', 0)).strip()
        if t_str.isdigit():
            val = int(t_str)
            if 11 <= val <= 70:
                lat = int(get_ev_attr(ev, 'latency'))
                audio_markers.append((t_str, lat))
                
    audio_markers.sort(key=lambda x: x[1])
    
    print("==================================================")
    print("=== TRIAL 1 TIMELINE (LATENCY BOUNDED) ==========")
    print("==================================================")
    
    if len(audio_markers) >= 2:
        t1_start = audio_markers[0][1]
        t2_start = audio_markers[1][1]
        
        print(f"Trial 1 Audio Marker: {audio_markers[0][0]} at latency {t1_start}")
        print(f"Trial 2 Audio Marker: {audio_markers[1][0]} at latency {t2_start}")
        print("Events between t1_start and t2_start:")
        
        for ev in events:
            lat = int(get_ev_attr(ev, 'latency') or 0)
            if t1_start <= lat < t2_start:
                t_str = str(get_ev_attr(ev, 'type', 0)).strip()
                ep = str(get_ev_attr(ev, 'epoch'))
                t_sec = (lat - t1_start) / 128.0  # Assuming 128Hz original sampling rate
                print(f"  {t_sec:6.2f}s | Type: {t_str:<5} | Epoch field: '{ep}'")
                
    print("\nIf 179 and 184 don't show up here, they don't exist in AASD!")

if __name__ == "__main__":
    main()
