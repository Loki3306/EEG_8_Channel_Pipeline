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
    print("=== PHASE 28.17 LOCATE MARKERS ==================")
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
    
    print("Searching for 179, 184, 254, 255 in events...\n")
    
    found = 0
    for i, ev in enumerate(events):
        t_str = str(get_ev_attr(ev, 'type', 0)).strip()
        if t_str in ['179', '184', '254', '255']:
            lat = str(get_ev_attr(ev, 'latency')).strip()
            ep = str(get_ev_attr(ev, 'epoch')).strip()
            print(f"Event {i:4d}: Type = {t_str:<3} | Latency = {lat:<6} | Epoch = '{ep}'")
            found += 1
            if found >= 20:
                print("... stopping after 20 found to keep logs short.")
                break
                
    if found == 0:
        print("CRITICAL ERROR: Markers not found in events array!")
        
    # Also print the first 20 events of ANY type to see the structure of epoch fields
    print("\n==================================================")
    print("First 20 events in the dataset:")
    for i, ev in enumerate(events[:20]):
        t_str = str(get_ev_attr(ev, 'type', 0)).strip()
        lat = str(get_ev_attr(ev, 'latency')).strip()
        ep = str(get_ev_attr(ev, 'epoch')).strip()
        print(f"Event {i:4d}: Type = {t_str:<3} | Latency = {lat:<6} | Epoch = '{ep}'")

if __name__ == "__main__":
    main()
