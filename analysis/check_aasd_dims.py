import scipy.io
import os
import glob
import numpy as np

def main():
    data_dir = "/kaggle/input/datasets/lokeshgile/aasd-processed-eeg/Processed EEG"
    mat_files = glob.glob(os.path.join(data_dir, '*', '*.mat'))
    
    print(f"Found {len(mat_files)} subjects.")
    
    for mf in sorted(mat_files):
        subj = os.path.basename(os.path.dirname(mf))
        mat = scipy.io.loadmat(mf, squeeze_me=True, struct_as_record=False)
        eeg_var = [k for k in mat.keys() if not k.startswith('__')][0]
        data = mat[eeg_var].data
        events = mat[eeg_var].event
        
        print(f"--- {subj} ---")
        print(f"Data shape: {data.shape}")
        
        if events.ndim > 1:
            first_ev = events[0]
            lat = float(first_ev[1])
            typ = str(first_ev[0]).strip()
        else:
            first_ev = events
            lat = float(getattr(first_ev, 'latency', 0))
            typ = str(getattr(first_ev, 'type', '')).strip()
            
        print(f"First event latency: {lat} (type: {typ})")
        if data.ndim == 2:
            print(f"Implied time (if 128Hz): {lat/128.0:.2f}s")
            print(f"Implied time (if 512Hz): {lat/512.0:.2f}s")

if __name__ == "__main__":
    main()
