import scipy.io
import os
import glob
import numpy as np

def main():
    data_dir = "/kaggle/input/datasets/lokeshgile/aasd-processed-eeg/Processed EEG"
    mat_files = glob.glob(os.path.join(data_dir, '*', '*.mat'))
    
    if not mat_files:
        print("No mat files found.")
        return
        
    mf = mat_files[0]
    subj = os.path.basename(os.path.dirname(mf))
    mat = scipy.io.loadmat(mf, squeeze_me=True, struct_as_record=False)
    eeg_var = [k for k in mat.keys() if not k.startswith('__')][0]
    
    chan_names = []
    try:
        chanlocs = mat[eeg_var].chanlocs
        if isinstance(chanlocs, np.ndarray):
            for c in chanlocs:
                lbl = getattr(c, 'labels', '')
                if isinstance(lbl, (list, np.ndarray)) and len(lbl) > 0:
                    lbl = lbl[0]
                chan_names.append(str(lbl).strip().upper())
        print(f"Channels for {subj}:")
        for i, c in enumerate(chan_names):
            print(f"{i}: {c}")
            
        target_channels = ['T7', 'C2', 'FT8', 'P7', 'CPz', 'Fp1', 'TP8', 'C3']
        print("\nTarget Indices:")
        for tc in target_channels:
            if tc.upper() in chan_names:
                print(f"{tc}: {chan_names.index(tc.upper())}")
            else:
                print(f"{tc}: NOT FOUND")
    except Exception as e:
        print(f"Failed to read chanlocs: {e}")

if __name__ == "__main__":
    main()
