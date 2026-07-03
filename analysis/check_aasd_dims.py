import scipy.io
import os
import glob
import numpy as np

def main():
    data_dir = "/kaggle/input/datasets/lokeshgile/aasd-processed-eeg/Processed EEG"
    mat_files = glob.glob(os.path.join(data_dir, '*', '*.mat'))
    
    for mf in sorted(mat_files)[:3]:
        subj = os.path.basename(os.path.dirname(mf))
        mat = scipy.io.loadmat(mf, squeeze_me=True, struct_as_record=False)
        eeg_var = [k for k in mat.keys() if not k.startswith('__')][0]
        data = mat[eeg_var].data
        events = mat[eeg_var].event
        
        print(f"--- {subj} ---")
        if events.ndim > 1:
            epoch1_events = [str(ev[0]).strip() for ev in events if int(ev[4]) == 1]
        else:
            epoch1_events = [str(getattr(events, 'type', '')).strip()] if int(getattr(events, 'epoch', 0)) == 1 else []
            
        print(f"Epoch 1 Event Types: {epoch1_events}")

if __name__ == "__main__":
    main()
