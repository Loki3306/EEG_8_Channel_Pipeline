import os
import scipy.io
import numpy as np

data_root = '/kaggle/input/datasets/lokeshgile/aasd-processed-eeg/Processed EEG'
if not os.path.exists(data_root):
    data_root = '/kaggle/input/aasd-processed-eeg/Processed EEG'
    
import glob
mat_files = sorted(glob.glob(os.path.join(data_root, 'S*', 'S*.mat')))

print("Analyzing TIMING of FIRST button press of each trial...")
print(f"{'Subject':<10} {'Median Time (s)':<18} {'Min Time (s)':<18}")
print("-" * 50)

for mat_path in mat_files:
    subj_name = os.path.basename(os.path.dirname(mat_path))
    mat = scipy.io.loadmat(mat_path, simplify_cells=True)
    events = mat['EEG_new']['event']
    
    first_times = []
    
    current_trial = None
    first_button_found = False
    trial_start_latency = 0
    
    for ev in events:
        ev_type = str(ev[0])
        latency = int(ev[1])
        
        if ev_type.isdigit() and 1 <= int(ev_type) <= 60:
            current_trial = int(ev_type)
            trial_start_latency = latency
            first_button_found = False
            continue
            
        if current_trial is not None and not first_button_found:
            if ev_type in ['179', '184']:
                time_sec = (latency - trial_start_latency) / 128.0
                first_times.append(time_sec)
                first_button_found = True

    if first_times:
        median_t = np.median(first_times)
        min_t = np.min(first_times)
        print(f"{subj_name:<10} {median_t:<18.2f} {min_t:<18.2f}")
    else:
        print(f"{subj_name:<10} N/A                N/A")
