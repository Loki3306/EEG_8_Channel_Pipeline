import os
import scipy.io

data_root = '/kaggle/input/datasets/lokeshgile/aasd-processed-eeg/Processed EEG'
if not os.path.exists(data_root):
    data_root = '/kaggle/input/aasd-processed-eeg/Processed EEG'
    
import glob
mat_files = sorted(glob.glob(os.path.join(data_root, 'S*', 'S*.mat')))

print("Analyzing FIRST button press of each trial...")
print(f"{'Subject':<10} {'First=179 (Left)':<18} {'First=184 (Right)':<18}")
print("-" * 50)

for mat_path in mat_files:
    subj_name = os.path.basename(os.path.dirname(mat_path))
    mat = scipy.io.loadmat(mat_path, simplify_cells=True)
    events = mat['EEG_new']['event']
    
    first_179 = 0
    first_184 = 0
    
    # We need to group events by trial
    # A trial starts with an event from '1' to '60'
    current_trial = None
    first_button_found = False
    
    for ev in events:
        ev_type = str(ev[0])
        
        if ev_type.isdigit() and 1 <= int(ev_type) <= 60:
            current_trial = int(ev_type)
            first_button_found = False
            continue
            
        if current_trial is not None and not first_button_found:
            if ev_type == '179':
                first_179 += 1
                first_button_found = True
            elif ev_type == '184':
                first_184 += 1
                first_button_found = True

    print(f"{subj_name:<10} {first_179:<18} {first_184:<18}")
