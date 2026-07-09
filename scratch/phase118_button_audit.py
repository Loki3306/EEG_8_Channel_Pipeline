import os
import scipy.io

data_root = '/kaggle/input/datasets/lokeshgile/aasd-processed-eeg/Processed EEG'
if not os.path.exists(data_root):
    data_root = '/kaggle/input/aasd-processed-eeg/Processed EEG'
    
import glob
mat_files = sorted(glob.glob(os.path.join(data_root, 'S*', 'S*.mat')))

print(f"Found {len(mat_files)} subjects. Analyzing button presses...")

print(f"{'Subject':<10} {'179 (Left?)':<15} {'184 (Right?)':<15} {'Other Buttons':<15}")
print("-" * 55)

for mat_path in mat_files:
    subj_name = os.path.basename(os.path.dirname(mat_path))
    
    mat = scipy.io.loadmat(mat_path, simplify_cells=True)
    events = mat['EEG_new']['event']
    
    count_179 = 0
    count_184 = 0
    count_other = {}
    
    for ev in events:
        ev_type = str(ev[0])
        if ev_type == '179':
            count_179 += 1
        elif ev_type == '184':
            count_184 += 1
        elif not ev_type.isdigit():
            # Trigger codes are usually digits in this dataset, skip others if any
            pass
        else:
            # Check if it's an audio trial start (1-60)
            if int(ev_type) > 60:
                count_other[ev_type] = count_other.get(ev_type, 0) + 1
                
    other_str = ", ".join([f"{k}:{v}" for k, v in count_other.items()])
    if not other_str: other_str = "None"
    
    print(f"{subj_name:<10} {count_179:<15} {count_184:<15} {other_str:<15}")
