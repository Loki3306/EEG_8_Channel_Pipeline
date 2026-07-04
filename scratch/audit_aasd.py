import os
import scipy.io
import numpy as np

def audit_aasd():
    print("=======================================================")
    print(" AASD DATASET DEEP AUDIT ")
    print("=======================================================")
    
    data_root = '/kaggle/input/datasets/lokeshgile/aasd-processed-eeg/Processed EEG'
    if not os.path.exists(data_root):
        print(f"Directory {data_root} not found. Ensure this is run on Kaggle.")
        return
        
    mat_files = []
    for root, dirs, files in os.walk(data_root):
        for file in files:
            if file.endswith('.mat') and not file.startswith('._'):
                mat_files.append(os.path.join(root, file))
                
    mat_files.sort()
    
    total_subjects = len(mat_files)
    total_trials = 0
    total_samples = 0
    total_seconds = 0.0
    
    fs = 64 # Based on DATASETS.md
    
    print(f"Found {total_subjects} subject files.")
    print("-" * 55)
    
    for mat_file in mat_files:
        try:
            mat_data = scipy.io.loadmat(mat_file, simplify_cells=True)
            subject_name = os.path.basename(mat_file)
            
            eeg_var = [k for k in mat_data.keys() if not k.startswith('__')][0]
            
            # Since simplify_cells=True, mat_data[eeg_var] is a dict
            eeg_struct = mat_data[eeg_var]
            
            if 'data' not in eeg_struct:
                print(f"  {subject_name}: Could not find 'data' field in struct {eeg_var}.")
                continue
                
            eeg_data = eeg_struct['data']
            
            if not isinstance(eeg_data, np.ndarray):
                print(f"  {subject_name}: 'data' is not a numpy array.")
                continue
                
            if eeg_data.ndim == 3:
                # Shape is usually (Channels, Time, Trials) in EEGLAB
                num_trials = eeg_data.shape[2]
                subject_samples = eeg_data.shape[1] * num_trials
                # The data in AASD is stored at 128 Hz before resampling in the script, wait!
                # Actually, in the preprocessing script it says downsample to 64Hz.
                # Let's just calculate based on actual samples.
                # In phase32_5_spatial_fix.py, they did `resample_poly(trial_eeg, 1, 2, axis=1)` meaning original fs is 128Hz!
                actual_fs = 128
            else:
                print(f"  {subject_name}: Unexpected EEG shape {eeg_data.shape}")
                continue
                
            subject_seconds = subject_samples / actual_fs
            
            print(f"  {subject_name:<10}: {num_trials:>3} trials, {subject_samples:>8} samples ({subject_seconds:>8.2f} sec) [Orig Fs={actual_fs}Hz]")
            
            total_trials += num_trials
            total_samples += subject_samples
            total_seconds += subject_seconds
            
        except Exception as e:
            print(f"  Error reading {os.path.basename(mat_file)}: {e}")
            
    print("-" * 55)
    print(f"TOTAL SUBJECTS: {total_subjects}")
    print(f"TOTAL TRIALS  : {total_trials}")
    print(f"TOTAL SAMPLES : {total_samples}")
    print(f"TOTAL TIME    : {total_seconds:.2f} seconds ({total_seconds / 3600:.2f} hours)")
    print("=======================================================")

if __name__ == '__main__':
    audit_aasd()
