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
            
            eeg_data = None
            if 'data' in mat_data:
                eeg_data = mat_data['data'].get('eeg', None)
            
            if eeg_data is None:
                print(f"  {subject_name}: Could not find 'eeg' field in 'data' struct.")
                continue
                
            # If it's a MATLAB cell array, in python it comes out as a list/array of arrays
            if isinstance(eeg_data, np.ndarray) and eeg_data.dtype == object:
                # Array of trials
                num_trials = len(eeg_data)
                subject_samples = 0
                for trial in eeg_data:
                    if hasattr(trial, 'shape'):
                        subject_samples += trial.shape[0] # Assuming (T, C)
                
            elif isinstance(eeg_data, list):
                num_trials = len(eeg_data)
                subject_samples = 0
                for trial in eeg_data:
                    if hasattr(trial, 'shape'):
                        subject_samples += trial.shape[0]
            else:
                # Maybe shape is (T, C, Trials)?
                print(f"  {subject_name}: Unexpected EEG shape {eeg_data.shape}")
                continue
                
            subject_seconds = subject_samples / fs
            
            print(f"  {subject_name:<10}: {num_trials:>3} trials, {subject_samples:>8} samples ({subject_seconds:>8.2f} sec)")
            
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
