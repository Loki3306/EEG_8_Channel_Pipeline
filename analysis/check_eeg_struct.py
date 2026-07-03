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
        
    s18_path = next((mf for mf in mat_files if 'S18' in mf), mat_files[0])
    mat = scipy.io.loadmat(s18_path, squeeze_me=True, struct_as_record=False)
    eeg_var = [k for k in mat.keys() if not k.startswith('__')][0]
    
    print(f"Keys in mat: {mat.keys()}")
    
    eeg_obj = mat[eeg_var]
    print(f"Attributes of eeg_obj:")
    for attr in dir(eeg_obj):
        if not attr.startswith('_'):
            print(attr)
            
if __name__ == "__main__":
    main()
