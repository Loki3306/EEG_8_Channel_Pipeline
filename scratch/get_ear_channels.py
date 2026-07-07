import scipy.io
import os
import glob

def main():
    # Find any subject's MAT file
    base_dir = '/kaggle/input/datasets/lokeshgile/aasd-processed-eeg/Processed EEG'
    mat_files = glob.glob(f'{base_dir}/*/*.mat')
    
    if not mat_files:
        print("Could not find MAT files. Are you on Kaggle?")
        return
        
    mat_path = mat_files[0]
    print(f"Loading {mat_path} to read channel labels...")
    
    mat = scipy.io.loadmat(mat_path, squeeze_me=True, struct_as_record=False)
    eeg_var = [k for k in mat.keys() if not k.startswith('__')][0]
    eeg_obj = mat[eeg_var]
    
    if not hasattr(eeg_obj, 'chanlocs'):
        print("No chanlocs found!")
        return
        
    chanlocs = eeg_obj.chanlocs
    channel_names = [getattr(ch, 'labels', str(ch)).upper() for ch in chanlocs]
    
    target_channels = ['T7', 'T8', 'TP7', 'TP8', 'FT7', 'FT8', 'P7', 'P8']
    
    print("\nTarget Ear-EEG Channels:")
    indices = []
    for target in target_channels:
        if target in channel_names:
            idx = channel_names.index(target)
            indices.append(idx)
            print(f"Found {target} at Index {idx}")
        else:
            print(f"WARNING: {target} not found in montage!")
            
    print(f"\nPhase 59 Python Array:\nEAR_CHANNEL_INDICES = {indices}")

if __name__ == "__main__":
    main()
