import scipy.io
import os
import glob

def main():
    base_dir = '/kaggle/input/datasets/lokeshgile/aasd-processed-eeg/Processed EEG'
    mat_files = glob.glob(f'{base_dir}/*/*.mat')
    
    if not mat_files:
        print("Could not find MAT files. Are you on Kaggle?")
        return
        
    mat_path = mat_files[0]
    print(f"Loading {mat_path} to read channel labels...")
    
    # Use simplify_cells=True to properly parse the nested structs
    mat = scipy.io.loadmat(mat_path, simplify_cells=True)
    
    if 'EEG' in mat and 'chanlocs' in mat['EEG']:
        chanlocs = mat['EEG']['chanlocs']
        channel_names = [c['labels'].upper() for c in chanlocs]
    else:
        print("No chanlocs found in mat['EEG']!")
        return
        
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
