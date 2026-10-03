import scipy.io
import glob
import json
import numpy as np
from pathlib import Path
import os

def unwrap_singleton(value):
    current = value
    while isinstance(current, np.ndarray) and current.size == 1:
        current = current[0, 0] if current.ndim == 2 else current.flat[0]
    return current

def get_authoritative_mapping():
    # Support Kaggle path or fallback
    if Path("/kaggle/input/datasets/lokeshgile/dataset-eeg").exists():
        base_dir = '/kaggle/input/datasets/lokeshgile/dataset-eeg'
    else:
        base_dir = '/kaggle/input'
        
    mat_files = glob.glob(f'{base_dir}/**/*.mat', recursive=True)
    
    if not mat_files:
        print(f"Could not find MAT files in {base_dir}")
        return
        
    mat_path = mat_files[0]
    print(f"Reading authoritative chanlocs from: {mat_path}\n")
    
    mat = scipy.io.loadmat(mat_path, squeeze_me=False, struct_as_record=False)
    
    try:
        data = mat["data"][0, 0]
        chan = data.dim[0, 0].chan[0, 0].eeg[0, 0]
        
        channel_names = []
        for index in range(chan.shape[1]):
            item = chan[0, index]
            item = unwrap_singleton(item)
            if isinstance(item, np.ndarray):
                channel_names.append(str(np.asarray(item).squeeze()).upper())
            else:
                channel_names.append(str(item).upper())
                
        print(f"Total Channels Found: {len(channel_names)}\n")
        
        mapping = {}
        for idx, name in enumerate(channel_names):
            mapping[idx] = name
            print(f"Index {idx:2d} -> {name}")
            
        print("\n--- Python Dictionary Format ---")
        print(json.dumps(mapping, indent=4))
        
        # Save to disk as well
        out_path = Path("dtu_64_channel_mapping.json")
        with open(out_path, "w") as f:
            json.dump(mapping, f, indent=4)
        print(f"\nSaved mapping to {out_path.absolute()}")
        
    except Exception as e:
        print(f"Error extracting channels: {e}")

if __name__ == "__main__":
    get_authoritative_mapping()
