import scipy.io as sio
import h5py
import numpy as np

def inspect_mat(file_path):
    print(f"Inspecting {file_path} ...")
    try:
        mat = sio.loadmat(file_path)
        print("Successfully loaded with scipy.io.loadmat")
        for key in mat.keys():
            if not key.startswith('__'):
                val = mat[key]
                print(f"Key: {key}, Type: {type(val)}, Shape: {getattr(val, 'shape', 'N/A')}")
                if isinstance(val, np.ndarray) and val.dtype.names is not None:
                    print(f"  Fields: {val.dtype.names}")
    except Exception as e:
        print(f"scipy.io.loadmat failed: {e}")
        try:
            with h5py.File(file_path, 'r') as f:
                print("Successfully loaded with h5py")
                for key in f.keys():
                    print(f"Key: {key}, Type: {type(f[key])}")
                    if isinstance(f[key], h5py.Group):
                        for subkey in f[key].keys():
                            print(f"  Subkey: {subkey}, Shape: {getattr(f[key][subkey], 'shape', 'N/A')}")
        except Exception as e2:
            print(f"h5py failed as well: {e2}")

if __name__ == '__main__':
    inspect_mat('/kaggle/input/datasets/lokeshgile/aasd-processed-eeg/Processed EEG/S1/S1.mat')
    # Or try alternative paths if that fails
    import os
    if not os.path.exists('/kaggle/input/datasets/lokeshgile/aasd-processed-eeg/Processed EEG/S1/S1.mat'):
        print("\nCould not find file at exact path. Searching...")
        import glob
        files = glob.glob('/kaggle/input/**/*.mat', recursive=True)
        if len(files) > 0:
            print(f"Found alternative: {files[0]}")
            inspect_mat(files[0])
