"""
Phase 1: Dataset Forensics
Determines what is actually stored in the raw EEG files.
Automatically inspects keys, shapes, sampling rate, channels, event markers.
"""

import os
import sys
import h5py
import scipy.io as sio
from pathlib import Path

def inspect_mat(file_path):
    print(f"Inspecting {file_path}...")
    
    # Try loading with scipy.io first (older MATLAB format)
    try:
        mat = sio.loadmat(file_path, squeeze_me=True, struct_as_record=False)
        print("Loaded successfully with scipy.io (MATLAB v7.2 or older).")
        return _inspect_scipy_mat(mat)
    except NotImplementedError:
        print("scipy.io failed (likely v7.3). Trying h5py...")
        # v7.3 is HDF5
        try:
            with h5py.File(file_path, 'r') as f:
                print("Loaded successfully with h5py (MATLAB v7.3).")
                return _inspect_h5_mat(f)
        except Exception as e:
            return {"error": f"Failed to load with h5py: {str(e)}"}
    except Exception as e:
        return {"error": f"Failed to load with scipy.io: {str(e)}"}

def _inspect_scipy_mat(mat):
    info = {"keys": list(mat.keys())}
    # Usually EEGLAB or Fieldtrip data is in a struct called 'EEG' or 'data'
    target_keys = ['EEG', 'data', 'raw']
    for k in target_keys:
        if k in mat:
            d = mat[k]
            info[k] = {}
            if hasattr(d, '_fieldnames'):
                info[k]['fields'] = d._fieldnames
                for field in d._fieldnames:
                    val = getattr(d, field)
                    if hasattr(val, 'shape'):
                        info[k][f'{field}_shape'] = val.shape
                    elif isinstance(val, (int, float, str)):
                        info[k][f'{field}_val'] = val
            break
    return info

def _inspect_h5_mat(f):
    info = {"keys": list(f.keys())}
    def visitor(name, node):
        if isinstance(node, h5py.Dataset):
            info[f"Dataset: {name}"] = {"shape": node.shape, "dtype": str(node.dtype)}
            if node.size < 10:
                try:
                    info[f"Dataset: {name}"]["val"] = node[()]
                except:
                    pass
    f.visititems(visitor)
    return info

def generate_report(info, out_path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        f.write("# Raw EEG Structure Report\n\n")
        if "error" in info:
            f.write(f"**Error**: {info['error']}\n")
            return
            
        f.write("## File Keys\n")
        f.write(f"- {', '.join(info.get('keys', []))}\n\n")
        
        f.write("## Extracted Information\n")
        for k, v in info.items():
            if k == "keys": continue
            f.write(f"### {k}\n")
            if isinstance(v, dict):
                for sub_k, sub_v in v.items():
                    f.write(f"- **{sub_k}**: {sub_v}\n")
            else:
                f.write(f"{v}\n")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir", type=str, default="/kaggle/input/datasets/lokeshgile/raw-eeg", help="Directory containing raw S1.mat")
    parser.add_argument("--out_dir", type=str, default="/kaggle/working/reports", help="Directory to save report")
    args = parser.parse_args()
    
    raw_dir = Path(args.raw_dir)
    s1_path = raw_dir / "S1.mat"
    
    if not s1_path.exists():
        # Maybe it's directly in raw_dir
        mats = list(raw_dir.glob("*.mat"))
        if mats:
            s1_path = mats[0]
            print(f"S1.mat not found, using {s1_path.name} instead.")
        else:
            print(f"Error: No .mat files found in {raw_dir}")
            sys.exit(1)
            
    info = inspect_mat(s1_path)
    generate_report(info, Path(args.out_dir) / "raw_eeg_structure.md")
    print(f"Report saved to {args.out_dir}/raw_eeg_structure.md")
