import os
import argparse
import scipy.io
import numpy as np

def explore_struct(obj, prefix="", max_depth=3, current_depth=0):
    """Recursively explore a scipy.io loaded MATLAB struct/array."""
    if current_depth > max_depth:
        print(f"{prefix}...")
        return
        
    if isinstance(obj, np.ndarray):
        if obj.dtype.names is not None:
            print(f"{prefix}Struct Array: shape {obj.shape}, fields: {obj.dtype.names}")
            if obj.size == 1:
                scalar_obj = obj.item()
                for name in obj.dtype.names:
                    print(f"{prefix}  -> .{name}")
                    explore_struct(scalar_obj[name], prefix + "    ", max_depth, current_depth + 1)
        else:
            print(f"{prefix}Array: shape {obj.shape}, dtype {obj.dtype}")
            if obj.size > 0 and obj.size <= 20:
                print(f"{prefix}Values: {obj}")
            elif obj.ndim == 2 and obj.shape[1] <= 10:
                print(f"{prefix}First 5 rows:")
                for i in range(min(5, obj.shape[0])):
                    print(f"{prefix}  {obj[i]}")
    elif isinstance(obj, (int, float, str, np.number)):
        print(f"{prefix}Value: {obj} ({type(obj).__name__})")
    else:
        print(f"{prefix}Type: {type(obj)}")

def audit_eeglab_mat(file_path):
    print(f"====================================================")
    print(f"DEEP MAT FORENSICS (V2): {os.path.basename(file_path)}")
    print(f"====================================================")
    
    try:
        mat = scipy.io.loadmat(file_path, squeeze_me=True, struct_as_record=False)
    except Exception as e:
        print(f"Failed to load {file_path}: {e}")
        return

    eeg_var = None
    for k in mat.keys():
        if not k.startswith('__'):
            eeg_var = k
            break
            
    if not eeg_var:
        return
        
    eeg_struct = mat[eeg_var]
    
    print("\n--- CORE METADATA ---")
    for f in ['srate', 'nbchan', 'trials', 'pnts', 'xmin', 'xmax', 'fs', 'sampling_rate']:
        if hasattr(eeg_struct, f):
            print(f"{f}: {getattr(eeg_struct, f)}")
            
    print("\n--- DATA TENSOR ---")
    if hasattr(eeg_struct, 'data'):
        data = eeg_struct.data
        if isinstance(data, np.ndarray):
            print(f"Shape: {data.shape}")
            print(f"Dtype: {data.dtype}")
            print(f"Data matches 128Hz? {data.shape[1] / 60.0 == 128.0 if len(data.shape) > 1 else 'N/A'}")
            
    print("\n--- EVENTS ---")
    if hasattr(eeg_struct, 'event'):
        events = eeg_struct.event
        if isinstance(events, np.ndarray):
            print(f"Event Array Shape: {events.shape}")
            print(f"Event Array Dtype: {events.dtype}")
            if events.dtype.names is not None:
                print(f"Event Fields: {events.dtype.names}")
                print("\nFirst 20 Events:")
                fields = events.dtype.names
                for i in range(min(20, events.size)):
                    ev = events[i] if events.ndim == 1 else events[i, 0]
                    vals = [f"{getattr(ev, f, 'N/A')}" for f in fields]
                    print(f"{i}: " + " | ".join(vals))
            else:
                print("\nFirst 20 Events (Raw Array):")
                for i in range(min(20, events.shape[0])):
                    print(f"{i}: {events[i]}")
                    
                # If it's a 2D object array, maybe elements are inside
                if events.dtype == 'O' and events.size > 0:
                    print("\nInspecting first event element types:")
                    print([type(x) for x in events[0]])
    else:
        print("No 'event' field found.")

def main():
    parser = argparse.ArgumentParser(description="Deep MAT Forensics V2")
    parser.add_argument("--mat_file", type=str, required=True, help="Path to a single .mat file")
    args = parser.parse_args()
    
    if os.path.exists(args.mat_file):
        audit_eeglab_mat(args.mat_file)
    else:
        print(f"File not found: {args.mat_file}")
    
if __name__ == "__main__":
    main()
