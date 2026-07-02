import os
import argparse
import scipy.io
import numpy as np

def explore_struct(obj, prefix=""):
    """Recursively explore a scipy.io loaded MATLAB struct/array."""
    if isinstance(obj, np.ndarray):
        if obj.dtype.names is not None:
            # It's a struct array
            print(f"{prefix}Struct Array: shape {obj.shape}, fields: {obj.dtype.names}")
            # If it's a scalar struct (shape () or (1,1)), go deeper
            if obj.size == 1:
                scalar_obj = obj.item()
                for name in obj.dtype.names:
                    print(f"{prefix}  -> .{name}")
                    explore_struct(scalar_obj[name], prefix + "    ")
        else:
            # Regular numeric/cell array
            print(f"{prefix}Array: shape {obj.shape}, dtype {obj.dtype}")
            if obj.size > 0 and obj.size < 10:
                print(f"{prefix}Values: {obj}")
    elif isinstance(obj, (int, float, str, np.number)):
        print(f"{prefix}Value: {obj} ({type(obj).__name__})")
    else:
        print(f"{prefix}Type: {type(obj)}")

def audit_eeglab_mat(file_path):
    print(f"====================================================")
    print(f"DEEP MAT FORENSICS: {os.path.basename(file_path)}")
    print(f"====================================================")
    
    try:
        mat = scipy.io.loadmat(file_path, squeeze_me=True, struct_as_record=False)
    except Exception as e:
        print(f"Failed to load {file_path}: {e}")
        return

    # Find the main EEG variable
    eeg_var = None
    for k in mat.keys():
        if not k.startswith('__'):
            eeg_var = k
            break
            
    if not eeg_var:
        print("No data variables found.")
        return
        
    print(f"\nMain Variable: {eeg_var}")
    eeg_struct = mat[eeg_var]
    
    # Extract standard EEGLAB fields if they exist
    print("\n--- CORE METADATA ---")
    fields = dir(eeg_struct)
    
    for f in ['srate', 'nbchan', 'trials', 'pnts', 'xmin', 'xmax']:
        if hasattr(eeg_struct, f):
            print(f"{f}: {getattr(eeg_struct, f)}")
            
    print("\n--- DATA TENSOR ---")
    if hasattr(eeg_struct, 'data'):
        data = eeg_struct.data
        print(f"Shape: {data.shape}")
        print(f"Dtype: {data.dtype}")
    else:
        print("No 'data' field found.")
        
    print("\n--- EVENTS ---")
    if hasattr(eeg_struct, 'event'):
        events = eeg_struct.event
        print(f"Event Array Shape: {events.shape if isinstance(events, np.ndarray) else 'scalar'}")
        
        if isinstance(events, np.ndarray) and events.size > 0:
            sample_event = events[0]
            fields = dir(sample_event) if not hasattr(events, 'dtype') or events.dtype.names is None else events.dtype.names
            if hasattr(sample_event, '_fieldnames'):
                fields = sample_event._fieldnames
            
            print(f"Event Fields: {fields}")
            print("\nFirst 20 Events:")
            print(f"{'Index':<6} | " + " | ".join([f"{f:<15}" for f in fields]))
            print("-" * 80)
            
            for i in range(min(20, events.size)):
                ev = events[i]
                vals = []
                for f in fields:
                    val = getattr(ev, f, 'N/A')
                    # truncate long strings/arrays
                    val_str = str(val)
                    if len(val_str) > 15:
                        val_str = val_str[:12] + "..."
                    vals.append(f"{val_str:<15}")
                print(f"{i:<6} | " + " | ".join(vals))
                
            # Count event types
            if 'type' in fields or hasattr(events[0], 'type'):
                print("\nEvent Type Distribution:")
                type_counts = {}
                for ev in events:
                    t = str(getattr(ev, 'type', 'N/A'))
                    type_counts[t] = type_counts.get(t, 0) + 1
                for t, c in type_counts.items():
                    print(f"  {t}: {c}")
    else:
        print("No 'event' field found.")
        
    print("\n--- FULL STRUCTURE TREE ---")
    # Doing a full scipy struct_as_record=True load to safely explore types
    mat_rec = scipy.io.loadmat(file_path, squeeze_me=True, struct_as_record=True)
    explore_struct(mat_rec[eeg_var])

def main():
    parser = argparse.ArgumentParser(description="Deep MAT Forensics")
    parser.add_argument("--mat_file", type=str, required=True, help="Path to a single .mat file (e.g. S1.mat)")
    args = parser.parse_args()
    
    if not os.path.exists(args.mat_file):
        print(f"File not found: {args.mat_file}")
        return
        
    audit_eeglab_mat(args.mat_file)
    
if __name__ == "__main__":
    main()
