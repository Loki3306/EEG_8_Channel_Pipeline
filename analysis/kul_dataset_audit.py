import os
import argparse
import numpy as np

try:
    import scipy.io as sio
except ImportError:
    print("Please install scipy")

try:
    import h5py
except ImportError:
    pass

def is_mat_struct(obj):
    return isinstance(obj, sio.matlab.mio5_params.mat_struct)

def get_fields(obj):
    if is_mat_struct(obj):
        return obj._fieldnames
    elif isinstance(obj, dict):
        return list(obj.keys())
    return []

def get_value(obj, field):
    if is_mat_struct(obj):
        return getattr(obj, field)
    elif isinstance(obj, dict):
        return obj[field]
    return None

def analyze_structure(obj, name="root", depth=0, max_depth=10, meta_hunt_keywords=None, found_meta=None):
    if depth > max_depth:
        return
        
    indent = "    " * depth
    if meta_hunt_keywords is None:
        meta_hunt_keywords = []
    if found_meta is None:
        found_meta = []
        
    lower_name = name.lower()
    for kw in meta_hunt_keywords:
        if kw in lower_name:
            found_meta.append(name)
            break

    if is_mat_struct(obj) or isinstance(obj, dict):
        fields = get_fields(obj)
        print(f"{indent}├── {name.split('.')[-1]} (struct/dict, {len(fields)} fields)")
        for f in fields:
            val = get_value(obj, f)
            analyze_structure(val, name + "." + f, depth + 1, max_depth, meta_hunt_keywords, found_meta)
    elif isinstance(obj, np.ndarray):
        print(f"{indent}├── {name.split('.')[-1]} (ndarray, shape={obj.shape}, dtype={obj.dtype})")
        if obj.dtype == object and obj.size > 0:
            val = obj.flatten()[0]
            if is_mat_struct(val) or isinstance(val, dict):
                print(f"{indent}│   [First element structure:]")
                analyze_structure(val, name + "[0]", depth + 1, max_depth, meta_hunt_keywords, found_meta)
    else:
        val_str = str(obj)
        if len(val_str) > 50:
            val_str = val_str[:47] + "..."
        print(f"{indent}├── {name.split('.')[-1]} ({type(obj).__name__}: {val_str})")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mat", type=str, required=True, help="Path to S1.mat")
    args = parser.parse_args()
    
    mat_path = args.mat
    if not os.path.exists(mat_path):
        print(f"File {mat_path} not found.")
        # If it's a directory, maybe find the .mat file
        if os.path.isdir(mat_path):
            print("Provided path is a directory. Searching for .mat files...")
            for f in os.listdir(mat_path):
                if f.endswith('.mat'):
                    print(f"Found {f}, trying to load it...")
                    mat_path = os.path.join(mat_path, f)
                    break
            else:
                return
        else:
            return
        
    print(f"Loading MAT file: {mat_path}")
    try:
        mat = sio.loadmat(mat_path, squeeze_me=True, struct_as_record=False)
        is_h5 = False
    except NotImplementedError:
        print("v7.3 detected. Loading with h5py...")
        mat = h5py.File(mat_path, 'r')
        is_h5 = True
        
    if is_h5:
        print("HDF5 recursive parsing...")
        keywords = ['attend', 'attention', 'ear', 'left', 'right', 'target', 'track', 'speaker', 'condition', 'experiment', 'label', 'correct']
        found_meta = []
        
        def h5_visitor(name, obj):
            indent = "    " * name.count('/')
            base_name = name.split('/')[-1]
            
            lower_name = base_name.lower()
            for kw in keywords:
                if kw in lower_name:
                    found_meta.append(name)
                    break
                    
            if isinstance(obj, h5py.Group):
                print(f"{indent}├── {base_name} (Group)")
            elif isinstance(obj, h5py.Dataset):
                print(f"{indent}├── {base_name} (Dataset, shape={obj.shape}, dtype={obj.dtype})")
                
        print("\n" + "="*50)
        print("SECTION A: MAT FILE STRUCTURE TREE")
        print("="*50)
        mat.visititems(h5_visitor)
        
        print("\n" + "="*50)
        print("SECTION D: METADATA HUNT")
        print("="*50)
        for m in sorted(list(set(found_meta))):
            print(f"MATCH: {m}")
            
        print("\nNote: For v7.3 HDF5 files, Sections B, C, E require manual indexing based on the printed tree above.")
        return

    clean_mat = {k: v for k, v in mat.items() if not k.startswith('__')}
    keywords = ['attend', 'attention', 'ear', 'left', 'right', 'target', 'track', 'speaker', 'condition', 'experiment', 'label', 'correct']
    found_meta = []
    
    print("\n" + "="*50)
    print("SECTION A: MAT FILE STRUCTURE TREE")
    print("="*50)
    for k, v in clean_mat.items():
        analyze_structure(v, k, depth=0, meta_hunt_keywords=keywords, found_meta=found_meta)
        
    print("\n" + "="*50)
    print("SECTION B: TRIAL INVENTORY")
    print("="*50)
    
    trials = None
    if 'trials' in clean_mat:
        trials = clean_mat['trials']
    elif 'trial' in clean_mat:
        trials = clean_mat['trial']
        
    if trials is not None:
        try:
            length = len(trials)
        except TypeError:
            length = 1
            trials = [trials]
            
        print(f"Number of trials: {length}")
        for i in range(min(3, length)):
            print(f"\n--- Trial {i} ---")
            t = trials[i]
            for f in get_fields(t):
                val = get_value(t, f)
                if isinstance(val, np.ndarray):
                    print(f"  {f}: ndarray, shape={val.shape}, dtype={val.dtype}")
                else:
                    print(f"  {f}: {type(val).__name__}, value={str(val)[:50]}")
    else:
        print("Variable 'trials' not found in root.")
        
    print("\n" + "="*50)
    print("SECTION C: EEG AUDIT")
    print("="*50)
    
    eeg_found = False
    if trials is not None and len(trials) > 0:
        t0 = trials[0]
        raw = get_value(t0, 'RawData')
        if raw is not None:
            eeg = get_value(raw, 'EegData')
            if eeg is not None:
                eeg_found = True
                print("Found trial[0].RawData.EegData")
                print(f"Shape: {eeg.shape}")
                print(f"Dtype: {eeg.dtype}")
                print(f"Min:   {np.min(eeg)}")
                print(f"Max:   {np.max(eeg)}")
                
                sh = eeg.shape
                if len(sh) >= 2:
                    ch = min(sh)
                    samples = max(sh)
                    print(f"Inferred Channels: {ch}")
                    print(f"Inferred Samples:  {samples}")
                else:
                    print("1D EEG array?")
                
                fs = get_value(raw, 'fs') or get_value(raw, 'Fs') or get_value(t0, 'fs') or get_value(t0, 'Fs')
                if fs is not None:
                    print(f"Found Sampling Rate (fs): {fs}")
                    try:
                        print(f"Inferred Trial Duration: {samples / float(fs):.2f} seconds")
                    except:
                        pass
                else:
                    print("Sampling rate 'fs' not found explicitly.")
                    
    if not eeg_found:
        print("trial.RawData.EegData not found directly. Will require manual path inspection.")
        
    print("\n" + "="*50)
    print("SECTION D: METADATA HUNT")
    print("="*50)
    
    unique_meta = sorted(list(set(found_meta)))
    for m in unique_meta:
        print(f"MATCH: {m}")
        
    print("\n" + "="*50)
    print("SECTION E: STIMULUS MAPPING AUDIT")
    print("="*50)
    
    if trials is not None:
        for i in range(min(10, len(trials))):
            t = trials[i]
            print(f"Trial {i}:")
            for f in get_fields(t):
                lower_f = f.lower()
                if 'stim' in lower_f or 'audio' in lower_f or 'part' in lower_f or 'track' in lower_f:
                    val = get_value(t, f)
                    if isinstance(val, np.ndarray) and val.dtype.kind in {'U', 'S'}:
                        print(f"  {f}: {val}")
                    elif isinstance(val, (str, int, float)):
                        print(f"  {f}: {val}")
                        
            for f in get_fields(t):
                val = get_value(t, f)
                if is_mat_struct(val):
                    for sub_f in get_fields(val):
                        lower_sub = sub_f.lower()
                        if 'stim' in lower_sub or 'audio' in lower_sub or 'part' in lower_sub or 'track' in lower_sub or 'name' in lower_sub or 'file' in lower_sub:
                            sub_val = get_value(val, sub_f)
                            if isinstance(sub_val, (str, int, float)) or (isinstance(sub_val, np.ndarray) and sub_val.dtype.kind in {'U','S'}):
                                print(f"  {f}.{sub_f}: {sub_val}")

if __name__ == "__main__":
    main()
