import os
import re
import glob
import scipy.io as sio
import collections
import numpy as np

DATA_DIR = "/kaggle/input/datasets/lokeshgile/eeg-audio"

def print_tree(startpath, max_depth=2):
    print(f"\n--- FOLDER STRUCTURE (Max Depth {max_depth}) ---")
    if not os.path.exists(startpath):
        print(f"Path not found: {startpath}")
        return
    for root, dirs, files in os.walk(startpath):
        level = root.replace(startpath, '').count(os.sep)
        if level > max_depth:
            continue
        indent = ' ' * 4 * (level)
        print(f"{indent}{os.path.basename(root)}/")
        subindent = ' ' * 4 * (level + 1)
        ext_counts = collections.Counter([os.path.splitext(f)[1].lower() for f in files])
        for ext, count in ext_counts.items():
            if ext == '': ext = 'no_extension'
            print(f"{subindent}{count} files of type '{ext}'")

def enumerate_wavs(startpath):
    print("\n--- AUDIO FILES INVENTORY ---")
    wav_files = []
    for root, dirs, files in os.walk(startpath):
        for f in files:
            if f.lower().endswith('.wav'):
                wav_files.append(os.path.join(root, f))
                
    print(f"Total WAV files found: {len(wav_files)}")
    
    # Try common patterns for DTU dataset
    # E.g., 'aske_story6_part2.wav', 'story1_part1.wav', 'male_story2.wav'
    pattern = re.compile(r'(?P<speaker>[a-zA-Z]+)?_?story_?(?P<story>\d+)[_\-]?(?:part|trial|chunk)?_?(?P<chunk>\d+)?', re.IGNORECASE)
    pattern_alt = re.compile(r'(?P<speaker>male|female|m|f|aske|dina|etc)[_\-]?(?P<story>\d+)[_\-]?(?P<chunk>\d+)?', re.IGNORECASE)

    stories = collections.defaultdict(list)
    speakers = set()
    chunks = set()
    
    unmatched = []
    for w in wav_files:
        basename = os.path.basename(w)
        match = pattern.search(basename)
        if not match:
            match = pattern_alt.search(basename)
            
        if match:
            spk = match.group('speaker') or 'unknown'
            sty = match.group('story')
            chk = match.group('chunk') or '0'
            speakers.add(spk.lower())
            stories[(spk.lower(), sty)].append(chk)
            chunks.add(f"{spk}_story{sty}_chunk{chk}")
        else:
            unmatched.append(basename)
            
    print(f"Unique speakers extracted: {len(speakers)} {list(speakers)}")
    print(f"Unique stories extracted: {len(stories)}")
    print(f"Unique story chunks extracted: {len(chunks)}")
    
    if unmatched:
        print(f"\nCould not parse {len(unmatched)} filenames with standard regex. First 15 examples:")
        for u in unmatched[:15]:
            print("  " + u)
        
    print("\n--- STORY INVENTORY ---")
    for (spk, sty), chks in sorted(stories.items()):
        print(f"Story {sty:>2} | Speaker: {spk:<10} | Chunks: {len(chks):>2} | {sorted(chks)}")
        
def search_metadata(startpath):
    print("\n--- METADATA SEARCH ---")
    mat_files = []
    for root, dirs, files in os.walk(startpath):
        for f in files:
            if f.lower().endswith('.mat'):
                mat_files.append(os.path.join(root, f))
                
    print(f"Found {len(mat_files)} .mat files. Inspecting a subset for trial information...")
    
    found_info = False
    for mat in mat_files[:15]: # Check first 15 MAT files
        try:
            # Load with squeeze_me and struct_as_record to easily parse Matlab structs
            data = sio.loadmat(mat, squeeze_me=True, struct_as_record=False)
            keys = [k for k in data.keys() if not k.startswith('__')]
            
            # Print keys for the first few files to see general structure
            if mat_files.index(mat) < 3:
                print(f"File: {os.path.basename(mat)} | Keys: {keys}")
                for k in keys:
                    val = data[k]
                    if hasattr(val, '_fieldnames'):
                        print(f"  Struct '{k}' fields: {val._fieldnames}")
            
            # Look for explicit trial info fields
            for k in keys:
                val = data[k]
                if isinstance(val, np.ndarray) and hasattr(val, 'dtype') and val.dtype.names:
                    fields = val.dtype.names
                elif hasattr(val, '_fieldnames'):
                    fields = val._fieldnames
                else:
                    fields = []
                    
                target_fields = ['expinfo', 'wavfile_male', 'wavfile_female', 'attend_mf', 'attend_lr', 'trialinfo']
                found_targets = [f for f in target_fields if f in fields or f == k]
                
                if found_targets:
                    found_info = True
                    print(f"\nFound trial metadata in {os.path.basename(mat)} -> {k}: {found_targets}")
                    # Try to extract the first few rows if it's an array of structs
                    if isinstance(val, np.ndarray) and val.size > 0:
                        print("  Example row 1:")
                        try:
                            item = val[0]
                            for tf in found_targets:
                                if hasattr(item, tf):
                                    print(f"    {tf}: {getattr(item, tf)}")
                        except Exception:
                            pass
        except Exception as e:
            pass
            
    if not found_info:
        print("Could not locate explicit 'expinfo' or 'wavfile' metadata fields in the inspected .mat files.")

def main():
    print("===========================================")
    print("STEP 4.8: STIMULUS REUSE AUDIT")
    print("===========================================")
    
    startpath = DATA_DIR if os.path.exists(DATA_DIR) else "."
    if startpath == ".":
         print(f"WARNING: Data directory {DATA_DIR} not found. Running locally.")
         
    print_tree(startpath, max_depth=3)
    enumerate_wavs(startpath)
    search_metadata(startpath)
    
    print("\n===========================================")
    print("AUDIT COMPLETE.")
    print("Please provide this output to the reviewer.")

if __name__ == "__main__":
    main()
