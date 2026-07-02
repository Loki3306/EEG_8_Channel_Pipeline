import os
import sys
import glob
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import scipy.io
import traceback

def scan_directory(name, path):
    print(f"\n====================================================")
    print(f"PART 1: FILE STRUCTURE AUDIT - {name}")
    print(f"====================================================")
    if not os.path.exists(path):
        print(f"Path does not exist: {path}")
        return []
        
    all_files = []
    for root, dirs, files in os.walk(path):
        for f in files:
            all_files.append(os.path.join(root, f))
            
    print(f"Total files found: {len(all_files)}")
    
    # Group by extension
    ext_counts = {}
    for f in all_files:
        ext = os.path.splitext(f)[1].lower()
        ext_counts[ext] = ext_counts.get(ext, 0) + 1
        
    print("\nExtensions distribution:")
    for ext, count in ext_counts.items():
        print(f"  {ext if ext else '<no extension>'}: {count} files")
        
    print("\nSample files (up to 10):")
    for f in all_files[:10]:
        print(f"  {os.path.relpath(f, path)}")
        
    return all_files

def inspect_eeg(files):
    print(f"\n====================================================")
    print(f"PART 2 & 7: EEG & PREPROCESSING AUDIT")
    print(f"====================================================")
    
    eeg_files = [f for f in files if f.endswith(('.mat', '.npy', '.edf', '.bdf', '.set', '.fif'))]
    if not eeg_files:
        print("No standard EEG files (.mat, .npy, .edf, .bdf, .set, .fif) found.")
        return
        
    sample_file = eeg_files[0]
    print(f"Inspecting sample EEG file: {os.path.basename(sample_file)}")
    
    ext = os.path.splitext(sample_file)[1].lower()
    try:
        if ext == '.mat':
            mat = scipy.io.loadmat(sample_file, squeeze_me=True)
            print("MATLAB Keys:")
            for k in mat.keys():
                if not k.startswith('__'):
                    if isinstance(mat[k], np.ndarray):
                        print(f"  {k}: shape {mat[k].shape}, dtype {mat[k].dtype}")
                    else:
                        print(f"  {k}: {type(mat[k])}")
        elif ext == '.npy':
            arr = np.load(sample_file)
            print(f"NPY Array: shape {arr.shape}, dtype {arr.dtype}")
        else:
            try:
                import mne
                raw = mne.io.read_raw(sample_file, preload=False, verbose=False)
                print(f"MNE Raw Info:")
                print(f"  Channels: {len(raw.ch_names)}")
                print(f"  Sampling Freq: {raw.info['sfreq']} Hz")
                print(f"  Duration: {raw.times[-1]:.2f} seconds")
                print(f"  Channel Names (first 10): {raw.ch_names[:10]}")
            except ImportError:
                print("MNE not installed. Cannot parse EDF/BDF/SET files natively. Please install mne (`pip install mne`).")
    except Exception as e:
        print(f"Error inspecting EEG: {e}")
        traceback.print_exc()

def inspect_labels_and_events(files):
    print(f"\n====================================================")
    print(f"PART 3 & 4: LABEL & SWITCH EVENT AUDIT")
    print(f"====================================================")
    
    label_files = [f for f in files if f.endswith(('.csv', '.tsv', '.txt'))]
    if not label_files:
        print("No text-based label files (.csv, .tsv, .txt) found.")
        # Check if events are in .mat
        mat_files = [f for f in files if f.endswith('.mat')]
        if mat_files:
            print("Assuming labels might be embedded in .mat files.")
        return
        
    # Prioritize files with 'event', 'label', 'annot' in name
    target_files = [f for f in label_files if any(x in f.lower() for x in ['event', 'label', 'annot', 'log'])]
    if not target_files:
        target_files = label_files
        
    sample_file = target_files[0]
    print(f"Inspecting sample label/event file: {os.path.basename(sample_file)}")
    
    try:
        sep = '\t' if sample_file.endswith('.tsv') else ','
        df = pd.read_csv(sample_file, sep=sep)
        print(f"Columns: {list(df.columns)}")
        print(f"Total Rows: {len(df)}")
        print("\nFirst 5 rows:")
        print(df.head(5).to_string())
    except Exception as e:
        print(f"Error reading labels: {e}")

def inspect_audio(files):
    print(f"\n====================================================")
    print(f"PART 5: AUDIO AUDIT")
    print(f"====================================================")
    
    audio_files = [f for f in files if f.endswith(('.wav', '.flac', '.mp3'))]
    if not audio_files:
        print("No audio files (.wav, .flac) found.")
        return
        
    sample_file = audio_files[0]
    print(f"Inspecting sample Audio file: {os.path.basename(sample_file)}")
    
    try:
        import soundfile as sf
        info = sf.info(sample_file)
        print(f"  Sampling Freq: {info.samplerate} Hz")
        print(f"  Channels: {info.channels} ({'Stereo' if info.channels == 2 else 'Mono'})")
        print(f"  Duration: {info.duration:.2f} seconds")
        print(f"  Format: {info.format}")
    except ImportError:
        print("soundfile not installed. Cannot inspect audio metadata. Please install soundfile (`pip install soundfile`).")
    except Exception as e:
        print(f"Error inspecting audio: {e}")

def main():
    parser = argparse.ArgumentParser(description="AASD Dataset Forensic Audit")
    parser.add_argument("--processed_eeg", type=str, help="Path to processed EEG directory")
    parser.add_argument("--original_eeg", type=str, help="Path to original raw EEG directory")
    parser.add_argument("--audio", type=str, help="Path to stimuli audio directory")
    args = parser.parse_args()
    
    all_processed = []
    all_original = []
    all_audio = []
    
    if args.processed_eeg:
        all_processed = scan_directory("PROCESSED EEG", args.processed_eeg)
        inspect_eeg(all_processed)
        inspect_labels_and_events(all_processed)
        
    if args.original_eeg:
        all_original = scan_directory("ORIGINAL EEG", args.original_eeg)
        if not args.processed_eeg:
            inspect_eeg(all_original)
            inspect_labels_and_events(all_original)
            
    if args.audio:
        all_audio = scan_directory("STIMULI AUDIO", args.audio)
        inspect_audio(all_audio)
        
    print("\n====================================================")
    print("AUDIT SCRIPT COMPLETE")
    print("Please paste this entire console output back to the agent.")
    print("====================================================")

if __name__ == "__main__":
    main()
