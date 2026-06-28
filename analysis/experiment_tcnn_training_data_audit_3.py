import os
import sys
import numpy as np
import torch
import scipy.io
from pathlib import Path
import re

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from data.kul_cached_dataset import KULCachedLoader
from preprocessing.build_kul_cache import get_kul_subject_files, load_kul_trials

FS = 64
TRAIN_HOP_SEC = 2
DECISION_WINDOW_SEC = 10

def chunk_data_classification(x_len, window_sec, hop_sec, fs=FS):
    win_samples = int(window_sec * fs)
    hop_samples = int(hop_sec * fs)
    if x_len < win_samples:
        return 0
    return ((x_len - win_samples) // hop_samples) + 1

def load_extended_metadata():
    # Load .mat files to get Part, Repetition, and Stimuli that aren't in .pt cache
    subject_files = get_kul_subject_files()
    extended_meta = {}
    
    for filepath in subject_files:
        sub_id = re.search(r'(S\d+)', filepath.name, re.IGNORECASE).group(1).upper()
        trials = load_kul_trials(str(filepath))
        
        extended_meta[sub_id] = {}
        for i, t in enumerate(trials):
            trial_id = getattr(t, "TrialID", i+1)
            
            part = getattr(t, "part", "N/A")
            rep = getattr(t, "repetition", "N/A")
            stimuli = getattr(t, "stimuli", ["N/A", "N/A"])
            
            # Format stimuli
            if len(stimuli) >= 2:
                stim_l = str(stimuli[0]).strip()
                stim_r = str(stimuli[1]).strip()
            else:
                stim_l = str(stimuli)
                stim_r = "N/A"
                
            extended_meta[sub_id][trial_id] = {
                "part": str(part),
                "repetition": str(rep),
                "stim_l": stim_l,
                "stim_r": stim_r
            }
    return extended_meta

def main():
    print("Loading KUL Cache and Raw MAT files...")
    loader = KULCachedLoader(REPO_ROOT / "data" / "processed_kul")
    all_subject_data = loader.load_all()
    
    try:
        extended_meta = load_extended_metadata()
    except Exception as e:
        print(f"Warning: Could not load raw MAT files for extended metadata: {e}")
        extended_meta = {}
    
    subject_ids = sorted(list(all_subject_data.keys()))
    if not subject_ids:
        print("No subjects found.")
        return
        
    held_out_subject = "S1"
    val_subject = subject_ids[1] # S10
    
    train_data = []
    for sub in subject_ids:
        if sub != held_out_subject and sub != val_subject:
            train_data.extend(all_subject_data[sub])
            
    c1_before = [t for t in train_data if str(t["meta"].get("attended_track")) == '1']
    c2_before = [t for t in train_data if str(t["meta"].get("attended_track")) == '2']
    
    min_class = min(len(c1_before), len(c2_before))
    
    np.random.seed(42)
    np.random.shuffle(c1_before)
    np.random.shuffle(c2_before)
    
    balanced_train = c1_before[:min_class] + c2_before[:min_class]
    
    print("\n==================================================")
    print("1. Trial Metadata (All selected training trials)")
    print("==================================================")
    
    story_segments = {}
    exp_stats = {}
    
    for t in balanced_train:
        meta = t["meta"]
        sub = meta.get("Subject", "Unknown")
        trial_id = meta.get("TrialID", "Unknown")
        att_track = str(meta.get("attended_track", "Unknown"))
        exp_id = str(meta.get("experiment", "Unknown")).strip("[]")
        
        eeg_len = t["eeg"].shape[1]
        duration = eeg_len / float(FS)
        windows = chunk_data_classification(eeg_len, DECISION_WINDOW_SEC, TRAIN_HOP_SEC)
        
        ext = extended_meta.get(sub, {}).get(trial_id, {"part": "N/A", "repetition": "N/A", "stim_l": "N/A", "stim_r": "N/A"})
        
        print(f"Subject    : {sub}")
        print(f"Experiment : {exp_id}")
        print(f"Part       : {ext['part']}")
        print(f"Repetition : {ext['repetition']}")
        print(f"Track      : {att_track}")
        print(f"Stimuli    : {ext['stim_l']}")
        print(f"             {ext['stim_r']}")
        print(f"EEG Samples: {eeg_len}")
        print(f"Duration   : {duration:.1f} sec")
        print(f"Windows    : {windows}")
        print("-" * 40)
        
        # Track experiment stats
        if exp_id not in exp_stats:
            exp_stats[exp_id] = {"trials": 0, "windows": 0}
        exp_stats[exp_id]["trials"] += 1
        exp_stats[exp_id]["windows"] += windows
        
        # Track story segments
        story_key = f"Exp{exp_id}_Part{ext['part']}_Rep{ext['repetition']}_{ext['stim_l']}_{ext['stim_r']}"
        if story_key not in story_segments:
            story_segments[story_key] = {"selections": 0, "windows": 0}
        story_segments[story_key]["selections"] += 1
        story_segments[story_key]["windows"] += windows

    print("\n==================================================")
    print("4. Unique Story Segments Dominance")
    print("==================================================")
    for segment, data in sorted(story_segments.items(), key=lambda x: x[1]["windows"], reverse=True):
        print(f"Segment: {segment}")
        print(f"  Selected {data['selections']} times")
        print(f"  Total Windows: {data['windows']}")
        print()
        
    print("\n==================================================")
    print("5. Experiment Contribution")
    print("==================================================")
    total_windows = sum(stats["windows"] for stats in exp_stats.values())
    
    for exp_id in sorted(exp_stats.keys()):
        stats = exp_stats[exp_id]
        print(f"Experiment {exp_id}")
        print(f"  {stats['trials']} trials")
        print(f"  {stats['windows']} windows")
        print(f"  {stats['windows']/total_windows*100:.1f}%")
        print()

if __name__ == "__main__":
    main()
