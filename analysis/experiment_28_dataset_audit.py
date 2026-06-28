import os
import sys
import glob
from collections import Counter
from pathlib import Path

# Add project root to path
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from training.train_kul_matchnet_loso import get_kul_subject_files, load_kul_trials

def main():
    print(f"Running Phase B: Empirical Dataset Audit")
    
    subject_paths = get_kul_subject_files()
    if not subject_paths:
        print("No KUL .mat files found.")
        return
        
    print(f"Found {len(subject_paths)} subject files.")
    
    global_track_counter = Counter()
    global_exp_counter = Counter()
    
    for p in sorted(subject_paths):
        sub_name = p.name
        try:
            trials = load_kul_trials(str(p))
            
            sub_track_counter = Counter()
            for t in trials:
                track = getattr(t, "attended_track", "Unknown")
                exp = getattr(t, "experiment", "Unknown")
                sub_track_counter[str(track)] += 1
                global_track_counter[str(track)] += 1
                global_exp_counter[str(exp)] += 1
                
            print(f"Subject {sub_name:<15}: {len(trials)} trials -> Track 1: {sub_track_counter.get('1', 0):<2} | Track 2: {sub_track_counter.get('2', 0):<2}")
            
        except Exception as e:
            print(f"Error loading {sub_name}: {e}")
            
    print(f"\n==================================================")
    print(f"GLOBAL DATASET DISTRIBUTION (Training Pool)")
    print(f"==================================================")
    
    total_trials = sum(global_track_counter.values())
    print(f"Total Trials: {total_trials}")
    
    print("\n--- By Track ---")
    for track, count in sorted(global_track_counter.items()):
        print(f"Track {track}: {count} trials ({count/total_trials*100:.1f}%)")
        
    print("\n--- By Experiment ---")
    for exp, count in sorted(global_exp_counter.items()):
        print(f"Experiment {exp}: {count} trials ({count/total_trials*100:.1f}%)")

if __name__ == "__main__":
    main()
