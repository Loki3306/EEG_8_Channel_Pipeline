import os
import sys
import numpy as np
import torch
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from data.kul_cached_dataset import KULCachedLoader

FS = 64
TRAIN_HOP_SEC = 2
DECISION_WINDOW_SEC = 10

def chunk_data_classification(x, attended_track, window_sec, hop_sec, fs=FS):
    win_samples = int(window_sec * fs)
    hop_samples = int(hop_sec * fs)
    chunks_x, labels = [], []
    
    label = 0 if str(attended_track) == '1' else 1
    
    start = 0
    while start + win_samples <= x.shape[1]:
        end = start + win_samples
        chunks_x.append(x[:, start:end])
        labels.append(label)
        start += hop_samples
    return chunks_x, labels

def main():
    print("Loading KUL Cache...")
    loader = KULCachedLoader(REPO_ROOT / "data" / "processed_kul")
    all_subject_data = loader.load_all()
    
    subject_ids = sorted(list(all_subject_data.keys()))
    if not subject_ids:
        print("No subjects found.")
        return
        
    # We simulate Fold 1 (Held-out S1)
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
    
    exp_trial_counts = {"1": 0, "2": 0, "3": 0, "Unknown": 0}
    exp_window_counts = {"1": 0, "2": 0, "3": 0, "Unknown": 0}
    
    # Store Exp 3 details
    exp3_details = {}
    
    total_t1_windows = 0
    total_t2_windows = 0
    
    for t in balanced_train:
        meta = t["meta"]
        att_track = str(meta.get("attended_track", "Unknown"))
        exp_id = str(meta.get("experiment", "Unknown"))
        trial_id = meta.get("TrialID", "Unknown")
        
        # In KUL .mat files, experiment might be a number 1, 2, 3 or array
        # Try to parse it to a clean string
        if isinstance(exp_id, str) and exp_id.startswith("["):
            exp_id = exp_id.strip("[]")
        exp_id = exp_id.strip()
        
        if exp_id not in exp_trial_counts:
            exp_trial_counts[exp_id] = 0
            exp_window_counts[exp_id] = 0
            
        exp_trial_counts[exp_id] += 1
        
        _, chunk_labels = chunk_data_classification(t["eeg"].numpy(), att_track, DECISION_WINDOW_SEC, TRAIN_HOP_SEC)
        num_windows = len(chunk_labels)
        
        exp_window_counts[exp_id] += num_windows
        
        if att_track == '1':
            total_t1_windows += num_windows
        else:
            total_t2_windows += num_windows
            
        if "3" in exp_id:
            # We don't have explicit part/rep, but TrialID uniquely identifies the trial across subjects
            key = f"TrialID_{trial_id}"
            if key not in exp3_details:
                exp3_details[key] = []
            exp3_details[key].append(num_windows)

    total_windows = total_t1_windows + total_t2_windows

    print("\n==================================================")
    print("Investigation 1: Distribution by Experiment")
    print("==================================================")
    for exp_id in sorted(exp_trial_counts.keys()):
        tc = exp_trial_counts[exp_id]
        wc = exp_window_counts[exp_id]
        print(f"Experiment {exp_id}:")
        print(f"  Trials  : {tc}")
        print(f"  Windows : {wc} ({wc/total_windows*100:.1f}%)")

    print("\n==================================================")
    print("Investigation 2: Experiment 3 Details")
    print("==================================================")
    # Since we can't easily parse part/rep from KUL without looking at stimuli arrays, 
    # we print it grouped by TrialID.
    for trial_id in sorted(exp3_details.keys(), key=lambda x: int(x.split('_')[1]) if x.split('_')[1].isdigit() else 0):
        window_list = exp3_details[trial_id]
        print(f"Exp 3 - {trial_id} | Selected {len(window_list)} times | Windows per selection: {window_list}")

    print("\n==================================================")
    print("Investigation 3: Final Training Statistics")
    print("==================================================")
    print(f"Total Training Windows: {total_windows}")
    print(f"Track 1 Windows: {total_t1_windows} ({total_t1_windows/total_windows*100:.1f}%)")
    print(f"Track 2 Windows: {total_t2_windows} ({total_t2_windows/total_windows*100:.1f}%)")
    
    for exp_id in sorted(exp_window_counts.keys()):
        wc = exp_window_counts[exp_id]
        print(f"Experiment {exp_id} Windows: {wc} ({wc/total_windows*100:.1f}%)")

if __name__ == "__main__":
    main()
