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
    
    # 0 for Track 1, 1 for Track 2
    label = 0 if str(attended_track) == '1' else 1
    
    start = 0
    while start + win_samples <= x.shape[1]:
        end = start + win_samples
        chunks_x.append(x[:, start:end])
        labels.append(label)
        start += hop_samples
    return chunks_x, labels

def create_dataset_and_audit(data_list, window_sec, hop_sec):
    all_x, all_y = [], []
    for t in data_list:
        meta = t["meta"]
        attended_track = str(meta.get('attended_track', 'Unknown'))
        if attended_track not in ['1', '2']:
            continue
            
        chunks_x, labels = chunk_data_classification(t["eeg"].numpy(), attended_track, window_sec, hop_sec)
        all_x.extend(chunks_x)
        all_y.extend(labels)
        
    return all_x, all_y

def main():
    print("Loading KUL Cache...")
    loader = KULCachedLoader(REPO_ROOT / "data" / "processed_kul")
    all_subject_data = loader.load_all()
    
    subject_ids = sorted(list(all_subject_data.keys()))
    if not subject_ids:
        print("No subjects found.")
        return
        
    # We will just simulate Fold 1 (Held-out S1)
    held_out_subject = "S1"
    val_subject = subject_ids[1] # S10 typically
    
    train_data = []
    for sub in subject_ids:
        if sub != held_out_subject and sub != val_subject:
            train_data.extend(all_subject_data[sub])
            
    print(f"\n--- 1. Before Balancing ---")
    c1_before = [t for t in train_data if str(t["meta"].get("attended_track")) == '1']
    c2_before = [t for t in train_data if str(t["meta"].get("attended_track")) == '2']
    print(f"Total Trials: {len(train_data)}")
    print(f"Class 1 (Track 1) Trials: {len(c1_before)}")
    print(f"Class 2 (Track 2) Trials: {len(c2_before)}")
    
    min_class = min(len(c1_before), len(c2_before))
    
    # Fix random seed to ensure deterministic output for audit
    np.random.seed(42)
    np.random.shuffle(c1_before)
    np.random.shuffle(c2_before)
    
    balanced_train = c1_before[:min_class] + c2_before[:min_class]
    
    print(f"\n--- 2. After Trial Balancing ---")
    c1_after = [t for t in balanced_train if str(t["meta"].get("attended_track")) == '1']
    c2_after = [t for t in balanced_train if str(t["meta"].get("attended_track")) == '2']
    print(f"Total Trials: {len(balanced_train)}")
    print(f"Class 1 (Track 1) Trials: {len(c1_after)}")
    print(f"Class 2 (Track 2) Trials: {len(c2_after)}")
    
    print(f"\n--- 3. After Chunking ---")
    all_x, all_y = create_dataset_and_audit(balanced_train, DECISION_WINDOW_SEC, TRAIN_HOP_SEC)
    
    c1_windows = sum(1 for y in all_y if y == 0)
    c2_windows = sum(1 for y in all_y if y == 1)
    
    print(f"Total Windows: {len(all_y)}")
    print(f"Class 1 (Track 1) Windows: {c1_windows} ({c1_windows/len(all_y)*100:.1f}%)")
    print(f"Class 2 (Track 2) Windows: {c2_windows} ({c2_windows/len(all_y)*100:.1f}%)")
    
    print(f"\n--- 4. Verify Window Inherits Correct Trial Label ---")
    print("Checking the first 5 trials in the balanced set...")
    for idx, t in enumerate(balanced_train[:5]):
        meta = t["meta"]
        att_track = str(meta.get('attended_track', 'Unknown'))
        expected_label = 0 if att_track == '1' else 1
        
        _, chunk_labels = chunk_data_classification(t["eeg"].numpy(), att_track, DECISION_WINDOW_SEC, TRAIN_HOP_SEC)
        
        all_match = all(y == expected_label for y in chunk_labels)
        print(f"Trial {idx+1} (Expected Class {expected_label}): Generated {len(chunk_labels)} chunks. All match? {all_match}")
        
    print(f"\n--- 5. Print first 100 labels fed to DataLoader (after shuffling) ---")
    tx = torch.FloatTensor(np.array(all_x))
    ty = torch.LongTensor(np.array(all_y))
    dataset = torch.utils.data.TensorDataset(tx, ty)
    
    torch.manual_seed(42) # For reproducible DataLoader shuffle
    loader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=True)
    
    first_batch_x, first_batch_y = next(iter(loader))
    first_100_labels = first_batch_y[:100].numpy()
    
    print("First 100 Labels:")
    print("".join(str(y) for y in first_100_labels))
    
    b_c1 = sum(1 for y in first_100_labels if y == 0)
    b_c2 = sum(1 for y in first_100_labels if y == 1)
    print(f"Batch distribution (first 100): Class 1={b_c1}, Class 2={b_c2}")

if __name__ == "__main__":
    main()
