import os
import sys
import re
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from copy import deepcopy
from pathlib import Path

# Add project root to path
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from models.eeg_classifier import EEGClassifier

FS = 64
TRAIN_WINDOW_SEC = 5
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

def evaluate_fold(model, test_data, device, window_sec=DECISION_WINDOW_SEC, fs=FS):
    model.eval()
    win_samples = int(window_sec * fs)
    
    total_windows = 0
    correct_windows = 0.0
    
    total_trials = len(test_data)
    correct_trials = 0.0
    total_trials_processed = 0
    
    with torch.no_grad():
        for t in test_data:
            x = t["eeg"].numpy()
            meta = t["meta"]
            attended_track = str(meta.get('attended_track', 'Unknown'))
            if attended_track not in ['1', '2']:
                continue
                
            true_label = 0 if attended_track == '1' else 1
            
            start = 0
            trial_logits = []
            
            while start + win_samples <= x.shape[1]:
                end = start + win_samples
                cx = torch.FloatTensor(x[:, start:end]).unsqueeze(0).to(device)
                
                logits = model(cx) # [1, 2]
                
                pred_label = logits.argmax(dim=-1).item()
                if pred_label == true_label:
                    correct_windows += 1.0
                total_windows += 1
                
                trial_logits.append(logits.squeeze(0).cpu().numpy()) # [2]
                
                start += win_samples
                
            if trial_logits:
                # Mean over windows
                mean_logits = np.mean(trial_logits, axis=0)
                trial_pred = np.argmax(mean_logits)
                
                pred_str = "CORRECT" if trial_pred == true_label else "WRONG"
                print(f"    Trial {meta.get('TrialID', 0):02d} | Exp: {meta.get('experiment', 'Unknown')} | Track Attended: {attended_track} | Pred: {trial_pred+1} ({pred_str})")
                
                if trial_pred == true_label:
                    correct_trials += 1.0
                total_trials_processed += 1
                
    win_acc = correct_windows / max(total_windows, 1)
    trial_acc = correct_trials / max(total_trials_processed, 1)
    
    print(f"  [EVAL SUMMARY] Total Trials Evaluated: {total_trials_processed}")
    print(f"  [EVAL SUMMARY] Correct: {correct_trials}, Accuracy: {trial_acc*100:.2f}%")
    
    return win_acc, trial_acc, total_trials_processed

def train_eeg_only_loso(target_fold=None, epochs=100, batch_size=128, lr=1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Starting EEG-Only LOSO Pipeline on {device}...")
    
    os.makedirs(REPO_ROOT / "checkpoints", exist_ok=True)
    
    from data.kul_cached_dataset import KULCachedLoader
    
    loader = KULCachedLoader(REPO_ROOT / "data" / "processed_kul")
    try:
        all_subject_data = loader.load_all()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please run `python preprocessing/build_kul_cache.py` first!")
        return
        
    subject_paths = list(all_subject_data.keys())
    
    results = {}
    
    for held_out_idx, held_out_id in enumerate(sorted(subject_paths)):
        if target_fold is not None and held_out_id != target_fold:
            continue
            
        print(f"\n==================================================")
        print(f"Evaluating fold with held-out subject: {held_out_id}")
        print(f"==================================================")
        
        # Build Train and Val sets
        train_data_full = []
        for other_id in sorted(subject_paths):
            if other_id != held_out_id:
                train_data_full.extend(all_subject_data[other_id])
                
        test_data = all_subject_data[held_out_id]
        
        # 10% Validation split from the training pool (trial level)
        np.random.seed(42)
        np.random.shuffle(train_data_full)
        val_split = int(len(train_data_full) * 0.1)
        
        val_data = train_data_full[:val_split]
        train_data = train_data_full[val_split:]
        
        print(f"\n--- Balancing Training Data ---")
        track1_trials = [t for t in train_data if str(t["meta"].get('attended_track')) == '1']
        track2_trials = [t for t in train_data if str(t["meta"].get('attended_track')) == '2']
        
        print(f"Original Pool -> Track 1: {len(track1_trials)}, Track 2: {len(track2_trials)}")
        
        # Balance to the minority class
        min_class_size = min(len(track1_trials), len(track2_trials))
        
        np.random.shuffle(track1_trials)
        np.random.shuffle(track2_trials)
        
        balanced_train_data = track1_trials[:min_class_size] + track2_trials[:min_class_size]
        np.random.shuffle(balanced_train_data)
        
        print(f"Balanced Pool -> Track 1: {min_class_size}, Track 2: {min_class_size}")
        print(f"Total balanced training trials: {len(balanced_train_data)}")
        print(f"-------------------------------\n")
        
        # Chunk training data
        tr_x, tr_labels = [], []
        for t in balanced_train_data:
            cx, clabels = chunk_data_classification(
                t["eeg"].numpy(), 
                t["meta"].get("attended_track"), 
                TRAIN_WINDOW_SEC, 
                TRAIN_HOP_SEC
            )
            tr_x.extend(cx)
            tr_labels.extend(clabels)
            
        print(f"Training on {len(tr_x)} chunks | Validating on {len(val_data)} full trials...")
        
        train_loader = DataLoader(
            TensorDataset(torch.FloatTensor(np.stack(tr_x)), 
                          torch.LongTensor(np.stack(tr_labels))),
            batch_size=batch_size, shuffle=True, pin_memory=True
        )
        
        model = EEGClassifier().to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        criterion = torch.nn.CrossEntropyLoss()
        
        if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
            scaler = torch.amp.GradScaler('cuda')
            use_amp = True
        else:
            scaler = torch.cuda.amp.GradScaler()
            use_amp = False
            
        best_val_acc = 0.0
        best_weights = deepcopy(model.state_dict())
        patience = 5
        epochs_no_improve = 0
        
        for epoch in range(epochs):
            model.train()
            train_loss = 0.0
            for bx, blabels in train_loader:
                bx, blabels = bx.to(device, non_blocking=True), blabels.to(device, non_blocking=True)
                optimizer.zero_grad()
                
                if use_amp:
                    with torch.amp.autocast('cuda'):
                        logits = model(bx)
                        loss = criterion(logits, blabels)
                else:
                    with torch.cuda.amp.autocast():
                        logits = model(bx)
                        loss = criterion(logits, blabels)
                        
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                train_loss += loss.item()
                
            win_acc, _, _ = evaluate_fold(model, val_data, device, window_sec=DECISION_WINDOW_SEC)
            
            sys.stdout.write(f"\r  Epoch {epoch+1:02d} | Train Loss: {train_loss/len(train_loader):.4f} | Val Window Acc: {win_acc*100:.2f}% | No Improve: {epochs_no_improve}")
            sys.stdout.flush()
            
            if win_acc > best_val_acc:
                best_val_acc = win_acc
                best_weights = deepcopy(model.state_dict())
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    break
        print()
        
        model.load_state_dict(best_weights)
        ckpt_path = REPO_ROOT / "checkpoints" / f"eeg_classifier_fold_{held_out_id}_best.pth"
        torch.save(model.state_dict(), ckpt_path)
        
        print("\nEvaluating on Held-Out Test Set:")
        win_acc, trial_acc, num_trials = evaluate_fold(model, test_data, device, window_sec=DECISION_WINDOW_SEC)
        
        results[held_out_id] = {
            "win_acc": win_acc,
            "trial_acc": trial_acc,
            "trials": num_trials
        }
        print(f"Fold {held_out_id} Results -> Window Acc: {win_acc*100:.2f}% | Trial Acc: {trial_acc*100:.2f}%\n")
        
    print("==================================================")
    print("LOSO CROSS-VALIDATION SUMMARY")
    print("==================================================")
    
    avg_win = np.mean([res["win_acc"] for res in results.values()])
    avg_trial = np.mean([res["trial_acc"] for res in results.values()])
    
    for sub, res in results.items():
        print(f"{sub}: Window {res['win_acc']*100:.2f}% | Trial {res['trial_acc']*100:.2f}% ({res['trials']} trials)")
        
    print("--------------------------------------------------")
    print(f"AVERAGE: Window {avg_win*100:.2f}% | Trial {avg_trial*100:.2f}%")
    print("==================================================")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=str, default=None, help="Specific subject fold to run (e.g. S1)")
    args = parser.parse_args()
    
    train_eeg_only_loso(target_fold=args.fold)
