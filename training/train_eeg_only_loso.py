import os
import sys
import re
import json
import csv
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

def evaluate_fold(model, test_data, device, window_sec=DECISION_WINDOW_SEC, fs=FS, criterion=None):
    model.eval()
    win_samples = int(window_sec * fs)
    
    total_windows = 0
    correct_windows = 0.0
    val_loss_sum = 0.0
    
    total_trials = len(test_data)
    correct_trials = 0.0
    total_trials_processed = 0
    
    window_predictions = []
    
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
                
                if criterion is not None:
                    c_label = torch.LongTensor([true_label]).to(device)
                    loss = criterion(logits, c_label)
                    val_loss_sum += loss.item()
                
                probs = F.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
                pred_label = np.argmax(probs)
                
                if pred_label == true_label:
                    correct_windows += 1.0
                total_windows += 1
                
                window_predictions.append({
                    "trial_id": meta.get('TrialID', 0),
                    "true_label": true_label,
                    "pred_label": pred_label,
                    "p_track1": float(probs[0]),
                    "p_track2": float(probs[1]),
                    "is_correct": int(pred_label == true_label)
                })
                
                trial_logits.append(logits.squeeze(0).cpu().numpy()) # [2]
                
                start += win_samples
                
            if trial_logits:
                # Mean over windows
                mean_logits = np.mean(trial_logits, axis=0)
                trial_pred = np.argmax(mean_logits)
                
                if trial_pred == true_label:
                    correct_trials += 1.0
                total_trials_processed += 1
                
    win_acc = correct_windows / max(total_windows, 1)
    trial_acc = correct_trials / max(total_trials_processed, 1)
    val_loss = val_loss_sum / max(total_windows, 1) if criterion is not None else 0.0
    
    return win_acc, trial_acc, val_loss, total_trials_processed, window_predictions, total_windows

def train_eeg_only_loso(target_fold=None, epochs=100, batch_size=128, lr=1e-3, window_sec=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Starting EEG-Only LOSO Pipeline on {device} (Window: {window_sec}s)...")
    
    os.makedirs(REPO_ROOT / "checkpoints", exist_ok=True)
    stats_dir = REPO_ROOT / "analysis" / "summaries"
    os.makedirs(stats_dir, exist_ok=True)
    
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
        
        # Balance to the minority class
        min_class_size = min(len(track1_trials), len(track2_trials))
        
        np.random.shuffle(track1_trials)
        np.random.shuffle(track2_trials)
        
        balanced_train_data = track1_trials[:min_class_size] + track2_trials[:min_class_size]
        np.random.shuffle(balanced_train_data)
        
        # Chunk training data
        tr_x, tr_labels = [], []
        for t in balanced_train_data:
            cx, clabels = chunk_data_classification(
                t["eeg"].numpy(), 
                t["meta"].get("attended_track"), 
                window_sec, 
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
        
        training_history = []
        
        for epoch in range(epochs):
            model.train()
            train_loss = 0.0
            correct_train = 0
            total_train = 0
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
                pred_train = logits.argmax(dim=-1)
                correct_train += (pred_train == blabels).sum().item()
                total_train += blabels.size(0)
                
            epoch_train_loss = train_loss / len(train_loader)
            epoch_train_acc = correct_train / total_train
            
            win_acc, _, val_loss, _, _, val_windows_cnt = evaluate_fold(model, val_data, device, window_sec=window_sec, criterion=criterion)
            
            training_history.append({
                "epoch": epoch + 1,
                "train_loss": epoch_train_loss,
                "train_acc": epoch_train_acc,
                "val_loss": val_loss,
                "val_acc": win_acc
            })
            
            sys.stdout.write(f"\r  Epoch {epoch+1:02d} | Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc*100:.2f}% | Val Loss: {val_loss:.4f} | Val Acc: {win_acc*100:.2f}% | No Improve: {epochs_no_improve}")
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
        
        # Save training history
        with open(stats_dir / f"diagnostics_fold_{held_out_id}_{window_sec}s.json", "w") as f:
            json.dump(training_history, f, indent=4)
            
        model.load_state_dict(best_weights)
        ckpt_path = REPO_ROOT / "checkpoints" / f"eeg_classifier_fold_{held_out_id}_{window_sec}s_best.pth"
        torch.save(model.state_dict(), ckpt_path)
        
        print("\nEvaluating on Held-Out Test Set:")
        win_acc, trial_acc, _, num_trials, window_predictions, test_windows_cnt = evaluate_fold(model, test_data, device, window_sec=window_sec)
        
        # Confusion Analysis
        tp, fp, tn, fn = 0, 0, 0, 0
        for p in window_predictions:
            if p["true_label"] == 0 and p["pred_label"] == 0: tp += 1
            if p["true_label"] == 1 and p["pred_label"] == 0: fp += 1
            if p["true_label"] == 1 and p["pred_label"] == 1: tn += 1
            if p["true_label"] == 0 and p["pred_label"] == 1: fn += 1
            
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * (precision * recall) / max(precision + recall, 1e-6)
        
        print(f"Confusion Matrix (Track 1 is Positive Class):")
        print(f"  TP: {tp} | FP: {fp}")
        print(f"  FN: {fn} | TN: {tn}")
        print(f"  Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}")
        
        # Save Confidence data
        csv_path = stats_dir / f"confidence_fold_{held_out_id}_{window_sec}s.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["trial_id", "true_label", "pred_label", "p_track1", "p_track2", "is_correct"])
            writer.writeheader()
            for p in window_predictions:
                writer.writerow(p)
        
        results[held_out_id] = {
            "val_acc_best": best_val_acc,
            "win_acc": win_acc,
            "trial_acc": trial_acc,
            "train_windows": len(tr_x),
            "val_windows": val_windows_cnt,
            "test_windows": test_windows_cnt,
            "precision": precision,
            "recall": recall,
            "f1": f1
        }
        print(f"Fold {held_out_id} Results -> Window Acc: {win_acc*100:.2f}% | Trial Acc: {trial_acc*100:.2f}%\n")
        
    print("==================================================")
    print("LOSO CROSS-VALIDATION SUMMARY")
    print("==================================================")
    
    val_accs = [res["val_acc_best"] for res in results.values()]
    win_accs = [res["win_acc"] for res in results.values()]
    trial_accs = [res["trial_acc"] for res in results.values()]
    
    for sub, res in results.items():
        print(f"{sub}: Val Win {res['val_acc_best']*100:.2f}% | Test Win {res['win_acc']*100:.2f}% | Test Trial {res['trial_acc']*100:.2f}% (Tr:{res['train_windows']} Val:{res['val_windows']} Te:{res['test_windows']})")
        
    print("--------------------------------------------------")
    print(f"MEAN VAL WIN: {np.mean(val_accs)*100:.2f}% ± {np.std(val_accs)*100:.2f}% | Median: {np.median(val_accs)*100:.2f}%")
    print(f"MEAN TEST WIN: {np.mean(win_accs)*100:.2f}% ± {np.std(win_accs)*100:.2f}% | Median: {np.median(win_accs)*100:.2f}%")
    print(f"MEAN TEST TRIAL: {np.mean(trial_accs)*100:.2f}% ± {np.std(trial_accs)*100:.2f}% | Median: {np.median(trial_accs)*100:.2f}%")
    print("==================================================")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=str, default=None, help="Specific subject fold to run (e.g. S1)")
    parser.add_argument("--window_sec", type=int, default=5, help="Length of the EEG window in seconds")
    args = parser.parse_args()
    
    train_eeg_only_loso(target_fold=args.fold, window_sec=args.window_sec)
