import os
import sys
import json
import csv
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from copy import deepcopy
from pathlib import Path
import matplotlib.pyplot as plt

# Add project root to path
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from models.tcnn import TCNN
from data.kul_cached_dataset import KULCachedLoader

FS = 64
TRAIN_HOP_SEC = 2
DECISION_WINDOW_SEC = 10
BATCH_SIZE = 128
LEARNING_RATE = 1e-4
MAX_EPOCHS = 30
PATIENCE = 5

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

def evaluate_fold(model, test_data, device, window_sec=DECISION_WINDOW_SEC, fs=FS, criterion=None):
    model.eval()
    win_samples = int(window_sec * fs)
    
    total_windows = 0
    correct_windows = 0.0
    val_loss_sum = 0.0
    
    total_trials = 0
    correct_trials = 0.0
    
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
                
                trial_logits.append(logits.squeeze(0).cpu().numpy()) # [2]
                start += win_samples
                
            if trial_logits:
                mean_logits = np.mean(trial_logits, axis=0)
                trial_pred = np.argmax(mean_logits)
                if trial_pred == true_label:
                    correct_trials += 1.0
                total_trials += 1
                
    win_acc = correct_windows / total_windows if total_windows > 0 else 0.0
    trial_acc = correct_trials / total_trials if total_trials > 0 else 0.0
    avg_loss = val_loss_sum / total_windows if total_windows > 0 else 0.0
    
    return avg_loss, win_acc, trial_acc

def create_dataset(data_list, window_sec, hop_sec):
    all_x, all_y = [], []
    for t in data_list:
        meta = t["meta"]
        attended_track = str(meta.get('attended_track', 'Unknown'))
        if attended_track not in ['1', '2']:
            continue
            
        chunks_x, labels = chunk_data_classification(t["eeg"].numpy(), attended_track, window_sec, hop_sec)
        all_x.extend(chunks_x)
        all_y.extend(labels)
        
    if not all_x:
        return None
        
    tx = torch.FloatTensor(np.array(all_x))
    ty = torch.LongTensor(np.array(all_y))
    return TensorDataset(tx, ty)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Starting TCNN LOSO Pipeline on {device} (Window: {DECISION_WINDOW_SEC}s)...")
    
    loader = KULCachedLoader(REPO_ROOT / "data" / "processed_kul")
    try:
        all_subject_data = loader.load_all()
    except FileNotFoundError:
        print("KUL cache not found. Run preprocessing/build_kul_cache.py first.")
        return
        
    subject_ids = sorted(list(all_subject_data.keys()))
    if not subject_ids:
        print("No subjects loaded.")
        return
        
    out_dir = REPO_ROOT / "results" / "tcnn_loso"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    results = []
    
    global_total_trials = 0
    global_correct_trials = 0
    
    # Check if we have valid labels before training
    has_labels = False
    for sub in subject_ids:
        for t in all_subject_data[sub]:
            if str(t["meta"].get("attended_track", "Unknown")) in ['1', '2']:
                has_labels = True
                break
        if has_labels: break
    if not has_labels:
        print("\n[ERROR] No 'attended_track' labels (1 or 2) found in KUL cache metadata!")
        print("Please rebuild the KUL cache (build_kul_cache.py) with the updated metadata saving 'attended_track'.")
        return
    
    for held_out_idx, held_out_subject in enumerate(subject_ids):
        print(f"\n{'='*50}")
        print(f"Evaluating fold with held-out subject: {held_out_subject} ({held_out_idx+1}/{len(subject_ids)})")
        print(f"{'='*50}")
        
        train_data = []
        val_subject = subject_ids[(held_out_idx + 1) % len(subject_ids)]
        val_data = all_subject_data[val_subject]
        test_data = all_subject_data[held_out_subject]
        
        for sub in subject_ids:
            if sub != held_out_subject and sub != val_subject:
                train_data.extend(all_subject_data[sub])
                
        # Balancing classes
        c1_data = [t for t in train_data if str(t["meta"].get("attended_track")) == '1']
        c2_data = [t for t in train_data if str(t["meta"].get("attended_track")) == '2']
        
        if len(c1_data) == 0 or len(c2_data) == 0:
            print(f"Skipping fold {held_out_subject} due to missing classes in training set.")
            continue
            
        min_class = min(len(c1_data), len(c2_data))
        np.random.shuffle(c1_data)
        np.random.shuffle(c2_data)
        balanced_train = c1_data[:min_class] + c2_data[:min_class]
        
        print(f"--- Balancing Training Data ---")
        print(f"Class 1 Trials: {len(c1_data)} -> {min_class}")
        print(f"Class 2 Trials: {len(c2_data)} -> {min_class}")
        
        train_dataset = create_dataset(balanced_train, DECISION_WINDOW_SEC, TRAIN_HOP_SEC)
        if train_dataset is None:
            continue
            
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
        print(f"Training on {len(train_dataset)} chunks | Validating on {val_subject} ({len(val_data)} trials)...")
        
        model = TCNN(in_channels=8, num_classes=2).to(device)
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
        criterion = torch.nn.CrossEntropyLoss()
        
        best_val_loss = float('inf')
        best_model_state = None
        patience_counter = 0
        
        train_losses = []
        val_losses = []
        
        for epoch in range(MAX_EPOCHS):
            model.train()
            epoch_loss = 0.0
            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                
                optimizer.zero_grad()
                logits = model(batch_x)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                
            train_loss = epoch_loss / len(train_loader)
            val_loss, val_win_acc, val_trial_acc = evaluate_fold(model, val_data, device, DECISION_WINDOW_SEC, FS, criterion)
            
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            
            print(f"  Epoch {epoch+1:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Trial Acc: {val_trial_acc*100:.1f}%")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = deepcopy(model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= PATIENCE:
                print(f"  Early stopping at epoch {epoch+1}")
                break
                
        # Save learning curve
        plt.figure()
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Val Loss')
        plt.legend()
        plt.title(f"Learning Curve (Held out: {held_out_subject})")
        plt.savefig(out_dir / f"learning_curve_{held_out_subject}.png")
        plt.close()
        
        # Load best model and evaluate on test set
        model.load_state_dict(best_model_state)
        torch.save(model.state_dict(), out_dir / f"best_model_{held_out_subject}.pt")
        
        _, test_win_acc, test_trial_acc = evaluate_fold(model, test_data, device, DECISION_WINDOW_SEC, FS)
        print(f"\n--> Fold Result ({held_out_subject}): Window Acc: {test_win_acc*100:.2f}% | Trial Acc: {test_trial_acc*100:.2f}%")
        
        results.append({
            "held_out_subject": held_out_subject,
            "window_acc": test_win_acc,
            "trial_acc": test_trial_acc
        })
        
        global_total_trials += len(test_data)
        global_correct_trials += (test_trial_acc * len(test_data))
        
    print(f"\n{'='*50}")
    print("FINAL LOSO RESULTS (TCNN)")
    print(f"{'='*50}")
    
    with open(out_dir / "tcnn_loso_summary.csv", "w", newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["held_out_subject", "window_acc", "trial_acc"])
        writer.writeheader()
        
        for r in results:
            print(f"Subject {r['held_out_subject']}: Window = {r['window_acc']*100:.1f}% | Trial = {r['trial_acc']*100:.1f}%")
            writer.writerow(r)
            
    mean_trial_acc = global_correct_trials / global_total_trials if global_total_trials > 0 else 0
    print(f"\nOverall Mean Trial Accuracy: {mean_trial_acc*100:.2f}%")

if __name__ == "__main__":
    main()
