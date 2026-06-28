import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from pathlib import Path
from copy import deepcopy
import csv
import argparse

# Add repo root to path
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from data.kul_cached_dataset import KULCachedLoader
from models.kuruvila_original import KuruvilaOriginalCNNLSTM

FS = 64
DECISION_WINDOW_SEC = 3.0   # 192 samples
TRAIN_HOP_SEC = 1.0         # 64 samples
EVAL_HOP_SEC = 1.0
BATCH_SIZE = 4              # mini_batch_size = 4 in original repo
LEARNING_RATE = 5e-4        # exact from repo
MAX_EPOCHS = 100
PATIENCE = 10

class LazyKULKuruvilaDataset(torch.utils.data.Dataset):
    def __init__(self, chunks, trials, win_samples):
        self.chunks = chunks
        self.trials = trials
        self.win_samples = win_samples
        
    def __len__(self):
        return len(self.chunks)
        
    def __getitem__(self, idx):
        trial_idx, start, label = self.chunks[idx]
        t = self.trials[trial_idx]
        
        attended_track = str(t["meta"].get('attended_track', 'Unknown'))
        if attended_track == '1':
            audio_spk1 = t["audio_a"]
            audio_spk2 = t["audio_b"]
        else:
            audio_spk1 = t["audio_b"]
            audio_spk2 = t["audio_a"]
            
        end = start + self.win_samples
        cx = t["eeg"][:, start:end]
        ca1 = audio_spk1[:, start:end]
        ca2 = audio_spk2[:, start:end]
        
        return cx, ca1, ca2, label

def evaluate_fold(model, eval_data, device, window_sec, fs, criterion=None):
    model.eval()
    
    total_loss = 0.0
    total_batches = 0
    
    total_wins = 0
    correct_wins = 0
    correct_trials = 0
    
    pred_t1_count = 0
    pred_t2_count = 0
    
    win_samples = int(window_sec * fs)
    hop_samples = int(EVAL_HOP_SEC * fs)
    
    with torch.no_grad():
        for t in eval_data:
            meta = t["meta"]
            attended_track = str(meta.get('attended_track', 'Unknown'))
            if attended_track not in ['1', '2']:
                continue
                
            label = 0 if attended_track == '1' else 1
            
            if attended_track == '1':
                audio_spk1 = t["audio_a"].to(device)
                audio_spk2 = t["audio_b"].to(device)
            else:
                audio_spk1 = t["audio_b"].to(device)
                audio_spk2 = t["audio_a"].to(device)
                
            eeg = t["eeg"].to(device)
            
            start = 0
            cx_list, ca1_list, ca2_list, cy_list = [], [], [], []
            while start + win_samples <= eeg.shape[1]:
                end = start + win_samples
                cx_list.append(eeg[:, start:end])
                ca1_list.append(audio_spk1[:, start:end])
                ca2_list.append(audio_spk2[:, start:end])
                cy_list.append(label)
                start += hop_samples
                
            if not cx_list:
                continue
                
            cx = torch.stack(cx_list)
            ca1 = torch.stack(ca1_list)
            ca2 = torch.stack(ca2_list)
            cy = torch.LongTensor(cy_list).to(device)
            
            # One-hot encode targets for BCELoss exactly as in original repo
            one_hot_y = F.one_hot(cy, num_classes=2).float()
            
            logits = model(cx, ca1, ca2)
            
            if criterion:
                loss = criterion(logits, one_hot_y)
                total_loss += loss.item()
                total_batches += 1
                
            preds = torch.argmax(logits, dim=1)
            
            c_t1 = (preds == 0).sum().item()
            c_t2 = (preds == 1).sum().item()
            pred_t1_count += c_t1
            pred_t2_count += c_t2
            
            correct = (preds == cy).sum().item()
            correct_wins += correct
            total_wins += len(preds)
            
            if c_t1 > c_t2:
                trial_pred = 0
            elif c_t2 > c_t1:
                trial_pred = 1
            else:
                trial_pred = np.random.choice([0, 1])
                
            if trial_pred == label:
                correct_trials += 1
                
    avg_loss = total_loss / total_batches if total_batches > 0 else 0
    win_acc = correct_wins / total_wins if total_wins > 0 else 0
    trial_acc = correct_trials / len(eval_data) if len(eval_data) > 0 else 0
    
    return avg_loss, win_acc, trial_acc, pred_t1_count, pred_t2_count

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke_test", action="store_true", help="Run 1 fold, 1 batch, print shapes, and exit")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Starting Exact Kuruvila 2021 Reproduction LOSO Pipeline on {device}")
    print(f"Window: {DECISION_WINDOW_SEC}s, Hop: {TRAIN_HOP_SEC}s, Batch: {BATCH_SIZE}, LR: {LEARNING_RATE}")
    
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
        
    out_dir = REPO_ROOT / "results" / "kuruvila_original_loso"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    results = []
    
    global_total_trials = 0
    global_correct_trials = 0
    
    win_samples = int(DECISION_WINDOW_SEC * FS)
    hop_samples = int(TRAIN_HOP_SEC * FS)
    
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
                for t in all_subject_data[sub]:
                    t["meta"]["Subject"] = sub
                train_data.extend(all_subject_data[sub])
                
        all_train_chunks = []
        all_train_meta = []
        
        for i, t in enumerate(train_data):
            meta = t["meta"]
            attended_track = str(meta.get('attended_track', 'Unknown'))
            if attended_track not in ['1', '2']:
                continue
                
            label = 0 if attended_track == '1' else 1
            sub = meta.get("Subject", "Unknown")
            tid = meta.get("TrialID", "Unknown")
            
            eeg_len = t["eeg"].shape[1]
            start = 0
            while start + win_samples <= eeg_len:
                all_train_chunks.append((i, start, label))
                all_train_meta.append((sub, tid))
                start += hop_samples
                
        chunks_c1 = [i for i, c in enumerate(all_train_chunks) if c[2] == 0]
        chunks_c2 = [i for i, c in enumerate(all_train_chunks) if c[2] == 1]
        
        print("\n========================================")
        print("Window Distribution BEFORE balancing")
        print(f"Track1 windows: {len(chunks_c1)}")
        print(f"Track2 windows: {len(chunks_c2)}")
        
        min_windows = min(len(chunks_c1), len(chunks_c2))
        if min_windows == 0:
            print(f"Skipping fold {held_out_subject} due to missing classes.")
            continue
            
        rng = np.random.default_rng(42)
        rng.shuffle(chunks_c1)
        rng.shuffle(chunks_c2)
        
        idx_c1_bal = chunks_c1[:min_windows]
        idx_c2_bal = chunks_c2[:min_windows]
        
        bal_idx = np.concatenate([idx_c1_bal, idx_c2_bal])
        rng.shuffle(bal_idx)
        
        bal_chunks = [all_train_chunks[i] for i in bal_idx]
        bal_meta = [all_train_meta[i] for i in bal_idx]
        
        unique_subs = set(m[0] for m in bal_meta)
        unique_trials = set(f"{m[0]}_{m[1]}" for m in bal_meta)
        
        print("\nWindow Distribution AFTER balancing")
        print(f"Track1 windows: {len(idx_c1_bal)}")
        print(f"Track2 windows: {len(idx_c2_bal)}")
        print(f"\nFinal Training Windows: {len(bal_chunks)}")
        print(f"Generated from {len(unique_trials)} unique trials, {len(unique_subs)} subjects")
        print("========================================\n")
        
        train_dataset = LazyKULKuruvilaDataset(bal_chunks, train_data, win_samples)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
        
        model = KuruvilaOriginalCNNLSTM(eeg_channels=8, audio_channels=28, num_spkr=2).to(device)
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total Trainable Parameters: {total_params:,}")

        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        # Exact loss from repo
        criterion = nn.BCELoss()
        
        if args.smoke_test:
            print("\n--- RUNNING SMOKE TEST (1 BATCH) ---")
            model.train()
            batch_x, batch_a1, batch_a2, batch_y = next(iter(train_loader))
            batch_x, batch_a1, batch_a2 = batch_x.to(device), batch_a1.to(device), batch_a2.to(device)
            batch_y = batch_y.to(device)
            one_hot_y = F.one_hot(batch_y, num_classes=2).float()
            
            optimizer.zero_grad()
            logits = model(batch_x, batch_a1, batch_a2, verbose=True)
            loss = criterion(logits, one_hot_y)
            loss.backward()
            optimizer.step()
            print(f"Smoke Test Forward/Backward Pass Complete. Loss: {loss.item():.4f}")
            return
        
        print(f"Printing first 5 training batches to verify balance:")
        loader_iter = iter(train_loader)
        for b_idx in range(5):
            try:
                _, _, _, b_y = next(loader_iter)
                b_c1 = (b_y == 0).sum().item()
                b_c2 = (b_y == 1).sum().item()
                print(f"Batch {b_idx + 1}")
                print(f"Track1 windows: {b_c1}")
                print(f"Track2 windows: {b_c2}")
            except StopIteration:
                break
        print(f"\nTraining on {len(train_dataset)} chunks | Validating on {val_subject} ({len(val_data)} trials)...")
        
        model = KuruvilaOriginalCNNLSTM(eeg_channels=8, audio_channels=28, num_spkr=2).to(device)
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        criterion = nn.BCELoss()
        
        best_val_loss = float('inf')
        best_model_state = None
        patience_counter = 0
        
        train_losses = []
        val_losses = []
        
        for epoch in range(MAX_EPOCHS):
            model.train()
            epoch_loss = 0.0
            for batch_x, batch_a1, batch_a2, batch_y in train_loader:
                batch_x = batch_x.to(device)
                batch_a1 = batch_a1.to(device)
                batch_a2 = batch_a2.to(device)
                batch_y = batch_y.to(device)
                
                one_hot_y = F.one_hot(batch_y, num_classes=2).float()
                
                optimizer.zero_grad()
                logits = model(batch_x, batch_a1, batch_a2)
                loss = criterion(logits, one_hot_y)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                
            train_loss = epoch_loss / len(train_loader)
            val_loss, val_win_acc, val_trial_acc, _, _ = evaluate_fold(model, val_data, device, DECISION_WINDOW_SEC, FS, criterion)
            
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
        
        _, test_win_acc, test_trial_acc, p_t1, p_t2 = evaluate_fold(model, test_data, device, DECISION_WINDOW_SEC, FS)
        print(f"\n--> Fold Result ({held_out_subject}): Window Acc: {test_win_acc*100:.2f}% | Trial Acc: {test_trial_acc*100:.2f}%")
        
        print("\n--- Forensic Analysis on Held-out Subject ---")
        print(f"Total Predicted Track 1 (Class 0): {p_t1}")
        print(f"Total Predicted Track 2 (Class 1): {p_t2}")
        if p_t1 == 0 or p_t2 == 0:
            print("[WARNING] Model COLLAPSED into a single class prediction on the test set!")
            
        results.append({
            "held_out_subject": held_out_subject,
            "window_acc": test_win_acc,
            "trial_acc": test_trial_acc,
            "pred_t1": p_t1,
            "pred_t2": p_t2
        })
        
        global_total_trials += len(test_data)
        global_correct_trials += (test_trial_acc * len(test_data))
        
    print(f"\n{'='*50}")
    print("FINAL LOSO RESULTS (Kuruvila CNN-LSTM)")
    print(f"{'='*50}")
    
    with open(out_dir / "kuruvila_loso_summary.csv", "w", newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["held_out_subject", "window_acc", "trial_acc", "pred_t1", "pred_t2"])
        writer.writeheader()
        
        for r in results:
            print(f"Subject {r['held_out_subject']}: Window = {r['window_acc']*100:.1f}% | Trial = {r['trial_acc']*100:.1f}% | T1/T2: {r['pred_t1']}/{r['pred_t2']}")
            writer.writerow(r)
            
    mean_trial_acc = global_correct_trials / global_total_trials if global_total_trials > 0 else 0
    print(f"\nOverall Mean Trial Accuracy: {mean_trial_acc*100:.2f}%")

if __name__ == "__main__":
    main()
