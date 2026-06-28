import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from pathlib import Path
from copy import deepcopy
import csv

# Add repo root to path
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from data.kul_cached_dataset import KULCachedLoader, chunk_data
from models.kuruvila_cnn_lstm import KuruvilaCNNLSTM

FS = 64
DECISION_WINDOW_SEC = 10.0
TRAIN_HOP_SEC = 0.5
EVAL_HOP_SEC = 0.5
BATCH_SIZE = 32
LEARNING_RATE = 5e-4
MAX_EPOCHS = 100
PATIENCE = 10

def chunk_data_kuruvila(eeg, audio_spk1, audio_spk2, label, window_sec, hop_sec, fs=FS):
    win_samples = int(window_sec * fs)
    hop_samples = int(hop_sec * fs)
    
    chunks_x = []
    chunks_a1 = []
    chunks_a2 = []
    labels = []
    
    start = 0
    # Shape of eeg: [Channels, Time]
    while start + win_samples <= eeg.shape[1]:
        end = start + win_samples
        chunks_x.append(eeg[:, start:end])
        chunks_a1.append(audio_spk1[:, start:end])
        chunks_a2.append(audio_spk2[:, start:end])
        labels.append(label)
        start += hop_samples
        
    return chunks_x, chunks_a1, chunks_a2, labels

def evaluate_fold(model, eval_data, device, window_sec, fs, criterion=None):
    model.eval()
    
    total_loss = 0.0
    total_batches = 0
    
    total_wins = 0
    correct_wins = 0
    
    correct_trials = 0
    
    pred_t1_count = 0
    pred_t2_count = 0
    
    with torch.no_grad():
        for t in eval_data:
            meta = t["meta"]
            attended_track = str(meta.get('attended_track', 'Unknown'))
            if attended_track not in ['1', '2']:
                continue
                
            label = 0 if attended_track == '1' else 1
            
            # REMAP: Input 1 must ALWAYS be Track 1 (Speaker 1). Input 2 must ALWAYS be Track 2 (Speaker 2).
            # The cache stores audio_a as Attended and audio_b as Unattended.
            if attended_track == '1':
                audio_spk1 = t["audio_a"].numpy()
                audio_spk2 = t["audio_b"].numpy()
            else:
                audio_spk1 = t["audio_b"].numpy()
                audio_spk2 = t["audio_a"].numpy()
                
            cx, ca1, ca2, clabels = chunk_data_kuruvila(t["eeg"].numpy(), audio_spk1, audio_spk2, label, window_sec, EVAL_HOP_SEC, fs)
            
            if not cx:
                continue
                
            cx = torch.FloatTensor(np.array(cx)).to(device)
            ca1 = torch.FloatTensor(np.array(ca1)).to(device)
            ca2 = torch.FloatTensor(np.array(ca2)).to(device)
            cy = torch.LongTensor(np.array(clabels)).to(device)
            
            logits = model(cx, ca1, ca2)
            
            if criterion:
                loss = criterion(logits, cy)
                total_loss += loss.item()
                total_batches += 1
                
            preds = torch.argmax(logits, dim=1)
            
            # Count window predictions
            c_t1 = (preds == 0).sum().item()
            c_t2 = (preds == 1).sum().item()
            pred_t1_count += c_t1
            pred_t2_count += c_t2
            
            correct = (preds == cy).sum().item()
            correct_wins += correct
            total_wins += len(preds)
            
            # Majority vote for trial
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Starting Kuruvila Joint CNN-BiLSTM LOSO Pipeline on {device} (Window: {DECISION_WINDOW_SEC}s)...")
    
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
        
    out_dir = REPO_ROOT / "results" / "kuruvila_loso"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    results = []
    
    global_total_trials = 0
    global_correct_trials = 0
    
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
                
        # Extract all chunks without trial-level balancing
        all_train_x = []
        all_train_a1 = []
        all_train_a2 = []
        all_train_y = []
        all_train_meta = []
        
        for t in train_data:
            meta = t["meta"]
            attended_track = str(meta.get('attended_track', 'Unknown'))
            if attended_track not in ['1', '2']:
                continue
                
            label = 0 if attended_track == '1' else 1
            
            # REMAP: Input 1 must ALWAYS be Track 1 (Speaker 1). Input 2 must ALWAYS be Track 2 (Speaker 2).
            if attended_track == '1':
                audio_spk1 = t["audio_a"].numpy()
                audio_spk2 = t["audio_b"].numpy()
            else:
                audio_spk1 = t["audio_b"].numpy()
                audio_spk2 = t["audio_a"].numpy()
                
            cx, ca1, ca2, clabels = chunk_data_kuruvila(t["eeg"].numpy(), audio_spk1, audio_spk2, label, DECISION_WINDOW_SEC, TRAIN_HOP_SEC, FS)
            
            all_train_x.extend(cx)
            all_train_a1.extend(ca1)
            all_train_a2.extend(ca2)
            all_train_y.extend(clabels)
            
            sub = meta.get("Subject", "Unknown")
            tid = meta.get("TrialID", "Unknown")
            all_train_meta.extend([(sub, tid)] * len(clabels))
            
        all_train_x = np.array(all_train_x)
        all_train_a1 = np.array(all_train_a1)
        all_train_a2 = np.array(all_train_a2)
        all_train_y = np.array(all_train_y)
        
        idx_c1 = np.where(all_train_y == 0)[0]
        idx_c2 = np.where(all_train_y == 1)[0]
        
        print("\n========================================")
        print("Window Distribution BEFORE balancing")
        print(f"Track1 windows: {len(idx_c1)}")
        print(f"Track2 windows: {len(idx_c2)}")
        
        min_windows = min(len(idx_c1), len(idx_c2))
        if min_windows == 0:
            print(f"Skipping fold {held_out_subject} due to missing classes.")
            continue
            
        # Randomly downsample reproducibly
        rng = np.random.default_rng(42)
        rng.shuffle(idx_c1)
        rng.shuffle(idx_c2)
        idx_c1_bal = idx_c1[:min_windows]
        idx_c2_bal = idx_c2[:min_windows]
        
        bal_idx = np.concatenate([idx_c1_bal, idx_c2_bal])
        rng.shuffle(bal_idx)
        
        bal_train_x = all_train_x[bal_idx]
        bal_train_a1 = all_train_a1[bal_idx]
        bal_train_a2 = all_train_a2[bal_idx]
        bal_train_y = all_train_y[bal_idx]
        bal_train_meta = [all_train_meta[i] for i in bal_idx]
        
        unique_subs = set(m[0] for m in bal_train_meta)
        unique_trials = set(f"{m[0]}_{m[1]}" for m in bal_train_meta)
        
        print("\nWindow Distribution AFTER balancing")
        print(f"Track1 windows: {len(idx_c1_bal)}")
        print(f"Track2 windows: {len(idx_c2_bal)}")
        print(f"\nFinal Training Windows: {len(bal_train_y)}")
        print(f"\nGenerated from")
        print(f"{len(unique_trials)} unique trials")
        print(f"{len(unique_subs)} subjects")
        print("========================================\n")
        
        tx = torch.FloatTensor(bal_train_x)
        ta1 = torch.FloatTensor(bal_train_a1)
        ta2 = torch.FloatTensor(bal_train_a2)
        ty = torch.LongTensor(bal_train_y)
        train_dataset = TensorDataset(tx, ta1, ta2, ty)
        
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
        
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
        
        model = KuruvilaCNNLSTM(eeg_channels=8, audio_channels=28, num_classes=2).to(device)
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        criterion = nn.CrossEntropyLoss()
        
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
                
                optimizer.zero_grad()
                logits = model(batch_x, batch_a1, batch_a2)
                loss = criterion(logits, batch_y)
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
