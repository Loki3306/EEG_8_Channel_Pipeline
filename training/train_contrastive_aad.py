import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from tqdm import tqdm
import numpy as np
from sklearn.linear_model import LogisticRegression
import argparse
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from data.kul_cached_dataset import KULCachedLoader
from models.contrastive_aad import ContrastiveAADModel

class ContrastiveDataset(Dataset):
    """
    Randomly samples windows from trials for pretraining.
    Provides (EEG, Attended_Audio, Unattended_Audio) for hard negatives.
    """
    def __init__(self, subject_data_dict, test_sub, window_sec=10.0, fs=64, steps_per_epoch=200, batch_size=128):
        self.trials = []
        for sub, trials in subject_data_dict.items():
            if sub != test_sub:
                self.trials.extend(trials)
                
        self.win_samples = int(window_sec * fs)
        self.num_samples = steps_per_epoch * batch_size
        
        # Pre-standardize all trials to avoid re-computing per window
        self.std_trials = []
        for t in self.trials:
            eeg = t["eeg"]
            a = t["audio_a"]
            b = t["audio_b"]
            
            eeg = (eeg - eeg.mean(dim=1, keepdim=True)) / (eeg.std(dim=1, keepdim=True) + 1e-12)
            a = (a - a.mean(dim=1, keepdim=True)) / (a.std(dim=1, keepdim=True) + 1e-12)
            b = (b - b.mean(dim=1, keepdim=True)) / (b.std(dim=1, keepdim=True) + 1e-12)
            
            self.std_trials.append((eeg, a, b))

    def __len__(self):
        return self.num_samples
        
    def __getitem__(self, idx):
        trial_idx = torch.randint(0, len(self.std_trials), (1,)).item()
        eeg, a, b = self.std_trials[trial_idx]
        
        max_start = eeg.shape[1] - self.win_samples
        start = torch.randint(0, max_start + 1, (1,)).item()
        end = start + self.win_samples
        
        return eeg[:, start:end], a[:, start:end], b[:, start:end]


def evaluate_ablated_probes(model, all_subject_data, test_sub, device, window_sec=10.0, hop_sec=1.0, fs=64):
    """
    Trains and evaluates multiple linear probes to measure where information originates.
    Note: EEG-only probe was removed as it is mathematically invalid for this task 
    (there is no audio-independent 'attended' label for the EEG itself).
    Returns dictionary with metrics for Audio-Only and Joint representations.
    """
    model.eval()
    win_samples = int(window_sec * fs)
    hop_samples = int(hop_sec * fs)
    
    train_features_joint, train_features_aud = [], []
    train_labels = []
    
    with torch.no_grad():
        for sub, trials in all_subject_data.items():
            if sub == test_sub: continue
            
            for t in trials:
                eeg = t["eeg"]
                a = t["audio_a"]
                b = t["audio_b"]
                
                eeg = (eeg - eeg.mean(dim=1, keepdim=True)) / (eeg.std(dim=1, keepdim=True) + 1e-12)
                a = (a - a.mean(dim=1, keepdim=True)) / (a.std(dim=1, keepdim=True) + 1e-12)
                b = (b - b.mean(dim=1, keepdim=True)) / (b.std(dim=1, keepdim=True) + 1e-12)
                
                eeg_wins, a_wins, b_wins = [], [], []
                start = 0
                while start + win_samples <= eeg.shape[1]:
                    end = start + win_samples
                    eeg_wins.append(eeg[:, start:end])
                    a_wins.append(a[:, start:end])
                    b_wins.append(b[:, start:end])
                    start += win_samples # Non-overlapping for training
                    
                if not eeg_wins: continue
                eeg_wins = torch.stack(eeg_wins).to(device)
                a_wins = torch.stack(a_wins).to(device)
                b_wins = torch.stack(b_wins).to(device)
                
                e_rep, a_rep = model.get_representations(eeg_wins, a_wins)
                _, b_rep = model.get_representations(eeg_wins, b_wins)
                
                e_rep = e_rep.cpu().numpy()
                a_rep = a_rep.cpu().numpy()
                b_rep = b_rep.cpu().numpy()
                
                # Positive pairs
                train_features_joint.append(e_rep * a_rep)
                train_features_aud.append(a_rep)
                train_labels.append(np.ones(len(e_rep)))
                
                # Negative pairs
                train_features_joint.append(e_rep * b_rep)
                train_features_aud.append(b_rep)
                train_labels.append(np.zeros(len(e_rep)))
                
    X_joint = np.concatenate(train_features_joint, axis=0)
    X_aud = np.concatenate(train_features_aud, axis=0)
    y_train = np.concatenate(train_labels, axis=0)
    
    clf_joint = LogisticRegression(max_iter=1000, n_jobs=-1).fit(X_joint, y_train)
    clf_aud = LogisticRegression(max_iter=1000, n_jobs=-1).fit(X_aud, y_train)
    
    # --- Evaluation ---
    total_trials = len(all_subject_data[test_sub])
    
    metrics = {
        "joint": {"windows": 0, "windows_correct": 0, "trials_correct": 0},
        "aud": {"windows": 0, "windows_correct": 0, "trials_correct": 0}
    }
    
    with torch.no_grad():
        for t in all_subject_data[test_sub]:
            eeg = t["eeg"]
            a = t["audio_a"]
            b = t["audio_b"]
            
            eeg = (eeg - eeg.mean(dim=1, keepdim=True)) / (eeg.std(dim=1, keepdim=True) + 1e-12)
            a = (a - a.mean(dim=1, keepdim=True)) / (a.std(dim=1, keepdim=True) + 1e-12)
            b = (b - b.mean(dim=1, keepdim=True)) / (b.std(dim=1, keepdim=True) + 1e-12)
            
            eeg_wins, a_wins, b_wins = [], [], []
            start = 0
            while start + win_samples <= eeg.shape[1]:
                end = start + win_samples
                eeg_wins.append(eeg[:, start:end])
                a_wins.append(a[:, start:end])
                b_wins.append(b[:, start:end])
                start += hop_samples
                
            if not eeg_wins: continue
            
            eeg_wins = torch.stack(eeg_wins).to(device)
            a_wins = torch.stack(a_wins).to(device)
            b_wins = torch.stack(b_wins).to(device)
            
            e_rep, a_rep = model.get_representations(eeg_wins, a_wins)
            _, b_rep = model.get_representations(eeg_wins, b_wins)
            
            e_rep = e_rep.cpu().numpy()
            a_rep = a_rep.cpu().numpy()
            b_rep = b_rep.cpu().numpy()
            
            num_wins = len(e_rep)
            
            for key, (feat_a, feat_b, clf) in [
                ("joint", (e_rep * a_rep, e_rep * b_rep, clf_joint)),
                ("aud", (a_rep, b_rep, clf_aud))
            ]:
                prob_a = clf.predict_proba(feat_a)[:, 1]
                prob_b = clf.predict_proba(feat_b)[:, 1]
                
                wins_correct = (prob_a > prob_b).sum()
                metrics[key]["windows"] += num_wins
                metrics[key]["windows_correct"] += wins_correct
                
                if wins_correct > num_wins / 2.0:
                    metrics[key]["trials_correct"] += 1
                    
    results = {}
    for k in metrics.keys():
        results[k] = {
            "win_acc": metrics[k]["windows_correct"] / metrics[k]["windows"],
            "trial_acc": metrics[k]["trials_correct"] / total_trials
        }
        
    return results

def main():
    parser = argparse.ArgumentParser(description="Contrastive AAD Training with Ablations")
    parser.add_argument("--smoke", action="store_true", help="Run fast smoke test (fewer epochs/steps)")
    parser.add_argument("--subjects", nargs="+", default=None, help="Specific subjects to test (e.g., S1 S2). Default is all.")
    args = parser.parse_args()

    print("="*70)
    print("   REVISED CONTRASTIVE AAD TRAINING (InfoNCE + Ablation Probes)")
    print("="*70)
    
    if args.smoke:
        print(">>> SMOKE TEST MODE ENABLED <<<")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Parameters
    batch_size = 64 if args.smoke else 128
    epochs = 5 if args.smoke else 40
    lr = 3e-4
    window_sec = 10.0
    hop_sec = 1.0
    steps_per_epoch = 25 if args.smoke else 150

    # Load Data
    loader = KULCachedLoader(REPO_ROOT / "data" / "processed_kul")
    try:
        all_subject_data = loader.load_all()
    except FileNotFoundError:
        print("Data cache not found. Run build_kul_cache.py first.")
        return

    if args.subjects:
        subs = args.subjects
    else:
        subs = sorted(list(all_subject_data.keys()), key=lambda x: int(x[1:]))

    out_dir = REPO_ROOT / "results" / "contrastive"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    all_results = []
    
    eval_epochs = [5] if args.smoke else [5, 10, 20, 40]

    for test_sub in subs:
        print(f"\n--- Fold: Test Subject {test_sub} ---")
        
        train_ds = ContrastiveDataset(all_subject_data, test_sub, window_sec=window_sec, steps_per_epoch=steps_per_epoch, batch_size=batch_size)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True)
        
        print(f"  Training windows per epoch: {len(train_ds)}")
        print(f"  Testing trials: {len(all_subject_data[test_sub])}")
        
        # 1. Evaluate Random Encoder Baseline (Epoch 0)
        print("\n  [Epoch  0] Evaluating Random Encoder Baseline...")
        model = ContrastiveAADModel().to(device)
        random_results = evaluate_ablated_probes(model, all_subject_data, test_sub, device, window_sec=window_sec, hop_sec=hop_sec)
        
        all_results.append({
            "Subject": test_sub,
            "Epoch": 0,
            "Audio_Only_Win": random_results["aud"]["win_acc"],
            "Audio_Only_Trial": random_results["aud"]["trial_acc"],
            "Joint_Win": random_results["joint"]["win_acc"],
            "Joint_Trial": random_results["joint"]["trial_acc"],
        })
        print(f"    Audio-only | Window: {random_results['aud']['win_acc']*100:.1f}% | Trial: {random_results['aud']['trial_acc']*100:.1f}%")
        print(f"    Joint      | Window: {random_results['joint']['win_acc']*100:.1f}% | Trial: {random_results['joint']['trial_acc']*100:.1f}%")
        
        # 2. Train Model and Evaluate at Intervals
        print("\n  Pretraining Contrastive Encoders...")
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        
        for epoch in range(1, epochs + 1):
            model.train()
            total_loss = 0.0
            for eeg_batch, a_pos_batch, a_neg_batch in train_loader:
                eeg_batch = eeg_batch.to(device)
                a_pos_batch = a_pos_batch.to(device)
                a_neg_batch = a_neg_batch.to(device)
                
                optimizer.zero_grad()
                loss = model(eeg_batch, a_pos_batch, a_neg_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                total_loss += loss.item()
                
            scheduler.step()
            avg_loss = total_loss / len(train_loader)
            
            if epoch == 1 or epoch % (1 if args.smoke else 10) == 0:
                print(f"    Epoch {epoch:2d}/{epochs} | InfoNCE Loss: {avg_loss:.4f} | Temp: {model.criterion.logit_scale.exp().clamp(max=100.0).item():.2f}")
                
            if epoch in eval_epochs:
                print(f"\n  [Epoch {epoch:2d}] Evaluating Trained Probes...")
                trained_results = evaluate_ablated_probes(model, all_subject_data, test_sub, device, window_sec=window_sec, hop_sec=hop_sec)
                
                all_results.append({
                    "Subject": test_sub,
                    "Epoch": epoch,
                    "Audio_Only_Win": trained_results["aud"]["win_acc"],
                    "Audio_Only_Trial": trained_results["aud"]["trial_acc"],
                    "Joint_Win": trained_results["joint"]["win_acc"],
                    "Joint_Trial": trained_results["joint"]["trial_acc"],
                })
                print(f"    Audio-only | Window: {trained_results['aud']['win_acc']*100:.1f}% | Trial: {trained_results['aud']['trial_acc']*100:.1f}%")
                print(f"    Joint      | Window: {trained_results['joint']['win_acc']*100:.1f}% | Trial: {trained_results['joint']['trial_acc']*100:.1f}%")
                print()
        
    df = pd.DataFrame(all_results)
    csv_path = out_dir / "representation_ablation.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved representation ablations to {csv_path}")

if __name__ == "__main__":
    main()
