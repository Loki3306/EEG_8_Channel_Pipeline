import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import numpy as np
import argparse
import pandas as pd
import math

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from data.kul_cached_dataset import KULCachedLoader
from models.pearson_aad import PearsonAADModel, NegativePearsonLoss

class EnvelopeDataset(Dataset):
    """
    Randomly samples windows from trials for Pearson regression.
    Provides (EEG, Attended_Audio, Subject_Label).
    """
    def __init__(self, subject_data_dict, test_sub, sub_to_idx, window_sec=10.0, fs=64, steps_per_epoch=200, batch_size=128):
        self.trials = []
        for sub, trials in subject_data_dict.items():
            if sub != test_sub:
                for t in trials:
                    self.trials.append((sub_to_idx[sub], t))
                
        self.win_samples = int(window_sec * fs)
        self.num_samples = steps_per_epoch * batch_size
        
        self.std_trials = []
        for sub_label, t in self.trials:
            eeg = t["eeg"]
            a = t["audio_a"]
            
            # Standardize EEG per trial
            eeg = (eeg - eeg.mean(dim=1, keepdim=True)) / (eeg.std(dim=1, keepdim=True) + 1e-12)
            # Audio is NOT standardized to mean 0, variance 1 because Pearson correlation is scale invariant
            # but standardizing helps with gradient scales during neural training.
            a = (a - a.mean(dim=1, keepdim=True)) / (a.std(dim=1, keepdim=True) + 1e-12)
            
            self.std_trials.append((eeg, a, sub_label))

    def __len__(self):
        return self.num_samples
        
    def __getitem__(self, idx):
        trial_idx = torch.randint(0, len(self.std_trials), (1,)).item()
        eeg, a, sub_label = self.std_trials[trial_idx]
        
        max_start = eeg.shape[1] - self.win_samples
        start = torch.randint(0, max_start + 1, (1,)).item()
        end = start + self.win_samples
        
        return eeg[:, start:end], a[:, start:end], sub_label

def pearson_corr(x, y):
    x_mean = x.mean(axis=-1, keepdims=True)
    y_mean = y.mean(axis=-1, keepdims=True)
    x_c = x - x_mean
    y_c = y - y_mean
    cov = (x_c * y_c).sum(axis=-1)
    std = np.sqrt((x_c**2).sum(axis=-1) * (y_c**2).sum(axis=-1) + 1e-8)
    return cov / std

def evaluate_pearson(model, all_subject_data, test_sub, device, window_sec=10.0, hop_sec=1.0, fs=64):
    """
    Evaluates Pearson correlation against attended and unattended envelopes.
    Standard AAD Evaluation Protocol.
    """
    model.eval()
    win_samples = int(window_sec * fs)
    hop_samples = int(hop_sec * fs)
    
    total_trials = len(all_subject_data[test_sub])
    windows_total = 0
    windows_correct = 0
    trials_correct = 0
    
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
            a_wins = torch.stack(a_wins).cpu().numpy()
            b_wins = torch.stack(b_wins).cpu().numpy()
            
            # Predict attended envelope
            env_pred = model.predict(eeg_wins).cpu().numpy()
            
            # Compute correlations
            # Shape: [Batch, 28, Time]. We average the correlation over the 28 subbands.
            corr_a = pearson_corr(env_pred, a_wins).mean(axis=1)
            corr_b = pearson_corr(env_pred, b_wins).mean(axis=1)
            
            wins_correct = (corr_a > corr_b).sum()
            num_wins = len(env_pred)
            
            windows_total += num_wins
            windows_correct += wins_correct
            
            if wins_correct > num_wins / 2.0:
                trials_correct += 1
                
    return {
        "win_acc": windows_correct / windows_total,
        "trial_acc": trials_correct / total_trials
    }

def main():
    parser = argparse.ArgumentParser(description="Subject-Invariant Pearson Regression AAD Training")
    parser.add_argument("--smoke", action="store_true", help="Run fast smoke test (fewer epochs/steps)")
    parser.add_argument("--subjects", nargs="+", default=None, help="Specific subjects to test (e.g., S1 S2). Default is all.")
    parser.add_argument("--grl_lambda", type=float, default=1.0, help="Weight of the Gradient Reversal Layer subject discriminator loss.")
    parser.add_argument("--cache_dir", type=str, default=None, help="Path to KUL cache")
    args = parser.parse_args()

    print("="*70)
    print("   SUBJECT-INVARIANT PEARSON REGRESSION AAD")
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
    if args.cache_dir:
        cache_dir = Path(args.cache_dir)
    elif Path("/kaggle/input/datasets/lowk1ee/kul-preprocessed-cache/data/processed_kul").exists():
        cache_dir = Path("/kaggle/input/datasets/lowk1ee/kul-preprocessed-cache/data/processed_kul")
    elif Path("/kaggle/input/kul-preprocessed-cache/data/processed_kul").exists():
        cache_dir = Path("/kaggle/input/kul-preprocessed-cache/data/processed_kul")
    else:
        cache_dir = REPO_ROOT / "data" / "processed_kul"

    loader = KULCachedLoader(cache_dir)
    try:
        all_subject_data = loader.load_all()
    except FileNotFoundError:
        print("Data cache not found. Run build_kul_cache.py first.")
        return

    if args.subjects:
        subs = args.subjects
    else:
        subs = sorted(list(all_subject_data.keys()), key=lambda x: int(x[1:]))
        
    out_dir = REPO_ROOT / "results" / "pearson_regression"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    all_results = []
    
    eval_epochs = [5] if args.smoke else [5, 10, 20, 40]

    for test_sub in subs:
        print(f"\n--- Fold: Test Subject {test_sub} ---")
        
        train_subs = [s for s in all_subject_data.keys() if s != test_sub]
        train_subs = sorted(train_subs, key=lambda x: int(x[1:]))
        sub_to_idx = {s: i for i, s in enumerate(train_subs)}
        
        train_ds = EnvelopeDataset(all_subject_data, test_sub, sub_to_idx, window_sec=window_sec, steps_per_epoch=steps_per_epoch, batch_size=batch_size)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True)
        
        print(f"  Training windows per epoch: {len(train_ds)}")
        print(f"  Testing trials: {len(all_subject_data[test_sub])}")
        
        print("\n  [Epoch  0] Evaluating Random Encoder Baseline...")
        model = PearsonAADModel(num_subjects=len(train_subs)).to(device)
        model.grl.lam = 0.0 # No adversarial loss during eval
        random_results = evaluate_pearson(model, all_subject_data, test_sub, device, window_sec=window_sec, hop_sec=hop_sec)
        
        all_results.append({
            "Subject": test_sub,
            "Epoch": 0,
            "Window_Acc": random_results["win_acc"],
            "Trial_Acc": random_results["trial_acc"],
        })
        print(f"    Pearson | Window: {random_results['win_acc']*100:.1f}% | Trial: {random_results['trial_acc']*100:.1f}%")
        
        print("\n  Training Neural Pearson Encoder...")
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        
        criterion = NegativePearsonLoss()
        
        for epoch in range(1, epochs + 1):
            model.train()
            total_loss = 0.0
            total_grl = 0.0
            
            num_batches = len(train_loader)
            for step, (eeg_batch, a_pos_batch, subj_batch) in enumerate(train_loader):
                # DANN schedule for lambda
                p = float(epoch - 1 + step / num_batches) / epochs
                current_lambda = (2.0 / (1.0 + math.exp(-10.0 * p)) - 1.0) * args.grl_lambda
                model.grl.lam = current_lambda
                
                eeg_batch = eeg_batch.to(device)
                a_pos_batch = a_pos_batch.to(device)
                subj_batch = subj_batch.to(device)
                
                optimizer.zero_grad()
                env_pred, subj_logits = model(eeg_batch)
                
                pearson_loss = criterion(env_pred, a_pos_batch)
                grl_loss = torch.nn.functional.cross_entropy(subj_logits, subj_batch)
                
                loss = pearson_loss + grl_loss
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                total_loss += pearson_loss.item()
                total_grl += grl_loss.item()
                
            scheduler.step()
            avg_loss = total_loss / num_batches
            avg_grl = total_grl / num_batches
            
            if epoch == 1 or epoch % (1 if args.smoke else 10) == 0:
                print(f"    Epoch {epoch:2d}/{epochs} | Pearson Loss: {avg_loss:.4f} | Subj Loss: {avg_grl:.4f} | GRL lam: {current_lambda:.2f}")
                
            if epoch in eval_epochs:
                print(f"\n  [Epoch {epoch:2d}] Evaluating Trained Model...")
                trained_results = evaluate_pearson(model, all_subject_data, test_sub, device, window_sec=window_sec, hop_sec=hop_sec)
                
                all_results.append({
                    "Subject": test_sub,
                    "Epoch": epoch,
                    "Window_Acc": trained_results["win_acc"],
                    "Trial_Acc": trained_results["trial_acc"],
                })
                print(f"    Pearson | Window: {trained_results['win_acc']*100:.1f}% | Trial: {trained_results['trial_acc']*100:.1f}%")
                print()
        
    df = pd.DataFrame(all_results)
    csv_path = out_dir / "pearson_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved pearson results to {csv_path}")

if __name__ == "__main__":
    main()
