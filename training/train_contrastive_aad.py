import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from tqdm import tqdm
import numpy as np
from sklearn.linear_model import LogisticRegression

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from data.kul_cached_dataset import KULCachedLoader
from models.contrastive_aad import ContrastiveAADModel

class ContrastiveDataset(Dataset):
    """
    Randomly samples 10-second windows from trials for pretraining.
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
        # Randomly select a trial
        trial_idx = torch.randint(0, len(self.std_trials), (1,)).item()
        eeg, a, b = self.std_trials[trial_idx]
        
        # Randomly select a window
        max_start = eeg.shape[1] - self.win_samples
        start = torch.randint(0, max_start + 1, (1,)).item()
        end = start + self.win_samples
        
        return eeg[:, start:end], a[:, start:end], b[:, start:end]


def evaluate_linear_probe(model, all_subject_data, test_sub, device, window_sec=10.0, hop_sec=1.0, fs=64):
    """
    Evaluates the frozen encoder representation quality using a linear probe (Logistic Regression).
    """
    model.eval()
    win_samples = int(window_sec * fs)
    hop_samples = int(hop_sec * fs)
    
    print("    [1/2] Extracting frozen representations for Linear Probe...")
    train_features, train_labels = [], []
    
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
                
                pos_feat = (e_rep * a_rep).cpu().numpy()
                neg_feat = (e_rep * b_rep).cpu().numpy()
                
                train_features.append(pos_feat)
                train_labels.append(np.ones(len(pos_feat)))
                
                train_features.append(neg_feat)
                train_labels.append(np.zeros(len(neg_feat)))
                
    X_train = np.concatenate(train_features, axis=0)
    y_train = np.concatenate(train_labels, axis=0)
    
    print("    [2/2] Training Logistic Regression & Evaluating...")
    clf = LogisticRegression(max_iter=1000, n_jobs=-1)
    clf.fit(X_train, y_train)
    
    correct_trials = 0
    total_trials = len(all_subject_data[test_sub])
    total_windows_correct = 0
    total_windows = 0
    
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
            
            feat_a = (e_rep * a_rep).cpu().numpy()
            feat_b = (e_rep * b_rep).cpu().numpy()
            
            prob_a = clf.predict_proba(feat_a)[:, 1]
            prob_b = clf.predict_proba(feat_b)[:, 1]
            
            wins_correct = (prob_a > prob_b).sum()
            total_windows_correct += wins_correct
            total_windows += len(prob_a)
            
            if wins_correct > len(prob_a) / 2.0:
                correct_trials += 1
                
    return total_windows_correct / total_windows, correct_trials / total_trials


def main():
    print("="*70)
    print("   REVISED CONTRASTIVE AAD TRAINING (InfoNCE + Hard Negatives)")
    print("="*70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Parameters
    batch_size = 128
    epochs = 40
    lr = 3e-4
    window_sec = 10.0
    hop_sec = 1.0
    steps_per_epoch = 150

    # Load Data
    loader = KULCachedLoader(REPO_ROOT / "data" / "processed_kul")
    try:
        all_subject_data = loader.load_all()
    except FileNotFoundError:
        print("Data cache not found. Run build_kul_cache.py first.")
        return

    subs = sorted(list(all_subject_data.keys()), key=lambda x: int(x[1:]))

    all_window_accs = []
    all_trial_accs = []

    out_dir = REPO_ROOT / "results" / "contrastive"
    out_dir.mkdir(parents=True, exist_ok=True)

    for test_sub in subs:
        print(f"\n--- Fold: Test Subject {test_sub} ---")
        
        train_ds = ContrastiveDataset(all_subject_data, test_sub, window_sec=window_sec, steps_per_epoch=steps_per_epoch, batch_size=batch_size)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True)
        
        model = ContrastiveAADModel().to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        
        # Cosine Annealing with Warmup
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        
        # Training Loop
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
            
            if epoch == 1 or epoch % 10 == 0:
                print(f"  Epoch {epoch:2d}/{epochs} | InfoNCE Loss: {avg_loss:.4f} | Temp: {model.criterion.logit_scale.exp().clamp(max=100.0).item():.2f}")
                
        # Frozen Linear Probe Evaluation
        win_acc, trial_acc = evaluate_linear_probe(model, all_subject_data, test_sub, device, window_sec=window_sec, hop_sec=hop_sec)
        
        all_window_accs.append(win_acc)
        all_trial_accs.append(trial_acc)
        
        print(f"  --> Linear Probe Window Acc: {win_acc*100:.1f}% | Trial Acc: {trial_acc*100:.1f}%")
        
    print("\n" + "="*70)
    print(f"Linear Probe Median Trial Acc:  {np.median(all_trial_accs)*100:.1f}%")
    print(f"Linear Probe Mean Trial Acc:    {np.mean(all_trial_accs)*100:.1f}%")
    print(f"Linear Probe Mean Window Acc:   {np.mean(all_window_accs)*100:.1f}%")
    print("="*70)

if __name__ == "__main__":
    main()
