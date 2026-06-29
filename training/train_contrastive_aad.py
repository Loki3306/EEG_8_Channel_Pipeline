import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from data.kul_cached_dataset import KULCachedLoader
from models.contrastive_aad import ContrastiveAADModel

class ContrastiveDataset(Dataset):
    """
    Slices trials into fixed windows and returns (EEG, Attended_Audio) pairs for InfoNCE training.
    """
    def __init__(self, subject_data_dict, test_sub, is_train=True, window_sec=2.0, hop_sec=1.0, fs=64):
        self.samples = []
        win_samples = int(window_sec * fs)
        hop_samples = int(hop_sec * fs)
        
        for sub, trials in subject_data_dict.items():
            if is_train and sub == test_sub: continue
            if not is_train and sub != test_sub: continue
            
            for t in trials:
                eeg = t["eeg"] # [8, time]
                audio = t["audio_a"] # [28, time]
                
                # Standardize
                e_mean = eeg.mean(dim=1, keepdim=True)
                e_std = eeg.std(dim=1, keepdim=True) + 1e-12
                eeg = (eeg - e_mean) / e_std
                
                a_mean = audio.mean(dim=1, keepdim=True)
                a_std = audio.std(dim=1, keepdim=True) + 1e-12
                audio = (audio - a_mean) / a_std
                
                start = 0
                while start + win_samples <= eeg.shape[1]:
                    end = start + win_samples
                    self.samples.append((eeg[:, start:end], audio[:, start:end]))
                    start += hop_samples

    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        eeg, audio = self.samples[idx]
        return eeg, audio


def evaluate_zero_shot(model, all_subject_data, test_sub, device, window_sec=2.0, hop_sec=1.0, fs=64):
    """
    Evaluates the contrastive model using zero-shot cosine similarity on the test subject.
    """
    model.eval()
    correct_trials = 0
    total_trials = len(all_subject_data[test_sub])
    
    win_samples = int(window_sec * fs)
    hop_samples = int(hop_sec * fs)
    
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
            
            # Get normalized embeddings
            e_emb, a_emb = model.get_embeddings(eeg_wins, a_wins)
            _, b_emb = model.get_embeddings(eeg_wins, b_wins)
            
            # Compute cosine similarity (dot product of L2 normalized vectors)
            sim_a = (e_emb * a_emb).sum(dim=1)
            sim_b = (e_emb * b_emb).sum(dim=1)
            
            wins_correct = (sim_a > sim_b).sum().item()
            total_windows_correct += wins_correct
            total_windows += len(sim_a)
            
            if wins_correct > len(sim_a) / 2.0:
                correct_trials += 1
                
    return total_windows_correct / total_windows, correct_trials / total_trials


def main():
    print("="*60)
    print("      GLOBAL CONTRASTIVE AAD TRAINING (LOSO)")
    print("="*60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Parameters
    batch_size = 256 # Large batch size for InfoNCE
    epochs = 20
    lr = 3e-4
    window_sec = 2.0
    hop_sec = 1.0

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
        print(f"\n--- Testing Subject: {test_sub} ---")
        
        train_ds = ContrastiveDataset(all_subject_data, test_sub, is_train=True, window_sec=window_sec, hop_sec=hop_sec)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True)
        
        print(f"Train samples: {len(train_ds)}")
        
        model = ContrastiveAADModel().to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        
        # Training Loop
        for epoch in range(1, epochs + 1):
            model.train()
            total_loss = 0.0
            
            for eeg_batch, audio_batch in train_loader:
                eeg_batch = eeg_batch.to(device)
                audio_batch = audio_batch.to(device)
                
                optimizer.zero_grad()
                loss = model(eeg_batch, audio_batch)
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                total_loss += loss.item()
                
            avg_loss = total_loss / len(train_loader)
            if epoch == 1 or epoch % 5 == 0:
                print(f"  Epoch {epoch:2d}/{epochs} | InfoNCE Loss: {avg_loss:.4f} | Temp: {model.criterion.logit_scale.exp().item():.2f}")
                
        # Zero-Shot Evaluation
        win_acc, trial_acc = evaluate_zero_shot(model, all_subject_data, test_sub, device, window_sec=window_sec, hop_sec=hop_sec)
        
        all_window_accs.append(win_acc)
        all_trial_accs.append(trial_acc)
        
        print(f"  --> Window Acc: {win_acc*100:.1f}% | Trial Acc: {trial_acc*100:.1f}%")
        
    print("\n" + "="*60)
    print(f"Global LOSO Contrastive Median Trial Acc:  {np.median(all_trial_accs)*100:.1f}%")
    print(f"Global LOSO Contrastive Mean Trial Acc:    {np.mean(all_trial_accs)*100:.1f}%")
    print(f"Global LOSO Contrastive Mean Window Acc:   {np.mean(all_window_accs)*100:.1f}%")
    print("="*60)

if __name__ == "__main__":
    main()
