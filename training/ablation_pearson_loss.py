import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import numpy as np
import argparse
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from data.kul_cached_dataset import KULCachedLoader
from models.pearson_aad import PearsonAADModel

class DiscriminativePearsonLoss(nn.Module):
    """
    Computes Pearson correlation loss with a discriminative objective.
    Supports reconstruction, margin, and hinge-margin formulations.
    """
    def __init__(self, mode='hinge', target_margin=0.10):
        super().__init__()
        self.mode = mode
        self.target_margin = target_margin
        
    def _pearson(self, preds, targets):
        preds_mean = preds.mean(dim=-1, keepdim=True)
        targets_mean = targets.mean(dim=-1, keepdim=True)
        
        preds_centered = preds - preds_mean
        targets_centered = targets - targets_mean
        
        cov = (preds_centered * targets_centered).sum(dim=-1)
        
        preds_std = torch.sqrt((preds_centered**2).sum(dim=-1) + 1e-8)
        targets_std = torch.sqrt((targets_centered**2).sum(dim=-1) + 1e-8)
        
        corr = cov / (preds_std * targets_std)
        return corr.mean()
        
    def forward(self, preds, target_att, target_unatt):
        corr_att = self._pearson(preds, target_att)
        
        if self.mode == 'reconstruction' or target_unatt is None:
            # Baseline: Maximize attended correlation (minimize negative correlation)
            # This ignores unattended entirely
            return -corr_att, corr_att.item(), 0.0, 0.0
            
        corr_unatt = self._pearson(preds, target_unatt)
        margin = corr_att - corr_unatt
        
        if self.mode == 'margin':
            # Maximize margin (minimize negative margin)
            loss = -margin
        elif self.mode == 'hinge':
            # Hinge loss: only penalize if margin is below target_margin
            # max(0, target_margin - margin)
            loss = torch.clamp(self.target_margin - margin, min=0.0)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")
            
        return loss, corr_att.item(), corr_unatt.item(), margin.item()


class DiscriminativeEnvelopeDataset(Dataset):
    """
    Randomly samples windows from trials for discriminative Pearson regression.
    Provides (EEG, Attended_Audio, Unattended_Audio, Subject_Label).
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
            b = t["audio_b"]
            
            # Standardize
            eeg = (eeg - eeg.mean(dim=1, keepdim=True)) / (eeg.std(dim=1, keepdim=True) + 1e-12)
            a = (a - a.mean(dim=1, keepdim=True)) / (a.std(dim=1, keepdim=True) + 1e-12)
            b = (b - b.mean(dim=1, keepdim=True)) / (b.std(dim=1, keepdim=True) + 1e-12)
            
            self.std_trials.append((eeg, a, b, sub_label))

    def __len__(self):
        return self.num_samples
        
    def __getitem__(self, idx):
        trial_idx = torch.randint(0, len(self.std_trials), (1,)).item()
        eeg, a, b, sub_label = self.std_trials[trial_idx]
        
        max_start = eeg.shape[1] - self.win_samples
        start = torch.randint(0, max_start + 1, (1,)).item()
        end = start + self.win_samples
        
        return eeg[:, start:end], a[:, start:end], b[:, start:end], sub_label

def pearson_corr_np(x, y):
    x_mean = x.mean(axis=-1, keepdims=True)
    y_mean = y.mean(axis=-1, keepdims=True)
    x_c = x - x_mean
    y_c = y - y_mean
    cov = (x_c * y_c).sum(axis=-1)
    std = np.sqrt((x_c**2).sum(axis=-1) * (y_c**2).sum(axis=-1) + 1e-8)
    return cov / std

def evaluate_pearson(model, all_subject_data, test_sub, device, variant_name, window_sec=10.0, hop_sec=1.0, fs=64):
    """
    Evaluates Pearson correlation against attended and unattended envelopes.
    Prints per-trial diagnostics.
    """
    model.eval()
    win_samples = int(window_sec * fs)
    hop_samples = int(hop_sec * fs)
    
    total_trials = len(all_subject_data[test_sub])
    windows_total = 0
    windows_correct = 0
    trials_correct = 0
    
    all_corr_a = []
    all_corr_b = []
    
    print("\n  --- Per-Trial Diagnostics ---")
    
    with torch.no_grad():
        for i, t in enumerate(all_subject_data[test_sub]):
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
            
            env_pred = model.predict(eeg_wins).cpu().numpy()
            
            corr_a = pearson_corr_np(env_pred, a_wins).mean(axis=1)
            corr_b = pearson_corr_np(env_pred, b_wins).mean(axis=1)
            
            trial_corr_a = np.mean(corr_a)
            trial_corr_b = np.mean(corr_b)
            trial_margin = trial_corr_a - trial_corr_b
            
            print(f"  Trial {i+1:2d} | corr(att): {trial_corr_a:7.4f} | corr(unatt): {trial_corr_b:7.4f} | margin: {trial_margin:7.4f}")
            
            all_corr_a.extend(corr_a.tolist())
            all_corr_b.extend(corr_b.tolist())
            
            wins_correct = (corr_a > corr_b).sum()
            num_wins = len(env_pred)
            
            windows_total += num_wins
            windows_correct += wins_correct
            
            if wins_correct > num_wins / 2.0:
                trials_correct += 1
                
    return {
        "mean_pearson_att": np.mean(all_corr_a),
        "mean_pearson_unatt": np.mean(all_corr_b),
        "margin": np.mean(all_corr_a) - np.mean(all_corr_b),
        "win_acc": windows_correct / windows_total,
        "trial_acc": trials_correct / total_trials
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="Run fast smoke test")
    parser.add_argument("--cache_dir", type=str, default=None)
    args = parser.parse_args()

    # Reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    print("="*70)
    print("   DISCRIMINATIVE PEARSON LOSS ABLATION EXPERIMENT")
    print("="*70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Parameters
    batch_size = 64 if args.smoke else 128
    epochs = 5
    lr = 3e-4
    window_sec = 10.0
    hop_sec = 1.0
    steps_per_epoch = 10 if args.smoke else 100
    test_sub = "S1"

    # We use Variant B (16 Hz)
    temporal_pools = (4, 1)
    target_margin = 0.10

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

    out_dir = REPO_ROOT / "results" / "pearson_loss_ablation"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    variants = [
        ("Reconstruction Pearson", "reconstruction"),
        ("Margin Pearson", "margin"),
        ("Hinge Margin Pearson", "hinge")
    ]
    
    all_results = []
    
    train_subs = [s for s in all_subject_data.keys() if s != test_sub]
    train_subs = sorted(train_subs, key=lambda x: int(x[1:]))
    sub_to_idx = {s: i for i, s in enumerate(train_subs)}
    
    train_ds = DiscriminativeEnvelopeDataset(all_subject_data, test_sub, sub_to_idx, window_sec=window_sec, steps_per_epoch=steps_per_epoch, batch_size=batch_size)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True)
    
    print(f"\nTraining on {len(train_subs)} subjects. Testing on {test_sub}.")
    
    for v_name, mode in variants:
        print(f"\n" + "="*60)
        print(f" TESTING VARIANT: {v_name}")
        print("="*60)
        
        # Reset seed before each model initialization for perfectly controlled weights
        torch.manual_seed(42)
        model = PearsonAADModel(num_subjects=len(train_subs), temporal_pooling_factors=temporal_pools).to(device)
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        criterion = DiscriminativePearsonLoss(mode=mode, target_margin=target_margin)
        
        for epoch in range(1, epochs + 1):
            model.train()
            total_loss = 0.0
            
            # Epoch trackers
            ep_corr_att = []
            ep_corr_unatt = []
            ep_margin = []
            
            num_batches = len(train_loader)
            for step, (eeg_batch, a_pos_batch, b_neg_batch, subj_batch) in enumerate(train_loader):
                eeg_batch = eeg_batch.to(device)
                a_pos_batch = a_pos_batch.to(device)
                b_neg_batch = b_neg_batch.to(device)
                
                optimizer.zero_grad()
                env_pred, subj_logits = model(eeg_batch)
                
                loss, c_att, c_unatt, marg = criterion(env_pred, a_pos_batch, b_neg_batch)
                
                # Purely discriminative, disable GRL
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                total_loss += loss.item()
                ep_corr_att.append(c_att)
                ep_corr_unatt.append(c_unatt)
                ep_margin.append(marg)
                
            avg_loss = total_loss / num_batches
            
            print(f"  [{v_name}] Epoch {epoch}/{epochs} | Train Loss: {avg_loss:.4f}")
            print(f"    corr(att): {np.mean(ep_corr_att):.4f} | corr(unatt): {np.mean(ep_corr_unatt):.4f}")
            print(f"    mean margin: {np.mean(ep_margin):.4f} | median margin: {np.median(ep_margin):.4f}")
            print(f"    min margin: {np.min(ep_margin):.4f} | max margin: {np.max(ep_margin):.4f}")
            
        print("\n  Evaluating Trained Model...")
        res = evaluate_pearson(model, all_subject_data, test_sub, device, variant_name=v_name, window_sec=window_sec, hop_sec=hop_sec)
        
        print(f"\n  Final Results:")
        print(f"    Pearson(att): {res['mean_pearson_att']:.4f}")
        print(f"    Pearson(unatt): {res['mean_pearson_unatt']:.4f}")
        print(f"    Margin: {res['margin']:.4f}")
        print(f"    Window Acc: {res['win_acc']*100:.1f}%")
        print(f"    Trial Acc: {res['trial_acc']*100:.1f}%\n")
        
        all_results.append({
            "Loss": v_name,
            "Corr(att)": round(res['mean_pearson_att'], 4),
            "Corr(unatt)": round(res['mean_pearson_unatt'], 4),
            "Margin": round(res['margin'], 4),
            "Window Acc": f"{res['win_acc']*100:.1f}%",
            "Trial Acc": f"{res['trial_acc']*100:.1f}%"
        })
        
    df = pd.DataFrame(all_results)
    csv_path = out_dir / "ablation_results.csv"
    df.to_csv(csv_path, index=False)
    
    print("\n" + "="*60)
    print(" LOSS ABLATION SUMMARY")
    print("="*60)
    
    markdown_table = df.to_markdown(index=False)
    print(markdown_table)
    
    print(f"\nSaved full results to {csv_path}")

if __name__ == "__main__":
    main()
