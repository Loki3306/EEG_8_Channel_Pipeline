import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import numpy as np
import argparse
import pandas as pd
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from data.kul_cached_dataset import KULCachedLoader
from models.pearson_aad import PearsonAADModel

class DiscriminativePearsonLoss(nn.Module):
    def __init__(self, target_margin=0.10):
        super().__init__()
        self.target_margin = target_margin
        
    def _pearson(self, preds, targets):
        preds_mean = preds.mean(dim=-1, keepdim=True)
        targets_mean = targets.mean(dim=-1, keepdim=True)
        
        preds_centered = preds - preds_mean
        targets_centered = targets - targets_mean
        
        cov = (preds_centered * targets_centered).sum(dim=-1)
        
        preds_std = torch.sqrt((preds_centered**2).sum(dim=-1) + 1e-8)
        targets_std = torch.sqrt((targets_centered**2).sum(dim=-1) + 1e-8)
        
        return cov / (preds_std * targets_std)
        
    def forward(self, preds, target_att, target_unatt):
        corr_att = self._pearson(preds, target_att).mean()
        corr_unatt = self._pearson(preds, target_unatt).mean()
        margin = corr_att - corr_unatt
        loss = torch.clamp(self.target_margin - margin, min=0.0)
        return loss, corr_att.item(), corr_unatt.item(), margin.item()


class DiscriminativeEnvelopeDataset(Dataset):
    def __init__(self, subject_data_dict, test_sub, sub_to_idx, window_sec=10.0, fs=64, steps_per_epoch=200, batch_size=128, label_shuffle=False):
        self.trials = []
        for sub, trials in subject_data_dict.items():
            if sub != test_sub:
                for t in trials:
                    self.trials.append((sub_to_idx[sub], t))
                
        self.win_samples = int(window_sec * fs)
        self.num_samples = steps_per_epoch * batch_size
        self.label_shuffle = label_shuffle
        
        self.std_trials = []
        for sub_label, t in self.trials:
            eeg = t["eeg"]
            a = t["audio_a"]
            b = t["audio_b"]
            
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
        
        a_win = a[:, start:end]
        b_win = b[:, start:end]
        
        if self.label_shuffle and torch.rand(1).item() > 0.5:
            return eeg[:, start:end], b_win, a_win, sub_label
            
        return eeg[:, start:end], a_win, b_win, sub_label


def pearson_corr_np(x, y):
    x_mean = x.mean(axis=-1, keepdims=True)
    y_mean = y.mean(axis=-1, keepdims=True)
    x_c = x - x_mean
    y_c = y - y_mean
    cov = (x_c * y_c).sum(axis=-1)
    std = np.sqrt((x_c**2).sum(axis=-1) * (y_c**2).sum(axis=-1) + 1e-8)
    return cov / std


def run_experiment(mode_name, test_sub, all_subject_data, out_dir, device, 
                   label_shuffle=False, speaker_swap=False, random_encoder=False, 
                   epochs=5, window_sec=10.0, hop_sec=1.0, steps_per_epoch=100, batch_size=128, smoke=False):
    
    print(f"\n{'='*70}")
    print(f" EXPERIMENT: {mode_name} | Test Subject: {test_sub}")
    print(f"{'='*70}")
    
    # Reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    train_subs = [s for s in all_subject_data.keys() if s != test_sub]
    train_subs = sorted(train_subs, key=lambda x: int(x[1:]))
    sub_to_idx = {s: i for i, s in enumerate(train_subs)}
    
    train_ds = DiscriminativeEnvelopeDataset(all_subject_data, test_sub, sub_to_idx, 
                                             window_sec=window_sec, steps_per_epoch=steps_per_epoch, 
                                             batch_size=batch_size, label_shuffle=label_shuffle)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True)
    
    model = PearsonAADModel(num_subjects=len(train_subs), temporal_pooling_factors=(4, 1)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    criterion = DiscriminativePearsonLoss(target_margin=0.10)
    
    epoch_train_margins = []
    epoch_val_margins = []
    epoch_val_accs = []
    
    if not random_encoder:
        for epoch in range(1, epochs + 1):
            model.train()
            
            ep_margin = []
            ep_hinge_loss = []
            hinge_violations = 0
            total_samples = 0
            
            for eeg_batch, a_batch, b_batch, subj_batch in train_loader:
                eeg_batch = eeg_batch.to(device)
                a_batch = a_batch.to(device)
                b_batch = b_batch.to(device)
                
                optimizer.zero_grad()
                env_pred, _ = model(eeg_batch)
                
                # Manual loss to track hinge violations properly per sample
                # Calculate pearson per batch element to count violations
                c_att = criterion._pearson(env_pred, a_batch).mean(dim=1)
                c_unatt = criterion._pearson(env_pred, b_batch).mean(dim=1)
                margins = c_att - c_unatt
                
                violations = (margins < criterion.target_margin).sum().item()
                hinge_violations += violations
                total_samples += len(margins)
                
                loss = torch.clamp(criterion.target_margin - margins, min=0.0).mean()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                ep_margin.append(margins.mean().item())
                ep_hinge_loss.append(loss.item())
                
            mean_margin = np.mean(ep_margin)
            mean_hinge = np.mean(ep_hinge_loss)
            viol_pct = hinge_violations / total_samples * 100
            
            print(f"  Epoch {epoch}/{epochs} | Hinge Loss: {mean_hinge:.4f} | Mean Margin: {mean_margin:.4f} | Hinge Violations: {viol_pct:.1f}%")
            
            epoch_train_margins.append(mean_margin)
            
            # Fast val check
            val_res = evaluate(model, all_subject_data, test_sub, device, window_sec, hop_sec, speaker_swap, silent=True)
            epoch_val_margins.append(val_res['margin'])
            epoch_val_accs.append(val_res['trial_acc'] * 100)
            print(f"    Val Margin: {val_res['margin']:.4f} | Val Trial Acc: {val_res['trial_acc']*100:.1f}%")

        if not smoke:
            plt.figure(figsize=(10, 5))
            epochs_x = range(1, epochs + 1)
            plt.plot(epochs_x, epoch_train_margins, label="Train Margin", marker='o')
            plt.plot(epochs_x, epoch_val_margins, label="Val Margin", marker='s')
            plt.title(f"Learning Curve - {mode_name}")
            plt.xlabel("Epoch")
            plt.ylabel("Margin")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(out_dir / f"learning_curve_{mode_name.replace(' ', '_')}.png", dpi=150)
            plt.close()
    else:
        print("  Skipping training (Random Encoder Mode)")
        
    print("\n  --- Final Evaluation ---")
    res = evaluate(model, all_subject_data, test_sub, device, window_sec, hop_sec, speaker_swap, silent=False, save_hist_path=out_dir / f"hist_{mode_name.replace(' ', '_')}.png" if not smoke else None)
    
    return res

def evaluate(model, all_subject_data, test_sub, device, window_sec=10.0, hop_sec=1.0, speaker_swap=False, silent=False, save_hist_path=None):
    model.eval()
    fs = 64
    win_samples = int(window_sec * fs)
    hop_samples = int(hop_sec * fs)
    
    total_trials = len(all_subject_data[test_sub])
    windows_total = 0
    windows_correct = 0
    trials_correct = 0
    
    all_corr_a = []
    all_corr_b = []
    all_margins = []
    
    with torch.no_grad():
        for i, t in enumerate(all_subject_data[test_sub]):
            eeg = t["eeg"]
            a = t["audio_a"]
            b = t["audio_b"]
            
            eeg = (eeg - eeg.mean(dim=1, keepdim=True)) / (eeg.std(dim=1, keepdim=True) + 1e-12)
            a = (a - a.mean(dim=1, keepdim=True)) / (a.std(dim=1, keepdim=True) + 1e-12)
            b = (b - b.mean(dim=1, keepdim=True)) / (b.std(dim=1, keepdim=True) + 1e-12)
            
            if speaker_swap:
                a, b = b, a
            
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
            margin = corr_a - corr_b
            
            if not silent:
                print(f"  Trial {i+1:2d} | corr(att): {np.mean(corr_a):7.4f} | corr(unatt): {np.mean(corr_b):7.4f} | margin: {np.mean(margin):7.4f}")
                # Print window diagnostics for the first trial only
                if i == 0:
                    print(f"    -- Trial 1 Window Diagnostics --")
                    for w in range(min(5, len(corr_a))):
                        correct = "YES" if margin[w] > 0 else "NO"
                        print(f"    Win {w+1:2d} | att: {corr_a[w]:7.4f} | unatt: {corr_b[w]:7.4f} | margin: {margin[w]:7.4f} | Correct: {correct}")
            
            all_corr_a.extend(corr_a.tolist())
            all_corr_b.extend(corr_b.tolist())
            all_margins.extend(margin.tolist())
            
            wins_correct = (corr_a > corr_b).sum()
            num_wins = len(env_pred)
            
            windows_total += num_wins
            windows_correct += wins_correct
            
            if wins_correct > num_wins / 2.0:
                trials_correct += 1
                
    if save_hist_path:
        plt.figure(figsize=(10, 6))
        plt.hist(all_corr_a, bins=50, alpha=0.5, label='Corr(Attended)')
        plt.hist(all_corr_b, bins=50, alpha=0.5, label='Corr(Unattended)')
        plt.title('Distribution of Window-level Correlations')
        plt.xlabel('Pearson r')
        plt.ylabel('Frequency')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_hist_path, dpi=150)
        plt.close()
                
    return {
        "mean_pearson_att": np.mean(all_corr_a),
        "mean_pearson_unatt": np.mean(all_corr_b),
        "margin": np.mean(all_margins),
        "win_acc": windows_correct / windows_total,
        "trial_acc": trials_correct / total_trials
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="Run fast smoke test")
    parser.add_argument("--cache_dir", type=str, default=None)
    args = parser.parse_args()

    print("="*70)
    print("   DISCRIMINATIVE PEARSON LOSS VERIFICATION SUITE")
    print("="*70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Standard Parameters
    batch_size = 64 if args.smoke else 128
    epochs = 5
    steps_per_epoch = 10 if args.smoke else 100

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

    out_dir = REPO_ROOT / "results" / "pearson_verification"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    experiments = [
        # Multi-subject tests
        {"name": "Standard S1", "sub": "S1", "args": {}},
        {"name": "Standard S5", "sub": "S5", "args": {}},
        {"name": "Standard S9", "sub": "S9", "args": {}},
        {"name": "Standard S11", "sub": "S11", "args": {}},
        # Control tests on S1
        {"name": "Label Shuffle S1", "sub": "S1", "args": {"label_shuffle": True}},
        {"name": "Speaker Swap S1", "sub": "S1", "args": {"speaker_swap": True}},
        {"name": "Random Encoder S1", "sub": "S1", "args": {"random_encoder": True}},
    ]
    
    results = []
    for exp in experiments:
        res = run_experiment(exp["name"], exp["sub"], all_subject_data, out_dir, device, 
                             epochs=epochs, steps_per_epoch=steps_per_epoch, batch_size=batch_size, smoke=args.smoke, **exp["args"])
        results.append({
            "Experiment": exp["name"],
            "Corr(att)": round(res['mean_pearson_att'], 4),
            "Corr(unatt)": round(res['mean_pearson_unatt'], 4),
            "Margin": round(res['margin'], 4),
            "Window Acc": f"{res['win_acc']*100:.1f}%",
            "Trial Acc": f"{res['trial_acc']*100:.1f}%"
        })
        
    df = pd.DataFrame(results)
    csv_path = out_dir / "verification_results.csv"
    df.to_csv(csv_path, index=False)
    
    print("\n" + "="*70)
    print(" VERIFICATION SUMMARY")
    print("="*70)
    print(df.to_markdown(index=False))
    print(f"\nSaved full results to {csv_path}")

if __name__ == "__main__":
    main()
