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
from training.verify_pearson_discriminative import DiscriminativePearsonLoss, pearson_corr_np

class DiscriminativeEnvelopeDataset(Dataset):
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
            
        return eeg[:, start:end], a_win, b_win, sub_label


def prepare_evaluation_trials(all_subject_data, test_sub):
    """
    Extracts all trials for evaluation and pre-standardizes them.
    Also returns a mapping of (subject -> trials) for cross-subject sampling.
    """
    eval_trials = []
    
    # Store all trials by subject for cross-subject permutations
    all_trials_by_sub = {}
    
    for sub, trials in all_subject_data.items():
        all_trials_by_sub[sub] = []
        for i, t in enumerate(trials):
            eeg = t["eeg"]
            a = t["audio_a"]
            b = t["audio_b"]
            
            eeg = (eeg - eeg.mean(dim=1, keepdim=True)) / (eeg.std(dim=1, keepdim=True) + 1e-12)
            a = (a - a.mean(dim=1, keepdim=True)) / (a.std(dim=1, keepdim=True) + 1e-12)
            b = (b - b.mean(dim=1, keepdim=True)) / (b.std(dim=1, keepdim=True) + 1e-12)
            
            trial_data = {"eeg": eeg, "a": a, "b": b, "global_idx": f"{sub}_trial_{i}"}
            all_trials_by_sub[sub].append(trial_data)
            
            if sub == test_sub:
                eval_trials.append(trial_data)
                
    return eval_trials, all_trials_by_sub


def evaluate_perturbation(model, eval_trials, all_trials_by_sub, test_sub, device, 
                          mode="standard", shift_sec=0, fs=64, window_sec=10.0, hop_sec=1.0):
    model.eval()
    win_samples = int(window_sec * fs)
    hop_samples = int(hop_sec * fs)
    
    total_trials = len(eval_trials)
    windows_total = 0
    windows_correct = 0
    trials_correct = 0
    
    all_corr_a = []
    all_corr_b = []
    all_margins = []
    
    # Track permutations for assertions
    printed_pairing = False
    
    with torch.no_grad():
        for i, t in enumerate(eval_trials):
            eeg = t["eeg"]
            a = t["a"]
            b = t["b"]
            idx = t["global_idx"]
            
            a_eval = a
            b_eval = b
            
            if mode == "true_permutation":
                # Select ANY other trial from ANY subject, except 'idx'
                choices = []
                for sub, trials in all_trials_by_sub.items():
                    for t2 in trials:
                        if t2["global_idx"] != idx:
                            choices.append(t2)
                t_rand = np.random.choice(choices)
                assert t_rand["global_idx"] != idx, f"Leakage! Paired {idx} with {t_rand['global_idx']}"
                a_eval = t_rand["a"]
                b_eval = t_rand["b"]
                
                if not printed_pairing:
                    print(f"    Assertion Passed: Paired {idx} with {t_rand['global_idx']}")
                    printed_pairing = True
                    
            elif mode == "within_subject_permutation":
                # Select another trial from the SAME test_sub, except 'idx'
                choices = [t2 for t2 in all_trials_by_sub[test_sub] if t2["global_idx"] != idx]
                t_rand = np.random.choice(choices)
                assert t_rand["global_idx"] != idx, "Leakage! Same trial paired."
                a_eval = t_rand["a"]
                b_eval = t_rand["b"]
                
                if not printed_pairing:
                    print(f"    Assertion Passed: Paired {idx} with {t_rand['global_idx']} (Same sub)")
                    printed_pairing = True
                    
            elif mode == "cross_subject_permutation":
                # Select a trial from a DIFFERENT subject
                choices = []
                for sub, trials in all_trials_by_sub.items():
                    if sub != test_sub:
                        choices.extend(trials)
                t_rand = np.random.choice(choices)
                assert not t_rand["global_idx"].startswith(f"{test_sub}_"), "Leakage! Same subject paired."
                a_eval = t_rand["a"]
                b_eval = t_rand["b"]
                
                if not printed_pairing:
                    print(f"    Assertion Passed: Paired {idx} with {t_rand['global_idx']} (Cross sub)")
                    printed_pairing = True
                    
            elif mode == "random_gaussian":
                # Replace a and b with gaussian noise matching mean/var
                a_eval = torch.randn_like(a) * a.std() + a.mean()
                b_eval = torch.randn_like(b) * b.std() + b.mean()
                
            elif mode == "circular_shift":
                shift_samples = int(shift_sec * fs)
                a_eval = torch.roll(a, shifts=shift_samples, dims=1)
                b_eval = torch.roll(b, shifts=shift_samples, dims=1)
            
            
            # Now generate windows and predict
            # We must truncate to the shortest sequence if there's a length mismatch during permutation
            min_len = min(eeg.shape[1], a_eval.shape[1], b_eval.shape[1])
            
            eeg_wins, a_wins, b_wins = [], [], []
            start = 0
            while start + win_samples <= min_len:
                end = start + win_samples
                eeg_wins.append(eeg[:, start:end])
                a_wins.append(a_eval[:, start:end])
                b_wins.append(b_eval[:, start:end])
                start += hop_samples
                
            if not eeg_wins: continue
            
            eeg_wins = torch.stack(eeg_wins).to(device)
            a_wins = torch.stack(a_wins).cpu().numpy()
            b_wins = torch.stack(b_wins).cpu().numpy()
            
            env_pred = model.predict(eeg_wins).cpu().numpy()
            
            corr_a = pearson_corr_np(env_pred, a_wins).mean(axis=1)
            corr_b = pearson_corr_np(env_pred, b_wins).mean(axis=1)
            margin = corr_a - corr_b
            
            all_corr_a.extend(corr_a.tolist())
            all_corr_b.extend(corr_b.tolist())
            all_margins.extend(margin.tolist())
            
            wins_correct = (margin > 0).sum()
            num_wins = len(env_pred)
            
            windows_total += num_wins
            windows_correct += wins_correct
            
            if wins_correct > num_wins / 2.0:
                trials_correct += 1
                
    margin_arr = np.array(all_margins)
    
    metrics = {
        "mean_corr_att": np.mean(all_corr_a),
        "mean_corr_unatt": np.mean(all_corr_b),
        "mean_margin": np.mean(margin_arr),
        "median_margin": np.median(margin_arr),
        "std_margin": np.std(margin_arr),
        "pos_margin_frac": np.mean(margin_arr > 0),
        "neg_margin_frac": np.mean(margin_arr <= 0),
        "win_acc": windows_correct / windows_total,
        "trial_acc": trials_correct / total_trials,
        "raw_margins": margin_arr,
        "raw_corr_att": np.array(all_corr_a),
        "raw_corr_unatt": np.array(all_corr_b)
    }
    return metrics

def plot_histograms(results_dict, out_dir):
    """
    Plot comparative histograms for Standard vs True Permutation vs Gaussian Noise
    """
    plt.figure(figsize=(15, 10))
    
    targets = ["Standard", "True Audio Permutation", "Random Gaussian Target"]
    
    for i, tgt in enumerate(targets):
        if tgt not in results_dict: continue
        
        plt.subplot(3, 1, i+1)
        res = results_dict[tgt]
        plt.hist(res["raw_corr_att"], bins=50, alpha=0.5, label='Corr(Attended)', color='blue')
        plt.hist(res["raw_corr_unatt"], bins=50, alpha=0.5, label='Corr(Unattended)', color='red')
        plt.hist(res["raw_margins"], bins=50, alpha=0.3, label='Margin', color='green')
        
        plt.title(tgt)
        plt.xlabel('Pearson r')
        plt.ylabel('Frequency')
        plt.xlim(-0.2, 0.2)
        plt.axvline(0, color='black', linestyle='--', linewidth=1)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
    plt.tight_layout()
    plt.savefig(out_dir / "correlation_histograms.png", dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="Run fast smoke test")
    parser.add_argument("--cache_dir", type=str, default=None)
    args = parser.parse_args()

    print("="*70)
    print("   DISCRIMINATIVE AAD LEAKAGE VERIFICATION SUITE")
    print("="*70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Standard Parameters
    batch_size = 64 if args.smoke else 128
    epochs = 5
    steps_per_epoch = 10 if args.smoke else 100
    test_sub = "S1"

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

    out_dir = REPO_ROOT / "results" / "leakage_verification"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # ------------------------------------------------------------------
    # PHASE 1: Train a standard baseline model on S1
    # ------------------------------------------------------------------
    print("\n[PHASE 1] Training Standard Margin Pearson Model on S1 LOSO...")
    
    torch.manual_seed(42)
    np.random.seed(42)

    train_subs = [s for s in all_subject_data.keys() if s != test_sub]
    train_subs = sorted(train_subs, key=lambda x: int(x[1:]))
    sub_to_idx = {s: i for i, s in enumerate(train_subs)}
    
    train_ds = DiscriminativeEnvelopeDataset(all_subject_data, test_sub, sub_to_idx, 
                                             window_sec=10.0, steps_per_epoch=steps_per_epoch, 
                                             batch_size=batch_size)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True)
    
    model = PearsonAADModel(num_subjects=len(train_subs), temporal_pooling_factors=(4, 1)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    criterion = DiscriminativePearsonLoss(target_margin=0.10)
    
    for epoch in range(1, epochs + 1):
        model.train()
        for eeg_batch, a_batch, b_batch, subj_batch in train_loader:
            eeg_batch = eeg_batch.to(device)
            a_batch = a_batch.to(device)
            b_batch = b_batch.to(device)
            
            optimizer.zero_grad()
            env_pred, _ = model(eeg_batch)
            
            c_att = criterion._pearson(env_pred, a_batch).mean(dim=1)
            c_unatt = criterion._pearson(env_pred, b_batch).mean(dim=1)
            margins = c_att - c_unatt
            
            loss = torch.clamp(criterion.target_margin - margins, min=0.0).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
    print("Training Complete. Freezing Model.")
    
    # ------------------------------------------------------------------
    # PHASE 2: Comprehensive Evaluation Suite
    # ------------------------------------------------------------------
    
    eval_trials, all_trials_by_sub = prepare_evaluation_trials(all_subject_data, test_sub)
    
    experiments = [
        {"name": "Standard", "mode": "standard", "shift_sec": 0},
        {"name": "True Audio Permutation", "mode": "true_permutation", "shift_sec": 0},
        {"name": "Within-subject Permutation", "mode": "within_subject_permutation", "shift_sec": 0},
        {"name": "Cross-subject Permutation", "mode": "cross_subject_permutation", "shift_sec": 0},
        {"name": "Random Gaussian Target", "mode": "random_gaussian", "shift_sec": 0},
        {"name": "Circular Shift 5s", "mode": "circular_shift", "shift_sec": 5},
        {"name": "Circular Shift 10s", "mode": "circular_shift", "shift_sec": 10},
        {"name": "Circular Shift 20s", "mode": "circular_shift", "shift_sec": 20},
    ]
    
    print("\n[PHASE 2] Executing Negative Control Eval Suite...")
    
    full_results_dict = {}
    table_rows = []
    
    for exp in experiments:
        name = exp["name"]
        print(f"\n  >> Executing: {name}")
        
        metrics = evaluate_perturbation(
            model, eval_trials, all_trials_by_sub, test_sub, device,
            mode=exp["mode"], shift_sec=exp["shift_sec"]
        )
        
        full_results_dict[name] = metrics
        
        print(f"     Corr(att): {metrics['mean_corr_att']:.4f} | Corr(unatt): {metrics['mean_corr_unatt']:.4f}")
        print(f"     Margin Stats -> Mean: {metrics['mean_margin']:.4f} | Median: {metrics['median_margin']:.4f} | Std: {metrics['std_margin']:.4f}")
        print(f"     Margin Dist  -> Pos: {metrics['pos_margin_frac']*100:.1f}% | Neg: {metrics['neg_margin_frac']*100:.1f}%")
        print(f"     Window Acc: {metrics['win_acc']*100:.1f}% | Trial Acc: {metrics['trial_acc']*100:.1f}%")
        
        # Check against chance
        status = "AS EXPECTED"
        if name != "Standard" and metrics['trial_acc'] > 0.60:
            status = "LEAKAGE WARNING"
            
        table_rows.append({
            "Experiment": name,
            "Corr(att)": round(metrics['mean_corr_att'], 4),
            "Corr(unatt)": round(metrics['mean_corr_unatt'], 4),
            "Margin": round(metrics['mean_margin'], 4),
            "Window Acc": f"{metrics['win_acc']*100:.1f}%",
            "Trial Acc": f"{metrics['trial_acc']*100:.1f}%",
            "Status": status
        })

    # Plot histograms
    if not args.smoke:
        plot_histograms(full_results_dict, out_dir)
        print(f"\nSaved Histograms to {out_dir / 'correlation_histograms.png'}")
        
    df = pd.DataFrame(table_rows)
    csv_path = out_dir / "leakage_verification_results.csv"
    df.to_csv(csv_path, index=False)
    
    print("\n" + "="*70)
    print(" VERIFICATION SUMMARY")
    print("="*70)
    print(df.to_markdown(index=False))
    print(f"\nSaved full results to {csv_path}")

if __name__ == "__main__":
    main()
