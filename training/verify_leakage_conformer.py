import sys
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
import argparse
import pandas as pd
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from data.kul_cached_dataset import KULCachedLoader
from models.aad_conformer import AADConformer
from training.train_conformer_loso import prepare_data, safe_corr_np, evaluate_trial_majority_vote, custom_loss

def evaluate_perturbation(model, test_trials, all_trials_by_sub, test_sub, device, 
                          mode="standard", shift_sec=0, fs=64, window_sec=10.0, hop_sec=1.0):
    model.eval()
    
    windows_total = 0
    windows_correct = 0
    trials_correct = 0
    total_trials = len(test_trials)
    
    all_corr_a = []
    all_corr_b = []
    all_margins = []
    
    with torch.no_grad():
        for i, t in enumerate(test_trials):
            eeg = t["eeg"].unsqueeze(0)       # (1, 8, Time)
            a = t["audio_a"].unsqueeze(0)     # (1, 28, Time)
            b = t["audio_b"].unsqueeze(0)     # (1, 28, Time)
            idx = t["global_idx"]
            
            a_eval = a
            b_eval = b
            
            if mode == "true_permutation":
                choices = []
                for sub, trials in all_trials_by_sub.items():
                    for t2 in trials:
                        if t2["global_idx"] != idx:
                            choices.append(t2)
                t_rand = np.random.choice(choices)
                a_eval = t_rand["audio_a"].unsqueeze(0)
                b_eval = t_rand["audio_b"].unsqueeze(0)
                
            elif mode == "within_subject_permutation":
                choices = [t2 for t2 in all_trials_by_sub[test_sub] if t2["global_idx"] != idx]
                t_rand = np.random.choice(choices)
                a_eval = t_rand["audio_a"].unsqueeze(0)
                b_eval = t_rand["audio_b"].unsqueeze(0)
                
            elif mode == "cross_subject_permutation":
                choices = []
                for sub, trials in all_trials_by_sub.items():
                    if sub != test_sub:
                        choices.extend(trials)
                t_rand = np.random.choice(choices)
                a_eval = t_rand["audio_a"].unsqueeze(0)
                b_eval = t_rand["audio_b"].unsqueeze(0)
                
            elif mode == "random_gaussian":
                a_eval = torch.randn_like(a) * a.std() + a.mean()
                b_eval = torch.randn_like(b) * b.std() + b.mean()
                
            elif mode == "circular_shift":
                shift_samples = int(shift_sec * fs)
                a_eval = torch.roll(a, shifts=shift_samples, dims=2)
                b_eval = torch.roll(b, shifts=shift_samples, dims=2)
                
            min_len = min(eeg.shape[2], a_eval.shape[2], b_eval.shape[2])
            eeg = eeg[:, :, :min_len]
            a_eval = a_eval[:, :, :min_len]
            b_eval = b_eval[:, :, :min_len]
            
            a_eval = a_eval.mean(dim=1, keepdim=True)
            b_eval = b_eval.mean(dim=1, keepdim=True)
            
            eeg_mean = eeg.mean(dim=2, keepdim=True)
            eeg_std = eeg.std(dim=2, keepdim=True) + 1e-8
            eeg_norm = (eeg - eeg_mean) / eeg_std
            
            a_mean = a_eval.mean(dim=2, keepdim=True)
            a_std = a_eval.std(dim=2, keepdim=True) + 1e-8
            a_norm = (a_eval - a_mean) / a_std
            
            b_mean = b_eval.mean(dim=2, keepdim=True)
            b_std = b_eval.std(dim=2, keepdim=True) + 1e-8
            b_norm = (b_eval - b_mean) / b_std
            
            eeg_norm = eeg_norm.to(device)
            pred = model(eeg_norm)
            
            pred_np = pred.squeeze(0).cpu().numpy()
            wav_a_np = a_norm.squeeze(1).squeeze(0).cpu().numpy()
            wav_b_np = b_norm.squeeze(1).squeeze(0).cpu().numpy()
            
            c_att = safe_corr_np(pred_np, wav_a_np)
            c_unatt = safe_corr_np(pred_np, wav_b_np)
            margin = c_att - c_unatt
            
            all_corr_a.append(float(c_att))
            all_corr_b.append(float(c_unatt))
            all_margins.append(float(margin))
            
            trial_ok, n_win, c_win = evaluate_trial_majority_vote(pred_np, wav_a_np, wav_b_np, window_seconds=10, hop_seconds=1.0, fs=64)
            if trial_ok:
                trials_correct += 1
            windows_total += n_win
            windows_correct += c_win
            
    margin_arr = np.array(all_margins)
    
    metrics = {
        "mean_corr_att": float(np.mean(all_corr_a)),
        "mean_corr_unatt": float(np.mean(all_corr_b)),
        "mean_margin": float(np.mean(margin_arr)),
        "median_margin": float(np.median(margin_arr)),
        "std_margin": float(np.std(margin_arr)),
        "pos_margin_frac": float(np.mean(margin_arr > 0)),
        "neg_margin_frac": float(np.mean(margin_arr <= 0)),
        "win_acc": float(windows_correct / max(1, windows_total)),
        "trial_acc": float(trials_correct / max(1, total_trials)),
        "raw_margins": margin_arr,
        "raw_corr_att": np.array(all_corr_a),
        "raw_corr_unatt": np.array(all_corr_b)
    }
    return metrics

def plot_histograms(results_dict, out_dir):
    plt.figure(figsize=(15, 10))
    targets = ["Standard", "True Audio Permutation", "Random Gaussian Target"]
    
    for i, tgt in enumerate(targets):
        if tgt not in results_dict:
            continue
        plt.subplot(3, 1, i+1)
        res = results_dict[tgt]
        plt.hist(res["raw_corr_att"], bins=50, alpha=0.5, label='Corr(Attended)', color='blue')
        plt.hist(res["raw_corr_unatt"], bins=50, alpha=0.5, label='Corr(Unattended)', color='red')
        plt.hist(res["raw_margins"], bins=50, alpha=0.3, label='Margin', color='green')
        
        plt.title(tgt)
        plt.xlabel('Pearson r')
        plt.ylabel('Frequency')
        plt.axvline(0, color='black', linestyle='--', linewidth=1)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
    plt.tight_layout()
    plt.savefig(out_dir / "conformer_correlation_histograms.png", dpi=150)
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="Run fast smoke test")
    parser.add_argument("--cache_dir", type=str, default=None)
    args = parser.parse_args()

    print("="*70)
    print("   AAD-CONFORMER LEAKAGE VERIFICATION SUITE")
    print("="*70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    test_sub = "S11"
    
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

    all_trials_by_sub = {}
    for sub, trials in all_subject_data.items():
        all_trials_by_sub[sub] = []
        for i, t in enumerate(trials):
            t["global_idx"] = f"{sub}_trial_{i}"
            all_trials_by_sub[sub].append(t)

    out_dir = REPO_ROOT / "results" / "leakage_verification_conformer"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n[PHASE 1] Training Conformer Model on {test_sub} LOSO (S11)...")
    
    torch.manual_seed(42)
    np.random.seed(42)

    train_trials = []
    test_trials = all_subject_data[test_sub]
    for sub, trials in all_subject_data.items():
        if sub != test_sub:
            train_trials.extend(trials)
            
    if args.smoke:
        train_trials = train_trials[:2]
        test_trials = test_trials[:2]
        epochs = 1
    else:
        epochs = 10
            
    train_tensors = prepare_data(train_trials, window_sec=2, hop_sec=1, fs=64)
    X_train, Ya_train = train_tensors
    dataset = TensorDataset(X_train, Ya_train)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True)
    
    model = AADConformer(
        in_channels=8,
        temporal_filters=32,
        spatial_filters=64,
        embed_dim=64,
        num_heads=4,
        num_layers=2,
        dropout=0.3,
        stride=4
    ).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            pred = model(batch_x)
            loss = custom_loss(pred, batch_y, mse_weight=0.5, corr_weight=0.5)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item() * batch_x.size(0)
        print(f"Epoch {epoch}/{epochs} | Train Loss: {train_loss/len(dataset):.4f}")
        
    print("Training Complete. Freezing Model.")
    
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
            model, test_trials, all_trials_by_sub, test_sub, device,
            mode=exp["mode"], shift_sec=exp["shift_sec"]
        )
        
        full_results_dict[name] = metrics
        
        print(f"     Corr(att): {metrics['mean_corr_att']:.4f} | Corr(unatt): {metrics['mean_corr_unatt']:.4f}")
        print(f"     Margin Stats -> Mean: {metrics['mean_margin']:.4f} | Median: {metrics['median_margin']:.4f} | Std: {metrics['std_margin']:.4f}")
        print(f"     Margin Dist  -> Pos: {metrics['pos_margin_frac']*100:.1f}% | Neg: {metrics['neg_margin_frac']*100:.1f}%")
        print(f"     Window Acc: {metrics['win_acc']*100:.1f}% | Trial Acc: {metrics['trial_acc']*100:.1f}%")
        
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

    if not args.smoke:
        plot_histograms(full_results_dict, out_dir)
        print(f"\nSaved Histograms to {out_dir / 'conformer_correlation_histograms.png'}")
        
    df = pd.DataFrame(table_rows)
    csv_path = out_dir / "leakage_verification_conformer_results.csv"
    df.to_csv(csv_path, index=False)
    
    print("\n" + "="*70)
    print(" VERIFICATION SUMMARY")
    print("="*70)
    print(df.to_markdown(index=False))
    print(f"\nSaved full results to {csv_path}")

if __name__ == "__main__":
    main()
