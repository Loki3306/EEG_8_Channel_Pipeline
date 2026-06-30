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
import matplotlib.pyplot as plt

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
            corr_a = pearson_corr(env_pred, a_wins).mean(axis=1)
            corr_b = pearson_corr(env_pred, b_wins).mean(axis=1)
            
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

def visualize_latent(model, all_subject_data, test_sub, device, variant_name, out_dir):
    """
    Saves a plot of a validation trial aligning true envelope, predicted envelope, and a latent channel.
    """
    model.eval()
    
    # Pick the first trial of the test subject
    t = all_subject_data[test_sub][0]
    eeg = t["eeg"]
    a = t["audio_a"]
    
    # Use just the first 10 seconds for clarity
    fs = 64
    win = 10 * fs
    if eeg.shape[1] < win:
        win = eeg.shape[1]
        
    eeg_segment = eeg[:, :win]
    a_segment = a[:, :win]
    
    eeg_std = (eeg_segment - eeg_segment.mean(dim=1, keepdim=True)) / (eeg_segment.std(dim=1, keepdim=True) + 1e-12)
    a_std = (a_segment - a_segment.mean(dim=1, keepdim=True)) / (a_segment.std(dim=1, keepdim=True) + 1e-12)
    
    with torch.no_grad():
        eeg_input = eeg_std.unsqueeze(0).to(device)
        
        # Extract latent feature
        # eeg_encoder outputs [1, 16, 1, Time/N]
        x = eeg_input.unsqueeze(1)
        x = model.eeg_encoder.block1(x)
        x = model.eeg_encoder.block2(x)
        latent_features = x.squeeze(2).cpu().numpy()[0] # [16, Time/N]
        
        # Extract predicted envelope
        env_pred = model.predict(eeg_input).cpu().numpy()[0] # [28, Time]
        
    # We will plot the first channel of the gammatone envelope
    true_env = a_std[0].cpu().numpy()
    pred_env = env_pred[0]
    latent_channel = latent_features[0] # Pick the first latent channel
    
    # Create time axes
    t_env = np.arange(len(true_env)) / fs
    
    latent_len = len(latent_channel)
    # The latent is evenly spaced across the time window
    t_latent = np.linspace(0, len(true_env) / fs, latent_len)
    
    plt.figure(figsize=(12, 8))
    
    plt.subplot(3, 1, 1)
    plt.title(f"True Attended Envelope (Subband 0) - {variant_name}")
    plt.plot(t_env, true_env, color='black', alpha=0.8)
    plt.ylabel("Amplitude")
    
    plt.subplot(3, 1, 2)
    plt.title("Predicted Envelope (Interpolated)")
    plt.plot(t_env, pred_env, color='blue', alpha=0.8)
    plt.ylabel("Amplitude")
    
    plt.subplot(3, 1, 3)
    plt.title(f"Latent Sequence (Channel 0) [Length: {latent_len}]")
    # Use drawstyle steps-mid to show the resolution of the latent clearly
    plt.plot(t_latent, latent_channel, color='red', drawstyle='steps-mid', alpha=0.8)
    plt.scatter(t_latent, latent_channel, color='darkred', s=10)
    plt.ylabel("Amplitude")
    plt.xlabel("Time (s)")
    
    plt.tight_layout()
    out_path = out_dir / f"latent_viz_{variant_name.replace(' ', '_')}.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    
    print(f"Saved visualization to {out_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="Run fast smoke test")
    parser.add_argument("--cache_dir", type=str, default=None)
    args = parser.parse_args()

    # Reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    print("="*70)
    print("   TEMPORAL POOLING ABLATION EXPERIMENT")
    print("="*70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Parameters (Fixed for all variants)
    batch_size = 64 if args.smoke else 128
    epochs = 5 # 5 epochs as requested
    lr = 3e-4
    window_sec = 10.0
    hop_sec = 1.0
    steps_per_epoch = 10 if args.smoke else 100
    test_sub = "S1" # Ablation on S1 only

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

    out_dir = REPO_ROOT / "results" / "pearson_ablation"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Define Variants
    # Tuple: (Variant Name, (pool1, pool2), target_hz)
    variants = [
        ("Current 2 Hz", (4, 8), 2),
        ("Variant B 16 Hz", (4, 1), 16),
        ("Variant C 32 Hz", (2, 1), 32)
    ]
    
    all_results = []
    
    train_subs = [s for s in all_subject_data.keys() if s != test_sub]
    train_subs = sorted(train_subs, key=lambda x: int(x[1:]))
    sub_to_idx = {s: i for i, s in enumerate(train_subs)}
    
    train_ds = EnvelopeDataset(all_subject_data, test_sub, sub_to_idx, window_sec=window_sec, steps_per_epoch=steps_per_epoch, batch_size=batch_size)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True)
    
    print(f"\nTraining on {len(train_subs)} subjects. Testing on {test_sub}.")
    
    for v_name, pools, target_hz in variants:
        print(f"\n" + "="*50)
        print(f" TESTING VARIANT: {v_name} (Pooling {pools})")
        print("="*50)
        
        # Create model with specific pooling
        model = PearsonAADModel(num_subjects=len(train_subs), temporal_pooling_factors=pools).to(device)
        model.debug_shapes = True # Turn on instrumentation
        
        # Run a dummy batch to trigger debug shapes
        print("\n--- Model Dimensionality Trace ---")
        dummy_eeg = torch.randn(2, 8, int(64 * window_sec)).to(device)
        with torch.no_grad():
            model(dummy_eeg, fs=64)
        model.debug_shapes = False # Turn off for training
        print("----------------------------------\n")
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        criterion = NegativePearsonLoss()
        
        for epoch in range(1, epochs + 1):
            model.train()
            total_loss = 0.0
            
            num_batches = len(train_loader)
            for step, (eeg_batch, a_pos_batch, subj_batch) in enumerate(train_loader):
                # We fix GRL lambda to 1.0 for simplicity in this ablation
                model.grl.lam = 1.0
                
                eeg_batch = eeg_batch.to(device)
                a_pos_batch = a_pos_batch.to(device)
                subj_batch = subj_batch.to(device)
                
                optimizer.zero_grad()
                env_pred, subj_logits = model(eeg_batch)
                
                pearson_loss = criterion(env_pred, a_pos_batch)
                
                # We completely ignore GRL loss for this ablation to perfectly isolate temporal resolution
                loss = pearson_loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                total_loss += pearson_loss.item()
                
            avg_loss = total_loss / num_batches
            print(f"  [{v_name}] Epoch {epoch}/{epochs} | Pearson Loss: {avg_loss:.4f}")
            
        print("\n  Evaluating Trained Model...")
        res = evaluate_pearson(model, all_subject_data, test_sub, device, window_sec=window_sec, hop_sec=hop_sec)
        
        print(f"  Results:")
        print(f"    Pearson(att): {res['mean_pearson_att']:.4f}")
        print(f"    Pearson(unatt): {res['mean_pearson_unatt']:.4f}")
        print(f"    Margin: {res['margin']:.4f}")
        print(f"    Window Acc: {res['win_acc']*100:.1f}%")
        print(f"    Trial Acc: {res['trial_acc']*100:.1f}%\n")
        
        all_results.append({
            "Encoder": v_name,
            "Final Hz": f"{target_hz} Hz",
            "Pearson(att)": round(res['mean_pearson_att'], 4),
            "Pearson(unatt)": round(res['mean_pearson_unatt'], 4),
            "Margin": round(res['margin'], 4),
            "Window Acc": f"{res['win_acc']*100:.1f}%",
            "Trial Acc": f"{res['trial_acc']*100:.1f}%"
        })
        
        # Visualize latent preservation
        visualize_latent(model, all_subject_data, test_sub, device, v_name, out_dir)
        
    df = pd.DataFrame(all_results)
    csv_path = out_dir / "ablation_results.csv"
    df.to_csv(csv_path, index=False)
    
    print("\n" + "="*50)
    print(" ABLATION SUMMARY")
    print("="*50)
    
    markdown_table = df.to_markdown(index=False)
    print(markdown_table)
    
    print(f"\nSaved full results to {csv_path}")

if __name__ == "__main__":
    main()
