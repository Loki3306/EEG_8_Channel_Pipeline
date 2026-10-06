"""
USCAPES: Universal Subject-Calibrated Auditory Processing & EEG Steering
Universal Foundation Model Training Script.

Trains the CA-TCN (Causal-Anticausal Temporal Convolutional Network) base decoder
across all available cohort subjects.

Usage:
    python scripts/train_universal.py --epochs 30 --batch-size 32
"""

import argparse
import sys
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# Add USCAPES root to Python path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from uscapes.config import (
    CHECKPOINT_DIR,
    FS_AUDIO,
    FS_EEG,
    WINDOW_SEC,
    MONTAGE_CHANNELS,
)
from uscapes.models.catcn import CATCNDirectDecoder
from uscapes.dsp.causal_filters import StreamingCausalEEGFilter
from uscapes.dsp.causal_gammatone import StreamingCausalAudioGammatoneExtractor
from uscapes.pipeline.data_provider import StreamDataProvider


def main():
    parser = argparse.ArgumentParser(
        description="USCAPES Universal Foundation Model Pre-Trainer"
    )
    parser.add_argument(
        "--subjects", type=str, default="S1,S2,S3,S4,S5",
        help="Comma-separated list of subjects to train on, or 'all'"
    )
    parser.add_argument(
        "--epochs", type=int, default=30,
        help="Number of training epochs (default: 30)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=32,
        help="Batch size for training (default: 32)"
    )
    parser.add_argument(
        "--lr", type=float, default=5e-4,
        help="Initial learning rate (default: 0.0005)"
    )
    parser.add_argument(
        "--output-path", type=str, default=None,
        help="Destination path for universal backbone weights"
    )
    parser.add_argument(
        "--max-trials-per-sub", type=int, default=30,
        help="Maximum number of trials per subject to include (default: 30)"
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_path = Path(args.output_path) if args.output_path else CHECKPOINT_DIR / "universal_catcn_backbone.pt"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("  USCAPES UNIVERSAL FOUNDATION MODEL PRE-TRAINING")
    print(f"  • Compute Device:    {device}")
    print(f"  • Target Checkpoint: {out_path}")
    print(f"  • Epochs:            {args.epochs}")
    print(f"  • Batch Size:        {args.batch_size}")
    print("=" * 80)

    data_provider = StreamDataProvider()
    if args.subjects.lower() == "all":
        target_subs = [f"S{i}" for i in range(1, 19)]
    else:
        target_subs = [s.strip().upper() for s in args.subjects.split(",") if s.strip()]

    print(f"  Loading training data across {len(target_subs)} subjects: {', '.join(target_subs)}...")

    eeg_filter = StreamingCausalEEGFilter(fs=FS_EEG, lowcut=1.0, highcut=6.0, order=2, n_channels=len(MONTAGE_CHANNELS))
    gamma_a = StreamingCausalAudioGammatoneExtractor(audio_fs=FS_AUDIO, target_fs=FS_EEG)
    gamma_b = StreamingCausalAudioGammatoneExtractor(audio_fs=FS_AUDIO, target_fs=FS_EEG)

    win_samples = int(WINDOW_SEC * FS_EEG)  # 320 samples
    hop_samples = int(2.5 * FS_EEG)         # 2.5s hop for diverse dataset

    X_all, YA_all, YB_all = [], [], []

    for sub in target_subs:
        print(f"    Extracting windows for {sub}...", end="", flush=True)
        sub_windows = 0
        for tr_id in range(1, args.max_trials_per_sub + 1):
            try:
                audio_a, audio_b, raw_eeg, attended = data_provider.load_trial_data(
                    subject_id=sub, trial_id=tr_id, duration_sec=50.0, fs_audio=FS_AUDIO, fs_eeg=FS_EEG
                )
                eeg_filter.reset()
                gamma_a.reset()
                gamma_b.reset()

                filt_eeg = eeg_filter.process_chunk(raw_eeg)
                filt_eeg = (filt_eeg - np.mean(filt_eeg, axis=0, keepdims=True)) / (np.std(filt_eeg, axis=0, keepdims=True) + 1e-8)

                env_a = gamma_a.process_chunk(audio_a)
                env_b = gamma_b.process_chunk(audio_b)
                env_a = (env_a - np.mean(env_a)) / (np.std(env_a) + 1e-8)
                env_b = (env_b - np.mean(env_b)) / (np.std(env_b) + 1e-8)

                min_len = min(len(filt_eeg), len(env_a), len(env_b))
                s = 0
                while s + win_samples <= min_len:
                    e = s + win_samples
                    X_all.append(filt_eeg[s:e].T)
                    YA_all.append(env_a[s:e][np.newaxis, :])
                    YB_all.append(env_b[s:e][np.newaxis, :])
                    s += hop_samples
                    sub_windows += 1
            except Exception:
                continue
        print(f" {sub_windows} windows.")

    if not X_all:
        print("[ERROR] No training windows extracted. Please check dataset paths.")
        sys.exit(1)

    X_t = torch.from_numpy(np.stack(X_all, axis=0)).float()
    YA_t = torch.from_numpy(np.stack(YA_all, axis=0)).float()
    YB_t = torch.from_numpy(np.stack(YB_all, axis=0)).float()

    dataset = TensorDataset(X_t, YA_t, YB_t)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    print(f"\n  Total Training Dataset: {len(dataset):,} windows ({len(dataset) * 5.0 / 3600.0:.2f} hours)")

    # Model
    model = CATCNDirectDecoder(
        eeg_channels=len(MONTAGE_CHANNELS), audio_channels=1, hidden_dim=64, max_lag_samples=8
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)

    print(f"  Beginning Training across {args.epochs} epochs...\n")
    t0 = time.time()

    for ep in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        n_batches = 0
        correct = 0
        total_samples = 0

        for bx, bya, byb in loader:
            bx, bya, byb = bx.to(device), bya.to(device), byb.to(device)
            optimizer.zero_grad(set_to_none=True)

            delta, (la, lb), _ = model(bx, bya, byb)
            loss = torch.clamp(0.5 - (la - lb), min=0.0).mean()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += float(loss.item())
            correct += int((delta > 0).sum().item())
            total_samples += len(delta)
            n_batches += 1

        scheduler.step()
        ep_loss = total_loss / max(1, n_batches)
        ep_acc = (correct / max(1, total_samples)) * 100.0

        if ep % 5 == 0 or ep == args.epochs:
            print(f"    Epoch {ep:02d}/{args.epochs:02d} | Loss: {ep_loss:.4f} | Accuracy: {ep_acc:.1f}% | LR: {scheduler.get_last_lr()[0]:.6f}")

    # Save
    torch.save({
        "model_state_dict": model.state_dict(),
        "epochs": args.epochs,
        "subjects": target_subs,
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S")
    }, out_path)

    elapsed = time.time() - t0
    print(f"\n[SUCCESS] Universal backbone saved to: {out_path} ({elapsed:.1f}s)")


if __name__ == "__main__":
    main()
