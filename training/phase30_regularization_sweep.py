import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
import scipy.io
import scipy.signal
import glob
import math

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from models.aad_conformer import AADConformer
from training.train_conformer_loso import safe_corr_torch, custom_loss, safe_corr_np
from training.phase29_cross_subject_train import WindowedDataset, load_aasd_subject
from training.phase30_within_subject_train import NegativePearsonLoss, evaluate_trial_majority_vote, smart_load_checkpoint

def set_dropout(model, drop_rate=0.8):
    for name, module in model.named_modules():
        if isinstance(module, nn.Dropout):
            module.p = drop_rate

def run_config(config_name, args, train_ds, test_trials, device):
    print(f"\n{'='*50}")
    print(f"=== RUNNING CONFIG: {config_name} ===")
    print(f"{'='*50}")

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    model = AADConformer(in_channels=8).to(device)

    # Load pretrained KUL checkpoint
    if args['checkpoint'] and os.path.exists(args['checkpoint']):
        model = smart_load_checkpoint(model, args['checkpoint'], device)

    # Apply Configuration Logic
    lr = args['lr']
    weight_decay = args['weight_decay']

    if args['heavy_dropout']:
        set_dropout(model, 0.8)

    if args['stem_freeze']:
        print("[INFO] Freezing Spatial and Temporal Stem.")
        for name, param in model.named_parameters():
            if 'temporal_conv' in name or 'spatial_conv' in name or 'temporal_norm' in name or 'spatial_norm' in name:
                param.requires_grad = False

    criterion = NegativePearsonLoss().to(device)
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=weight_decay)

    best_test_auc = 0.0
    best_stats = {}

    for epoch in range(1, args['epochs'] + 1):
        model.train()
        train_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            pred = model(batch_x)
            loss = criterion(pred, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item() * batch_x.size(0)

        model.eval()
        
        # Calculate Test Metrics
        att_corrs, unatt_corrs = [], []
        with torch.no_grad():
            for t in test_trials:
                eeg = t["eeg"].unsqueeze(0).to(device)
                pred = model(eeg).squeeze(0).cpu().numpy()
                wav_a = t["audio_a"].squeeze(0).cpu().numpy()
                wav_b = t["audio_b"].squeeze(0).cpu().numpy()
                att_corrs.append(safe_corr_np(pred, wav_a))
                unatt_corrs.append(safe_corr_np(pred, wav_b))

        test_pearson = np.mean(att_corrs)
        test_acc = np.mean(np.array(att_corrs) > np.array(unatt_corrs))
        labels = np.concatenate([np.ones(len(att_corrs)), np.zeros(len(unatt_corrs))])
        scores = np.concatenate([att_corrs, unatt_corrs])
        from sklearn.metrics import roc_auc_score
        try:
            test_auc = roc_auc_score(labels, scores)
        except:
            test_auc = 0.5

        if test_auc > best_test_auc:
            best_test_auc = test_auc
            best_stats = {
                'epoch': epoch,
                'test_auc': test_auc,
                'test_pearson': test_pearson,
                'test_acc': test_acc,
                'train_loss': train_loss / len(train_ds)
            }

        print(f"Epoch {epoch} | Train Loss: {train_loss/len(train_ds):.4f} | Test AUC: {test_auc:.4f} | Test Pearson: {test_pearson:.4f}")

    print(f"--- Best {config_name} ---")
    print(best_stats)
    return best_stats


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device:", device)

    mat_files = glob.glob('/kaggle/input/datasets/lokeshgile/aasd-processed-eeg/Processed EEG/*/*.mat')
    if not mat_files:
        print("ERROR: No .mat files found. Please run on Kaggle.")
        return

    subject = "S18"
    sub_str = f"{subject}.mat"
    sub_path = next((p for p in mat_files if sub_str in p), None)

    if not sub_path:
        print("Subject not found")
        return

    b, a = scipy.signal.butter(4, [1.0/64.0, 8.0/64.0], btype='band')
    sel_idx = [23, 28, 22, 41, 36, 0, 40, 25] # fallback map
    audio_dir = '/kaggle/input/datasets/lokeshgile/aasd-audio-gammatones'

    print("Loading trials...")
    trials = load_aasd_subject(sub_path, b, a, sel_idx, audio_dir)
    print(f"Loaded {len(trials)} trials")

    # Split: Chronological 80/20
    split_idx = int(0.8 * len(trials))
    train_trials = trials[:split_idx]
    test_trials = trials[split_idx:]

    train_ds = WindowedDataset(train_trials, window_len=128, hop_len=64, censor_margin=256)

    checkpoint_path = "/kaggle/input/datasets/lowkieee/eeg-aad-conformer-seed1-checkpoints/checkpoints/seed_1/model_S1.pt"

    configs = {
        "Baseline": {
            "checkpoint": checkpoint_path,
            "lr": 1e-3,
            "weight_decay": 1e-4,
            "heavy_dropout": False,
            "stem_freeze": False,
            "epochs": 15
        },
        "Heavy_L2": {
            "checkpoint": checkpoint_path,
            "lr": 1e-3,
            "weight_decay": 1e-1,
            "heavy_dropout": False,
            "stem_freeze": False,
            "epochs": 15
        },
        "Heavy_Dropout": {
            "checkpoint": checkpoint_path,
            "lr": 1e-3,
            "weight_decay": 1e-4,
            "heavy_dropout": True,
            "stem_freeze": False,
            "epochs": 15
        },
        "Stem_Freeze": {
            "checkpoint": checkpoint_path,
            "lr": 1e-3,
            "weight_decay": 1e-4,
            "heavy_dropout": False,
            "stem_freeze": True,
            "epochs": 15
        }
    }

    results = {}
    for name, args in configs.items():
        results[name] = run_config(name, args, train_ds, test_trials, device)

    print("\n\n" + "="*50)
    print("FINAL SWEEP RESULTS")
    print("="*50)
    for name, stats in results.items():
        print(f"{name:15} | Test AUC: {stats.get('test_auc', 0):.4f} | Test Acc: {stats.get('test_acc', 0):.4f} | Train Loss: {stats.get('train_loss', 0):.4f}")

if __name__ == "__main__":
    main()
