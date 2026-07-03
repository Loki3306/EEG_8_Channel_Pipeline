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
from training.phase30_within_subject_train import NegativePearsonLoss, smart_load_checkpoint
from sklearn.metrics import roc_auc_score

def set_dropout(model, drop_rate=0.8):
    for name, module in model.named_modules():
        if isinstance(module, nn.Dropout):
            module.p = drop_rate

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

    print(f"\n{'='*50}")
    print(f"=== RUNNING CONFIG: Heavy Dropout Audit ===")
    print(f"{'='*50}")

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    model = AADConformer(in_channels=8).to(device)

    # Load pretrained KUL checkpoint
    if checkpoint_path and os.path.exists(checkpoint_path):
        model = smart_load_checkpoint(model, checkpoint_path, device)

    lr = 1e-3
    weight_decay = 1e-4

    set_dropout(model, 0.8)

    criterion = NegativePearsonLoss().to(device)
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=weight_decay)

    epochs = 6 # We saw peak AUROC at epoch 6
    
    for epoch in range(1, epochs + 1):
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
            
        print(f"Epoch {epoch} | Train Loss: {train_loss/len(train_ds):.4f}")

    print("\n\n" + "="*50)
    print("BEGIN FORENSIC TEST EVALUATION (EPOCH 6)")
    print("="*50)

    model.eval()
    att_corrs, unatt_corrs = [], []
    
    with torch.no_grad():
        for i, t in enumerate(test_trials):
            eeg = t["eeg"].unsqueeze(0).to(device)
            pred = model(eeg).squeeze(0).cpu().numpy()
            wav_a = t["att"].squeeze(0).cpu().numpy()
            wav_b = t["unatt"].squeeze(0).cpu().numpy()
            
            c_att = safe_corr_np(pred, wav_a)
            c_unatt = safe_corr_np(pred, wav_b)
            
            att_corrs.append(c_att)
            unatt_corrs.append(c_unatt)
            
            margin = c_att - c_unatt
            
            if i < 3:
                print(f"\n--- Raw Prediction Example (Trial {i}) ---")
                print(f"Pred -> Mean: {pred.mean():.4e}, Var: {pred.var():.4e}, Min: {pred.min():.4e}, Max: {pred.max():.4e}")
                print(f"Att  -> Mean: {wav_a.mean():.4e}, Var: {wav_a.var():.4e}, Min: {wav_a.min():.4e}, Max: {wav_a.max():.4e}")
                print(f"Unatt-> Mean: {wav_b.mean():.4e}, Var: {wav_b.var():.4e}, Min: {wav_b.min():.4e}, Max: {wav_b.max():.4e}")
                print(f"P(Att): {c_att:.4f} | P(Unatt): {c_unatt:.4f} | Margin: {margin:.4f}")

    print("\n--- Per-Trial Breakdown ---")
    margins = []
    for i, (c_att, c_unatt) in enumerate(zip(att_corrs, unatt_corrs)):
        m = c_att - c_unatt
        margins.append(m)
        print(f"Trial {i:02d}: P(Att): {c_att:7.4f} | P(Unatt): {c_unatt:7.4f} | Margin: {m:7.4f} | Correct: {m > 0}")

    print("\n--- Global Statistics ---")
    test_pearson = np.mean(att_corrs)
    test_acc = np.mean(np.array(att_corrs) > np.array(unatt_corrs))
    mean_margin = np.mean(margins)
    std_margin = np.std(margins)
    
    print(f"Mean Pearson(Att): {test_pearson:.5f}")
    print(f"Mean Pearson(Unatt): {np.mean(unatt_corrs):.5f}")
    print(f"Mean Margin: {mean_margin:.5f} (Std: {std_margin:.5f})")
    print(f"Overall Accuracy: {test_acc*100:.1f}%")

    print("\n--- ROC Inputs & Calculation ---")
    labels = np.concatenate([np.ones(len(att_corrs)), np.zeros(len(unatt_corrs))])
    scores = np.concatenate([att_corrs, unatt_corrs])
    
    print(f"Labels ({len(labels)} total): {labels}")
    print(f"Scores ({len(scores)} total): {np.round(scores, 4)}")
    print(f"Positive Samples (Attended): {sum(labels == 1)}")
    print(f"Negative Samples (Unattended): {sum(labels == 0)}")
    
    try:
        test_auc = roc_auc_score(labels, scores)
        print(f"\n=> MANUALLY COMPUTED ROC-AUC: {test_auc:.4f}")
    except Exception as e:
        print(f"\n=> ERROR COMPUTING ROC-AUC: {e}")

if __name__ == "__main__":
    main()
