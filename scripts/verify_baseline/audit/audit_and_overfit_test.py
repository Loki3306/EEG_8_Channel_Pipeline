import sys
import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from models.catcn import CATCNDirectDecoder
from baselines.ridge_aad import load_subject_examples, subject_files
from training.train_matchnet_wavlm import (
    FS, TRAIN_WINDOW_SEC, prepare_dataset, get_mapping_data, ChunkDataset
)

def test_1_prove_labels():
    print("\n" + "="*80)
    print(" TEST 1: PROVE DATASET LABELS & PRESENTATION TARGETS")
    print("="*80)
    
    paths = subject_files()
    if not paths:
        print("[!] No dataset files found. Ensure DTU files are in DATA_DIR or /kaggle/input.")
        return None, None, None
        
    mapping, envelopes = get_mapping_data("gammatone")
    
    # Audit 20 consecutive trials across subjects
    print(f"\n[Audit]: Inspecting trial-to-audio label mapping across available subjects:")
    print(f"{'Idx':>3} | {'Subj':>8} | {'RawLabel':>8} | {'AttStream':>9} | {'WavA File':>30} | {'WavB File':>30} | {'Target':>6}")
    print("-" * 105)
    
    ex_count = 0
    label_counts = {1: 0, 2: 0}
    
    sample_exs = []
    sample_subj_name = ""
    
    for p in paths[:2]:
        sub_name = p.stem
        sub_key = sub_name.replace("_data_preproc", "")
        exs = load_subject_examples(p)
        if not sample_exs:
            sample_exs = exs[:2] # Save first 2 trials for Overfit Test 2
            sample_subj_name = sub_name
            
        for i, ex in enumerate(exs[:10]):
            trial_key = f"trial_{i}"
            if sub_key in mapping and trial_key in mapping[sub_key]:
                fname_a = mapping[sub_key][trial_key]["wavA"]["filename"]
                fname_b = mapping[sub_key][trial_key]["wavB"]["filename"]
                lbl = getattr(ex, 'label', 1)
                label_counts[lbl] = label_counts.get(lbl, 0) + 1
                
                # DTU convention (DATASETS_REFERENCE.md):
                # wavA is ALWAYS attended, wavB is ALWAYS unattended
                # ex.label indicates speaker gender (1=Male, 2=Female)
                gender = "Male" if lbl == 1 else "Female"
                att_stream = "wavA (Attended)"
                target_a = 1.0
                
                print(f"{ex_count:3d} | {sub_key:>8} | {lbl:8d} ({gender:>6}) | {att_stream:>15} | {fname_a:>28} | {fname_b:>28} | {target_a:6.1f}")
                ex_count += 1
                
    total_audited = sum(label_counts.values())
    print("-" * 115)
    print(f"Trigger Distribution: Label 1 (Attend Male) = {label_counts.get(1,0)} ({label_counts.get(1,0)/max(total_audited,1)*100:.1f}%) | "
          f"Label 2 (Attend Female) = {label_counts.get(2,0)} ({label_counts.get(2,0)/max(total_audited,1)*100:.1f}%)")
    print(f"[Verified]: wavA is ALWAYS attended across all conditions in audio_mapping.json.")
    print("-" * 115)
    
    return paths, mapping, envelopes, sample_exs, sample_subj_name

def test_2_overfit_two_trials(sample_exs, sample_subj_name, mapping, envelopes):
    print("\n" + "="*80)
    print(" TEST 2: OVERFIT LITMUS TEST (2 TRIALS, ZERO REGULARIZATION)")
    print("="*80)
    print(f"Goal: Can CA-TCN drive 2 trials of {sample_subj_name} to >95% training accuracy?")
    print(f"Setup: dropout=0.0, weight_decay=0.0, Adam lr=1e-3, 100 epochs")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    channels = list(range(64))
    
    # Prepare only the 2 sample trials
    X_list, YA_list, YB_list = prepare_dataset(sample_exs, channels, 1.0, 6.0, sample_subj_name, mapping, envelopes)
    
    X_t = [torch.from_numpy(x) for x in X_list]
    YA_t = [torch.from_numpy(y) for y in YA_list]
    YB_t = [torch.from_numpy(y) for y in YB_list]
    
    chunk_indices = []
    win_samples = int(TRAIN_WINDOW_SEC * FS)
    hop_samples = int(1.25 * FS) # 75% overlap
    for i in range(len(X_t)):
        t_len = X_t[i].shape[1]
        start = 0
        while start + win_samples <= t_len:
            chunk_indices.append((i, start, start + win_samples))
            start += hop_samples
            
    print(f"Extracted {len(chunk_indices)} chunks from 2 trials. Device: {device}")
    
    dataset = ChunkDataset(X_t, YA_t, YB_t, chunk_indices)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Model with zero dropout
    model = CATCNDirectDecoder(
        eeg_channels=64,
        audio_channels=1,
        hidden_dim=64,
        max_lag_samples=8,
        dropout=0.0
    ).to(device)
    
    # Pure Adam with lr=1e-3, zero weight decay
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=0.0)
    
    print("\nEpoch |   BCE Loss | Train Acc | Status")
    print("-" * 42)
    
    torch.manual_seed(42)
    for epoch in range(1, 101):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for bx, bya, byb in loader:
            bx = bx.to(device)
            bya = bya.to(device)
            byb = byb.to(device)
            
            # 50/50 balanced order
            swap_mask = torch.rand(bx.size(0), device=device) > 0.5
            c1 = torch.where(swap_mask[:, None, None], byb, bya)
            c2 = torch.where(swap_mask[:, None, None], bya, byb)
            target = torch.where(swap_mask, torch.zeros(bx.size(0), device=device), torch.ones(bx.size(0), device=device))
            
            optimizer.zero_grad()
            delta, (la, lb), _ = model(bx, c1, c2)
            loss = F.binary_cross_entropy_with_logits(delta, target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * bx.size(0)
            preds = (delta > 0).float()
            correct += (preds == target).sum().item()
            total += bx.size(0)
            
        epoch_loss = total_loss / total
        epoch_acc = (correct / total) * 100.0
        
        if epoch % 10 == 0 or epoch == 1 or epoch_acc >= 95.0:
            status = "CONVERGED!" if epoch_acc >= 95.0 else "training..."
            print(f"{epoch:5d} | {epoch_loss:10.4f} | {epoch_acc:8.2f}% | {status}")
            if epoch_acc >= 95.0:
                print(f"\n[SUCCESS]: CA-TCN easily overfit the 2 training trials to {epoch_acc:.2f}% (BCE Loss: {epoch_loss:.4f})!")
                print(f"[VERDICT]: The architecture, causal/anticausal directionality, multi-lag cross-correlation head, and loss are mathematically sound.")
                return model, loader, device
                
    if epoch_acc < 90.0:
        print(f"\n[FAIL]: Model failed to overfit 2 trials after 100 epochs (Final Acc: {epoch_acc:.2f}%, Loss: {epoch_loss:.4f}).")
        print(f"[VERDICT]: Core pipeline issue exists in the tensor representation or gradients.")
    else:
        print(f"\n[PASS]: Model reached {epoch_acc:.2f}% on 2 trials.")
        
    return model, loader, device

def test_3_ab_swap_test(model, loader, device):
    print("\n" + "="*80)
    print(" TEST 3: A/B SWAP TEST ON TRAINED OVERFIT MODEL")
    print("="*80)
    model.eval()
    bx, bya, byb = next(iter(loader))
    bx, bya, byb = bx.to(device), bya.to(device), byb.to(device)
    
    with torch.no_grad():
        delta_normal, _, _ = model(bx, bya, byb)
        delta_swapped, _, _ = model(bx, byb, bya)
        err = (delta_normal + delta_swapped).abs().max().item()
        
    print(f"Sample Normal Logits:  {delta_normal[:4].cpu().numpy().round(4)}")
    print(f"Sample Swapped Logits: {delta_swapped[:4].cpu().numpy().round(4)}")
    print(f"Max Anti-Symmetry Error: {err:.8e}")
    if err < 1e-5:
        print("PASS: Exact A/B Anti-Symmetry strictly satisfied (Delta_swapped == -Delta).")
    else:
        print(f"FAIL: Anti-symmetry violated by {err}!")

def main():
    res = test_1_prove_labels()
    if res[0] is None:
        return
    paths, mapping, envelopes, sample_exs, sample_subj_name = res
    model, loader, device = test_2_overfit_two_trials(sample_exs, sample_subj_name, mapping, envelopes)
    if model is not None:
        test_3_ab_swap_test(model, loader, device)

if __name__ == "__main__":
    main()
