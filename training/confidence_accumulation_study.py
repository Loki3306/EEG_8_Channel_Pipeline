import os
import glob
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from copy import deepcopy
import sys
from pathlib import Path

# Add project root to path
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from baselines.ridge_aad import load_subject_examples, subject_files
from training.train_matchnet_loso import prepare_dataset, chunk_trial, get_mapping_data
from models.matchnet import ContrastiveMatchNet
from training.quick_loso import CHANNELS, LOWCUT, HIGHCUT, NUM_BANDS

def extract_similarities_for_trial(model, x, ya, yb, window_sec, device):
    """
    Chunks a 60s trial into non-overlapping windows of `window_sec`
    and computes sim_a and sim_b for each chunk.
    """
    # chunk_trial expects hop_sec. For non-overlapping, hop_sec = window_sec
    cx, cya, cyb = chunk_trial(x, ya, yb, window_sec, window_sec)
    
    if len(cx) == 0:
        return [], []
        
    x_tensor = torch.FloatTensor(np.stack(cx)).to(device)
    ya_tensor = torch.FloatTensor(np.stack(cya)).to(device)
    yb_tensor = torch.FloatTensor(np.stack(cyb)).to(device)
    
    with torch.no_grad():
        with torch.amp.autocast('cuda' if torch.cuda.is_available() else 'cpu'):
            z_eeg, z_a, z_b = model(x_tensor, ya_tensor, yb_tensor)
            
            if hasattr(model, 'compute_similarity'):
                batch_sim_a = model.compute_similarity(z_eeg, z_a).cpu().numpy()
                batch_sim_b = model.compute_similarity(z_eeg, z_b).cpu().numpy()
            else:
                import torch.nn.functional as F
                batch_sim_a = F.cosine_similarity(z_eeg, z_a, dim=1).mean(dim=1).cpu().numpy()
                batch_sim_b = F.cosine_similarity(z_eeg, z_b, dim=1).mean(dim=1).cpu().numpy()
                
    return batch_sim_a, batch_sim_b

def run_study():
    target_subjects = ["S8", "S9", "S11"]
    windows = [0.5, 1.0, 2.0, 3.0, 5.0]
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running Confidence Accumulation Study on {device}...")
    
    mapping, envelopes = get_mapping_data()
    all_paths = subject_files()
    
    ckpt_files = glob.glob(str(REPO_ROOT / "checkpoints" / "matchnet_fold_S*_best.pth"))
    if not ckpt_files:
        print("No checkpoints found. Ensure checkpoints are in 'checkpoints/' directory.")
        return
        
    # Data structures to store results
    # results[subject][window][method] = accuracy
    results = {s: {w: {} for w in windows} for s in target_subjects}
    
    # Store SPRT latency data: sprt_results[window][threshold] = {"acc": [], "latency": []}
    sprt_results = {w: {t: {"correct": [], "latency": []} for t in [1, 2, 3, 4]} for w in windows}
    
    # Static method lists
    static_methods = ["Majority Voting", "Confidence Accumulation", "Normalized Confidence", "Log-Likelihood"]
    
    for subj in target_subjects:
        ckpt_path = next((c for c in ckpt_files if f"_{subj}_" in os.path.basename(c)), None)
        if not ckpt_path:
            print(f"Missing checkpoint for {subj}, skipping.")
            continue
            
        print(f"\n--- Processing {subj} ---")
        
        test_path = next((p for p in all_paths if p.stem.split('_')[0] == subj), None)
        test_exs = load_subject_examples(test_path)
        X_te, YA_te, YB_te = prepare_dataset(test_exs, CHANNELS, LOWCUT, HIGHCUT, test_path.stem, mapping, envelopes)
        
        # Load model
        model = ContrastiveMatchNet(eeg_model_type="eegnet", eeg_channels=len(CHANNELS), audio_channels=NUM_BANDS, latent_dim=64, audio_model_type="standard").to(device)
        model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
        model.eval()
        
        for w in windows:
            print(f"  Evaluating window size {w}s...")
            method_correct = {m: 0 for m in static_methods}
            total_trials = len(X_te)
            
            for i in range(total_trials):
                sa, sb = extract_similarities_for_trial(model, X_te[i], YA_te[i], YB_te[i], w, device)
                if len(sa) == 0:
                    continue
                
                N = len(sa)
                
                # 1. Majority Voting
                votes = np.sign(sa - sb)
                votes[votes == 0] = -1 # Tie goes to unattended
                if np.sum(votes) > 0:
                    method_correct["Majority Voting"] += 1
                    
                # 2. Confidence Accumulation
                if np.sum(sa - sb) > 0:
                    method_correct["Confidence Accumulation"] += 1
                    
                # 3. Normalized Confidence
                norm_score = (sa - sb) / (np.abs(sa) + np.abs(sb) + 1e-8)
                if np.sum(norm_score) > 0:
                    method_correct["Normalized Confidence"] += 1
                    
                # 4. Log-Likelihood
                k = 5.0
                delta = sa - sb
                p = 1.0 / (1.0 + np.exp(-k * delta))
                p = np.clip(p, 1e-7, 1.0 - 1e-7)
                llr = np.log(p / (1.0 - p))
                if np.sum(llr) > 0:
                    method_correct["Log-Likelihood"] += 1
                    
                # 5. SPRT Style (across different thresholds)
                for thresh in [1, 2, 3, 4]:
                    running_llr = 0.0
                    decided = False
                    
                    for step_idx in range(N):
                        running_llr += llr[step_idx]
                        if running_llr > thresh:
                            sprt_results[w][thresh]["correct"].append(1)
                            sprt_results[w][thresh]["latency"].append((step_idx + 1) * w)
                            decided = True
                            break
                        elif running_llr < -thresh:
                            sprt_results[w][thresh]["correct"].append(0)
                            sprt_results[w][thresh]["latency"].append((step_idx + 1) * w)
                            decided = True
                            break
                            
                    if not decided:
                        sprt_results[w][thresh]["correct"].append(1 if running_llr > 0 else 0)
                        sprt_results[w][thresh]["latency"].append(N * w)
            
            for m in static_methods:
                results[subj][w][m] = method_correct[m] / max(total_trials, 1)

    # ------------------ OUTPUT TABLES ------------------
    print("\n\n" + "="*80)
    print("PER SUBJECT ACCURACY BY WINDOW SIZE")
    print("="*80)
    
    for w in windows:
        print(f"\nWindow: {w}s")
        print("| Subject | " + " | ".join([f"{m:23s}" for m in static_methods]) + " |")
        print("| ------- | " + " | ".join(["-"*23 for _ in static_methods]) + " |")
        for subj in target_subjects:
            if subj not in results or w not in results[subj] or len(results[subj][w]) == 0:
                continue
            row = f"| {subj:7s} | "
            row += " | ".join([f"{results[subj][w][m]*100:23.2f}" for m in static_methods]) + " |"
            print(row)
            
    print("\n\n" + "="*80)
    print("MEAN ACCURACY ACROSS SUBJECTS")
    print("="*80)
    print("| Window | " + " | ".join([f"{m:23s}" for m in static_methods]) + " |")
    print("| ------ | " + " | ".join(["-"*23 for _ in static_methods]) + " |")
    
    mean_method_accs = {w: {m: 0.0 for m in static_methods} for w in windows}
    for w in windows:
        row = f"| {w:5.1f}s | "
        for m in static_methods:
            accs = [results[s][w][m] for s in target_subjects if len(results[s][w]) > 0]
            m_acc = np.mean(accs) * 100 if len(accs) > 0 else 0.0
            mean_method_accs[w][m] = m_acc
            row += f"{m_acc:23.2f} | "
        print(row)
        
    print("\n\n" + "="*80)
    print("SPRT ADAPTIVE STOPPING ANALYSIS (Log-Likelihood)")
    print("="*80)
    print("| Window | Thresh | Accuracy | Mean Latency | Median Latency |")
    print("| ------ | ------ | -------- | ------------ | -------------- |")
    
    for w in windows:
        for t in [1, 2, 3, 4]:
            arr_c = sprt_results[w][t]["correct"]
            arr_l = sprt_results[w][t]["latency"]
            if len(arr_c) == 0: continue
            
            acc = np.mean(arr_c) * 100
            mlat = np.mean(arr_l)
            medlat = np.median(arr_l)
            
            print(f"| {w:5.1f}s | ±{t:<5d} | {acc:7.2f}% | {mlat:11.2f}s | {medlat:13.2f}s |")
            
    print("\n" + "="*80)
    print("SUCCESS CRITERIA & RECOMMENDATION")
    print("="*80)
    
    recommended = False
    for w in windows:
        if w > 5.0: continue
        
        base_acc = mean_method_accs[w]["Majority Voting"]
        for m in static_methods[1:]:
            gain = mean_method_accs[w][m] - base_acc
            if gain >= 3.0:
                print(f"STRONG PROMOTE: {m} at {w}s window yields +{gain:.2f}% (>= 3%) over baseline.")
                recommended = True
            elif gain >= 2.0:
                print(f"PROMOTE: {m} at {w}s window yields +{gain:.2f}% (>= 2%) over baseline.")
                recommended = True
                
    if not recommended:
        print("No static method achieved >= 2% gain at <= 5s windows.")

if __name__ == "__main__":
    run_study()
