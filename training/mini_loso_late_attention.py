import argparse
import os
import time
import numpy as np
import torch
import sys
from pathlib import Path

# Add project root to path
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from baselines.ridge_aad import load_subject_examples, subject_files
from training.train_matchnet_loso import prepare_dataset, get_mapping_data
from training.quick_loso import evaluate_model_version, CHANNELS, LOWCUT, HIGHCUT

def mini_loso(epochs, batch_size):
    target_subjects = ["S1", "S4", "S6", "S8", "S11", "S14"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Starting Mini-LOSO on device: {device} for {epochs} epochs")
    
    mapping, envelopes = get_mapping_data()
    all_paths = subject_files()
    if not all_paths:
        print("No subjects found.")
        return
        
    subject_examples = {str(p): load_subject_examples(p) for p in all_paths}
    
    results = {}
    
    for held_out_subj in target_subjects:
        held_out_path = next((p for p in all_paths if p.stem.split('_')[0] == held_out_subj), None)
        if not held_out_path:
            print(f"Warning: Subject {held_out_subj} not found, skipping.")
            continue
            
        print(f"\n--- Testing Subject {held_out_subj} ---")
        
        train_paths = [p for p in all_paths if p.stem.split('_')[0] != held_out_subj]
        
        train_exs = []
        for p in train_paths: train_exs.extend(subject_examples[str(p)])
        test_exs = subject_examples[str(held_out_path)]
        
        np.random.seed(42)
        np.random.shuffle(train_exs)
        
        X_tr_full, YA_tr_full, YB_tr_full = [], [], []
        X_va_full, YA_va_full, YB_va_full = [], [], []
        
        for p in train_paths:
            tX, tYA, tYB = prepare_dataset(subject_examples[str(p)], CHANNELS, LOWCUT, HIGHCUT, p.stem, mapping, envelopes)
            v_split_idx = int(0.1 * len(tX))
            X_va_full.extend(tX[:v_split_idx]); YA_va_full.extend(tYA[:v_split_idx]); YB_va_full.extend(tYB[:v_split_idx])
            X_tr_full.extend(tX[v_split_idx:]); YA_tr_full.extend(tYA[v_split_idx:]); YB_tr_full.extend(tYB[v_split_idx:])
            
        X_te_full, YA_te_full, YB_te_full = prepare_dataset(test_exs, CHANNELS, LOWCUT, HIGHCUT, held_out_path.stem, mapping, envelopes)
        
        print("Running Baseline EEGNet...")
        b_val, b_test, b_time = evaluate_model_version([], "eegnet", "standard", "contrastive", False, False, X_tr_full, YA_tr_full, YB_tr_full, X_va_full, YA_va_full, YB_va_full, X_te_full, YA_te_full, YB_te_full, device, epochs, batch_size)
        
        print("Running EEGNet + LateAudioAttention...")
        m_val, m_test, m_time = evaluate_model_version([], "eegnet", "standard", "contrastive", False, True, X_tr_full, YA_tr_full, YB_tr_full, X_va_full, YA_va_full, YB_va_full, X_te_full, YA_te_full, YB_te_full, device, epochs, batch_size)
        
        results[held_out_subj] = {
            "baseline": b_test,
            "multiscale": m_test,
            "b_time": b_time,
            "m_time": m_time
        }
        print(f"  Baseline   : {b_test*100:.2f}% ({b_time:.1f}s)")
        print(f"  MultiScale : {m_test*100:.2f}% ({m_time:.1f}s)")
        print(f"  Delta      : {(m_test - b_test)*100:+.2f}%")
        
    print("\n\n" + "="*60)
    print("MINI-LOSO VALIDATION SUMMARY (Accuracy %)")
    print("="*60)
    print("| Subject | Baseline | LateAttn   | Delta |")
    print("| ------- | -------- | ---------- | ----- |")
    
    b_list, m_list, d_list = [], [], []
    for subj in target_subjects:
        if subj not in results: continue
        res = results[subj]
        b, m = res['baseline']*100, res['multiscale']*100
        d = m - b
        b_list.append(b); m_list.append(m); d_list.append(d)
        print(f"| {subj:7s} | {b:8.2f} | {m:10.2f} | {d:+5.2f} |")
        
    print("| ------- | -------- | ---------- | ----- |")
    if len(d_list) > 0:
        mean_b = np.mean(b_list)
        mean_m = np.mean(m_list)
        mean_d = np.mean(d_list)
        median_d = np.median(d_list)
        std_d = np.std(d_list)
        
        print(f"| Mean    | {mean_b:8.2f} | {mean_m:10.2f} | {mean_d:+5.2f} |")
        print("="*60)
        print(f"Median Delta : {median_d:+.2f}%")
        print(f"StdDev Delta : {std_d:.2f}%")
        
        print("\nDECISION GATE:")
        if mean_d < 1.0:
            print(f"❌ Recommend: KILL (Mean Delta {mean_d:+.2f}% is < +1.0%)")
        elif 1.0 <= mean_d <= 2.0:
            print(f"⚠️ Recommend: OPTIONAL FULL LOSO (Mean Delta {mean_d:+.2f}% is between +1.0% and +2.0%)")
        else:
            print(f"✅ Recommend: PROMOTE TO FULL LOSO (Mean Delta {mean_d:+.2f}% is > +2.0%)")
    else:
        print("No valid results computed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mini-LOSO for MultiScaleEEGNet")
    parser.add_argument("--epochs", type=int, default=20, help="Epochs to train per model")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    args = parser.parse_args()
    
    mini_loso(args.epochs, args.batch_size)
