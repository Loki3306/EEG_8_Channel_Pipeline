import os
import sys
import numpy as np
import torch
from pathlib import Path
import json
import random
import pandas as pd
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from data.kul_cached_dataset import KULCachedLoader
from models.aad_conformer import AADConformer

def safe_corr_np(x, y, eps=1e-8):
    x_mean = x.mean()
    y_mean = y.mean()
    num = np.sum((x - x_mean) * (y - y_mean))
    den = np.sqrt(np.sum((x - x_mean)**2) * np.sum((y - y_mean)**2))
    return num / (den + eps)

def evaluate_trial_majority_vote(predicted: np.ndarray, wav_a: np.ndarray, wav_b: np.ndarray, window_seconds: int, hop_seconds: float = 1.0, fs: int = 64):
    win_samples = int(window_seconds * fs)
    hop_samples = int(hop_seconds * fs)
    
    if win_samples >= predicted.shape[0]:
        c_a = safe_corr_np(predicted, wav_a)
        c_b = safe_corr_np(predicted, wav_b)
        return c_a > c_b, 1, 1 if c_a > c_b else 0
        
    correct_windows = 0
    total_windows = 0
    
    for start in range(0, predicted.shape[0] - win_samples + 1, hop_samples):
        stop = start + win_samples
        c_a = safe_corr_np(predicted[start:stop], wav_a[start:stop])
        c_b = safe_corr_np(predicted[start:stop], wav_b[start:stop])
        if c_a > c_b:
            correct_windows += 1
        total_windows += 1
        
    if total_windows == 0:
        return False, 0, 0
        
    trial_correct = (correct_windows > total_windows / 2.0)
    return trial_correct, total_windows, correct_windows

def main():
    print("Loading KUL Cache...")
    cache_dir = REPO_ROOT / "data" / "processed_kul"
    if not cache_dir.exists():
        print(f"KUL Cache not found at {cache_dir}.")
        return
        
    loader = KULCachedLoader(cache_dir)
    all_subject_data = loader.load_all()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    checkpoint_dir = REPO_ROOT / "conformer_loso_results" / "checkpoints" / "seed_1"
    if not checkpoint_dir.exists():
        print(f"Checkpoints not found at {checkpoint_dir}")
        return
        
    out_dir = REPO_ROOT / "results" / "conformer_loso" / "falsification"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    subjects = sorted(list(all_subject_data.keys()))
    
    # Flat list of all trials for permutation experiments
    all_trials_flat = []
    for sub, trials in all_subject_data.items():
        for t in trials:
            all_trials_flat.append({"subject": sub, "trial": t})
            
    # Set up experiments tracking
    experiments = {
        1: {"name": "Standard Evaluation", "desc": "Reference baseline"},
        2: {"name": "True Audio Permutation", "desc": "Random trial audio"},
        3: {"name": "Within-Subject Permutation", "desc": "Audio from same subject"},
        4: {"name": "Cross-Subject Permutation", "desc": "Audio from diff subject, strict disjoint"},
        5: {"name": "Random Gaussian Envelope", "desc": "Gaussian noise audio"},
        6: {"name": "Zero EEG", "desc": "EEG = 0"},
        7: {"name": "Random EEG", "desc": "EEG = Gaussian noise"},
        8: {"name": "Circular Shift (2s)", "desc": "Shift audio 2s"},
        9: {"name": "Circular Shift (10s)", "desc": "Shift audio 10s"},
        10: {"name": "Label Shuffle", "desc": "Swap attended/unattended"}
    }
    
    results_store = {k: {"trial_correct": 0, "total_trials": 0, 
                         "windows_correct": 0, "total_windows": 0,
                         "margins": [], "p_att": [], "p_unatt": []} for k in experiments.keys()}
                         
    # For Exp 10 (Story Audit)
    story_overlap_cases = 0
    total_test_stories = 0
    
    # Load model structure
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
    
    for test_subject in subjects:
        print(f"Processing Test Subject: {test_subject}")
        checkpoint_path = checkpoint_dir / f"model_{test_subject}.pt"
        if not checkpoint_path.exists():
            print(f"Missing checkpoint for {test_subject}")
            continue
            
        model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
        model.eval()
        
        test_trials = all_subject_data[test_subject]
        
        # Determine training stories for Experiment 10
        train_stories = set()
        for sub, trials in all_subject_data.items():
            if sub != test_subject:
                for t in trials:
                    train_stories.add(t['meta']['stimuli_left'])
                    train_stories.add(t['meta']['stimuli_right'])
                    
        with torch.no_grad():
            for t_idx, t in enumerate(test_trials):
                # Exp 10 Logic
                s_left = t['meta']['stimuli_left']
                s_right = t['meta']['stimuli_right']
                total_test_stories += 2
                if s_left in train_stories: story_overlap_cases += 1
                if s_right in train_stories: story_overlap_cases += 1
                
                # Base Data
                eeg_base = t["eeg"].unsqueeze(0).to(device)
                audio_a_base = t["audio_a"].unsqueeze(0).to(device).mean(dim=1, keepdim=True)
                audio_b_base = t["audio_b"].unsqueeze(0).to(device).mean(dim=1, keepdim=True)
                
                for exp_id in experiments.keys():
                    # Reset to base for each exp
                    eeg = eeg_base.clone()
                    audio_a = audio_a_base.clone()
                    audio_b = audio_b_base.clone()
                    
                    if exp_id == 2:
                        # Random trial
                        rand_t = random.choice([x for x in all_trials_flat if x["trial"] is not t])["trial"]
                        audio_a = rand_t["audio_a"].unsqueeze(0).to(device).mean(dim=1, keepdim=True)
                        audio_b = rand_t["audio_b"].unsqueeze(0).to(device).mean(dim=1, keepdim=True)
                    elif exp_id == 3:
                        # Within subject
                        valid_within = [x for x in test_trials if x is not t]
                        if valid_within:
                            rand_t = random.choice(valid_within)
                            audio_a = rand_t["audio_a"].unsqueeze(0).to(device).mean(dim=1, keepdim=True)
                            audio_b = rand_t["audio_b"].unsqueeze(0).to(device).mean(dim=1, keepdim=True)
                    elif exp_id == 4:
                        # Cross subject STRICT
                        valid_cross = [x for x in all_trials_flat if x["subject"] != test_subject and 
                                       x["trial"]["meta"]["stimuli_left"] not in [s_left, s_right] and 
                                       x["trial"]["meta"]["stimuli_right"] not in [s_left, s_right]]
                        if valid_cross:
                            rand_t = random.choice(valid_cross)["trial"]
                            audio_a = rand_t["audio_a"].unsqueeze(0).to(device).mean(dim=1, keepdim=True)
                            audio_b = rand_t["audio_b"].unsqueeze(0).to(device).mean(dim=1, keepdim=True)
                    elif exp_id == 5:
                        # Random Gaussian Envelope
                        audio_a = torch.randn_like(audio_a)
                        audio_b = torch.randn_like(audio_b)
                    elif exp_id == 6:
                        # Zero EEG
                        eeg = torch.zeros_like(eeg)
                    elif exp_id == 7:
                        # Random EEG
                        eeg = torch.randn_like(eeg)
                    elif exp_id == 8:
                        # Circular Shift 2s
                        shift_samp = int(2 * 64)
                        audio_a = torch.roll(audio_a, shifts=shift_samp, dims=2)
                        audio_b = torch.roll(audio_b, shifts=shift_samp, dims=2)
                    elif exp_id == 9:
                        # Circular Shift 10s
                        shift_samp = int(10 * 64)
                        audio_a = torch.roll(audio_a, shifts=shift_samp, dims=2)
                        audio_b = torch.roll(audio_b, shifts=shift_samp, dims=2)
                    elif exp_id == 10:
                        # Label Shuffle (Swap A and B 50% of time)
                        if random.random() > 0.5:
                            audio_a, audio_b = audio_b, audio_a
                            
                    # Normalization
                    eeg_norm = (eeg - eeg.mean(dim=2, keepdim=True)) / (eeg.std(dim=2, keepdim=True) + 1e-8)
                    audio_a_norm = (audio_a - audio_a.mean(dim=2, keepdim=True)) / (audio_a.std(dim=2, keepdim=True) + 1e-8)
                    audio_b_norm = (audio_b - audio_b.mean(dim=2, keepdim=True)) / (audio_b.std(dim=2, keepdim=True) + 1e-8)
                    
                    min_len = min(eeg_norm.shape[2], audio_a_norm.shape[2], audio_b_norm.shape[2])
                    eeg_norm = eeg_norm[:, :, :min_len]
                    audio_a_norm = audio_a_norm[:, :, :min_len]
                    audio_b_norm = audio_b_norm[:, :, :min_len]
                    
                    pred = model(eeg_norm)
                    
                    pred_np = pred.squeeze(0).cpu().numpy()
                    wav_a_np = audio_a_norm.squeeze(1).squeeze(0).cpu().numpy()
                    wav_b_np = audio_b_norm.squeeze(1).squeeze(0).cpu().numpy()
                    
                    c_att = safe_corr_np(pred_np, wav_a_np)
                    c_unatt = safe_corr_np(pred_np, wav_b_np)
                    margin = c_att - c_unatt
                    
                    trial_ok, n_win, c_win = evaluate_trial_majority_vote(pred_np, wav_a_np, wav_b_np, window_seconds=10, hop_seconds=1.0, fs=64)
                    
                    results_store[exp_id]["total_trials"] += 1
                    if trial_ok: results_store[exp_id]["trial_correct"] += 1
                    results_store[exp_id]["total_windows"] += n_win
                    results_store[exp_id]["windows_correct"] += c_win
                    results_store[exp_id]["margins"].append(float(margin))
                    results_store[exp_id]["p_att"].append(float(c_att))
                    results_store[exp_id]["p_unatt"].append(float(c_unatt))

    # --- Write Results to CSV ---
    records = []
    for exp_id, stats in results_store.items():
        margins = np.array(stats["margins"])
        t_acc = stats["trial_correct"] / max(1, stats["total_trials"])
        w_acc = stats["windows_correct"] / max(1, stats["total_windows"])
        
        expected = experiments[exp_id]["desc"]
        if exp_id == 1: 
            expected_acc = 0.77
        else:
            expected_acc = 0.50
            
        pass_fail = "PASS" if abs(t_acc - expected_acc) < 0.10 else "FAIL"
        
        records.append({
            "Experiment": experiments[exp_id]["name"],
            "Description": experiments[exp_id]["desc"],
            "Trial Accuracy": f"{t_acc:.4f}",
            "Window Accuracy": f"{w_acc:.4f}",
            "Mean Pearson(att)": f"{np.mean(stats['p_att']):.4f}",
            "Mean Pearson(unatt)": f"{np.mean(stats['p_unatt']):.4f}",
            "Mean Margin": f"{np.mean(margins):.4f}",
            "Median Margin": f"{np.median(margins):.4f}",
            "Margin Std": f"{np.std(margins):.4f}",
            "Positive Margin %": f"{np.mean(margins > 0):.4f}",
            "Expected": expected_acc,
            "PASS/FAIL": pass_fail
        })
        
        # Plot Histogram
        plt.figure(figsize=(6, 4))
        plt.hist(margins, bins=50, color='skyblue', edgecolor='black')
        plt.axvline(0, color='red', linestyle='dashed', linewidth=2)
        plt.title(f"Margin Distribution: {experiments[exp_id]['name']}")
        plt.xlabel("Margin (Pearson_att - Pearson_unatt)")
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.savefig(out_dir / f"exp_{exp_id}_hist.png")
        plt.close()
        
    df = pd.DataFrame(records)
    csv_path = out_dir / "negative_control_summary.csv"
    df.to_csv(csv_path, index=False)
    
    print("\n--- STORY HOLDOUT AUDIT ---")
    print(f"Total Test Stories Evaluated: {total_test_stories}")
    print(f"Stories overlapping with training set: {story_overlap_cases}")
    print(f"Overlap Percentage: {story_overlap_cases / max(1, total_test_stories) * 100:.2f}%")
    print("\n")
    print(df.to_string())
    print(f"\nSaved falsification results to {out_dir}")

if __name__ == "__main__":
    main()
