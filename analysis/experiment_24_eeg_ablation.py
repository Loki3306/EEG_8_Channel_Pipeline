import os
import sys
import re
import torch
import numpy as np
import scipy.io
from pathlib import Path
import torch.nn.functional as F

# Add project root to path
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from models.matchnet import ContrastiveMatchNet
from training.train_kul_matchnet_loso import get_kul_subject_files, load_kul_trials, preprocess_trial

DECISION_WINDOW_SEC = 10
FS = 64

def evaluate_ablation(model, test_data, device, mode="control", window_sec=DECISION_WINDOW_SEC, fs=FS):
    model.eval()
    win_samples = int(window_sec * fs)
    
    total_trials = len(test_data)
    correct_trials = 0.0
    
    print(f"\n==================================================")
    print(f"EVALUATION MODE: {mode.upper()}")
    print(f"==================================================")
    
    with torch.no_grad():
        for i, t in enumerate(test_data):
            x = t["eeg"].numpy()
            ya = t["audio_a"].numpy()
            yb = t["audio_b"].numpy()
            meta = t["meta"]
            
            # Apply ablation to EEG (x)
            x_test = x.copy()
            ya_test = ya.copy()
            yb_test = yb.copy()
            
            if mode == "zero":
                x_test = np.zeros_like(x)
            elif mode == "random":
                x_test = np.random.randn(*x.shape)
            elif mode == "shuffle":
                # Pick a random other trial for EEG
                rand_idx = np.random.randint(0, len(test_data))
                x_other = test_data[rand_idx]["eeg"].numpy()
                
                # We need to make sure the lengths match. We truncate to the min length
                min_len = min(x_other.shape[1], x.shape[1])
                x_test = x_other[:, :min_len]
                ya_test = ya[:, :min_len]
                yb_test = yb[:, :min_len]
                
            start = 0
            trial_sim_a = []
            trial_sim_b = []
            
            while start + win_samples <= x_test.shape[1]:
                end = start + win_samples
                cx = torch.FloatTensor(x_test[:, start:end]).unsqueeze(0).to(device)
                cya = torch.FloatTensor(ya_test[:, start:end]).unsqueeze(0).to(device)
                cyb = torch.FloatTensor(yb_test[:, start:end]).unsqueeze(0).to(device)
                
                z_eeg, z_a, z_b = model(cx, cya, cyb)
                sim_a = F.cosine_similarity(z_eeg, z_a, dim=1).mean().item()
                sim_b = F.cosine_similarity(z_eeg, z_b, dim=1).mean().item()
                
                trial_sim_a.append(sim_a)
                trial_sim_b.append(sim_b)
                start += win_samples
                
            if trial_sim_a:
                mean_a = np.mean(trial_sim_a)
                mean_b = np.mean(trial_sim_b)
                margin = mean_a - mean_b
                
                pred = "CORRECT" if mean_a > mean_b else "WRONG" if mean_a < mean_b else "TIE"
                print(f"Trial {meta.get('TrialID', i+1):02d} | Exp {meta.get('experiment', 'Unknown')} | Track {meta.get('attended_track', 'Unknown')} | Margin: {margin:+.4f} | Pred: {pred}")
                
                if mean_a > mean_b: correct_trials += 1.0
                elif mean_a == mean_b: correct_trials += 0.5
                
    trial_acc = correct_trials / max(total_trials, 1)
    print(f"--------------------------------------------------")
    print(f"{mode.upper()} ACCURACY: {trial_acc*100:.2f}% ({correct_trials}/{total_trials})")
    print(f"==================================================\n")
    return trial_acc


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running Phase 1: EEG Ablation Experiments on {device}")
    
    # 1. We need a trained LOSO model.
    if len(sys.argv) > 1:
        ckpt_path = Path(sys.argv[1])
        if not ckpt_path.exists():
            print(f"Provided checkpoint not found: {ckpt_path}")
            return
    else:
        ckpt_dir = REPO_ROOT / "checkpoints"
        if not ckpt_dir.exists():
            print("No checkpoints found. Please run KUL LOSO training first.")
            return
            
        ckpts = list(ckpt_dir.glob("matchnet_kul_fold_S*_best.pth"))
        if not ckpts:
            print("No KUL LOSO checkpoints found.")
            return
            
        # Just grab the first one
        ckpt_path = ckpts[0]
    
    m = re.search(r"fold_(S\d+)_best", ckpt_path.name)
    if not m:
        print(f"Could not parse held-out subject from {ckpt_path.name}")
        return
        
    held_out_id = m.group(1)
    print(f"Found trained model for Held-Out Subject: {held_out_id}")
    
    # 2. Load Model
    model = ContrastiveMatchNet("eegnet", eeg_channels=8, audio_channels=28, latent_dim=64).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    
    # 3. Load the cached data for this specific held-out subject
    from data.kul_cached_dataset import KULCachedLoader
    loader = KULCachedLoader(REPO_ROOT / "data" / "processed_kul")
    try:
        all_subject_data = loader.load_all()
    except FileNotFoundError:
        print("Could not find cached KUL dataset. Run preprocessing/build_kul_cache.py first.")
        return
        
    if held_out_id not in all_subject_data:
        print(f"Could not find data for {held_out_id} in cache.")
        return
        
    test_data = all_subject_data[held_out_id]
    print(f"Loaded {len(test_data)} valid trials for {held_out_id}.")
    
    # 4. Run Ablations
    np.random.seed(42) # For reproducible shuffling
    evaluate_ablation(model, test_data, device, mode="control")
    evaluate_ablation(model, test_data, device, mode="zero")
    evaluate_ablation(model, test_data, device, mode="random")
    evaluate_ablation(model, test_data, device, mode="shuffle")

if __name__ == "__main__":
    main()
