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

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running Phase A: Similarity Probe on {device}")
    
    ckpt_dir = REPO_ROOT / "checkpoints"
    ckpts = list(ckpt_dir.glob("matchnet_kul_fold_S*_best.pth"))
    if not ckpts:
        print("No KUL LOSO checkpoints found.")
        return
        
    ckpt_path = ckpts[0]
    m = re.search(r"fold_(S\d+)_best", ckpt_path.name)
    held_out_id = m.group(1) if m else "S1"
    
    print(f"Loading weights from {ckpt_path.name} (Held-out: {held_out_id})")
    
    model = ContrastiveMatchNet("eegnet", eeg_channels=8, audio_channels=28, latent_dim=64).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    
    subject_paths = get_kul_subject_files()
    target_path = next((p for p in subject_paths if f"S{re.search(r'S(\d+)', p.name, re.IGNORECASE).group(1)}" == held_out_id), None)
            
    if not target_path:
        print(f"Could not find data for {held_out_id}")
        return
        
    trials = load_kul_trials(str(target_path))
    computed_envelope_cache = {}
    
    print(f"\nProbing raw similarities with ZERO EEG...")
    print(f"{'Trial':<6} | {'Exp':<5} | {'Track':<7} | {'Sim(Track 1)':<15} | {'Sim(Track 2)':<15}")
    print("-" * 65)
    
    with torch.no_grad():
        for i, t in enumerate(trials):
            x, ya, yb, _ = preprocess_trial(t, computed_envelope_cache, apply_car=True)
            if x is None:
                continue
                
            track_attended = getattr(t, "attended_track", "Unknown")
            exp = getattr(t, "experiment", "Unknown")
            
            # 1. Zero out EEG completely
            x_zero = np.zeros_like(x)
            
            win_samples = int(DECISION_WINDOW_SEC * FS)
            start = 0
            
            sim_a_list = []
            sim_b_list = []
            
            while start + win_samples <= x_zero.shape[1]:
                end = start + win_samples
                
                cx = torch.FloatTensor(x_zero[:, start:end]).unsqueeze(0).to(device)
                cya = torch.FloatTensor(ya[:, start:end]).unsqueeze(0).to(device)
                cyb = torch.FloatTensor(yb[:, start:end]).unsqueeze(0).to(device)
                
                # Get embeddings without projecting them to contrastive loss logic yet
                z_eeg, z_a, z_b = model(cx, cya, cyb)
                
                # Raw cosine similarity between ZERO EEG embedding and Audio embeddings
                sim_a = F.cosine_similarity(z_eeg, z_a, dim=1).mean().item()
                sim_b = F.cosine_similarity(z_eeg, z_b, dim=1).mean().item()
                
                sim_a_list.append(sim_a)
                sim_b_list.append(sim_b)
                start += win_samples
                
            if sim_a_list:
                mean_a = np.mean(sim_a_list)
                mean_b = np.mean(sim_b_list)
                
                # ya is ALWAYS the attended track, yb is ALWAYS the unattended track.
                # So if track_attended == 1, then sim_a corresponds to Track 1, sim_b to Track 2.
                # If track_attended == 2, then sim_a corresponds to Track 2, sim_b to Track 1.
                if str(track_attended) == '1':
                    sim_t1 = mean_a
                    sim_t2 = mean_b
                elif str(track_attended) == '2':
                    sim_t1 = mean_b
                    sim_t2 = mean_a
                else:
                    sim_t1 = mean_a
                    sim_t2 = mean_b
                    
                indicator = ">>> TRACK 1 HIGHER" if sim_t1 > sim_t2 else "<<< TRACK 2 HIGHER"
                print(f"{i+1:<6} | {exp:<5} | {track_attended:<7} | {sim_t1:<15.4f} | {sim_t2:<15.4f} | {indicator}")

if __name__ == "__main__":
    main()
