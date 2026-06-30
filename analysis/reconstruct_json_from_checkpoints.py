import sys
import json
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from data.kul_cached_dataset import KULCachedLoader
from models.aad_conformer import AADConformer
from training.train_conformer_loso import prepare_data, evaluate_trial_majority_vote, safe_corr_np

def main():
    print("Loading KUL Cache...")
    if Path("/kaggle/input/datasets/lowk1ee/kul-preprocessed-cache/data/processed_kul").exists():
        cache_dir = Path("/kaggle/input/datasets/lowk1ee/kul-preprocessed-cache/data/processed_kul")
    elif Path("/kaggle/input/kul-preprocessed-cache/data/processed_kul").exists():
        cache_dir = Path("/kaggle/input/kul-preprocessed-cache/data/processed_kul")
    else:
        cache_dir = REPO_ROOT / "data" / "processed_kul"
        
    if not cache_dir.exists():
        print(f"Error: KUL cache not found at {cache_dir}")
        return
        
    loader = KULCachedLoader(cache_dir)
    all_subject_data = loader.load_all()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    results_dir = REPO_ROOT / "results" / "conformer_loso"
    if not results_dir.exists():
        # Fallback for local testing if folder was renamed
        results_dir = REPO_ROOT / "conformer_loso_results"
        
    checkpoints_dir = results_dir / "checkpoints"
    
    if not checkpoints_dir.exists():
        print(f"Checkpoints directory not found at {checkpoints_dir}!")
        return
        
    seeds = [1, 7, 21, 42, 123]
    subjects = sorted(list(all_subject_data.keys()))
    
    all_seeds_results = {}
    
    for seed in seeds:
        seed_dir = checkpoints_dir / f"seed_{seed}"
        if not seed_dir.exists():
            print(f"Warning: Missing seed {seed}")
            continue
            
        print(f"\nReconstructing Seed {seed}...")
        loso_results = {}
        
        for subject in tqdm(subjects, desc=f"Seed {seed} Subjects"):
            ckpt_path = seed_dir / f"model_{subject}.pt"
            if not ckpt_path.exists():
                print(f"Missing checkpoint for {subject}")
                continue
                
            model = AADConformer(
                in_channels=8, temporal_filters=32, spatial_filters=64,
                embed_dim=64, num_heads=4, num_layers=2, dropout=0.3, stride=4
            ).to(device)
            
            try:
                model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
            except Exception as e:
                print(f"Error loading {ckpt_path}: {e}")
                continue
                
            model.eval()
            test_trials = all_subject_data[subject]
            
            total_test_samples = 0
            test_corr_att, test_corr_unatt = 0.0, 0.0
            mv_trial_correct, mv_windows_correct, mv_windows_total = 0, 0, 0
            trial_margins = []
            
            with torch.no_grad():
                for t in test_trials:
                    eeg = t["eeg"].unsqueeze(0).to(device)       
                    audio_a = t["audio_a"].unsqueeze(0).to(device) 
                    audio_b = t["audio_b"].unsqueeze(0).to(device)
                    
                    audio_a = audio_a.mean(dim=1, keepdim=True)
                    audio_b = audio_b.mean(dim=1, keepdim=True)
                    
                    eeg_mean = eeg.mean(dim=2, keepdim=True)
                    eeg_std = eeg.std(dim=2, keepdim=True) + 1e-8
                    eeg_norm = (eeg - eeg_mean) / eeg_std
                    
                    audio_a_mean = audio_a.mean(dim=2, keepdim=True)
                    audio_a_std = audio_a.std(dim=2, keepdim=True) + 1e-8
                    audio_a_norm = (audio_a - audio_a_mean) / audio_a_std
                    
                    audio_b_mean = audio_b.mean(dim=2, keepdim=True)
                    audio_b_std = audio_b.std(dim=2, keepdim=True) + 1e-8
                    audio_b_norm = (audio_b - audio_b_mean) / audio_b_std
                    
                    pred = model(eeg_norm)
                    
                    pred_np = pred.squeeze(0).cpu().numpy()
                    wav_a_np = audio_a_norm.squeeze(1).squeeze(0).cpu().numpy()
                    wav_b_np = audio_b_np = audio_b_norm.squeeze(1).squeeze(0).cpu().numpy()
                    
                    c_att = safe_corr_np(pred_np, wav_a_np)
                    c_unatt = safe_corr_np(pred_np, wav_b_np)
                    test_corr_att += c_att
                    test_corr_unatt += c_unatt
                    trial_margin = c_att - c_unatt
                    trial_margins.append(float(trial_margin))
                    
                    trial_ok, n_win, c_win = evaluate_trial_majority_vote(pred_np, wav_a_np, wav_b_np, window_seconds=10, hop_seconds=1.0, fs=64)
                    if trial_ok: mv_trial_correct += 1
                    mv_windows_total += n_win
                    mv_windows_correct += c_win
                    total_test_samples += 1
                    
            trial_acc = mv_trial_correct / total_test_samples if total_test_samples > 0 else 0
            win_acc = mv_windows_correct / mv_windows_total if mv_windows_total > 0 else 0
            
            loso_results[subject] = {
                "trial_accuracy": float(trial_acc),
                "window_accuracy": float(win_acc),
                "mean_pearson_att": float(test_corr_att / total_test_samples),
                "mean_pearson_unatt": float(test_corr_unatt / total_test_samples),
                "mean_margin": float(np.mean(trial_margins)),
                "median_margin": float(np.median(trial_margins)),
                "margin_std": float(np.std(trial_margins)),
                "fold_trial_margins": trial_margins,
                # Placeholders for history which we can't recover from just checkpoints
                "best_epoch": -1,
                "stopped_epoch": -1,
                "history": {}
            }
            
        all_seeds_results[str(seed)] = loso_results
        
    out_file = results_dir / "conformer_loso_multiseed_summary.json"
    with open(out_file, 'w') as f:
        json.dump(all_seeds_results, f, indent=4)
        
    print(f"\nSuccessfully reconstructed JSON and saved to {out_file}")

if __name__ == "__main__":
    main()
