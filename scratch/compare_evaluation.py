import sys
from pathlib import Path
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from baselines.ridge_aad import subject_files, load_subject_examples
from training.train_matchnet_loso import prepare_dataset, get_mapping_data, pearson_corr, ContrastiveMatchNet, evaluate_model, FS
from training.export_matchnet_predictions import chunk_trial_with_metadata

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running comparison on device: {device}")
    
    mapping, envelopes = get_mapping_data()
    all_paths = subject_files()
    
    # Pick S1
    s1_path = next(p for p in all_paths if "S1_" in p.name)
    subject_id = s1_path.stem.replace("_data_preproc", "")
    
    checkpoint_path = Path("checkpoints") / f"matchnet_fold_S1_data_preproc_best.pth"
    if not checkpoint_path.exists():
        print(f"WARNING: Checkpoint {checkpoint_path} not found. Ensure it is downloaded or trained.")
        return
        
    model = ContrastiveMatchNet(eeg_model_type="eegnet", eeg_channels=8, audio_channels=28, latent_dim=64).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    
    examples = load_subject_examples(s1_path)
    
    # 1. Run original prepare_dataset
    tX, tYA, tYB = prepare_dataset(examples, [13, 46, 43, 23, 50, 0, 52, 14], 1.0, 6.0, subject_id, mapping, envelopes)
    
    print("\n--- Running Original evaluate_model ---")
    window_sec = 10.0
    nc_norm, nt_norm = evaluate_model(model, tX, tYA, tYB, device, window_sec=window_sec, zero_eeg=False, shuffle_labels=False, metric="pearson")
    acc_orig = (nc_norm / nt_norm) * 100 if nt_norm > 0 else 0
    print(f"Original n_correct: {nc_norm}, n_total: {nt_norm}, accuracy: {acc_orig:.2f}%")
    
    # 2. Run Export script logic exactly
    print("\n--- Running Export Script Logic ---")
    n_correct_exp = 0.0
    n_total_exp = 0
    
    window_samples = int(window_sec * FS)
    
    divergence_point = None
    comparisons = []
    
    with torch.no_grad():
        for i in range(len(tX)):
            x_np, ya_np, yb_np = tX[i], tYA[i], tYB[i]
            ex = examples[i]
            
            chunks = chunk_trial_with_metadata(x_np, ya_np, yb_np, subject_id, i, ex.label, window_sec, window_sec, fs=FS)
            
            # Original evaluate_model does exactly this for chunking:
            # start = 0; while start + window_samples <= x_np.shape[1]
            # Let's compare window by window
            
            start = 0
            chunk_idx = 0
            while start + window_samples <= x_np.shape[1]:
                end = start + window_samples
                
                # Original extraction
                x_chunk_orig = torch.FloatTensor(x_np[:, start:end]).unsqueeze(0).to(device)
                ya_chunk_orig = torch.FloatTensor(ya_np[:, start:end]).unsqueeze(0).to(device)
                yb_chunk_orig = torch.FloatTensor(yb_np[:, start:end]).unsqueeze(0).to(device)
                
                z_eeg_orig, z_a_orig, z_b_orig = model(x_chunk_orig, ya_chunk_orig, yb_chunk_orig)
                sim_a_orig = pearson_corr(z_eeg_orig, z_a_orig, dim=1).mean().item()
                sim_b_orig = pearson_corr(z_eeg_orig, z_b_orig, dim=1).mean().item()
                
                pred_orig = 'A' if sim_a_orig > sim_b_orig else 'B'
                
                # Export extraction
                if chunk_idx < len(chunks):
                    chunk = chunks[chunk_idx]
                    x_t_exp = torch.FloatTensor(chunk['x']).unsqueeze(0).to(device)
                    ya_t_exp = torch.FloatTensor(chunk['ya']).unsqueeze(0).to(device)
                    yb_t_exp = torch.FloatTensor(chunk['yb']).unsqueeze(0).to(device)
                    
                    z_eeg_exp, z_a_exp, z_b_exp = model(x_t_exp, ya_t_exp, yb_t_exp)
                    sim_a_exp = pearson_corr(z_eeg_exp, z_a_exp, dim=1).mean().item()
                    sim_b_exp = pearson_corr(z_eeg_exp, z_b_exp, dim=1).mean().item()
                    
                    pred_exp = 'A' if sim_a_exp > sim_b_exp else 'B'
                    
                    if pred_exp == 'A':
                        n_correct_exp += 1.0
                    elif sim_a_exp == sim_b_exp:
                        n_correct_exp += 0.5
                    n_total_exp += 1
                else:
                    sim_a_exp, sim_b_exp, pred_exp = None, None, None
                    
                if len(comparisons) < 20:
                    comparisons.append({
                        'subject': subject_id,
                        'trial': i,
                        'window': chunk_idx,
                        'sim_A_orig': sim_a_orig,
                        'sim_B_orig': sim_b_orig,
                        'sim_A_exp': sim_a_exp,
                        'sim_B_exp': sim_b_exp,
                        'pred_orig': pred_orig,
                        'pred_exp': pred_exp
                    })
                    
                if divergence_point is None and (pred_orig != pred_exp or abs(sim_a_orig - sim_a_exp) > 1e-4):
                    divergence_point = comparisons[-1]
                    
                start += window_samples
                chunk_idx += 1

    acc_exp = (n_correct_exp / n_total_exp) * 100 if n_total_exp > 0 else 0
    print(f"Export n_correct: {n_correct_exp}, n_total: {n_total_exp}, accuracy: {acc_exp:.2f}%")
    
    print("\n--- First 20 Windows Comparison ---")
    print(f"{'Subj':<5} {'Trial':<5} {'Win':<4} | {'SimA (Orig)':<12} {'SimB (Orig)':<12} | {'SimA (Exp)':<12} {'SimB (Exp)':<12} | {'Pred (Orig)':<10} {'Pred (Exp)':<10}")
    print("-" * 110)
    for c in comparisons:
        print(f"{c['subject']:<5} {c['trial']:<5} {c['window']:<4} | {c['sim_A_orig']:<12.4f} {c['sim_B_orig']:<12.4f} | {c['sim_A_exp']:<12.4f} {c['sim_B_exp']:<12.4f} | {c['pred_orig']:<10} {c['pred_exp']:<10}")
        
    if divergence_point:
        print("\n--- DIVERGENCE FOUND ---")
        print(f"Divergence at Trial {divergence_point['trial']}, Window {divergence_point['window']}")
        print(f"Orig: SimA={divergence_point['sim_A_orig']:.6f}, SimB={divergence_point['sim_B_orig']:.6f}")
        print(f"Exp : SimA={divergence_point['sim_A_exp']:.6f}, SimB={divergence_point['sim_B_exp']:.6f}")
    else:
        print("\n--- NO DIVERGENCE FOUND ---")
        print("Both methods produced perfectly identical predictions and similarities.")

if __name__ == "__main__":
    main()
