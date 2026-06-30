import os
import sys
import numpy as np
import pandas as pd
import torch
import warnings
from pathlib import Path
from sklearn.metrics import brier_score_loss, roc_auc_score

import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from data.kul_cached_dataset import KULCachedLoader
from models.aad_conformer import AADConformer
from analysis.interpretability.utils import safe_corr_np, normalize_eeg, normalize_audio

warnings.filterwarnings("ignore")

def sliding_window_evaluation(model, eeg, wav_a, wav_b, win_samples, hop_samples):
    margins = []
    correct_list = []
    confidences = []
    
    for start in range(0, eeg.shape[-1] - win_samples + 1, hop_samples):
        stop = start + win_samples
        eeg_win = eeg[:, :, start:stop]
        
        # We query the model's learned confidence directly using late fusion
        pred, z_pool = model(eeg_win, return_features=True)
        pred = pred.squeeze(0).cpu().numpy()
        
        wa = wav_a[:, :, start:stop].squeeze(1).squeeze(0).cpu().numpy()
        wb = wav_b[:, :, start:stop].squeeze(1).squeeze(0).cpu().numpy()
        
        ca = safe_corr_np(pred, wa)
        cb = safe_corr_np(pred, wb)
        margin = ca - cb
        
        # Convert numpy scalars to tensors for the confidence head
        ca_t = torch.tensor([ca], dtype=torch.float32, device=eeg.device)
        cb_t = torch.tensor([cb], dtype=torch.float32, device=eeg.device)
        margin_t = torch.tensor([margin], dtype=torch.float32, device=eeg.device)
        
        conf = model.predict_confidence(z_pool, ca_t, cb_t, margin_t)
        conf = conf.squeeze().item()
        margins.append(margin)
        correct_list.append(margin > 0)
        confidences.append(conf)
        
    return margins, correct_list, confidences

def extract_subject_predictions(model, test_trials, device, win_samples, hop_samples, mode="clean"):
    all_margins = []
    all_correct = []
    all_conf = []
    
    model.eval()
    with torch.no_grad():
        for t in test_trials:
            eeg = t["eeg"].unsqueeze(0).to(device)
            wav_a = t["audio_a"].unsqueeze(0).to(device).mean(dim=1, keepdim=True)
            wav_b = t["audio_b"].unsqueeze(0).to(device).mean(dim=1, keepdim=True)
            
            # Apply corruptions for robustness evaluation
            if mode == "random":
                eeg = torch.randn_like(eeg)
            elif mode == "zero":
                eeg = torch.zeros_like(eeg)
            elif mode == "permuted":
                idx = torch.randperm(eeg.shape[-1], device=device)
                eeg = eeg[:, :, idx]
            
            eeg = normalize_eeg(eeg)
            wav_a = normalize_audio(wav_a)
            wav_b = normalize_audio(wav_b)
            
            min_len = min(eeg.shape[-1], wav_a.shape[-1], wav_b.shape[-1])
            eeg, wav_a, wav_b = eeg[:,:,:min_len], wav_a[:,:,:min_len], wav_b[:,:,:min_len]
            
            m, c, conf = sliding_window_evaluation(model, eeg, wav_a, wav_b, win_samples, hop_samples)
            all_margins.extend(m)
            all_correct.extend(c)
            all_conf.extend(conf)
            
    return all_margins, all_correct, all_conf

def compute_calibration(df, out_dir):
    correct = df['correct'].values.astype(float)
    conf = df['confidence'].values
    
    brier = brier_score_loss(correct, conf)
    
    bins = np.linspace(0.0, 1.0, 11)
    df['conf_bin'] = pd.cut(df['confidence'], bins=bins, include_lowest=True)
    
    calib_stats = df.groupby('conf_bin', observed=True).agg(
        count=('correct', 'size'),
        mean_conf=('confidence', 'mean'),
        accuracy=('correct', 'mean')
    ).reset_index()
    
    ece = 0.0
    total_samples = len(df)
    
    for _, row in calib_stats.iterrows():
        count = row['count']
        if count == 0: continue
        mean_conf = row['mean_conf']
        acc = row['accuracy']
        ece += (count / total_samples) * np.abs(acc - mean_conf)
        
    # Plot Reliability Diagram
    plt.figure(figsize=(8, 8))
    plot_stats = calib_stats[calib_stats['count'] > 0]
    
    plt.plot([0.0, 1.0], [0.0, 1.0], 'k--', label="Perfect Calibration")
    plt.plot(plot_stats['mean_conf'], plot_stats['accuracy'], marker='o', linewidth=2, color='blue', label="AAD-Conformer Learned Confidence")
    plt.bar(plot_stats['mean_conf'], plot_stats['count'] / total_samples, 
            width=0.08, alpha=0.3, color='gray', label="% of Samples")
            
    plt.title(f"Phase 7 Reliability Diagram (Global ECE = {ece:.4f})")
    plt.xlabel("Mean Predicted Confidence")
    plt.ylabel("Observed Accuracy / Sample Density")
    plt.legend()
    plt.grid(True)
    plt.savefig(out_dir / "fig_reliability_diagram.png", dpi=300)
    plt.close()
    
    return brier, ece, calib_stats

def run_selective_prediction(df, out_dir):
    # For learned confidence, the network output is a probability. We threshold from 0.5 to 0.95.
    thresholds = np.arange(0.50, 0.96, 0.05)
    total_predictions = len(df)
    results = []
    
    for th in thresholds:
        accepted = df[df['confidence'] >= th]
        rejected = df[df['confidence'] < th]
        
        cov = len(accepted) / total_predictions if total_predictions > 0 else 0
        acc = accepted['correct'].mean() if len(accepted) > 0 else np.nan
        rej_rate = len(rejected) / total_predictions if total_predictions > 0 else 0
        
        results.append({
            'Threshold': th,
            'Coverage': cov,
            'Accepted_Accuracy': acc,
            'Rejected_Rate': rej_rate,
            'Accepted_Count': len(accepted),
            'Mean_Margin': accepted['margin'].mean() if len(accepted) > 0 else 0
        })
        
    res_df = pd.DataFrame(results)
    res_df.to_csv(out_dir / "selective_prediction_thresholds.csv", index=False)
    
    plt.figure(figsize=(8, 6))
    plt.plot(res_df['Coverage'] * 100, res_df['Accepted_Accuracy'] * 100, marker='o', linewidth=2, color='blue')
    plt.gca().invert_xaxis()
    plt.title("Phase 7: Coverage vs Accepted Accuracy (Learned Confidence)")
    plt.xlabel("Coverage (%)")
    plt.ylabel("Accuracy (%)")
    plt.grid(True)
    plt.savefig(out_dir / "fig_coverage_vs_accuracy.png", dpi=300)
    plt.close()
    
    return res_df

def main():
    print("--- Phase 7: Production-Grade Confidence-Aware Selective AAD (Learned Confidence) ---")
    
    cache_dir = REPO_ROOT / "data" / "processed_kul"
    if Path("/kaggle/input/datasets/lowk1ee/kul-preprocessed-cache/data/processed_kul").exists():
        cache_dir = Path("/kaggle/input/datasets/lowk1ee/kul-preprocessed-cache/data/processed_kul")
    
    # In Phase 7, we evaluate the multitask checkpoints
    checkpoint_dir = REPO_ROOT / "results" / "run7_multitask_conformer_loso" / "checkpoints" / "seed_1"
    
    out_dir = REPO_ROOT / "results" / "run7_multitask_evaluation"
    os.makedirs(out_dir, exist_ok=True)
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    if not checkpoint_dir.exists():
        print(f"[Error] Multi-task checkpoints not found at {checkpoint_dir}.")
        print("Please run `training/train_conformer_multitask_loso.py` first.")
        return
        
    loader = KULCachedLoader(cache_dir)
    loader.load_all()
    
    subjects_to_test = list(loader.subjects_data.keys())
    print(f"Evaluating {len(subjects_to_test)} subjects...")
    
    fs = 64
    win_samples = 10 * fs
    hop_samples = fs
    
    all_data = []
    all_random = []
    all_zero = []
    
    for subj in subjects_to_test:
        print(f"Processing Subject {subj}...")
        ckpt_path = checkpoint_dir / f"model_{subj}.pt"
        if not ckpt_path.exists():
            print(f"  [Error] Checkpoint not found: {ckpt_path}. Skipping.")
            continue
            
        model = AADConformer(in_channels=8).to(device)
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        model.eval()
        
        test_trials = loader.subjects_data[subj]
        
        m_cln, c_cln, conf_cln = extract_subject_predictions(model, test_trials, device, win_samples, hop_samples, mode="clean")
        m_rnd, c_rnd, conf_rnd = extract_subject_predictions(model, test_trials, device, win_samples, hop_samples, mode="random")
        m_zro, c_zro, conf_zro = extract_subject_predictions(model, test_trials, device, win_samples, hop_samples, mode="zero")
        
        for m, c, cf in zip(m_cln, c_cln, conf_cln):
            all_data.append({'subject': subj, 'margin': m, 'correct': c, 'confidence': cf, 'mode': 'clean'})
        for m, c, cf in zip(m_rnd, c_rnd, conf_rnd):
            all_random.append({'subject': subj, 'margin': m, 'correct': c, 'confidence': cf, 'mode': 'random'})
        for m, c, cf in zip(m_zro, c_zro, conf_zro):
            all_zero.append({'subject': subj, 'margin': m, 'correct': c, 'confidence': cf, 'mode': 'zero'})
            
    df = pd.DataFrame(all_data)
    df_rnd = pd.DataFrame(all_random)
    df_zro = pd.DataFrame(all_zero)
    
    print("\n[Confidence Distribution - Clean Data]")
    print(df['confidence'].describe())
    
    # 1. Calibration
    print("\n[Running Calibration]")
    brier, ece, calib_stats = compute_calibration(df, out_dir)
    print(f"Global ECE: {ece:.4f} | Brier Score: {brier:.4f}")
    
    try:
        auroc = roc_auc_score(df['correct'], df['confidence'])
        print(f"Confidence AUROC (Discrimination Power): {auroc:.4f}")
    except ValueError:
        print("Confidence AUROC: Error (only one class present?)")
    
    # 2. Selective Prediction
    print("\n[Running Selective Prediction Thresholds]")
    res_df = run_selective_prediction(df, out_dir)
    print(res_df.to_string(index=False))
    
    # 3. Robustness Checks (Crucial Phase 7 Test)
    print("\n[Scientific Validation: Corrupted Input Robustness]")
    print("Does the learned confidence safely collapse under noise?")
    print(f"Mean Confidence (Clean):  {df['confidence'].mean():.4f}")
    print(f"Mean Confidence (Random): {df_rnd['confidence'].mean():.4f}")
    print(f"Mean Confidence (Zero):   {df_zro['confidence'].mean():.4f}")
    
    if df_rnd['confidence'].mean() < df['confidence'].mean() and df_zro['confidence'].mean() < df['confidence'].mean():
        print("=> SUCCESS: Learned Confidence properly collapses on corrupted inputs.")
    else:
        print("=> FAILURE: Learned Confidence is equal to or higher than Clean inputs.")
    
    print(f"\nResults saved to {out_dir}")

if __name__ == "__main__":
    main()
