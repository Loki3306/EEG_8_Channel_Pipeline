import os
import sys
import numpy as np
import pandas as pd
import torch
import warnings
from pathlib import Path
from sklearn.metrics import brier_score_loss, roc_auc_score, precision_recall_curve, auc, f1_score, balanced_accuracy_score
from scipy.stats import pearsonr, ttest_ind

import matplotlib.pyplot as plt
import seaborn as sns

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from data.kul_cached_dataset import KULCachedLoader
from models.aad_conformer import AADConformer
from analysis.interpretability.utils import safe_corr_np, normalize_eeg, normalize_audio

warnings.filterwarnings("ignore")

def compute_ece(correct, conf, bins=10):
    bin_boundaries = np.linspace(0, 1, bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0.0
    mce = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (conf >= bin_lower) & (conf < bin_upper)
        prop_in_bin = in_bin.mean()
        if prop_in_bin > 0:
            accuracy_in_bin = correct[in_bin].mean()
            avg_confidence_in_bin = conf[in_bin].mean()
            error = np.abs(avg_confidence_in_bin - accuracy_in_bin)
            ece += prop_in_bin * error
            mce = max(mce, error)
            
    return ece, mce

def get_predictions(model, eeg, wav_a, wav_b, win_samples, hop_samples):
    margins, correct_list, confidences, latent_norms = [], [], [], []
    ca_list, cb_list = [], []
    
    for start in range(0, eeg.shape[-1] - win_samples + 1, hop_samples):
        stop = start + win_samples
        eeg_win = eeg[:, :, start:stop]
        
        pred, z_pool = model(eeg_win, return_features=True)
        pred_np = pred.squeeze(0).cpu().numpy()
        
        wa = wav_a[:, :, start:stop].squeeze(1).squeeze(0).cpu().numpy()
        wb = wav_b[:, :, start:stop].squeeze(1).squeeze(0).cpu().numpy()
        
        ca = safe_corr_np(pred_np, wa)
        cb = safe_corr_np(pred_np, wb)
        margin = ca - cb
        
        ca_t = torch.tensor([ca], dtype=torch.float32, device=eeg.device)
        cb_t = torch.tensor([cb], dtype=torch.float32, device=eeg.device)
        margin_t = torch.tensor([margin], dtype=torch.float32, device=eeg.device)
        
        conf = model.predict_confidence(z_pool, ca_t, cb_t, margin_t).squeeze().item()
        z_norm = torch.norm(z_pool, dim=-1).squeeze().item()
        
        margins.append(margin)
        correct_list.append(margin > 0)
        confidences.append(conf)
        latent_norms.append(z_norm)
        ca_list.append(ca)
        cb_list.append(cb)
        
    return margins, correct_list, confidences, latent_norms, ca_list, cb_list

def corrupt_eeg(eeg, wav_a, wav_b, mode, device):
    if mode == "clean": return eeg, wav_a, wav_b
    elif mode == "random": return torch.randn_like(eeg), wav_a, wav_b
    elif mode == "zero": return torch.zeros_like(eeg), wav_a, wav_b
    elif mode == "gaussian": return eeg + torch.randn_like(eeg) * 0.5, wav_a, wav_b
    elif mode == "audio_permute": return eeg, wav_b, wav_a
    elif mode == "label_shuffle": return eeg, wav_a[:, :, torch.randperm(wav_a.shape[-1])], wav_b[:, :, torch.randperm(wav_b.shape[-1])]
    elif mode == "circular_shift": 
        shift = eeg.shape[-1] // 2
        return torch.roll(eeg, shifts=shift, dims=-1), wav_a, wav_b
    return eeg, wav_a, wav_b

import argparse

def main():
    parser = argparse.ArgumentParser(description="Phase 8: Scientific Falsification")
    parser.add_argument("--cache_dir", type=str, default=None, help="Path to KUL cache dir")
    parser.add_argument("--checkpoint_dir", type=str, default=None, help="Path to Phase 7 checkpoints (seed_1 dir)")
    args = parser.parse_args()

    print("--- Phase 8: Scientific Falsification ---")
    
    # 1. Resolve Cache Directory
    if args.cache_dir:
        cache_dir = Path(args.cache_dir)
    else:
        cache_dir = REPO_ROOT / "data" / "processed_kul"
        if Path("/kaggle/input/datasets/lowk1ee/kul-preprocessed-cache/data/processed_kul").exists():
            cache_dir = Path("/kaggle/input/datasets/lowk1ee/kul-preprocessed-cache/data/processed_kul")
            
    # 2. Resolve Checkpoint Directory
    if args.checkpoint_dir:
        checkpoint_dir = Path(args.checkpoint_dir)
    else:
        checkpoint_dir = REPO_ROOT / "results" / "run7_multitask_conformer_loso" / "checkpoints" / "seed_1"
        
    print(f"Using cache_dir: {cache_dir}")
    print(f"Using checkpoint_dir: {checkpoint_dir}")
    
    out_dir = REPO_ROOT / "results" / "run8_falsification"
    os.makedirs(out_dir, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    loader = KULCachedLoader(cache_dir)
    loader.load_all()
    subjects = list(loader.subjects_data.keys())
    
    win_samples, hop_samples = 640, 64 # 10s window, 1s hop
    modes = ["clean", "random", "zero", "gaussian", "audio_permute", "label_shuffle", "circular_shift"]
    all_results = {m: [] for m in modes}
    
    # 1. Generate Data
    for subj in subjects:
        print(f"Processing Subject {subj}...")
        ckpt_path = checkpoint_dir / f"model_{subj}.pt"
        if not ckpt_path.exists(): continue
            
        model = AADConformer(in_channels=8).to(device)
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        model.eval()
        
        test_trials = loader.subjects_data[subj]
        
        with torch.no_grad():
            for t in test_trials:
                eeg_base = t["eeg"].unsqueeze(0).to(device)
                wav_a_base = t["audio_a"].unsqueeze(0).to(device).mean(dim=1, keepdim=True)
                wav_b_base = t["audio_b"].unsqueeze(0).to(device).mean(dim=1, keepdim=True)
                
                for mode in modes:
                    eeg, wav_a, wav_b = corrupt_eeg(eeg_base, wav_a_base, wav_b_base, mode, device)
                    eeg, wav_a, wav_b = normalize_eeg(eeg), normalize_audio(wav_a), normalize_audio(wav_b)
                    min_len = min(eeg.shape[-1], wav_a.shape[-1], wav_b.shape[-1])
                    
                    m, c, conf, z_norm, ca, cb = get_predictions(model, eeg[:,:,:min_len], wav_a[:,:,:min_len], wav_b[:,:,:min_len], win_samples, hop_samples)
                    
                    for i in range(len(m)):
                        all_results[mode].append({
                            'subject': subj, 'margin': m[i], 'correct': c[i], 'confidence': conf[i],
                            'latent_norm': z_norm[i], 'ca': ca[i], 'cb': cb[i]
                        })

    # Save raw datasets
    df_clean = pd.DataFrame(all_results["clean"])
    df_clean.to_csv(out_dir / "clean_predictions.csv", index=False)
    
    print("\n[Stage 2 & 7] Confidence Robustness")
    robust_stats = []
    for mode in modes:
        df_mode = pd.DataFrame(all_results[mode])
        robust_stats.append({
            'Mode': mode,
            'Mean_Conf': df_mode['confidence'].mean(),
            'Median_Conf': df_mode['confidence'].median(),
            'Var_Conf': df_mode['confidence'].var()
        })
    rob_df = pd.DataFrame(robust_stats)
    print(rob_df.to_string(index=False))
    rob_df.to_csv(out_dir / "robustness_stats.csv", index=False)
    
    print("\n[Stage 3] Selective Prediction")
    thresholds = np.arange(0.50, 0.96, 0.05)
    sel_res = []
    for th in thresholds:
        accepted = df_clean[df_clean['confidence'] >= th]
        rejected = df_clean[df_clean['confidence'] < th]
        cov = len(accepted) / len(df_clean) if len(df_clean) > 0 else 0
        acc_acc = accepted['correct'].mean() if len(accepted) > 0 else np.nan
        rej_acc = rejected['correct'].mean() if len(rejected) > 0 else np.nan
        sel_res.append({'Threshold': th, 'Coverage': cov, 'Accepted_Acc': acc_acc, 'Rejected_Acc': rej_acc})
    sel_df = pd.DataFrame(sel_res)
    print(sel_df.to_string(index=False))
    sel_df.to_csv(out_dir / "selective_prediction.csv", index=False)
    
    print("\n[Stage 4 & 5 & 6] Statistical & Calibration Verification")
    correct = df_clean['correct'].values.astype(float)
    conf = df_clean['confidence'].values
    
    ece, mce = compute_ece(correct, conf)
    brier = brier_score_loss(correct, conf)
    auroc = roc_auc_score(correct, conf)
    precision, recall, _ = precision_recall_curve(correct, conf)
    auprc = auc(recall, precision)
    
    print(f"Global ECE: {ece:.4f} | MCE: {mce:.4f} | Brier: {brier:.4f}")
    print(f"AUROC: {auroc:.4f} | AUPRC: {auprc:.4f}")
    
    # Bootstrap CIs for AUROC
    n_bootstraps = 1000
    boot_aurocs = []
    rng = np.random.RandomState(42)
    for i in range(n_bootstraps):
        indices = rng.randint(0, len(conf), len(conf))
        if len(np.unique(correct[indices])) < 2: continue
        boot_aurocs.append(roc_auc_score(correct[indices], conf[indices]))
    
    ci_lower = np.percentile(boot_aurocs, 2.5)
    ci_upper = np.percentile(boot_aurocs, 97.5)
    print(f"AUROC 95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
    
    print("\n[Stage 9] Subject-wise Reliability")
    subj_res = []
    for subj in subjects:
        sdf = df_clean[df_clean['subject'] == subj]
        if len(sdf) == 0: continue
        s_corr = sdf['correct'].values.astype(float)
        s_conf = sdf['confidence'].values
        s_ece, _ = compute_ece(s_corr, s_conf)
        try: s_auroc = roc_auc_score(s_corr, s_conf)
        except: s_auroc = np.nan
        subj_res.append({'Subject': subj, 'ECE': s_ece, 'AUROC': s_auroc, 'Accuracy': s_corr.mean()})
    subj_df = pd.DataFrame(subj_res)
    subj_df.to_csv(out_dir / "subject_reliability.csv", index=False)
    
    print("\n[Stage 10] Correlation Analysis")
    corr_matrix = df_clean[['confidence', 'margin', 'latent_norm', 'ca', 'cb']].corr()
    print(corr_matrix)
    corr_matrix.to_csv(out_dir / "correlation_matrix.csv")
    
    print("\nDone. Results saved to", out_dir)

if __name__ == "__main__":
    main()
