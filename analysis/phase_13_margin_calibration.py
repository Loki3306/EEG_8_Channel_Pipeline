import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from scipy.optimize import minimize
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

# Setup Paths
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(REPO_ROOT))

from models.aad_conformer import AADConformer
from data.kul_cached_dataset import KULCachedLoader

def safe_corr_torch(x, y, eps=1e-8):
    x_mean = x.mean(dim=-1, keepdim=True)
    y_mean = y.mean(dim=-1, keepdim=True)
    x_centered = x - x_mean
    y_centered = y - y_mean
    cov = (x_centered * y_centered).sum(dim=-1)
    x_var = (x_centered ** 2).sum(dim=-1)
    y_var = (y_centered ** 2).sum(dim=-1)
    corr = cov / (torch.sqrt(x_var * y_var) + eps)
    return corr

def extract_margins(model, subject_data, device, window_sec=2, hop_sec=2, fs=64):
    win_samples = int(window_sec * fs)
    hop_samples = int(hop_sec * fs)
    
    margins = []
    
    with torch.no_grad():
        for trial_idx, t in enumerate(subject_data):
            eeg = t["eeg"]
            audio_a = t["audio_a"].mean(dim=0, keepdim=True)
            audio_b = t["audio_b"].mean(dim=0, keepdim=True)
            
            n_windows = (eeg.shape[1] - win_samples) // hop_samples + 1
            for w in range(max(1, n_windows)):
                start = w * hop_samples
                stop = start + win_samples
                if stop > eeg.shape[1]:
                    break
                    
                e = eeg[:, start:stop].unsqueeze(0).to(device)
                a = audio_a[:, start:stop].unsqueeze(0).to(device)
                b = audio_b[:, start:stop].unsqueeze(0).to(device)
                
                a = (a - a.mean(dim=-1, keepdim=True)) / (a.std(dim=-1, keepdim=True) + 1e-8)
                b = (b - b.mean(dim=-1, keepdim=True)) / (b.std(dim=-1, keepdim=True) + 1e-8)
                
                pred = model(e, return_features=False)
                
                ca = safe_corr_torch(pred, a).item()
                cb = safe_corr_torch(pred, b).item()
                
                # Randomly assign A and B to stream 1 and 2
                if np.random.rand() > 0.5:
                    c1, c2 = ca, cb
                    is_stream1_attended = 1
                else:
                    c1, c2 = cb, ca
                    is_stream1_attended = 0
                    
                margin = c1 - c2
                prediction = 1 if margin > 0 else 0
                correct = 1 if (prediction == is_stream1_attended) else 0
                
                margins.append({
                    "trial": trial_idx,
                    "window": w,
                    "corrA": c1,
                    "corrB": c2,
                    "margin": margin,
                    "ground_truth": is_stream1_attended,
                    "prediction": prediction,
                    "correct": correct
                })
    return pd.DataFrame(margins)

def temperature_scaling_nll(T, margins, labels):
    probs = 1 / (1 + np.exp(-margins / T))
    probs = np.clip(probs, 1e-7, 1 - 1e-7)
    return log_loss(labels, probs)

def fit_temperature_scaling(margins, labels):
    res = minimize(temperature_scaling_nll, x0=[1.0], args=(margins, labels), bounds=[(0.01, 100.0)])
    return res.x[0]

def expected_calibration_error(y_true, y_prob, n_bins=10):
    bins = np.linspace(0., 1., n_bins + 1)
    binids = np.digitize(y_prob, bins) - 1
    
    ece = 0.0
    for i in range(n_bins):
        bin_idx = (binids == i)
        if np.sum(bin_idx) > 0:
            bin_acc = np.mean(y_true[bin_idx])
            bin_conf = np.mean(y_prob[bin_idx])
            ece += np.abs(bin_acc - bin_conf) * np.sum(bin_idx)
    return ece / len(y_true)

def main():
    print("--- Phase 13: Margin Calibration (KUL Only) ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    cache_dir = REPO_ROOT / "data" / "processed_kul"
    if not cache_dir.exists():
        cache_dir = Path("/kaggle/input/datasets/lowk1ee/kul-preprocessed-cache/data/processed_kul")
    if not cache_dir.exists():
        cache_dir = Path("/kaggle/input/datasets/lokeshgile/kul-processed/data/processed_kul")
        
    loader = KULCachedLoader(cache_dir)
    all_subject_data = loader.load_all()
    subjects = sorted(list(all_subject_data.keys()))
    
    out_dir = REPO_ROOT / "results" / "phase13_margin_calibration"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    ckpt_dir = REPO_ROOT / "results" / "run1_baseline_conformer_loso" / "checkpoints" / "seed_1"
    if not ckpt_dir.exists():
        ckpt_dir = Path("/kaggle/input/datasets/lowkieee/eeg-aad-conformer-seed1-checkpoints/checkpoints/seed_1")
    if not ckpt_dir.exists():
        ckpt_dir = Path("/kaggle/input/datasets/lokeshgile/eeg-aad-conformer-seed1-checkpoints/checkpoints/seed_1")
        
    all_test_predictions = []
    
    for test_subject in subjects:
        print(f"\n{'-'*50}")
        print(f"Subject: {test_subject}")
        
        remaining_subjects = [s for s in subjects if s != test_subject]
        test_idx = subjects.index(test_subject)
        val_subject = remaining_subjects[test_idx % len(remaining_subjects)]
        train_subjects = [s for s in remaining_subjects if s != val_subject]
        
        ckpt_path = ckpt_dir / f"model_{test_subject}.pt"
        if not ckpt_path.exists():
            print(f"Skipping {test_subject}, checkpoint not found.")
            continue
            
        model = AADConformer(in_channels=8).to(device)
        model.load_state_dict(torch.load(ckpt_path, map_location=device), strict=False)
        model.eval()
        
        val_df = extract_margins(model, all_subject_data[val_subject], device)
        test_df = extract_margins(model, all_subject_data[test_subject], device)
        
        # Use 2 train subjects to simulate global calibration quickly
        train_dfs = []
        for s in train_subjects[:2]:
             train_dfs.append(extract_margins(model, all_subject_data[s], device))
        global_val_df = pd.concat([val_df] + train_dfs, ignore_index=True)
        
        calibration_scopes = {"per_subject": val_df, "global": global_val_df}
        margin_ablations = ["raw_margin", "absolute_margin", "normalized_abs_margin"]
        
        best_df = None
        
        for scope_name, scope_df in calibration_scopes.items():
            print(f"\n  Scope: {scope_name.upper()}")
            for ablation in margin_ablations:
                if ablation == "raw_margin":
                    X_cal = scope_df['margin'].values
                    X_test = test_df['margin'].values
                elif ablation == "absolute_margin":
                    X_cal = np.abs(scope_df['margin'].values)
                    X_test = np.abs(test_df['margin'].values)
                elif ablation == "normalized_abs_margin":
                    cal_abs = np.abs(scope_df['margin'].values)
                    test_abs = np.abs(test_df['margin'].values)
                    mu, sigma = cal_abs.mean(), cal_abs.std() + 1e-8
                    X_cal = (cal_abs - mu) / sigma
                    X_test = (test_abs - mu) / sigma
                    
                y_cal = scope_df['correct'].values
                y_test = test_df['correct'].values
                
                prob_raw = 1 / (1 + np.exp(-X_test))
                
                platt = LogisticRegression()
                platt.fit(X_cal.reshape(-1, 1), y_cal)
                prob_platt = platt.predict_proba(X_test.reshape(-1, 1))[:, 1]
                
                T = fit_temperature_scaling(X_cal, y_cal)
                prob_temp = 1 / (1 + np.exp(-X_test / T))
                
                iso = IsotonicRegression(out_of_bounds='clip')
                iso.fit(X_cal, y_cal)
                prob_iso = iso.predict(X_test)
                
                def get_metrics(probs):
                    ece = expected_calibration_error(y_test, probs)
                    brier = brier_score_loss(y_test, probs)
                    try:
                        auroc = roc_auc_score(y_test, probs)
                    except ValueError:
                        auroc = 0.5
                    return ece, brier, auroc
                    
                raw_ece, raw_brier, raw_auc = get_metrics(prob_raw)
                platt_ece, platt_brier, platt_auc = get_metrics(prob_platt)
                temp_ece, temp_brier, temp_auc = get_metrics(prob_temp)
                iso_ece, iso_brier, iso_auc = get_metrics(prob_iso)
                
                print(f"    [{ablation}] Platt a={platt.coef_[0][0]:.4f}, b={platt.intercept_[0]:.4f} | Temp T={T:.4f}")
                print(f"    [{ablation}] ECE  (Raw: {raw_ece:.4f} -> Platt: {platt_ece:.4f}, Temp: {temp_ece:.4f}, Iso: {iso_ece:.4f})")
                print(f"    [{ablation}] Brier(Raw: {raw_brier:.4f} -> Platt: {platt_brier:.4f}, Temp: {temp_brier:.4f}, Iso: {iso_brier:.4f})")
                
                if scope_name == "per_subject" and ablation == "absolute_margin":
                    res_df = test_df.copy()
                    res_df['subject'] = test_subject
                    res_df['ablation'] = f"{scope_name}_{ablation}"
                    res_df['prob_raw'] = prob_raw
                    res_df['prob_platt'] = prob_platt
                    res_df['prob_temp'] = prob_temp
                    res_df['prob_iso'] = prob_iso
                    best_df = res_df
        
        all_test_predictions.append(best_df)
        
        print(f"\n  Debug Sample (Absolute Margin):")
        sample = best_df.sample(5)
        for _, row in sample.iterrows():
            print(f"  Margin: {row['margin']:.4f} | Raw: {row['prob_raw']:.4f} | Platt: {row['prob_platt']:.4f} | Temp: {row['prob_temp']:.4f} | Iso: {row['prob_iso']:.4f} | Pred: {row['prediction']} | Truth: {row['ground_truth']} | Correct: {row['correct']}")
            
        print(f"{'-'*50}")

    if all_test_predictions:
        final_df = pd.concat(all_test_predictions, ignore_index=True)
        final_df.to_csv(out_dir / "calibration_predictions.csv", index=False)
        print(f"Saved predictions to {out_dir / 'calibration_predictions.csv'}")

if __name__ == "__main__":
    main()
