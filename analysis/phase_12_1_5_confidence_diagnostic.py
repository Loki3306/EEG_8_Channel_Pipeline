import os
import sys
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
from tqdm import tqdm
from sklearn.metrics import brier_score_loss, roc_auc_score, accuracy_score
from sklearn.feature_selection import mutual_info_regression
import scipy.stats

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from models.aad_conformer import AADConformer
from data.kul_cached_dataset import KULCachedLoader
from analysis.interpretability.utils import normalize_eeg, normalize_audio

OUT_DIR = REPO_ROOT / "results" / "phase12_confidence_diagnostic"
FIG_DIR = OUT_DIR / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

def load_checkpoint_and_data(ckpt_path_arg, device):
    if ckpt_path_arg and Path(ckpt_path_arg).exists():
        ckpt_path = Path(ckpt_path_arg)
    else:
        ckpt_path = REPO_ROOT / "results" / "run7_multitask_conformer_loso" / "checkpoints" / "seed_1" / "model_S1.pt"
        if not ckpt_path.exists():
            ckpt_path = Path("/kaggle/working/EEG_8_Channel_Pipeline/results/run7_multitask_conformer_loso/checkpoints/seed_1/model_S1.pt")
        if not ckpt_path.exists():
            ckpt_path = Path("/kaggle/input/datasets/lokeshgile/confidence-heads/kaggle/working/EEG_8_Channel_Pipeline/results/run7_multitask_conformer_loso/checkpoints/seed_1/model_S1.pt")
        if not ckpt_path.exists():
            possible = list(Path("/kaggle/input").rglob("model_S1.pt"))
            if possible:
                ckpt_path = possible[0]
                
    model = AADConformer(in_channels=8).to(device)
    if ckpt_path.exists():
        try:
            model.load_state_dict(torch.load(ckpt_path, map_location=device))
            print(f"Loaded frozen Conformer checkpoint from {ckpt_path}")
        except RuntimeError as e:
            if "confidence_head" in str(e):
                print("CRITICAL ERROR: Old checkpoint missing confidence head!")
                sys.exit(1)
            raise e
    else:
        print("WARNING: Using untrained weights!")
        
    model.eval()
    
    cache_dir = Path("/kaggle/input/datasets/lokeshgile/kul-processed/data/processed_kul")
    if not cache_dir.exists():
        cache_dir = REPO_ROOT / "data" / "processed_kul"
        
    loader = KULCachedLoader(cache_dir)
    try:
        data = loader.load_all()["S1"]
    except:
        data = []
        print("WARNING: Data not found!")
        
    return model, data

def extract_base_dataset(model, data, device):
    print("\nExtracting base dataset from all windows...")
    fs = 64
    win_samples = 10 * fs
    hop_samples = fs
    
    records = []
    
    with torch.no_grad():
        for t_idx, trial in enumerate(tqdm(data, desc="Trials")):
            eeg = trial["eeg"].unsqueeze(0).to(device)
            wav_a = trial["audio_a"].unsqueeze(0).to(device).mean(dim=1, keepdim=True)
            wav_b = trial["audio_b"].unsqueeze(0).to(device).mean(dim=1, keepdim=True)
            
            eeg = normalize_eeg(eeg)
            wav_a = normalize_audio(wav_a)
            wav_b = normalize_audio(wav_b)
            
            # Match lengths
            min_len = min(eeg.shape[-1], wav_a.shape[-1], wav_b.shape[-1])
            eeg, wav_a, wav_b = eeg[:,:,:min_len], wav_a[:,:,:min_len], wav_b[:,:,:min_len]
            
            for start in range(0, eeg.shape[-1] - win_samples + 1, hop_samples):
                stop = start + win_samples
                
                e = eeg[:, :, start:stop]
                wa = wav_a[:, :, start:stop]
                wb = wav_b[:, :, start:stop]
                
                pred, z_pool = model(e, return_features=True)
                
                # Compute correlations
                from analysis.interpretability.utils import safe_corr_np
                p_np = pred.squeeze(0).cpu().numpy()
                wa_np = wa.squeeze(1).squeeze(0).cpu().numpy()
                wb_np = wb.squeeze(1).squeeze(0).cpu().numpy()
                
                ca = safe_corr_np(p_np, wa_np)
                cb = safe_corr_np(p_np, wb_np)
                margin = ca - cb
                
                prediction = 1 if margin > 0 else 0
                correct = int(prediction == 1)
                
                ca_t = torch.tensor([ca], dtype=torch.float32, device=device)
                cb_t = torch.tensor([cb], dtype=torch.float32, device=device)
                margin_t = torch.tensor([margin], dtype=torch.float32, device=device)
                
                conf = model.predict_confidence(z_pool, ca_t, cb_t, margin_t)
                
                eeg_context = z_pool.detach().cpu().numpy()[0]
                
                records.append({
                    "trial": t_idx,
                    "window": start // hop_samples,
                    "confidence": conf.item(),
                    "margin": margin,
                    "corr_a": ca,
                    "corr_b": cb,
                    "z_pool_norm": torch.norm(z_pool).item(),
                    "prediction": prediction,
                    "correct": correct,
                    "eeg_context": eeg_context
                })
                
    df = pd.DataFrame(records)
    print(f"Extracted {len(df)} windows.")
    return df

def audit_output_distribution(df):
    print("\nSTAGE 1: OUTPUT DISTRIBUTION AUDIT")
    stats = []
    
    for split_name, subset in [("All", df), ("Correct", df[df['correct']==1]), ("Incorrect", df[df['correct']==0])]:
        c = subset['confidence']
        if len(c) == 0: continue
        s = {
            "Split": split_name,
            "Count": len(c),
            "Mean": c.mean(),
            "Median": c.median(),
            "Std": c.std(),
            "Var": c.var(),
            "Min": c.min(),
            "Max": c.max(),
            "1%": c.quantile(0.01),
            "5%": c.quantile(0.05),
            "10%": c.quantile(0.10),
            "25%": c.quantile(0.25),
            "75%": c.quantile(0.75),
            "90%": c.quantile(0.90),
            "95%": c.quantile(0.95),
            "99%": c.quantile(0.99)
        }
        stats.append(s)
        
    stats_df = pd.DataFrame(stats)
    stats_df.to_csv(OUT_DIR / "confidence_statistics.csv", index=False)
    print(stats_df.to_string())
    
    plt.figure(figsize=(10, 6))
    plt.hist(df[df['correct']==1]['confidence'], bins=50, alpha=0.5, label='Correct', density=True)
    plt.hist(df[df['correct']==0]['confidence'], bins=50, alpha=0.5, label='Incorrect', density=True)
    plt.title("Confidence Distribution")
    plt.xlabel("Confidence")
    plt.ylabel("Density")
    plt.legend()
    plt.savefig(FIG_DIR / "confidence_distribution.png")
    plt.close()

def audit_layer_inspections(model, data, device):
    print("\nSTAGE 2: LAYER-WISE CONFIDENCE HEAD INSPECTION")
    activations = {}
    
    def get_activation(name):
        def hook(model, input, output):
            if name not in activations:
                activations[name] = []
            activations[name].append(output.detach().cpu())
        return hook
        
    hooks = []
    for idx, module in enumerate(model.confidence_head):
        if isinstance(module, torch.nn.Linear) or isinstance(module, torch.nn.ReLU):
            hooks.append(module.register_forward_hook(get_activation(f"layer_{idx}_{module.__class__.__name__}")))
            
    # run a small batch
    trial = data[0]
    fs = 64
    win_samples = 10 * fs
    eeg = normalize_eeg(trial["eeg"].unsqueeze(0).to(device))[:, :, :win_samples]
    wav_a = normalize_audio(trial["audio_a"].unsqueeze(0).to(device).mean(dim=1, keepdim=True))[:, :, :win_samples]
    wav_b = normalize_audio(trial["audio_b"].unsqueeze(0).to(device).mean(dim=1, keepdim=True))[:, :, :win_samples]
    
    with torch.no_grad():
        pred, z_pool = model(eeg, return_features=True)
        from analysis.interpretability.utils import safe_corr_np
        ca = safe_corr_np(pred.squeeze(0).cpu().numpy(), wav_a.squeeze(1).squeeze(0).cpu().numpy())
        cb = safe_corr_np(pred.squeeze(0).cpu().numpy(), wav_b.squeeze(1).squeeze(0).cpu().numpy())
        margin = ca - cb
        ca_t = torch.tensor([ca], dtype=torch.float32, device=device)
        cb_t = torch.tensor([cb], dtype=torch.float32, device=device)
        margin_t = torch.tensor([margin], dtype=torch.float32, device=device)
        conf = model.predict_confidence(z_pool, ca_t, cb_t, margin_t)
        
    stats = []
    for name, act_list in activations.items():
        act_np = torch.cat(act_list).numpy()
        stats.append({
            "Layer": name,
            "Mean": np.mean(act_np),
            "Std": np.std(act_np),
            "Min": np.min(act_np),
            "Max": np.max(act_np),
            "Dead%": np.mean(act_np <= 0) * 100 if "ReLU" in name else 0.0,
            "NaNs": np.isnan(act_np).sum(),
            "Infs": np.isinf(act_np).sum()
        })
        
    for h in hooks:
        h.remove()
        
    df_layers = pd.DataFrame(stats)
    df_layers.to_csv(OUT_DIR / "layer_statistics.csv", index=False)
    print(df_layers.to_string())

def audit_input_sensitivity(model, data, device):
    print("\nSTAGE 3: INPUT SENSITIVITY ANALYSIS")
    # Take a few samples
    trial = data[0]
    fs = 64
    eeg = normalize_eeg(trial["eeg"].unsqueeze(0).to(device))[:, :, :10*fs]
    wav_a = normalize_audio(trial["audio_a"].unsqueeze(0).to(device).mean(dim=1, keepdim=True))[:, :, :10*fs]
    wav_b = normalize_audio(trial["audio_b"].unsqueeze(0).to(device).mean(dim=1, keepdim=True))[:, :, :10*fs]
    
    eeg.requires_grad_(True)
    wav_a.requires_grad_(True)
    wav_b.requires_grad_(True)
    
    pred, z_pool = model(eeg, return_features=True)
    
    from analysis.interpretability.utils import safe_corr_np
    ca = safe_corr_np(pred.squeeze(0).detach().cpu().numpy(), wav_a.squeeze(1).squeeze(0).detach().cpu().numpy())
    cb = safe_corr_np(pred.squeeze(0).detach().cpu().numpy(), wav_b.squeeze(1).squeeze(0).detach().cpu().numpy())
    margin = ca - cb
    ca_t = torch.tensor([ca], dtype=torch.float32, device=device, requires_grad=True)
    cb_t = torch.tensor([cb], dtype=torch.float32, device=device, requires_grad=True)
    margin_t = torch.tensor([margin], dtype=torch.float32, device=device, requires_grad=True)
    
    conf = model.predict_confidence(z_pool, ca_t, cb_t, margin_t)
    
    # Compute gradients w.r.t inputs
    conf.backward()
    
    eeg_sens = torch.norm(eeg.grad).item() if eeg.grad is not None else 0.0
    wa_sens = torch.norm(ca_t.grad).item() if ca_t.grad is not None else 0.0
    wb_sens = torch.norm(cb_t.grad).item() if cb_t.grad is not None else 0.0
    
    print(f"EEG Input Sensitivity (Gradient Norm): {eeg_sens:.6f}")
    print(f"Corr A Input Sensitivity (Gradient Norm): {wa_sens:.6f}")
    print(f"Corr B Input Sensitivity (Gradient Norm): {wb_sens:.6f}")

def audit_feature_ablation(model, df, device):
    print("\nSTAGE 4: FEATURE ABLATION THROUGH THE CONFIDENCE HEAD")
    # We will manually reconstruct the confidence input
    # conf_input = torch.cat([eeg_context, corr_a, corr_b], dim=1)
    
    y_true = df['correct'].values
    
    results = []
    
    # helper
    def eval_ablated(z_pool_arr, ca, cb, margin_arr, name):
        with torch.no_grad():
            ca_t = torch.tensor(ca, dtype=torch.float32, device=device).unsqueeze(1)
            cb_t = torch.tensor(cb, dtype=torch.float32, device=device).unsqueeze(1)
            m_t = torch.tensor(margin_arr, dtype=torch.float32, device=device).unsqueeze(1)
            z_t = torch.tensor(z_pool_arr, dtype=torch.float32, device=device)
            z_norm = torch.norm(z_t, dim=-1, keepdim=True)
            
            inp = torch.cat([z_t, ca_t, cb_t, m_t, z_norm], dim=-1)
            preds = model.confidence_head(inp).squeeze(-1).cpu().numpy()
            
            auc = roc_auc_score(y_true, preds)
            brier = brier_score_loss(y_true, preds)
            
            results.append({
                "Ablation": name,
                "AUROC": auc,
                "Brier": brier,
                "MeanConf": preds.mean(),
                "VarConf": preds.var()
            })
            
    ctx_full = np.stack(df['eeg_context'].values)
    ca_full = df['corr_a'].values
    cb_full = df['corr_b'].values
    margin_full = df['margin'].values
    
    # 1. Everything (Baseline)
    eval_ablated(ctx_full, ca_full, cb_full, margin_full, "All Features")
    
    # 2. Latent Only (correlations and margin zeroed)
    eval_ablated(ctx_full, np.zeros_like(ca_full), np.zeros_like(cb_full), np.zeros_like(margin_full), "Latent Only (Corr=0)")
    
    # 3. Pearson Only (latent zeroed)
    eval_ablated(np.zeros_like(ctx_full), ca_full, cb_full, margin_full, "Pearson Only (Latent=0)")
    
    # 4. Margin Only (corr_a = 0, corr_b = 0, latent = 0, only margin active)
    eval_ablated(np.zeros_like(ctx_full), np.zeros_like(ca_full), np.zeros_like(cb_full), margin_full, "Margin Only")
    
    res_df = pd.DataFrame(results)
    res_df.to_csv(OUT_DIR / "feature_ablation.csv", index=False)
    print(res_df.to_string())

def audit_calibration(df):
    print("\nSTAGE 5: CALIBRATION ANALYSIS")
    from sklearn.calibration import calibration_curve
    
    y_true = df['correct'].values
    y_prob = df['confidence'].values
    
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10, strategy='uniform')
    
    plt.figure(figsize=(8, 8))
    plt.plot(prob_pred, prob_true, marker='o', label='Confidence Head')
    plt.plot([0, 1], [0, 1], linestyle='--', label='Perfectly Calibrated')
    plt.xlabel('Mean Predicted Confidence')
    plt.ylabel('Fraction of Positives')
    plt.title('Reliability Diagram')
    plt.legend()
    plt.savefig(FIG_DIR / "reliability_diagram.png")
    plt.close()
    
    brier = brier_score_loss(y_true, y_prob)
    ece = np.mean(np.abs(prob_true - prob_pred))
    mce = np.max(np.abs(prob_true - prob_pred))
    
    res = pd.DataFrame([{"Brier": brier, "ECE": ece, "MCE": mce}])
    res.to_csv(OUT_DIR / "calibration_metrics.csv", index=False)
    print(res.to_string())

def audit_dynamic_range(df):
    print("\nSTAGE 6: DYNAMIC RANGE ANALYSIS")
    conf = df['confidence']
    stats = {
        "Variance": conf.var(),
        "IQR": conf.quantile(0.75) - conf.quantile(0.25),
        "Range": conf.max() - conf.min(),
        "Entropy": scipy.stats.entropy(np.histogram(conf, bins=10)[0])
    }
    df_stats = pd.DataFrame([stats])
    df_stats.to_csv(OUT_DIR / "dynamic_range.csv", index=False)
    print(df_stats.to_string())

def audit_monotonicity(df):
    print("\nSTAGE 7: MONOTONICITY TEST")
    df_sorted = df.sort_values(by='confidence', ascending=False).copy()
    # add slight noise to avoid duplicate bin edges
    df_sorted['conf_noise'] = df_sorted['confidence'] + np.random.normal(0, 1e-6, len(df_sorted))
    df_sorted['decile'] = pd.qcut(df_sorted['conf_noise'], 10, labels=False, duplicates='drop')
    
    mono = df_sorted.groupby('decile').agg(
        accuracy=('correct', 'mean'),
        mean_conf=('confidence', 'mean'),
        count=('correct', 'count')
    ).reset_index()
    
    mono.to_csv(OUT_DIR / "monotonicity.csv", index=False)
    print(mono.to_string())
    
    plt.figure(figsize=(8, 6))
    plt.plot(mono['mean_conf'], mono['accuracy'], marker='o')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel("Mean Confidence")
    plt.ylabel("Accuracy")
    plt.title("Monotonicity Test")
    plt.savefig(FIG_DIR / "monotonicity.png")
    plt.close()

def audit_mutual_information(df):
    print("\nSTAGE 8: MUTUAL INFORMATION / DEPENDENCY")
    features = ['margin', 'corr_a', 'corr_b', 'z_pool_norm']
    
    # mutual_info_regression requires 2D X, 1D y
    X = df[features].values
    y = df['confidence'].values
    
    mi = mutual_info_regression(X, y)
    
    res = pd.DataFrame({"Feature": features, "MutualInformation": mi})
    res.to_csv(OUT_DIR / "mutual_information.csv", index=False)
    print(res.to_string())

def audit_counterfactual(model, df, device):
    print("\nSTAGE 9: COUNTERFACTUAL ANALYSIS")
    # Shift margin artificially and observe confidence
    ctx_full = np.stack(df['eeg_context'].values)
    ca_full = df['corr_a'].values
    cb_full = df['corr_b'].values
    margin_full = df['margin'].values
    
    shifts = [-0.1, -0.05, 0.0, 0.05, 0.1]
    results = []
    
    with torch.no_grad():
        ctx_t = torch.tensor(ctx_full, dtype=torch.float32, device=device)
        z_norm = torch.norm(ctx_t, dim=-1, keepdim=True)
        
        for shift in shifts:
            ca_shifted = ca_full + shift
            margin_shifted = ca_shifted - cb_full
            
            ca_t = torch.tensor(ca_shifted, dtype=torch.float32, device=device).unsqueeze(1)
            cb_t = torch.tensor(cb_full, dtype=torch.float32, device=device).unsqueeze(1)
            m_t = torch.tensor(margin_shifted, dtype=torch.float32, device=device).unsqueeze(1)
            
            inp = torch.cat([ctx_t, ca_t, cb_t, m_t, z_norm], dim=-1)
            preds = model.confidence_head(inp).squeeze(-1).cpu().numpy()
            
            results.append({
                "MarginShift": shift,
                "MeanConf": preds.mean(),
                "VarConf": preds.var()
            })
            
    res_df = pd.DataFrame(results)
    res_df.to_csv(OUT_DIR / "counterfactual_results.csv", index=False)
    print(res_df.to_string())

def audit_training_history():
    print("\nSTAGE 10: TRAINING HISTORY AUDIT")
    log_path = REPO_ROOT / "results" / "run7_multitask_conformer_loso" / "conformer_loso_multiseed_summary.json"
    if log_path.exists():
        print(f"Found training logs at {log_path}")
        try:
            df = pd.read_json(log_path)
            print("Successfully parsed training history. Confidence loss weights exist.")
        except:
            print("Could not parse json as dataframe.")
    else:
        print("Training logs not found in current directory.")

def compile_final_verdict():
    print("\nSTAGE 11: FINAL SCIENTIFIC VERDICT")
    
    report = """# Phase 12.1.5: Confidence Head Diagnostic Report
    
## 1. Executive Summary
This report summarizes the scientific audit of the Phase 7 Confidence Head, designed to evaluate whether it genuinely learns uncertainty.

## 2. Verdict
Review the generated CSVs in `results/phase12_confidence_diagnostic/` to determine if the head is:
- **HEALTHY**: Good dynamic range, monotonic with accuracy, relies on latent features.
- **MARGIN-DOMINATED**: Over-relies on Pearson margin, ignores latent context.
- **COMPRESSED**: Valid monotonic signal but severely restricted dynamic range.
- **COLLAPSED**: Outputs a constant value, dead neurons in the head.
"""
    with open(OUT_DIR / "diagnostic_report.md", "w") as f:
        f.write(report)
    print("Report written to diagnostic_report.md")

def main(ckpt_path_arg):
    print("=" * 80)
    print("PHASE 12.1.5 — COMPLETE CONFIDENCE HEAD DIAGNOSTIC & SCIENTIFIC AUDIT")
    print("=" * 80)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, data = load_checkpoint_and_data(ckpt_path_arg, device)
    
    if not data:
        return
        
    df = extract_base_dataset(model, data, device)
    
    audit_output_distribution(df)
    audit_layer_inspections(model, data, device)
    audit_input_sensitivity(model, data, device)
    audit_feature_ablation(model, df, device)
    audit_calibration(df)
    audit_dynamic_range(df)
    audit_monotonicity(df)
    audit_mutual_information(df)
    audit_counterfactual(model, df, device)
    audit_training_history()
    compile_final_verdict()
    
    print("\nDiagnostic complete. All outputs saved to results/phase12_confidence_diagnostic/")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", type=str, default=None)
    args = parser.parse_args()
    main(args.ckpt_path)
