import os
import sys
import numpy as np
import torch
import warnings
from pathlib import Path

import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from data.kul_cached_dataset import KULCachedLoader
from models.aad_conformer import AADConformer
from analysis.interpretability.utils import safe_corr_np, normalize_eeg, normalize_audio
from analysis.interpretability.channel_ablation import run_leave_one_channel_out

warnings.filterwarnings("ignore")

def confidence_interval(data, confidence=0.95):
    """Computes the 95% CI (1.96 * std / sqrt(N))."""
    if len(data) == 0: return 0.0
    return 1.96 * np.std(data, ddof=1) / np.sqrt(len(data)) if len(data) > 1 else 0.0

def sliding_window_evaluation(model, eeg, wav_a, wav_b, win_samples, hop_samples):
    """Custom evaluator returning detailed margin statistics for each window."""
    margins = []
    correct_w = 0
    total_w = 0
    p_att = []
    p_unatt = []
    
    if win_samples >= eeg.shape[-1]:
        pred = model(eeg).squeeze(0).cpu().numpy()
        wa = wav_a.squeeze(1).squeeze(0).cpu().numpy()
        wb = wav_b.squeeze(1).squeeze(0).cpu().numpy()
        
        ca = safe_corr_np(pred, wa)
        cb = safe_corr_np(pred, wb)
        margin = ca - cb
        return ca > cb, 1, 1 if ca > cb else 0, [margin], [ca], [cb]
        
    for start in range(0, eeg.shape[-1] - win_samples + 1, hop_samples):
        stop = start + win_samples
        eeg_win = eeg[:, :, start:stop]
        pred = model(eeg_win).squeeze(0).cpu().numpy()
        wa = wav_a[:, :, start:stop].squeeze(1).squeeze(0).cpu().numpy()
        wb = wav_b[:, :, start:stop].squeeze(1).squeeze(0).cpu().numpy()
        
        ca = safe_corr_np(pred, wa)
        cb = safe_corr_np(pred, wb)
        margin = ca - cb
        
        margins.append(margin)
        p_att.append(ca)
        p_unatt.append(cb)
        
        if ca > cb:
            correct_w += 1
        total_w += 1
        
    trial_correct = (correct_w > total_w / 2.0)
    return trial_correct, total_w, correct_w, margins, p_att, p_unatt

def exp4_1_decision_windows(model, test_trials, device):
    windows_sec = [1, 2, 5, 10, 20, 30, 60]
    fs = 64
    hop_sec = 1.0
    
    results = {}
    
    for w_sec in windows_sec:
        win_samples = int(w_sec * fs)
        hop_samples = int(hop_sec * fs)
        
        t_corr = 0
        total_w = 0
        w_corr = 0
        all_margins = []
        all_p_att = []
        all_p_unatt = []
        
        model.eval()
        with torch.no_grad():
            for t in test_trials:
                eeg = t["eeg"].unsqueeze(0).to(device)
                wav_a = t["audio_a"].unsqueeze(0).to(device).mean(dim=1, keepdim=True)
                wav_b = t["audio_b"].unsqueeze(0).to(device).mean(dim=1, keepdim=True)
                
                eeg = normalize_eeg(eeg)
                wav_a = normalize_audio(wav_a)
                wav_b = normalize_audio(wav_b)
                
                min_len = min(eeg.shape[-1], wav_a.shape[-1], wav_b.shape[-1])
                eeg, wav_a, wav_b = eeg[:,:,:min_len], wav_a[:,:,:min_len], wav_b[:,:,:min_len]
                
                tc, tw, cw, m, pa, pb = sliding_window_evaluation(model, eeg, wav_a, wav_b, win_samples, hop_samples)
                if tc: t_corr += 1
                total_w += tw
                w_corr += cw
                all_margins.extend(m)
                all_p_att.extend(pa)
                all_p_unatt.extend(pb)
                
        results[f"{w_sec}s"] = {
            "Trial Acc": t_corr / len(test_trials),
            "Window Acc": w_corr / max(1, total_w),
            "Mean Margin": np.mean(all_margins) if len(all_margins) > 0 else 0,
            "Median Margin": np.median(all_margins) if len(all_margins) > 0 else 0,
            "Margin Std": np.std(all_margins) if len(all_margins) > 0 else 0,
            "Positive Margin %": np.mean(np.array(all_margins) > 0) * 100 if len(all_margins) > 0 else 0,
            "Mean P(att)": np.mean(all_p_att) if len(all_p_att) > 0 else 0,
            "Mean P(unatt)": np.mean(all_p_unatt) if len(all_p_unatt) > 0 else 0
        }
    return results

def inject_awgn(tensor, snr_db):
    if snr_db is None: # Clean
        return tensor
    
    signal_power = torch.var(tensor, dim=-1, keepdim=True)
    snr_linear = 10 ** (snr_db / 10.0)
    noise_power = signal_power / snr_linear
    noise = torch.randn_like(tensor) * torch.sqrt(noise_power)
    return tensor + noise

def exp4_2_eeg_noise(model, test_trials, device):
    snr_levels = [None, 20, 15, 10, 5, 0] # None = Clean
    fs = 64
    win_samples = 10 * fs
    hop_samples = fs
    
    results = {}
    
    for snr in snr_levels:
        t_corr = 0
        total_w = 0
        w_corr = 0
        all_margins = []
        
        model.eval()
        with torch.no_grad():
            for t in test_trials:
                eeg_clean = t["eeg"].unsqueeze(0).to(device)
                wav_a = t["audio_a"].unsqueeze(0).to(device).mean(dim=1, keepdim=True)
                wav_b = t["audio_b"].unsqueeze(0).to(device).mean(dim=1, keepdim=True)
                
                eeg_noisy = inject_awgn(eeg_clean, snr)
                
                eeg = normalize_eeg(eeg_noisy)
                wav_a = normalize_audio(wav_a)
                wav_b = normalize_audio(wav_b)
                
                min_len = min(eeg.shape[-1], wav_a.shape[-1], wav_b.shape[-1])
                eeg, wav_a, wav_b = eeg[:,:,:min_len], wav_a[:,:,:min_len], wav_b[:,:,:min_len]
                
                tc, tw, cw, m, pa, pb = sliding_window_evaluation(model, eeg, wav_a, wav_b, win_samples, hop_samples)
                if tc: t_corr += 1
                total_w += tw
                w_corr += cw
                all_margins.extend(m)
                
        label = f"{snr}dB" if snr is not None else "Clean"
        results[label] = {
            "Trial Acc": t_corr / len(test_trials),
            "Window Acc": w_corr / max(1, total_w),
            "Mean Margin": np.mean(all_margins) if len(all_margins) > 0 else 0
        }
    return results

def exp4_3_channel_count(model, test_trials, device, ranked_channels):
    """Progressive permutation ablation down to 1 channel."""
    counts = [8, 7, 6, 5, 4, 3, 2, 1]
    results = {}
    
    fs = 64
    win_samples = 10 * fs
    hop_samples = fs
    
    for n in counts:
        keep_channels = ranked_channels[:n]
        
        t_corr = 0
        total_w = 0
        w_corr = 0
        all_margins = []
        
        model.eval()
        with torch.no_grad():
            for t in test_trials:
                eeg = t["eeg"].unsqueeze(0).to(device).clone()
                
                for ch in range(eeg.shape[1]):
                    if ch not in keep_channels:
                        idx = torch.randperm(eeg.shape[-1], device=device)
                        eeg[:, ch, :] = eeg[:, ch, idx]
                        
                wav_a = t["audio_a"].unsqueeze(0).to(device).mean(dim=1, keepdim=True)
                wav_b = t["audio_b"].unsqueeze(0).to(device).mean(dim=1, keepdim=True)
                
                eeg = normalize_eeg(eeg)
                wav_a = normalize_audio(wav_a)
                wav_b = normalize_audio(wav_b)
                
                min_len = min(eeg.shape[-1], wav_a.shape[-1], wav_b.shape[-1])
                eeg, wav_a, wav_b = eeg[:,:,:min_len], wav_a[:,:,:min_len], wav_b[:,:,:min_len]
                
                tc, tw, cw, m, pa, pb = sliding_window_evaluation(model, eeg, wav_a, wav_b, win_samples, hop_samples)
                if tc: t_corr += 1
                total_w += tw
                w_corr += cw
                all_margins.extend(m)
                
        results[f"{n} Chs"] = {
            "Trial Acc": t_corr / len(test_trials),
            "Window Acc": w_corr / max(1, total_w),
            "Mean Margin": np.mean(all_margins) if len(all_margins) > 0 else 0
        }
    return results

def get_calib_data(model, test_trials, device):
    fs = 64
    win_samples = 10 * fs
    hop_samples = fs
    
    all_margins = []
    all_correct = []
    
    model.eval()
    with torch.no_grad():
        for t in test_trials:
            eeg = t["eeg"].unsqueeze(0).to(device)
            wav_a = t["audio_a"].unsqueeze(0).to(device).mean(dim=1, keepdim=True)
            wav_b = t["audio_b"].unsqueeze(0).to(device).mean(dim=1, keepdim=True)
            
            eeg = normalize_eeg(eeg)
            wav_a = normalize_audio(wav_a)
            wav_b = normalize_audio(wav_b)
            
            min_len = min(eeg.shape[-1], wav_a.shape[-1], wav_b.shape[-1])
            eeg, wav_a, wav_b = eeg[:,:,:min_len], wav_a[:,:,:min_len], wav_b[:,:,:min_len]
            
            tc, tw, cw, m, pa, pb = sliding_window_evaluation(model, eeg, wav_a, wav_b, win_samples, hop_samples)
            all_margins.extend(m)
            all_correct.extend([margin > 0 for margin in m])
            
    return all_margins, all_correct

def aggregate_results(subj_results, metric_keys):
    """
    subj_results: dict mapping subj_id -> dict mapping condition -> dict mapping metric -> val
    Returns aggregated dictionary: condition -> metric -> {Mean, Median, Std, CI95}
    """
    subjects = list(subj_results.keys())
    if len(subjects) == 0: return {}
    
    conditions = list(subj_results[subjects[0]].keys())
    
    agg = {}
    for cond in conditions:
        agg[cond] = {}
        for m in metric_keys:
            vals = [subj_results[s][cond][m] for s in subjects if cond in subj_results[s] and m in subj_results[s][cond]]
            agg[cond][m] = {
                "Mean": np.mean(vals),
                "Median": np.median(vals),
                "Std": np.std(vals, ddof=1) if len(vals) > 1 else 0.0,
                "CI95": confidence_interval(vals)
            }
    return agg

def print_agg_table(agg_dict, metric_key, title):
    print(f"\n{title}")
    cols = ["Condition", "Mean", "Median", "Std", "95% CI"]
    print("| " + " | ".join(cols) + " |")
    print("| " + " | ".join(["---"] * len(cols)) + " |")
    for cond, metrics in agg_dict.items():
        if metric_key not in metrics: continue
        d = metrics[metric_key]
        row = [cond, f"{d['Mean']:.4f}", f"{d['Median']:.4f}", f"{d['Std']:.4f}", f"±{d['CI95']:.4f}"]
        print("| " + " | ".join(row) + " |")

def plot_exp4_1(agg_data, out_dir):
    windows = [1, 2, 5, 10, 20, 30, 60]
    keys = [f"{w}s" for w in windows]
    
    acc_mean = [agg_data[k]["Window Acc"]["Mean"] for k in keys]
    acc_ci = [agg_data[k]["Window Acc"]["CI95"] for k in keys]
    
    margin_mean = [agg_data[k]["Mean Margin"]["Mean"] for k in keys]
    margin_ci = [agg_data[k]["Mean Margin"]["CI95"] for k in keys]
    
    pos_margin_mean = [agg_data[k]["Positive Margin %"]["Mean"] for k in keys]
    pos_margin_ci = [agg_data[k]["Positive Margin %"]["CI95"] for k in keys]
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Accuracy
    axes[0].plot(windows, acc_mean, 'bo-', label='Window Accuracy')
    axes[0].fill_between(windows, np.array(acc_mean)-np.array(acc_ci), np.array(acc_mean)+np.array(acc_ci), color='b', alpha=0.2)
    axes[0].set_xlabel("Decision Window (s)")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_title("Window Length vs. Accuracy")
    axes[0].grid(True)
    
    # Margin
    axes[1].plot(windows, margin_mean, 'go-', label='Mean Margin')
    axes[1].fill_between(windows, np.array(margin_mean)-np.array(margin_ci), np.array(margin_mean)+np.array(margin_ci), color='g', alpha=0.2)
    axes[1].set_xlabel("Decision Window (s)")
    axes[1].set_ylabel("Margin")
    axes[1].set_title("Window Length vs. Mean Margin")
    axes[1].grid(True)
    
    # Positive Margin %
    axes[2].plot(windows, pos_margin_mean, 'ro-', label='Positive Margin %')
    axes[2].fill_between(windows, np.array(pos_margin_mean)-np.array(pos_margin_ci), np.array(pos_margin_mean)+np.array(pos_margin_ci), color='r', alpha=0.2)
    axes[2].set_xlabel("Decision Window (s)")
    axes[2].set_ylabel("Positive Margin (%)")
    axes[2].set_title("Window Length vs. Positive Margin %")
    axes[2].grid(True)
    
    plt.tight_layout()
    plt.savefig(out_dir / "exp4_1_windows.png", dpi=300)
    plt.close()

def plot_exp4_2(agg_data, out_dir):
    snrs = ["Clean", "20dB", "15dB", "10dB", "5dB", "0dB"]
    x_labels = ["Clean", "20", "15", "10", "5", "0"]
    
    acc_mean = [agg_data[k]["Window Acc"]["Mean"] for k in snrs]
    acc_ci = [agg_data[k]["Window Acc"]["CI95"] for k in snrs]
    
    plt.figure(figsize=(8, 6))
    plt.plot(x_labels, acc_mean, 'bo-')
    plt.fill_between(x_labels, np.array(acc_mean)-np.array(acc_ci), np.array(acc_mean)+np.array(acc_ci), color='b', alpha=0.2)
    plt.xlabel("SNR (dB)")
    plt.ylabel("Window Accuracy")
    plt.title("EEG Noise Robustness (SNR vs. Accuracy)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_dir / "exp4_2_noise.png", dpi=300)
    plt.close()

def plot_exp4_3(agg_data, out_dir):
    channels = [8, 7, 6, 5, 4, 3, 2, 1]
    keys = [f"{c} Chs" for c in channels]
    
    acc_mean = [agg_data[k]["Trial Acc"]["Mean"] for k in keys]
    acc_ci = [agg_data[k]["Trial Acc"]["CI95"] for k in keys]
    
    plt.figure(figsize=(8, 6))
    plt.plot(channels, acc_mean, 'bo-')
    plt.fill_between(channels, np.array(acc_mean)-np.array(acc_ci), np.array(acc_mean)+np.array(acc_ci), color='b', alpha=0.2)
    plt.gca().invert_xaxis()
    plt.xlabel("Number of Channels Retained")
    plt.ylabel("Trial Accuracy")
    plt.title("Channel Count Robustness")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_dir / "exp4_3_channels.png", dpi=300)
    plt.close()

def compute_and_plot_calibration(all_margins, all_correct, out_dir):
    all_margins = np.array(all_margins)
    all_correct = np.array(all_correct, dtype=float)
    
    conf_probs = (all_margins + 2.0) / 4.0
    conf_probs = np.clip(conf_probs, 0.0, 0.9999)
    
    bins = np.linspace(0, 1, 11)
    bin_indices = np.digitize(conf_probs, bins) - 1
    
    calibration_curve = []
    ece = 0.0
    total_samples = len(conf_probs)
    
    bin_centers = []
    emp_accs = []
    
    if total_samples > 0:
        for i in range(10):
            mask = (bin_indices == i)
            if np.sum(mask) > 0:
                bin_acc = np.mean(all_correct[mask])
                bin_conf = np.mean(conf_probs[mask])
                n_bin = np.sum(mask)
                ece += np.abs(bin_acc - bin_conf) * (n_bin / total_samples)
                
                mean_margin = (bin_conf * 4.0) - 2.0
                calibration_curve.append({
                    "Bin": i,
                    "Count": n_bin,
                    "Mean Margin": mean_margin,
                    "Empirical Acc": bin_acc,
                    "Mean Conf": bin_conf
                })
                bin_centers.append(bin_conf)
                emp_accs.append(bin_acc)
                
    plt.figure(figsize=(8, 8))
    plt.plot([0, 1], [0, 1], "k--", label="Perfect Calibration")
    plt.plot(bin_centers, emp_accs, "bo-", label="Empirical Calibration")
    plt.xlabel("Mean Confidence (Normalized Margin)")
    plt.ylabel("Empirical Accuracy")
    plt.title(f"Reliability Diagram (Global ECE = {ece:.4f})")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_dir / "exp4_4_calibration.png", dpi=300)
    plt.close()
    
    return calibration_curve, ece

def main():
    print("--- Phase 4: Full Robustness Validation Benchmark (16 Subjects) ---")
    
    cache_dir = REPO_ROOT / "data" / "processed_kul"
    if Path("/kaggle/input/datasets/lowk1ee/kul-preprocessed-cache/data/processed_kul").exists():
        cache_dir = Path("/kaggle/input/datasets/lowk1ee/kul-preprocessed-cache/data/processed_kul")
    
    checkpoint_dir = REPO_ROOT / "conformer_loso_results" / "checkpoints" / "seed_1"
    kaggle_ckpt = Path("/kaggle/input/datasets/lowkieee/eeg-aad-conformer-seed1-checkpoints/checkpoints/seed_1")
    if not kaggle_ckpt.exists():
        kaggle_ckpt = Path("/kaggle/input/eeg-aad-conformer-seed1-checkpoints/checkpoints/seed_1")
    if kaggle_ckpt.exists():
        checkpoint_dir = kaggle_ckpt
        
    out_dir = REPO_ROOT / "results" / "run4_robustness_final"
    os.makedirs(out_dir, exist_ok=True)
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    loader = KULCachedLoader(cache_dir)
    loader.load_all()
    
    subjects_to_test = list(loader.subjects_data.keys())
    print(f"Evaluating {len(subjects_to_test)} subjects: {subjects_to_test}")
    
    all_res_4_1 = {}
    all_res_4_2 = {}
    all_res_4_3 = {}
    
    global_margins = []
    global_correct = []
    
    for subj in subjects_to_test:
        print(f"\n=== Processing Subject {subj} ===")
        ckpt_path = checkpoint_dir / f"model_{subj}.pt"
        if not ckpt_path.exists():
            print(f"  [Error] Checkpoint not found: {ckpt_path}. Skipping.")
            continue
            
        model = AADConformer(in_channels=8).to(device)
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        model.eval()
        
        test_trials = loader.subjects_data[subj] # Full trial set
        
        print("  - Running Exp 4.1: Decision Window Robustness...")
        res_4_1 = exp4_1_decision_windows(model, test_trials, device)
        all_res_4_1[subj] = res_4_1
        
        print("  - Running Exp 4.2: EEG Noise Robustness...")
        res_4_2 = exp4_2_eeg_noise(model, test_trials, device)
        all_res_4_2[subj] = res_4_2
        
        print("  - Running Exp 4.3: Channel Count Robustness...")
        loco_results, ranked_channels = run_leave_one_channel_out(model, test_trials, device)
        
        CHANNEL_NAMES = ['T7', 'C2', 'FT8', 'P7', 'CPz', 'Fp1', 'TP8', 'C3']
        ranked_indices = []
        for ch_val in ranked_channels:
            if isinstance(ch_val, str) and ch_val in CHANNEL_NAMES:
                ranked_indices.append(CHANNEL_NAMES.index(ch_val))
            elif isinstance(ch_val, int):
                ranked_indices.append(ch_val)
                
        res_4_3 = exp4_3_channel_count(model, test_trials, device, ranked_indices)
        all_res_4_3[subj] = res_4_3
        
        print("  - Running Exp 4.4: Accumulating Confidence Calibration Data...")
        subj_margins, subj_correct = get_calib_data(model, test_trials, device)
        global_margins.extend(subj_margins)
        global_correct.extend(subj_correct)
        
    print("\n\n" + "="*80)
    print("PHASE 4 CONSOLIDATED SCIENTIFIC REPORT (16 SUBJECTS)")
    print("="*80 + "\n")
    
    # Aggregate 4.1
    agg_4_1 = aggregate_results(all_res_4_1, ["Trial Acc", "Window Acc", "Mean Margin", "Positive Margin %"])
    print("### Experiment 4.1: Decision Window Robustness")
    print_agg_table(agg_4_1, "Window Acc", "Aggregated Window Accuracy")
    print_agg_table(agg_4_1, "Mean Margin", "Aggregated Mean Margin")
    plot_exp4_1(agg_4_1, out_dir)
    print(f"-> Saved Exp 4.1 plot to {out_dir}/exp4_1_windows.png")
    
    # Aggregate 4.2
    agg_4_2 = aggregate_results(all_res_4_2, ["Trial Acc", "Window Acc", "Mean Margin"])
    print("\n### Experiment 4.2: EEG Noise Robustness")
    print_agg_table(agg_4_2, "Window Acc", "Aggregated Window Accuracy vs SNR")
    plot_exp4_2(agg_4_2, out_dir)
    print(f"-> Saved Exp 4.2 plot to {out_dir}/exp4_2_noise.png")
    
    # Aggregate 4.3
    agg_4_3 = aggregate_results(all_res_4_3, ["Trial Acc", "Window Acc", "Mean Margin"])
    print("\n### Experiment 4.3: Channel Count Robustness")
    print_agg_table(agg_4_3, "Trial Acc", "Aggregated Trial Accuracy vs Channel Count")
    plot_exp4_3(agg_4_3, out_dir)
    print(f"-> Saved Exp 4.3 plot to {out_dir}/exp4_3_channels.png")
    
    # Aggregate 4.4
    print("\n### Experiment 4.4: Confidence Calibration (Population Level)")
    calib_curve, global_ece = compute_and_plot_calibration(global_margins, global_correct, out_dir)
    print(f"**Global Expected Calibration Error (ECE): {global_ece:.4f}**")
    print("| Bin | Count | Mean Margin | Empirical Acc |")
    print("|---|---|---|---|")
    for row in calib_curve:
        print(f"| {row['Bin']} | {row['Count']} | {row['Mean Margin']:.4f} | {row['Empirical Acc']:.4f} |")
    print(f"-> Saved Exp 4.4 plot to {out_dir}/exp4_4_calibration.png")
    
    print("\n### Cross-Experiment Scientific Conclusions")
    print("""
1. **How quickly can attention be decoded?**
   (See Exp 4.1). Review the point where the Window Accuracy curve sharply degrades.
2. **How robust is the model to EEG noise?**
   (See Exp 4.2). Identifies the SNR limit at which the attention signal is completely masked.
3. **What is the minimum usable channel count?**
   (See Exp 4.3). Check when Trial Accuracy drops below acceptable operating thresholds (e.g. 60-65%).
4. **Is the confidence estimate reliable?**
   (See Exp 4.4). A monotonically increasing Reliability Diagram implies high confidence is trustworthy.
""")
    print("\n[Phase 4 Benchmark Complete]")

if __name__ == "__main__":
    main()
