import os
import sys
import numpy as np
import torch
import json
import warnings
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from data.kul_cached_dataset import KULCachedLoader
from models.aad_conformer import AADConformer
from analysis.interpretability.utils import safe_corr_np, normalize_eeg, normalize_audio
from analysis.interpretability.channel_ablation import run_leave_one_channel_out

warnings.filterwarnings("ignore")

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
            "Mean Margin": np.mean(all_margins),
            "Median Margin": np.median(all_margins),
            "Margin Std": np.std(all_margins),
            "Positive Margin %": np.mean(np.array(all_margins) > 0) * 100,
            "Mean P(att)": np.mean(all_p_att),
            "Mean P(unatt)": np.mean(all_p_unatt)
        }
    return results

def inject_awgn(tensor, snr_db):
    if snr_db is None: # Clean
        return tensor
    
    # Calculate signal power (variance) per channel
    signal_power = torch.var(tensor, dim=-1, keepdim=True)
    # Calculate required noise power
    snr_linear = 10 ** (snr_db / 10.0)
    noise_power = signal_power / snr_linear
    # Generate noise
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
                
                # Inject noise BEFORE normalization
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
            "Mean Margin": np.mean(all_margins)
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
            "Mean Margin": np.mean(all_margins)
        }
    return results

def exp4_4_confidence_calibration(model, test_trials, device):
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
            
    all_margins = np.array(all_margins)
    all_correct = np.array(all_correct, dtype=float)
    
    conf_probs = (all_margins + 2.0) / 4.0
    
    bins = np.linspace(0, 1, 11)
    bin_indices = np.digitize(conf_probs, bins) - 1
    
    calibration_curve = []
    ece = 0.0
    total_samples = len(conf_probs)
    
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
                    "Empirical Acc": bin_acc
                })
            
    return calibration_curve, ece

def print_markdown_table(data_dict, cols):
    print("| " + " | ".join(cols) + " |")
    print("| " + " | ".join(["---"] * len(cols)) + " |")
    for key, vals in data_dict.items():
        row = [key]
        for c in cols[1:]:
            val = vals.get(c, "")
            if isinstance(val, float):
                row.append(f"{val:.4f}")
            else:
                row.append(str(val))
        print("| " + " | ".join(row) + " |")
    print()

def main():
    print("--- Phase 4: Robustness & Real-World Generalization Validation ---")
    
    cache_dir = REPO_ROOT / "data" / "processed_kul"
    if Path("/kaggle/input/datasets/lowk1ee/kul-preprocessed-cache/data/processed_kul").exists():
        cache_dir = Path("/kaggle/input/datasets/lowk1ee/kul-preprocessed-cache/data/processed_kul")
    
    checkpoint_dir = REPO_ROOT / "conformer_loso_results" / "checkpoints" / "seed_1"
    kaggle_ckpt = Path("/kaggle/input/datasets/lowkieee/eeg-aad-conformer-seed1-checkpoints/checkpoints/seed_1")
    if not kaggle_ckpt.exists():
        kaggle_ckpt = Path("/kaggle/input/eeg-aad-conformer-seed1-checkpoints/checkpoints/seed_1")
    if kaggle_ckpt.exists():
        checkpoint_dir = kaggle_ckpt
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    loader = KULCachedLoader(cache_dir)
    loader.load_all()
    
    subjects_to_test = ['S11', 'S16', 'S13'] # Strong, Average, Weak
    
    all_res_4_1 = {}
    all_res_4_2 = {}
    all_res_4_3 = {}
    all_res_4_4 = {}
    
    for subj in subjects_to_test:
        print(f"\n=== Processing Subject {subj} ===")
        ckpt_path = checkpoint_dir / f"model_{subj}.pt"
        if not ckpt_path.exists():
            print(f"  [Error] Checkpoint not found: {ckpt_path}. Skipping.")
            continue
            
        model = AADConformer(in_channels=8).to(device)
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        model.eval()
        
        exs = loader.get_subject_data(subj)
        test_trials = exs[-3:] if len(exs) > 3 else exs # Last 3 trials for testing per protocol
        
        print("  - Running Exp 4.1: Decision Window Robustness...")
        res_4_1 = exp4_1_decision_windows(model, test_trials, device)
        all_res_4_1[subj] = res_4_1
        
        print("  - Running Exp 4.2: EEG Noise Robustness...")
        res_4_2 = exp4_2_eeg_noise(model, test_trials, device)
        all_res_4_2[subj] = res_4_2
        
        print("  - Running Exp 4.3: Channel Count Robustness...")
        # Get ranked channels (indices) directly
        loco_results, ranked_channels = run_leave_one_channel_out(model, test_trials, device)
        
        # ranked_channels might contain string names if channel_ablation.py was fully updated to return names
        # We need indices for exp4_3. Let's ensure we use indices.
        CHANNEL_NAMES = ['T7', 'C2', 'FT8', 'P7', 'CPz', 'Fp1', 'TP8', 'C3']
        ranked_indices = []
        for ch_val in ranked_channels:
            if isinstance(ch_val, str) and ch_val in CHANNEL_NAMES:
                ranked_indices.append(CHANNEL_NAMES.index(ch_val))
            elif isinstance(ch_val, int):
                ranked_indices.append(ch_val)
        
        res_4_3 = exp4_3_channel_count(model, test_trials, device, ranked_indices)
        all_res_4_3[subj] = res_4_3
        
        print("  - Running Exp 4.4: Confidence Calibration...")
        calib_curve, ece = exp4_4_confidence_calibration(model, test_trials, device)
        all_res_4_4[subj] = {"Curve": calib_curve, "ECE": ece}
        
    print("\n\n" + "="*80)
    print("PHASE 4 CONSOLIDATED SCIENTIFIC REPORT")
    print("="*80 + "\n")
    
    print("### Experiment 4.1: Decision Window Robustness")
    print("Measures how temporal context affects accuracy.\n")
    for subj, res in all_res_4_1.items():
        print(f"**Subject {subj}**")
        cols = ["Window", "Trial Acc", "Window Acc", "Mean Margin", "Positive Margin %"]
        print_markdown_table(res, cols)
        
    print("\n### Experiment 4.2: EEG Noise Robustness")
    print("Measures tolerance to Additive White Gaussian Noise (AWGN).\n")
    for subj, res in all_res_4_2.items():
        print(f"**Subject {subj}**")
        cols = ["SNR", "Trial Acc", "Window Acc", "Mean Margin"]
        print_markdown_table(res, cols)
        
    print("\n### Experiment 4.3: Channel Count Robustness")
    print("Graceful degradation via permutation feature importance (Top 8 to 1).\n")
    for subj, res in all_res_4_3.items():
        print(f"**Subject {subj}**")
        cols = ["Channels", "Trial Acc", "Window Acc", "Mean Margin"]
        print_markdown_table(res, cols)
        
    print("\n### Experiment 4.4: Confidence Calibration")
    print("Measures if higher prediction margin correlates with higher accuracy.\n")
    for subj, res in all_res_4_4.items():
        print(f"**Subject {subj} - ECE: {res['ECE']:.4f}**")
        print("| Bin | Count | Mean Margin | Empirical Acc |")
        print("|---|---|---|---|")
        for row in res['Curve']:
            print(f"| {row['Bin']} | {row['Count']} | {row['Mean Margin']:.4f} | {row['Empirical Acc']:.4f} |")
        print()
        
    print("\n### Cross-Experiment Scientific Conclusions")
    print("""
1. **How quickly can attention be decoded?**
   (See Exp 4.1 tables). Look for the inflection point where 'Window Acc' collapses.

2. **How robust is the model to EEG noise?**
   (See Exp 4.2 tables). Compare 'Clean' performance to SNR thresholds to identify the breakpoint.

3. **How many channels are actually necessary?**
   (See Exp 4.3 tables). The point where 'Trial Acc' drops below 60% indicates the minimum viable hardware bound.

4. **Is prediction confidence meaningful?**
   (See Exp 4.4 tables). If 'Empirical Acc' monotonically increases with 'Mean Margin', the model is well-calibrated and highly reliable for real-world deployment.

5. **Under what conditions does the model fail?**
   Based on the data above, you can precisely define the failure thresholds (e.g., < 2s windows, < 10dB SNR, < 3 channels).

6. **Which experiment represents the largest practical limitation?**
   The steepest performance cliff across the 4 experiments determines the true hardware/environmental constraint for a viable hearing-aid.
""")

if __name__ == "__main__":
    main()
