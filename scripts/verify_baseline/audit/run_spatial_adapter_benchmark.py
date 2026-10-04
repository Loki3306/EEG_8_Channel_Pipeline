import argparse
import sys
import os
import json
import time
from pathlib import Path
from copy import deepcopy
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

REPO_ROOT = Path(__file__).resolve().parents[3]
VERIFY_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(VERIFY_ROOT) not in sys.path:
    sys.path.insert(0, str(VERIFY_ROOT))

from models.catcn import CATCNDirectDecoder
from src.models.spatial_adapter import SpatialEEGAdapter
from src.streaming.causal_filters import StreamingCausalEEGFilter
from training.montages import MONTAGES
from training.train_matchnet_wavlm import get_mapping_data, prepare_dataset, FS
from baselines.ridge_aad import load_subject_examples, subject_files

from src.selective_aad.core import TemperatureCalibrator
from src.selective_aad.metrics import (
    calculate_selective_metrics,
    compute_temporal_stability_metrics,
)
from src.selective_aad.evaluator import SelectiveAADEvaluator
from src.selective_aad.temporal_gate import SignalQualityMonitor, StickyHysteresisGate

def process_subject_trials(examples, montage_channels, sub_id, mapping, envelopes, causal_filter, fs):
    _, ya_raw, yb_raw = prepare_dataset(examples, montage_channels, 1.0, 6.0, sub_id, mapping, envelopes)
    ya_clean = [ya.mean(axis=0, keepdims=True).squeeze(0).astype(np.float32) if ya.shape[0] > 1 else ya.squeeze(0).astype(np.float32) for ya in ya_raw]
    yb_clean = [yb.mean(axis=0, keepdims=True).squeeze(0).astype(np.float32) if yb.shape[0] > 1 else yb.squeeze(0).astype(np.float32) for yb in yb_raw]
    
    eeg_out, ya_out, yb_out = [], [], []
    for idx in range(min(len(examples), len(ya_clean))):
        raw_eeg = examples[idx].eeg[:, montage_channels].astype(np.float32)
        min_len = min(len(raw_eeg), len(ya_clean[idx]), len(yb_clean[idx]))
        raw_eeg = raw_eeg[:min_len]
        
        causal_filter.reset()
        eeg_c = causal_filter.process_chunk(raw_eeg)
        eeg_c = (eeg_c - np.mean(eeg_c, axis=0, keepdims=True)) / (np.std(eeg_c, axis=0, keepdims=True) + 1e-8)
        
        eeg_out.append(eeg_c)
        ya_out.append(ya_clean[idx][:min_len])
        yb_out.append(yb_clean[idx][:min_len])
        
    return eeg_out, ya_out, yb_out

def extract_windows_dataset(eeg_list, ya_list, yb_list, window_sec, step_sec, fs):
    """
    Slices trials into [B, C, T] sliding windows for calibration or evaluation.
    """
    window_samples = int(window_sec * fs)
    step_samples = int(step_sec * fs)
    
    x_chunks, ya_chunks, yb_chunks = [], [], []
    for eeg, ya, yb in zip(eeg_list, ya_list, yb_list):
        t_len = min(len(eeg), len(ya), len(yb))
        curr_start = 0
        while curr_start + window_samples <= t_len:
            curr_end = curr_start + window_samples
            w_e = eeg[curr_start:curr_end]
            w_a = ya[curr_start:curr_end]
            w_b = yb[curr_start:curr_end]
            
            # Causal standardization per window
            w_e_std = (w_e - np.mean(w_e, axis=0, keepdims=True)) / (np.std(w_e, axis=0, keepdims=True) + 1e-8)
            w_a_std = (w_a - np.mean(w_a)) / (np.std(w_a) + 1e-8)
            w_b_std = (w_b - np.mean(w_b)) / (np.std(w_b) + 1e-8)
            
            x_chunks.append(w_e_std.T)
            ya_chunks.append(np.expand_dims(w_a_std, axis=0))
            yb_chunks.append(np.expand_dims(w_b_std, axis=0))
            curr_start += step_samples
            
    if not x_chunks:
        return np.zeros((0, 8, window_samples), dtype=np.float32), np.zeros((0, 1, window_samples), dtype=np.float32), np.zeros((0, 1, window_samples), dtype=np.float32)
    return np.stack(x_chunks, axis=0).astype(np.float32), np.stack(ya_chunks, axis=0).astype(np.float32), np.stack(yb_chunks, axis=0).astype(np.float32)

def train_spatial_adapter(adapter, frozen_backbone, calib_x, calib_ya, calib_yb, epochs=15, lr=1e-3, l2_identity=0.05, device="cpu"):
    """
    Trains ONLY the 64-parameter SpatialEEGAdapter while keeping CA-TCN 100% frozen.
    Symmetric dual-target augmentation (stream A and stream B).
    """
    frozen_backbone.eval()
    for p in frozen_backbone.parameters():
        p.requires_grad = False
        
    adapter.train()
    optimizer = optim.AdamW(adapter.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()
    
    # Symmetric data augmentation (Target 1 for attended A, Target 0 for attended B)
    # Aug 1: (x, ya, yb) -> label 1.0
    # Aug 2: (x, yb, ya) -> label 0.0
    N = len(calib_x)
    x_aug = np.concatenate([calib_x, calib_x], axis=0)
    ya_aug = np.concatenate([calib_ya, calib_yb], axis=0)
    yb_aug = np.concatenate([calib_yb, calib_ya], axis=0)
    labels_aug = np.concatenate([np.ones(N, dtype=np.float32), np.zeros(N, dtype=np.float32)], axis=0)
    
    ds = TensorDataset(torch.from_numpy(x_aug), torch.from_numpy(ya_aug), torch.from_numpy(yb_aug), torch.from_numpy(labels_aug))
    loader = DataLoader(ds, batch_size=32, shuffle=True)
    
    for ep in range(epochs):
        for bx, bya, byb, blabel in loader:
            bx, bya, byb, blabel = bx.to(device), bya.to(device), byb.to(device), blabel.to(device)
            optimizer.zero_grad()
            
            # Forward through 8x8 adapter
            bx_adapted = adapter(bx)
            
            # Forward through frozen backbone
            with torch.no_grad():
                # Extract audio features through frozen audio encoder
                za = frozen_backbone.audio_encoder(bya)
                zb = frozen_backbone.audio_encoder(byb)
                
            # EEG encoder takes adapted EEG with gradient flowing to adapter
            ze = frozen_backbone.eeg_encoder(bx_adapted)
            score_a = frozen_backbone.classifier_head(ze, za)
            score_b = frozen_backbone.classifier_head(ze, zb)
            margin = score_a - score_b
            
            loss_task = criterion(margin, blabel)
            loss_reg = l2_identity * adapter.identity_regularization_loss()
            loss = loss_task + loss_reg
            
            loss.backward()
            optimizer.step()
            
    adapter.eval()
    return adapter

def evaluate_trials(model, adapter, eeg_list, ya_list, yb_list, window_sec, step_sec, fs, device):
    """
    Extracts rolling window margins on sequential test trials.
    adapter=None evaluates zero-shot baseline.
    """
    model.eval()
    if adapter is not None:
        adapter.eval()
        
    window_samples = int(window_sec * fs)
    step_samples = int(step_sec * fs)
    
    trials_margins = []
    trials_labels = []
    trials_raw_eeg = []
    
    for eeg, ya, yb in zip(eeg_list, ya_list, yb_list):
        t_len = min(len(eeg), len(ya), len(yb))
        win_e, win_a, win_b = [], [], []
        raw_e = []
        
        curr_start = 0
        while curr_start + window_samples <= t_len:
            curr_end = curr_start + window_samples
            w_e = eeg[curr_start:curr_end]
            w_a = ya[curr_start:curr_end]
            w_b = yb[curr_start:curr_end]
            
            raw_e.append(w_e)
            w_e_std = (w_e - np.mean(w_e, axis=0, keepdims=True)) / (np.std(w_e, axis=0, keepdims=True) + 1e-8)
            w_a_std = (w_a - np.mean(w_a)) / (np.std(w_a) + 1e-8)
            w_b_std = (w_b - np.mean(w_b)) / (np.std(w_b) + 1e-8)
            
            win_e.append(w_e_std.T)
            win_a.append(np.expand_dims(w_a_std, axis=0))
            win_b.append(np.expand_dims(w_b_std, axis=0))
            curr_start += step_samples
            
        if win_e:
            t_e = torch.from_numpy(np.stack(win_e)).float().to(device)
            t_a = torch.from_numpy(np.stack(win_a)).float().to(device)
            t_b = torch.from_numpy(np.stack(win_b)).float().to(device)
            
            with torch.no_grad():
                if adapter is not None:
                    t_e = adapter(t_e)
                d, _, _ = model(t_e, t_a, t_b)
                
            trial_m = d.detach().cpu().numpy().tolist()
            trials_margins.append(np.array(trial_m, dtype=np.float64))
            trials_labels.append(np.ones(len(trial_m), dtype=np.int64))
            trials_raw_eeg.append(raw_e)
            
    return trials_margins, trials_labels, trials_raw_eeg

def main():
    parser = argparse.ArgumentParser(description="Subject-Specific 8x8 Spatial Adapter Benchmark")
    parser.add_argument("--checkpoint_dir", type=str, default="/kaggle/working/loso_checkpoints", help="Path to checkpoint directory")
    parser.add_argument("--all_folds", action="store_true", help="Run across all 18 subjects (S1-S18)")
    parser.add_argument("--folds", type=str, default="", help="Comma-separated target subjects (e.g. S1,S4,S10,S16)")
    parser.add_argument("--subject", type=str, default="", help="Single target subject (e.g. S1)")
    parser.add_argument("--calib_trials", type=int, default=12, help="Number of calibration trials")
    parser.add_argument("--epochs_adapter", type=int, default=15, help="Epochs for 64-parameter adapter training")
    parser.add_argument("--lr_adapter", type=float, default=1e-3, help="Learning rate for adapter")
    parser.add_argument("--l2_identity", type=float, default=0.05, help="Identity shrinkage regularization")
    parser.add_argument("--montage", type=str, default="near_ear_expanded", help="Electrode montage")
    parser.add_argument("--window_sec", type=float, default=5.0, help="Window size in seconds (Frozen target)")
    parser.add_argument("--step_sec", type=float, default=0.5, help="Control step size in seconds (2 Hz)")
    parser.add_argument("--smoke_test", action="store_true", help="Fast smoke test")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 115)
    print("  SUBJECT-SPECIFIC 8x8 SPATIAL ADAPTER BENCHMARK (FROZEN 5.0s CA-TCN)")
    print(f"  Device: {device} | Calibration Trials: {args.calib_trials} | Adapter Params: 64 | L2 Identity: {args.l2_identity}")
    print("=" * 115)
    
    montage_channels = MONTAGES[args.montage]
    all_paths = subject_files()
    if not all_paths:
        print("[ERROR] No DTU files found.")
        return
        
    mapping, envelopes = get_mapping_data("gammatone")
    if args.all_folds:
        target_subs = [f"S{i}" for i in range(1, 19)]
    elif args.folds:
        target_subs = [s.strip() for s in args.folds.split(",") if s.strip()]
    elif args.subject:
        target_subs = [args.subject.strip()]
    else:
        target_subs = ["S1"]
        
    causal_filter = StreamingCausalEEGFilter(fs=FS, lowcut=1.0, highcut=6.0, order=2, n_channels=len(montage_channels))
    sq_monitor = SignalQualityMonitor()
    
    results = {}
    
    for target_sub in target_subs:
        print(f"\n==========================================================================")
        print(f"  TARGET SUBJECT: {target_sub}")
        print(f"==========================================================================")
        
        target_path = next((p for p in all_paths if p.stem.split("_")[0] == target_sub), None)
        if not target_path:
            continue
            
        # 1. Load Pre-Trained Universal Backbone
        backbone_candidates = [
            Path(args.checkpoint_dir) / f"catcn_loso_{target_sub}.pt",
            Path(args.checkpoint_dir) / f"catcn_univ_heldout_{target_sub}.pt",
            Path(f"/kaggle/working/loso_checkpoints/catcn_loso_{target_sub}.pt"),
            Path(f"/kaggle/working/checkpoints/catcn_loso_{target_sub}.pt"),
            Path(f"/kaggle/working/checkpoints/catcn_univ_heldout_{target_sub}.pt"),
            Path(f"checkpoints/loso/catcn_loso_{target_sub}.pt"),
            Path(f"checkpoints/adaptation/catcn_univ_heldout_{target_sub}.pt"),
        ]
        found_ckpt = next((p for p in backbone_candidates if p.exists()), None)
        if not found_ckpt:
            print(f"[ERROR] No checkpoint found for {target_sub}. Run training first.")
            continue
            
        univ_model = CATCNDirectDecoder(eeg_channels=len(montage_channels), audio_channels=1, hidden_dim=64, max_lag_samples=8).to(device)
        univ_model.load_state_dict(torch.load(found_ckpt, map_location=device))
        univ_model.eval()
        for p in univ_model.parameters():
            p.requires_grad = False
            
        print(f"  [CHECKPOINT] Loaded frozen backbone from: {found_ckpt}")
        
        # 2. Partition Subject Trials
        target_exs = load_subject_examples(target_path)
        if args.smoke_test:
            target_exs = target_exs[:min(18, len(target_exs))]
        eeg_all, ya_all, yb_all = process_subject_trials(target_exs, montage_channels, target_sub, mapping, envelopes, causal_filter, FS)
        
        K = min(args.calib_trials, len(eeg_all) // 2) if not args.smoke_test else 4
        eeg_calib, ya_calib, yb_calib = eeg_all[:K], ya_all[:K], yb_all[:K]
        eeg_test, ya_test, yb_test = eeg_all[K:], ya_all[K:], yb_all[K:]
        print(f"  [DATA] Partitioned {len(eeg_calib)} Calibration Trials, {len(eeg_test)} Test Trials")
        
        # 3. Extract Calibration Dataset
        calib_x, calib_ya, calib_yb = extract_windows_dataset(eeg_calib, ya_calib, yb_calib, args.window_sec, args.step_sec, FS)
        
        # 4. Train 8x8 Spatial Adapter (64 parameters)
        adapter = SpatialEEGAdapter(channels=len(montage_channels)).to(device)
        adapter_epochs = args.epochs_adapter if not args.smoke_test else 3
        print(f"  [ADAPTER] Training 8x8 Spatial Adapter (64 params, {adapter_epochs} epochs, LR={args.lr_adapter})...")
        adapter = train_spatial_adapter(
            adapter, univ_model, calib_x, calib_ya, calib_yb,
            epochs=adapter_epochs, lr=args.lr_adapter, l2_identity=args.l2_identity, device=device
        )
        
        # Log learned spatial projection matrix
        W_mat = adapter.get_weight_matrix().numpy()
        diag_mean = float(np.mean(np.diag(W_mat)))
        offdiag_mean = float(np.mean(np.abs(W_mat - np.diag(np.diag(W_mat)))))
        print(f"  [ADAPTER FIT] Diagonal Gain: {diag_mean:.3f} | Off-Diagonal Cross-Mix: {offdiag_mean:.3f}")
        
        # 5. Extract Evaluation Margins on Unseen Test Trials
        # A) Zero-Shot (adapter = None)
        zs_test_margins, zs_test_labels, _ = evaluate_trials(univ_model, None, eeg_test, ya_test, yb_test, args.window_sec, args.step_sec, FS, device)
        flat_zs_m = np.concatenate(zs_test_margins)
        flat_test_l = np.concatenate(zs_test_labels)
        gt_test_str = np.where(flat_test_l == 1, "A", "B")
        
        # B) Adapted 8x8 (adapter = adapter)
        ad_test_margins, _, ad_test_raw_eeg = evaluate_trials(univ_model, adapter, eeg_test, ya_test, yb_test, args.window_sec, args.step_sec, FS, device)
        flat_ad_m = np.concatenate(ad_test_margins)
        
        # -----------------------------------------------------------------
        # TIER 1: Raw Zero-Shot 5.0s Decoder (Baseline)
        # -----------------------------------------------------------------
        t1_preds = np.where(flat_zs_m >= 0, "A", "B")
        m_t1 = calculate_selective_metrics(t1_preds, gt_test_str)
        t1_fsw = float(np.mean([compute_temporal_stability_metrics(np.where(m >= 0, "A", "B"), np.where(l == 1, "A", "B"), args.step_sec)["false_switches_per_minute"] for m, l in zip(zs_test_margins, zs_test_labels)]))
        
        # -----------------------------------------------------------------
        # TIER 2: Raw Adapted 8x8 5.0s Decoder (Core Scientific Hypothesis)
        # -----------------------------------------------------------------
        t2_preds = np.where(flat_ad_m >= 0, "A", "B")
        m_t2 = calculate_selective_metrics(t2_preds, gt_test_str)
        t2_fsw = float(np.mean([compute_temporal_stability_metrics(np.where(m >= 0, "A", "B"), np.where(l == 1, "A", "B"), args.step_sec)["false_switches_per_minute"] for m, l in zip(ad_test_margins, zs_test_labels)]))
        raw_gain_pp = (m_t2["selective_accuracy"] - m_t1["selective_accuracy"]) * 100.0
        
        # -----------------------------------------------------------------
        # TIER 3: Adapted 8x8 + Sticky Hysteresis Gate (Product Controller)
        # -----------------------------------------------------------------
        # Calibrate gating parameters on ADAPTED calibration set
        ad_calib_margins, ad_calib_labels, _ = evaluate_trials(univ_model, adapter, eeg_calib, ya_calib, yb_calib, args.window_sec, args.step_sec, FS, device)
        flat_ad_calib_m = np.concatenate(ad_calib_margins)
        flat_ad_calib_l = np.concatenate(ad_calib_labels)
        
        calibrator = TemperatureCalibrator()
        fitted_t = calibrator.fit(flat_ad_calib_m, flat_ad_calib_l, bounds=(0.05, 10.0))
        
        ema_sweep = SelectiveAADEvaluator.sweep_ema_parameters(ad_calib_margins, ad_calib_labels, alpha_candidates=[0.75, 0.85])
        best_alpha = ema_sweep["best_alpha"]
        
        hyst_sweep = SelectiveAADEvaluator.sweep_hysteresis_parameters(
            ad_calib_margins, ad_calib_labels, alpha=best_alpha,
            switch_candidates=[0.25, 0.35, 0.50], confirm_candidates=[1, 2], step_sec=args.step_sec
        )
        best_hyst = hyst_sweep["best_config"]
        
        t3_dec_list = []
        t3_gains_attended = []
        for m_seq, eeg_trial in zip(ad_test_margins, ad_test_raw_eeg):
            gate_t3 = StickyHysteresisGate(
                alpha=best_alpha,
                threshold_switch=best_hyst["threshold_switch"],
                threshold_maintain=best_hyst["threshold_maintain"],
                n_confirm=best_hyst["n_confirm"],
                deadband_timeout_steps=24,
                temperature=fitted_t
            )
            trial_decs = []
            trial_gains = []
            for v, w_eeg in zip(m_seq, eeg_trial):
                sq = sq_monitor.check_eeg_window(w_eeg)
                out = gate_t3.update(v, is_artifact=not sq["is_valid"])
                trial_decs.append(out["decision"])
                trial_gains.append(out["gain_a"])
            t3_dec_list.append(np.array(trial_decs))
            t3_gains_attended.append(np.array(trial_gains))
            
        flat_t3 = np.concatenate(t3_dec_list)
        m_t3 = calculate_selective_metrics(flat_t3, gt_test_str)
        t3_fsw = float(np.mean([compute_temporal_stability_metrics(d, np.where(l == 1, "A", "B"), args.step_sec)["false_switches_per_minute"] for d, l in zip(t3_dec_list, zs_test_labels)]))
        t3_useful_cov = float(np.mean(np.concatenate(t3_gains_attended) >= 0.80) * 100.0)
        
        results[target_sub] = {
            "T1_raw_zero_acc": m_t1["selective_accuracy"] * 100.0,
            "T1_raw_zero_fsw": t1_fsw,
            "T2_raw_adapt_acc": m_t2["selective_accuracy"] * 100.0,
            "T2_raw_adapt_fsw": t2_fsw,
            "raw_gain_pp": raw_gain_pp,
            "T3_sticky_adapt_acc": m_t3["selective_accuracy"] * 100.0,
            "T3_sticky_adapt_cov": t3_useful_cov,
            "T3_sticky_adapt_hold": m_t3["abstention_rate"] * 100.0,
            "T3_sticky_adapt_fsw": t3_fsw,
        }
        
        print(f"  RESULTS for {target_sub}:")
        print(f"    Tier 1: Raw Zero-Shot (5s):     {m_t1['selective_accuracy']*100:.1f}% | FalseSw: {t1_fsw:.2f}/m")
        print(f"    Tier 2: Raw 8x8 Adapted (5s):   {m_t2['selective_accuracy']*100:.1f}% | FalseSw: {t2_fsw:.2f}/m | ΔRaw: {raw_gain_pp:+.1f} pp")
        print(f"    Tier 3: 8x8 Adapt + Sticky Gate:{m_t3['selective_accuracy']*100:.1f}% | FalseSw: {t3_fsw:.2f}/m | Useful Boost: {t3_useful_cov:.1f}% (HOLD: {m_t3['abstention_rate']*100:.1f}%)")
        
        del univ_model, adapter
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        import gc
        gc.collect()

    # Grand Comparison Table
    print("\n" + "=" * 115)
    print("  GRAND SUMMARY: SUBJECT-SPECIFIC 8x8 SPATIAL ADAPTER BENCHMARK (5.0s)")
    print("=" * 115)
    print(f"  {'Subject':<8} | {'1. Raw Zero-Shot':<17} | {'2. Raw 8x8 Adapted':<22} | {'3. Adapted + Sticky Gate':<24}")
    print(f"  {'':<8} | {'Acc':<7} {'F.Sw':<8} | {'Acc':<7} {'ΔRaw':<6} {'F.Sw':<6} | {'Acc':<7} {'Boost%':<7} {'F.Sw':<7}")
    print("  " + "-" * 111)
    
    for sub, r in results.items():
        print(f"  {sub:<8} | {r['T1_raw_zero_acc']:>5.1f}% {r['T1_raw_zero_fsw']:>6.2f}/m | {r['T2_raw_adapt_acc']:>5.1f}% {r['raw_gain_pp']:>+5.1f} {r['T2_raw_adapt_fsw']:>5.2f} | {r['T3_sticky_adapt_acc']:>5.1f}% {r['T3_sticky_adapt_cov']:>6.1f}% {r['T3_sticky_adapt_fsw']:>6.2f}/m")
        
    if len(results) > 1:
        m_t1_acc = np.mean([r["T1_raw_zero_acc"] for r in results.values()])
        m_t1_fsw = np.mean([r["T1_raw_zero_fsw"] for r in results.values()])
        m_t2_acc = np.mean([r["T2_raw_adapt_acc"] for r in results.values()])
        m_t2_gain = np.mean([r["raw_gain_pp"] for r in results.values()])
        m_t2_fsw = np.mean([r["T2_raw_adapt_fsw"] for r in results.values()])
        m_t3_acc = np.mean([r["T3_sticky_adapt_acc"] for r in results.values()])
        m_t3_cov = np.mean([r["T3_sticky_adapt_cov"] for r in results.values()])
        m_t3_fsw = np.mean([r["T3_sticky_adapt_fsw"] for r in results.values()])
        
        print("  " + "-" * 111)
        print(f"  {'AVERAGE':<8} | {m_t1_acc:>5.1f}% {m_t1_fsw:>6.2f}/m | {m_t2_acc:>5.1f}% {m_t2_gain:>+5.1f} {m_t2_fsw:>5.2f} | {m_t3_acc:>5.1f}% {m_t3_cov:>6.1f}% {m_t3_fsw:>6.2f}/m")
    print("=" * 115)
    
    out_file = Path("/kaggle/working/spatial_adapter_benchmark_results.json") if Path("/kaggle/working").exists() else Path("spatial_adapter_benchmark_results.json")
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[SAVED] Benchmark metrics saved to: {out_file}")

if __name__ == "__main__":
    main()
