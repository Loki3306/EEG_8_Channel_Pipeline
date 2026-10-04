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
from src.streaming.causal_filters import StreamingCausalEEGFilter
from training.montages import MONTAGES
from training.train_matchnet_wavlm import get_mapping_data, prepare_dataset, FS
from baselines.ridge_aad import load_subject_examples, subject_files

from src.selective_aad.core import (
    TemperatureCalibrator,
)
from src.selective_aad.metrics import (
    calculate_selective_metrics,
    compute_temporal_stability_metrics,
)
from src.selective_aad.evaluator import SelectiveAADEvaluator
from src.selective_aad.streaming_gate import SelectiveStreamingGate

def butter_lowpass_sosfilt(data: np.ndarray, cutoff: float, fs: float, order: int = 2) -> np.ndarray:
    from scipy import signal
    sos = signal.butter(order, cutoff, btype='low', fs=fs, output='sos')
    return signal.sosfilt(sos, data).astype(np.float32)

def process_subject_trials(examples, montage_channels, sub_id, mapping, envelopes, causal_filter, fs):
    """
    Extracts, causal-filters, and standardizes continuous trials for a subject.
    Returns: list of eeg arrays, list of ya arrays, list of yb arrays.
    """
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
        
        ya_c = butter_lowpass_sosfilt(ya_clean[idx][:min_len], 8.0, fs, order=2).astype(np.float32)
        yb_c = butter_lowpass_sosfilt(yb_clean[idx][:min_len], 8.0, fs, order=2).astype(np.float32)
        ya_c = (ya_c - np.mean(ya_c)) / (np.std(ya_c) + 1e-8)
        yb_c = (yb_c - np.mean(yb_c)) / (np.std(yb_c) + 1e-8)
        
        eeg_out.append(eeg_c)
        ya_out.append(ya_c)
        yb_out.append(yb_c)
        
    return eeg_out, ya_out, yb_out

def chunk_trials(eeg_list, ya_list, yb_list, win_sec, hop_sec, fs):
    win_samples = int(win_sec * fs)
    hop_samples = int(hop_sec * fs)
    x_chunks, ya_chunks, yb_chunks = [], [], []
    
    for eeg, ya, yb in zip(eeg_list, ya_list, yb_list):
        t_len = min(len(eeg), len(ya), len(yb))
        x_t = eeg.T
        ya_t = np.expand_dims(ya, axis=0)
        yb_t = np.expand_dims(yb, axis=0)
        
        start = 0
        while start + win_samples <= t_len:
            end = start + win_samples
            x_chunks.append(x_t[:, start:end])
            ya_chunks.append(ya_t[:, start:end])
            yb_chunks.append(yb_t[:, start:end])
            start += hop_samples
            
    if len(x_chunks) == 0:
        return np.zeros((0, x_t.shape[0], win_samples)), np.zeros((0, 1, win_samples)), np.zeros((0, 1, win_samples))
    return np.stack(x_chunks, axis=0), np.stack(ya_chunks, axis=0), np.stack(yb_chunks, axis=0)

def train_backbone(train_paths, montage_channels, mapping, envelopes, causal_filter, args, device):
    """
    Trains the 17-subject Universal CA-TCN backbone.
    """
    print(f"\n  [STAGE 1] Pre-training Universal CA-TCN Backbone on {len(train_paths)} subjects...")
    t0 = time.time()
    all_x, all_ya, all_yb = [], [], []
    
    for p in train_paths:
        sub_id = p.stem.split("_")[0]
        exs = list(load_subject_examples(p))
        if args.smoke_test:
            exs = exs[:6]
        eeg_l, ya_l, yb_l = process_subject_trials(exs, montage_channels, sub_id, mapping, envelopes, causal_filter, FS)
        cx, cya, cyb = chunk_trials(eeg_l, ya_l, yb_l, args.window_sec, args.hop_sec, FS)
        all_x.append(cx)
        all_ya.append(cya)
        all_yb.append(cyb)
        
    X_tr = np.concatenate(all_x, axis=0)
    YA_tr = np.concatenate(all_ya, axis=0)
    YB_tr = np.concatenate(all_yb, axis=0)
    print(f"  [DATA] Prepared {len(X_tr)} universal training chunks in {time.time() - t0:.1f}s.")
    
    model = CATCNDirectDecoder(eeg_channels=len(montage_channels), audio_channels=1, hidden_dim=args.hidden_dim, max_lag_samples=8).to(device)
    train_ds = TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(YA_tr), torch.from_numpy(YB_tr))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=2 if os.name != 'nt' else 0)
    
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    epochs = args.epochs_pretrain if not args.smoke_test else 2
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    scaler = torch.amp.GradScaler('cuda' if torch.cuda.is_available() else 'cpu')
    
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, n_batches = 0.0, 0
        for bx, bya, byb in train_loader:
            bx, bya, byb = bx.to(device, non_blocking=True), bya.to(device, non_blocking=True), byb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda' if torch.cuda.is_available() else 'cpu'):
                delta, (la, lb), _ = model(bx, bya, byb)
                loss = torch.clamp(0.5 - (la - lb), min=0.0).mean()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()
            n_batches += 1
        scheduler.step()
        if epoch % 2 == 0 or epoch == epochs:
            print(f"    * Backbone Epoch {epoch:02d}/{epochs:02d} | Loss: {total_loss/max(1, n_batches):.4f} | LR: {scheduler.get_last_lr()[0]:.2e}")
            
    print(f"  [BACKBONE] Universal training finished in {(time.time() - t0)/60.0:.1f} minutes.")
    return model

def adapt_spatial_model(base_model, calib_loader, epochs, lr, device):
    """
    Trains only the spatial & BN layers of the CA-TCN model.
    Locks frozen temporal TCN blocks and classification head in eval mode
    to preserve universal feature representations and prevent BatchNorm drift.
    """
    spatial_model = deepcopy(base_model)
    
    # Freeze temporal & classification head
    for p in spatial_model.audio_encoder.parameters():
        p.requires_grad = False
    for p in spatial_model.eeg_encoder.blocks.parameters():
        p.requires_grad = False
    for p in spatial_model.classifier_head.parameters():
        p.requires_grad = False
        
    # Unfreeze spatial projection and spatial batchnorm
    for p in spatial_model.eeg_encoder.spatial_proj.parameters():
        p.requires_grad = True
    for p in spatial_model.eeg_encoder.bn_spatial.parameters():
        p.requires_grad = True
        
    trainable_params = [p for p in spatial_model.parameters() if p.requires_grad]
    opt_spatial = optim.AdamW(trainable_params, lr=lr, weight_decay=1e-4)
    sched_spatial = optim.lr_scheduler.CosineAnnealingLR(opt_spatial, T_max=epochs, eta_min=1e-5)
    
    spatial_model.train()
    # Explicitly lock frozen submodules into eval mode
    spatial_model.audio_encoder.eval()
    for block in spatial_model.eeg_encoder.blocks:
        block.eval()
    spatial_model.classifier_head.eval()
    # Ensure spatial parameters are in training mode
    spatial_model.eeg_encoder.spatial_proj.train()
    spatial_model.eeg_encoder.bn_spatial.train()
    
    for ep in range(1, epochs + 1):
        for bx, bya, byb in calib_loader:
            bx, bya, byb = bx.to(device), bya.to(device), byb.to(device)
            opt_spatial.zero_grad(set_to_none=True)
            delta, (la, lb), _ = spatial_model(bx, bya, byb)
            loss = torch.clamp(0.5 - (la - lb), min=0.0).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            opt_spatial.step()
        sched_spatial.step()
        
    return spatial_model

def extract_margins(model, eeg_list, ya_list, yb_list, window_sec, step_sec, fs, device):
    """
    Extracts rolling window predictions for streaming evaluation.
    Vectorized per trial for 30x faster GPU execution.
    """
    model.eval()
    window_samples = int(window_sec * fs)
    step_samples = int(step_sec * fs)
    
    trials_margins = []
    trials_labels = []
    
    for t_idx, (eeg, ya, yb) in enumerate(zip(eeg_list, ya_list, yb_list)):
        t_len = min(len(eeg), len(ya), len(yb))
        win_e, win_a, win_b = [], [], []
        
        curr_start = 0
        while curr_start + window_samples <= t_len:
            curr_end = curr_start + window_samples
            w_e = eeg[curr_start:curr_end]
            w_a = ya[curr_start:curr_end]
            w_b = yb[curr_start:curr_end]
            
            # Causal window standardization (matching streaming ring buffer)
            w_e = (w_e - np.mean(w_e, axis=0, keepdims=True)) / (np.std(w_e, axis=0, keepdims=True) + 1e-8)
            w_a = (w_a - np.mean(w_a)) / (np.std(w_a) + 1e-8)
            w_b = (w_b - np.mean(w_b)) / (np.std(w_b) + 1e-8)
            
            win_e.append(w_e.T)
            win_a.append(np.expand_dims(w_a, axis=0))
            win_b.append(np.expand_dims(w_b, axis=0))
            curr_start += step_samples
            
        if win_e:
            t_e = torch.from_numpy(np.stack(win_e)).float().to(device)
            t_a = torch.from_numpy(np.stack(win_a)).float().to(device)
            t_b = torch.from_numpy(np.stack(win_b)).float().to(device)
            with torch.no_grad():
                d, _, _ = model(t_e, t_a, t_b)
            trial_m = d.detach().cpu().numpy().tolist()
            trials_margins.append(np.array(trial_m, dtype=np.float64))
            # In DTU preprocessed convention, wavA is attended stream (ground truth = 1)
            trials_labels.append(np.ones(len(trial_m), dtype=np.int64))
            
    return trials_margins, trials_labels

def main():
    parser = argparse.ArgumentParser(description="Selective Few-Shot Adaptation Benchmark")
    parser.add_argument("--all_folds", action="store_true", help="Run all 18 subjects (S1-S18)")
    parser.add_argument("--folds", type=str, default="", help="Comma-separated target subjects (e.g. S1,S2)")
    parser.add_argument("--subject", type=str, default="", help="Single target subject (e.g. S1)")
    parser.add_argument("--calib_trials", type=int, default=12, help="Number of calibration trials")
    parser.add_argument("--montage", type=str, default="near_ear_expanded", help="Electrode montage")
    parser.add_argument("--window_sec", type=float, default=5.0, help="Window size in seconds")
    parser.add_argument("--hop_sec", type=float, default=2.5, help="Training hop size in seconds")
    parser.add_argument("--step_sec", type=float, default=0.5, help="Inference step size in seconds")
    parser.add_argument("--epochs_pretrain", type=int, default=8, help="Backbone epochs")
    parser.add_argument("--epochs_calib", type=int, default=10, help="Few-shot epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Backbone LR")
    parser.add_argument("--lr_calib_spatial", type=float, default=2e-4, help="Spatial adaptation LR")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--hidden_dim", type=int, default=64, help="Hidden dimension")
    parser.add_argument("--smoke_test", action="store_true", help="Fast smoke test")
    parser.add_argument("--force_retrain", action="store_true", help="Force retrain universal backbone")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 105)
    print(f"  SELECTIVE FEW-SHOT ADAPTATION BENCHMARK (CROSS-VALIDATED GATING)")
    print(f"  Device: {device} | Calibration Trials: {args.calib_trials}")
    print("=" * 105)
    
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
    
    grand_results = {}
    
    for target_sub in target_subs:
        print(f"\n==========================================================================")
        print(f"  TARGET SUBJECT: {target_sub}")
        print(f"==========================================================================")
        
        target_path = next((p for p in all_paths if p.stem.split("_")[0] == target_sub), None)
        if not target_path:
            print(f"[WARNING] Skipping {target_sub} (not found).")
            continue
            
        train_paths = [p for p in all_paths if p.stem.split("_")[0] != target_sub]
        
        # 1. Universal Backbone (Check existing checkpoints first)
        backbone_candidates = [
            Path(f"/kaggle/working/loso_checkpoints/catcn_loso_{target_sub}.pt"),
            Path(f"/kaggle/working/checkpoints/catcn_loso_{target_sub}.pt"),
            Path(f"/kaggle/working/checkpoints/catcn_univ_heldout_{target_sub}.pt"),
            Path(f"checkpoints/loso/catcn_loso_{target_sub}.pt"),
            Path(f"checkpoints/adaptation/catcn_univ_heldout_{target_sub}.pt"),
        ]
        found_ckpt = next((p for p in backbone_candidates if p.exists()), None)
        
        if found_ckpt and not args.force_retrain:
            print(f"  [CHECKPOINT] Reusing pre-trained universal backbone from: {found_ckpt}")
            univ_model = CATCNDirectDecoder(eeg_channels=len(montage_channels), audio_channels=1, hidden_dim=args.hidden_dim, max_lag_samples=8).to(device)
            univ_model.load_state_dict(torch.load(found_ckpt, map_location=device))
        else:
            univ_model = train_backbone(train_paths, montage_channels, mapping, envelopes, causal_filter, args, device)
            save_dir = Path("/kaggle/working/checkpoints") if Path("/kaggle/working").exists() else Path("checkpoints/adaptation")
            save_dir.mkdir(parents=True, exist_ok=True)
            torch.save(univ_model.state_dict(), save_dir / f"catcn_univ_heldout_{target_sub}.pt")
            
        # 2. Extract Data
        target_exs = list(load_subject_examples(target_path))
        if args.smoke_test:
            target_exs = target_exs[:min(18, len(target_exs))]
        eeg_all, ya_all, yb_all = process_subject_trials(target_exs, montage_channels, target_sub, mapping, envelopes, causal_filter, FS)
        
        K = min(args.calib_trials, len(eeg_all) // 2) if not args.smoke_test else 4
        eeg_calib, ya_calib, yb_calib = eeg_all[:K], ya_all[:K], yb_all[:K]
        eeg_test, ya_test, yb_test = eeg_all[K:], ya_all[K:], yb_all[K:]
        
        print(f"  [DATA] Partitioned {len(eeg_calib)} Calibration Trials, {len(eeg_test)} Test Trials")
        
        # 3. Inner Cross-Validation for Unbiased Margins
        print(f"  [GATE CALIBRATION] Running 3-fold inner cross-validation on calibration trials...")
        cv_margins = []
        cv_labels = []
        
        n_folds = min(3, len(eeg_calib))
        fold_indices = np.array_split(np.arange(len(eeg_calib)), n_folds)
        
        for fold, val_idx in enumerate(fold_indices):
            tr_idx = np.setdiff1d(np.arange(len(eeg_calib)), val_idx)
            if len(tr_idx) == 0:
                tr_idx = val_idx
                
            val_eeg = [eeg_calib[i] for i in val_idx]
            val_ya = [ya_calib[i] for i in val_idx]
            val_yb = [yb_calib[i] for i in val_idx]
            
            tr_eeg = [eeg_calib[i] for i in tr_idx]
            tr_ya = [ya_calib[i] for i in tr_idx]
            tr_yb = [yb_calib[i] for i in tr_idx]
            
            # Train inner spatial model
            X_in, YA_in, YB_in = chunk_trials(tr_eeg, tr_ya, tr_yb, args.window_sec, args.hop_sec, FS)
            in_ds = TensorDataset(torch.from_numpy(X_in), torch.from_numpy(YA_in), torch.from_numpy(YB_in))
            in_loader = DataLoader(in_ds, batch_size=args.batch_size, shuffle=True)
            
            inner_epochs = args.epochs_calib if not args.smoke_test else 2
            inner_model = adapt_spatial_model(univ_model, in_loader, inner_epochs, args.lr_calib_spatial, device)
            
            # Extract unbiased margins
            vm, vl = extract_margins(inner_model, val_eeg, val_ya, val_yb, args.window_sec, args.step_sec, FS, device)
            cv_margins.extend(vm)
            cv_labels.extend(vl)
            del inner_model
            
        # Fit Selective AAD Parameters on Unbiased Margins
        flat_calib_m = np.concatenate(cv_margins)
        flat_calib_l = np.concatenate(cv_labels)
        
        calibrator = TemperatureCalibrator()
        fitted_t = calibrator.fit(flat_calib_m, flat_calib_l, bounds=(0.05, 10.0))
        
        ema_sweep = SelectiveAADEvaluator.sweep_ema_parameters(cv_margins, cv_labels, alpha_candidates=[0.5, 0.7, 0.85])
        best_alpha = ema_sweep["best_alpha"]
        
        hyst_sweep = SelectiveAADEvaluator.sweep_hysteresis_parameters(
            cv_margins, cv_labels, alpha=best_alpha,
            switch_candidates=[0.15, 0.25, 0.35, 0.50], confirm_candidates=[1, 2], step_sec=args.step_sec
        )
        best_hyst = hyst_sweep["best_config"]
        print(f"  [GATE FITTED] T: {fitted_t:.3f} | Alpha: {best_alpha} | Switch: {best_hyst['threshold_switch']:.2f} | Confirm: {best_hyst['n_confirm']}")
        
        # 4. Final Spatial Adaptation on ALL Calibration Trials
        print(f"  [FINAL ADAPT] Training final personalized model on all {len(eeg_calib)} calibration trials...")
        X_all, YA_all, YB_all = chunk_trials(eeg_calib, ya_calib, yb_calib, args.window_sec, args.hop_sec, FS)
        all_ds = TensorDataset(torch.from_numpy(X_all), torch.from_numpy(YA_all), torch.from_numpy(YB_all))
        all_loader = DataLoader(all_ds, batch_size=args.batch_size, shuffle=True)
        
        final_epochs = args.epochs_calib if not args.smoke_test else 2
        final_model = adapt_spatial_model(univ_model, all_loader, final_epochs, args.lr_calib_spatial, device)
        
        # 5. Streaming Evaluation on Sequestered Test Set
        print(f"  [INFERENCE] Running test trials through streaming pipeline...")
        
        # Zero-shot Baseline predictions
        z_margins, z_labels = extract_margins(univ_model, eeg_test, ya_test, yb_test, args.window_sec, args.step_sec, FS, device)
        flat_zm = np.concatenate(z_margins)
        flat_zl = np.concatenate(z_labels)
        gt_str = np.where(flat_zl == 1, "A", "B")
        
        z_preds = np.where(flat_zm >= 0, "A", "B")
        m_zero = calculate_selective_metrics(z_preds, gt_str)
        z_false_sw = float(np.mean([compute_temporal_stability_metrics(np.where(m>=0,"A","B"), np.where(l==1,"A","B"), args.step_sec)["false_switches_per_minute"] for m,l in zip(z_margins, z_labels)]))
        
        # Final Adapted predictions (Forced Choice)
        a_margins, a_labels = extract_margins(final_model, eeg_test, ya_test, yb_test, args.window_sec, args.step_sec, FS, device)
        flat_am = np.concatenate(a_margins)
        
        a_preds = np.where(flat_am >= 0, "A", "B")
        m_adapt = calculate_selective_metrics(a_preds, gt_str)
        a_false_sw = float(np.mean([compute_temporal_stability_metrics(np.where(m>=0,"A","B"), np.where(l==1,"A","B"), args.step_sec)["false_switches_per_minute"] for m,l in zip(a_margins, a_labels)]))
        
        # Selective Adapted predictions (Method F)
        test_f_dec_list = []
        for m_seq in a_margins:
            stream_gate = SelectiveStreamingGate(
                alpha=best_alpha,
                threshold_switch=best_hyst["threshold_switch"],
                threshold_maintain=best_hyst["threshold_maintain"],
                n_confirm=best_hyst["n_confirm"],
                temperature=fitted_t
            )
            decs = [stream_gate.update(val)["decision"] for val in m_seq]
            test_f_dec_list.append(np.array(decs))
            
        flat_f_dec = np.concatenate(test_f_dec_list)
        m_sel = calculate_selective_metrics(flat_f_dec, gt_str)
        f_false_sw = float(np.mean([compute_temporal_stability_metrics(d, np.where(l==1,"A","B"), args.step_sec)["false_switches_per_minute"] for d,l in zip(test_f_dec_list, a_labels)]))
        
        print(f"    - Zero-Shot Base Acc:     {m_zero['selective_accuracy']*100:.1f}% | False Sw: {z_false_sw:.2f}/m")
        print(f"    - Few-Shot Adapted Acc:   {m_adapt['selective_accuracy']*100:.1f}% | False Sw: {a_false_sw:.2f}/m")
        print(f"    - Selective Adapted Acc:  {m_sel['selective_accuracy']*100:.1f}% | False Sw: {f_false_sw:.2f}/m | HOLD: {m_sel['abstention_rate']*100:.1f}%")
        
        grand_results[target_sub] = {
            "zero_acc": m_zero["selective_accuracy"]*100,
            "zero_fsw": z_false_sw,
            "adapt_acc": m_adapt["selective_accuracy"]*100,
            "adapt_fsw": a_false_sw,
            "sel_acc": m_sel["selective_accuracy"]*100,
            "sel_fsw": f_false_sw,
            "sel_hold": m_sel["abstention_rate"]*100
        }
        
        # GPU Memory cleanup
        del univ_model, final_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        import gc
        gc.collect()
        
    print("\n" + "=" * 115)
    print("  GRAND SUMMARY: FEW-SHOT ADAPTATION + SELECTIVE AAD")
    print("=" * 115)
    print(f"  {'Subject':<10} | {'Zero-Shot Acc':<14} | {'Adapted Acc':<12} | {'Selective Acc':<14} | {'HOLD %':<8} | {'False Sw (Sel)'}")
    print("  " + "-" * 111)
    
    for sub, r in grand_results.items():
        print(f"  {sub:<10} | {r['zero_acc']:>13.1f}% | {r['adapt_acc']:>11.1f}% | {r['sel_acc']:>13.1f}% | {r['sel_hold']:>7.1f}% | {r['sel_fsw']:>14.2f}/m")
        
    if len(grand_results) > 1:
        mean_zero = float(np.mean([r['zero_acc'] for r in grand_results.values()]))
        mean_adapt = float(np.mean([r['adapt_acc'] for r in grand_results.values()]))
        mean_sel = float(np.mean([r['sel_acc'] for r in grand_results.values()]))
        mean_hold = float(np.mean([r['sel_hold'] for r in grand_results.values()]))
        mean_fsw = float(np.mean([r['sel_fsw'] for r in grand_results.values()]))
        print("  " + "-" * 111)
        print(f"  {'AVERAGE':<10} | {mean_zero:>13.1f}% | {mean_adapt:>11.1f}% | {mean_sel:>13.1f}% | {mean_hold:>7.1f}% | {mean_fsw:>14.2f}/m")
        print(f"  GAIN (Adapted vs Zero-Shot):    {mean_adapt - mean_zero:+5.1f}%")
        print(f"  GAIN (Selective vs Adapted):    {mean_sel - mean_adapt:+5.1f}%")
        print(f"  TOTAL GAIN (Selective vs Zero): {mean_sel - mean_zero:+5.1f}%")
    print("=" * 115)
    
    # Save results to disk
    res_path = Path("/kaggle/working/selective_adaptation_results.json") if Path("/kaggle/working").exists() else Path("selective_adaptation_results.json")
    with open(res_path, "w") as f:
        json.dump(grand_results, f, indent=2)
    print(f"\n[SAVED] Benchmark metrics saved to: {res_path}")

if __name__ == "__main__":
    main()
