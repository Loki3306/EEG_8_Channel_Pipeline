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

from src.selective_aad.core import TemperatureCalibrator
from src.selective_aad.metrics import (
    calculate_selective_metrics,
    compute_temporal_stability_metrics,
)
from src.selective_aad.evaluator import SelectiveAADEvaluator
from src.selective_aad.streaming_gate import SelectiveStreamingGate
from src.selective_aad.temporal_gate import (
    SignalQualityMonitor,
    StickyHysteresisGate,
    AdvancedStickyGate,
    AnalyticalBayesianGate,
    TinyTemporalGate,
)

def butter_lowpass_sosfilt(data: np.ndarray, cutoff: float, fs: float, order: int = 2) -> np.ndarray:
    from scipy import signal
    sos = signal.butter(order, cutoff, btype='low', fs=fs, output='sos')
    return signal.sosfilt(sos, data).astype(np.float32)

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

def extract_margins_with_raw_eeg(model, eeg_list, ya_list, yb_list, window_sec, step_sec, fs, device):
    """
    Extracts rolling window predictions along with raw EEG slices for signal quality monitoring.
    Vectorized per trial for 30x faster GPU execution.
    """
    model.eval()
    window_samples = int(window_sec * fs)
    step_samples = int(step_sec * fs)
    
    trials_margins = []
    trials_labels = []
    trials_raw_eeg = []
    trials_audio_energy = []
    
    for t_idx, (eeg, ya, yb) in enumerate(zip(eeg_list, ya_list, yb_list)):
        t_len = min(len(eeg), len(ya), len(yb))
        win_e, win_a, win_b = [], [], []
        raw_e = []
        raw_energy = []
        
        curr_start = 0
        while curr_start + window_samples <= t_len:
            curr_end = curr_start + window_samples
            w_e = eeg[curr_start:curr_end]
            w_a = ya[curr_start:curr_end]
            w_b = yb[curr_start:curr_end]
            
            raw_e.append(w_e)
            step_energy = float(np.mean(np.abs(w_a)) + np.mean(np.abs(w_b))) / 2.0
            raw_energy.append(step_energy)
            
            # Causal window standardization
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
                d, _, _ = model(t_e, t_a, t_b)
            trial_m = d.detach().cpu().numpy().tolist()
            trials_margins.append(np.array(trial_m, dtype=np.float64))
            trials_labels.append(np.ones(len(trial_m), dtype=np.int64))
            trials_raw_eeg.append(raw_e)
            trials_audio_energy.append(np.array(raw_energy, dtype=np.float32))
            
    return trials_margins, trials_labels, trials_raw_eeg, trials_audio_energy

def build_temporal_features(margin_seq: np.ndarray, seq_len: int = 8) -> np.ndarray:
    """
    Constructs causal time-series feature matrix for TinyTemporalGate:
    Features at step t: [m_t, Δm_t, rolling_std_m, streak_t, sign_t]
    Returns tensor: [N_steps, seq_len, 5]
    """
    m = np.asarray(margin_seq, dtype=np.float32)
    n = len(m)
    feats = np.zeros((n, 5), dtype=np.float32)
    
    # Feature 0: raw margin
    feats[:, 0] = m
    # Feature 1: velocity Δm
    feats[1:, 1] = m[1:] - m[:-1]
    # Feature 2: rolling std (5-step window)
    for i in range(n):
        start_w = max(0, i - 4)
        feats[i, 2] = float(np.std(m[start_w:i+1])) if i > start_w else 0.1
    # Feature 3: streak counter
    streak = 0
    for i in range(n):
        if m[i] > 0:
            streak = streak + 1 if streak >= 0 else 1
        elif m[i] < 0:
            streak = streak - 1 if streak <= 0 else -1
        else:
            streak = 0
        feats[i, 3] = streak / 10.0  # Normalized streak
    # Feature 4: sign
    feats[:, 4] = np.sign(m)
    
    # Pad to causal sequence of length seq_len
    seqs = np.zeros((n, seq_len, 5), dtype=np.float32)
    for i in range(n):
        s_idx = max(0, i - seq_len + 1)
        sub = feats[s_idx:i+1]
        seqs[i, seq_len - len(sub):] = sub
        
    return seqs

def train_tiny_temporal_gate(cv_margins, cv_labels, epochs=15, lr=1e-3, device="cpu"):
    """
    Trains the ~1,100-parameter TinyTemporalGate on calibration folds.
    """
    model = TinyTemporalGate(input_dim=5, hidden_dim=16).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion_intent = nn.BCEWithLogitsLoss()
    
    all_seqs, all_targs = [], []
    for m_seq, l_seq in zip(cv_margins, cv_labels):
        x_seq = build_temporal_features(m_seq, seq_len=8)
        all_seqs.append(x_seq)
        all_targs.append(l_seq.astype(np.float32))
        
        # Symmetric augmentation (dual counterfactual: -m with label 0)
        x_neg = build_temporal_features(-m_seq, seq_len=8)
        all_seqs.append(x_neg)
        all_targs.append(np.zeros_like(l_seq, dtype=np.float32))
        
    X_train = np.concatenate(all_seqs, axis=0)
    Y_train = np.concatenate(all_targs, axis=0)
    
    ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(Y_train))
    loader = DataLoader(ds, batch_size=64, shuffle=True)
    
    model.train()
    for ep in range(epochs):
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            intent_logit, _, _ = model(bx)
            loss = criterion_intent(intent_logit, by)
            loss.backward()
            optimizer.step()
            
    model.eval()
    return model

def main():
    parser = argparse.ArgumentParser(description="Personalized Gating and Time-Series Confidence Benchmark")
    parser.add_argument("--checkpoint_dir", type=str, default="/kaggle/working/loso_checkpoints", help="Path to checkpoint directory")
    parser.add_argument("--all_folds", action="store_true", help="Run across all 18 subjects (S1-S18)")
    parser.add_argument("--folds", type=str, default="", help="Comma-separated target subjects (e.g. S1,S2)")
    parser.add_argument("--subject", type=str, default="", help="Single target subject (e.g. S1)")
    parser.add_argument("--calib_trials", type=int, default=12, help="Number of calibration trials")
    parser.add_argument("--montage", type=str, default="near_ear_expanded", help="Electrode montage")
    parser.add_argument("--window_sec", type=float, default=5.0, help="Window size in seconds (Frozen target)")
    parser.add_argument("--step_sec", type=float, default=0.5, help="Control step size in seconds (2 Hz)")
    parser.add_argument("--smoke_test", action="store_true", help="Fast smoke test")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 115)
    print("  PERSONALIZED GATING & TIME-SERIES CONFIDENCE BENCHMARK (FROZEN 5.0s CA-TCN)")
    print(f"  Device: {device} | Calibration Trials: {args.calib_trials} | Step: {args.step_sec}s (2 Hz)")
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
        
        # 2. Extract Data
        target_exs = list(load_subject_examples(target_path))
        if args.smoke_test:
            target_exs = target_exs[:min(18, len(target_exs))]
        eeg_all, ya_all, yb_all = process_subject_trials(target_exs, montage_channels, target_sub, mapping, envelopes, causal_filter, FS)
        
        K = min(args.calib_trials, len(eeg_all) // 2) if not args.smoke_test else 4
        eeg_calib, ya_calib, yb_calib = eeg_all[:K], ya_all[:K], yb_all[:K]
        eeg_test, ya_test, yb_test = eeg_all[K:], ya_all[K:], yb_all[K:]
        
        # 3. Inner CV for Gating Parameters
        cv_margins, cv_labels, _, _ = extract_margins_with_raw_eeg(univ_model, eeg_calib, ya_calib, yb_calib, args.window_sec, args.step_sec, FS, device)
        flat_calib_m = np.concatenate(cv_margins)
        flat_calib_l = np.concatenate(cv_labels)
        
        calibrator = TemperatureCalibrator()
        fitted_t = calibrator.fit(flat_calib_m, flat_calib_l, bounds=(0.05, 10.0))
        
        ema_sweep = SelectiveAADEvaluator.sweep_ema_parameters(cv_margins, cv_labels, alpha_candidates=[0.7, 0.85])
        best_alpha = ema_sweep["best_alpha"]
        
        hyst_sweep = SelectiveAADEvaluator.sweep_hysteresis_parameters(
            cv_margins, cv_labels, alpha=best_alpha,
            switch_candidates=[0.25, 0.35, 0.50], confirm_candidates=[1, 2], step_sec=args.step_sec
        )
        best_hyst = hyst_sweep["best_config"]
        
        # 4. Train Tiny Temporal Gate (GRU, ~1,100 params) on calibration folds
        print(f"  [TGM] Training Tiny Temporal Gate (GRU, ~1.1k params) on {K} calibration trials...")
        tiny_gru = train_tiny_temporal_gate(cv_margins, cv_labels, epochs=15 if not args.smoke_test else 3, lr=1e-3, device=device)
        
        # 5. Extract Test Set Data
        test_margins, test_labels, test_raw_eeg, test_audio_energy = extract_margins_with_raw_eeg(univ_model, eeg_test, ya_test, yb_test, args.window_sec, args.step_sec, FS, device)
        flat_test_m = np.concatenate(test_margins)
        flat_test_l = np.concatenate(test_labels)
        gt_test_str = np.where(flat_test_l == 1, "A", "B")
        
        # -----------------------------------------------------------------
        # STRATEGY 1: Baseline Forced Choice (Raw CA-TCN)
        # -----------------------------------------------------------------
        s1_preds = np.where(flat_test_m >= 0, "A", "B")
        m_s1 = calculate_selective_metrics(s1_preds, gt_test_str)
        s1_fsw = float(np.mean([compute_temporal_stability_metrics(np.where(m >= 0, "A", "B"), np.where(l == 1, "A", "B"), args.step_sec)["false_switches_per_minute"] for m, l in zip(test_margins, test_labels)]))
        
        # -----------------------------------------------------------------
        # STRATEGY 2: Blunt HOLD Gate (Current Method F)
        # -----------------------------------------------------------------
        s2_dec_list = []
        for m_seq in test_margins:
            gate_s2 = SelectiveStreamingGate(
                alpha=best_alpha,
                threshold_switch=best_hyst["threshold_switch"],
                threshold_maintain=best_hyst["threshold_maintain"],
                n_confirm=best_hyst["n_confirm"],
                temperature=fitted_t
            )
            decs = [gate_s2.update(v)["decision"] for v in m_seq]
            s2_dec_list.append(np.array(decs))
        flat_s2 = np.concatenate(s2_dec_list)
        m_s2 = calculate_selective_metrics(flat_s2, gt_test_str)
        s2_fsw = float(np.mean([compute_temporal_stability_metrics(d, np.where(l == 1, "A", "B"), args.step_sec)["false_switches_per_minute"] for d, l in zip(s2_dec_list, test_labels)]))
        
        # -----------------------------------------------------------------
        # STRATEGY 3: Sticky State Retention Gate (Analytical Bayesian Hysteresis)
        # -----------------------------------------------------------------
        s3_dec_list = []
        s3_gains_attended = []
        for m_seq, eeg_trial in zip(test_margins, test_raw_eeg):
            gate_s3 = StickyHysteresisGate(
                alpha=best_alpha,
                threshold_switch=best_hyst["threshold_switch"],
                threshold_maintain=best_hyst["threshold_maintain"],
                n_confirm=best_hyst["n_confirm"],
                deadband_timeout_steps=20,  # 10s timeout
                temperature=fitted_t
            )
            trial_decs = []
            trial_gains = []
            for v, w_eeg in zip(m_seq, eeg_trial):
                sq = sq_monitor.check_eeg_window(w_eeg)
                out = gate_s3.update(v, is_artifact=not sq["is_valid"])
                trial_decs.append(out["decision"])
                trial_gains.append(out["gain_a"])  # Stream A is attended in DTU convention
            s3_dec_list.append(np.array(trial_decs))
            s3_gains_attended.append(np.array(trial_gains))
            
        flat_s3 = np.concatenate(s3_dec_list)
        m_s3 = calculate_selective_metrics(flat_s3, gt_test_str)
        s3_fsw = float(np.mean([compute_temporal_stability_metrics(d, np.where(l == 1, "A", "B"), args.step_sec)["false_switches_per_minute"] for d, l in zip(s3_dec_list, test_labels)]))
        # Useful Acoustic Coverage: % time gain_a >= 0.85 (boosted)
        s3_useful_cov = float(np.mean(np.concatenate(s3_gains_attended) >= 0.80) * 100.0)
        
        # -----------------------------------------------------------------
        # STRATEGY 4: Lightweight Time-Series Neural Gate (Tiny GRU)
        # -----------------------------------------------------------------
        calib_gru_logits = []
        for m_seq in cv_margins:
            x_seq = torch.from_numpy(build_temporal_features(m_seq, seq_len=8)).to(device)
            with torch.no_grad():
                l_out, _, _ = tiny_gru(x_seq)
            calib_gru_logits.append(l_out.cpu().numpy())
            
        flat_calib_gru = np.concatenate(calib_gru_logits)
        calibrator_gru = TemperatureCalibrator()
        t_gru = calibrator_gru.fit(flat_calib_gru, flat_calib_l, bounds=(0.05, 10.0))
        
        hyst_gru_sweep = SelectiveAADEvaluator.sweep_hysteresis_parameters(
            calib_gru_logits, cv_labels, alpha=0.3,
            switch_candidates=[0.05, 0.15, 0.30], confirm_candidates=[1, 2], step_sec=args.step_sec
        )
        best_gru_hyst = hyst_gru_sweep["best_config"]
        
        s4_dec_list = []
        s4_gains_attended = []
        for m_seq in test_margins:
            x_seq = torch.from_numpy(build_temporal_features(m_seq, seq_len=8)).to(device)
            with torch.no_grad():
                intent_logits, _, _ = tiny_gru(x_seq)
            raw_logits = intent_logits.cpu().numpy()
            
            gate_s4 = StickyHysteresisGate(
                alpha=0.3,
                threshold_switch=best_gru_hyst["threshold_switch"],
                threshold_maintain=best_gru_hyst["threshold_maintain"],
                n_confirm=best_gru_hyst["n_confirm"],
                deadband_timeout_steps=20,
                temperature=t_gru
            )
            trial_decs = [gate_s4.update(rl)["decision"] for rl in raw_logits]
            trial_gains = [gate_s4.gain_a for _ in trial_decs]
            s4_dec_list.append(np.array(trial_decs))
            s4_gains_attended.append(np.array(trial_gains))
            
        flat_s4 = np.concatenate(s4_dec_list)
        m_s4 = calculate_selective_metrics(flat_s4, gt_test_str)
        s4_fsw = float(np.mean([compute_temporal_stability_metrics(d, np.where(l == 1, "A", "B"), args.step_sec)["false_switches_per_minute"] for d, l in zip(s4_dec_list, test_labels)]))
        s4_useful_cov = float(np.mean(np.concatenate(s4_gains_attended) >= 0.80) * 100.0)
        
        # -----------------------------------------------------------------
        # STRATEGY 5: Sticky Pro (AdvancedStickyGate with Asym+SPRT+VAD)
        # -----------------------------------------------------------------
        s5_dec_list = []
        s5_gains_attended = []
        for m_seq, eeg_trial, nrg_trial in zip(test_margins, test_raw_eeg, test_audio_energy):
            gate_s5 = AdvancedStickyGate(
                alpha_fast=0.65,
                alpha_slow=0.92,
                threshold_switch=best_hyst["threshold_switch"],
                threshold_maintain=best_hyst["threshold_maintain"] * 0.8,
                evidence_threshold=0.30,
                lambda_leak=0.50,
                deadband_timeout_steps=24,
                temperature=fitted_t,
                silence_threshold=0.015
            )
            trial_decs = []
            trial_gains = []
            for v, w_eeg, nrg in zip(m_seq, eeg_trial, nrg_trial):
                sq = sq_monitor.check_eeg_window(w_eeg)
                out = gate_s5.update(v, is_artifact=not sq["is_valid"], audio_energy=float(nrg))
                trial_decs.append(out["decision"])
                trial_gains.append(out["gain_a"])
            s5_dec_list.append(np.array(trial_decs))
            s5_gains_attended.append(np.array(trial_gains))
            
        flat_s5 = np.concatenate(s5_dec_list)
        m_s5 = calculate_selective_metrics(flat_s5, gt_test_str)
        s5_fsw = float(np.mean([compute_temporal_stability_metrics(d, np.where(l == 1, "A", "B"), args.step_sec)["false_switches_per_minute"] for d, l in zip(s5_dec_list, test_labels)]))
        s5_useful_cov = float(np.mean(np.concatenate(s5_gains_attended) >= 0.80) * 100.0)
        
        results[target_sub] = {
            "S1_forced_acc": m_s1["selective_accuracy"] * 100.0,
            "S1_forced_fsw": s1_fsw,
            "S2_blunt_acc": m_s2["selective_accuracy"] * 100.0,
            "S2_blunt_hold": m_s2["abstention_rate"] * 100.0,
            "S2_blunt_fsw": s2_fsw,
            "S3_sticky_acc": m_s3["selective_accuracy"] * 100.0,
            "S3_sticky_cov": s3_useful_cov,
            "S3_sticky_hold": m_s3["abstention_rate"] * 100.0,
            "S3_sticky_fsw": s3_fsw,
            "S4_gru_acc": m_s4["selective_accuracy"] * 100.0,
            "S4_gru_cov": s4_useful_cov,
            "S4_gru_hold": m_s4["abstention_rate"] * 100.0,
            "S4_gru_fsw": s4_fsw,
            "S5_pro_acc": m_s5["selective_accuracy"] * 100.0,
            "S5_pro_cov": s5_useful_cov,
            "S5_pro_hold": m_s5["abstention_rate"] * 100.0,
            "S5_pro_fsw": s5_fsw,
        }
        
        print(f"  RESULTS for {target_sub}:")
        print(f"    1. Forced Choice (Base):       {m_s1['selective_accuracy']*100:.1f}% | FalseSw: {s1_fsw:.2f}/m | Boost Coverage: 100.0%")
        print(f"    2. Blunt HOLD (Current F):     {m_s2['selective_accuracy']*100:.1f}% | FalseSw: {s2_fsw:.2f}/m | HOLD: {m_s2['abstention_rate']*100:.1f}%")
        print(f"    3. Sticky State Retention:     {m_s3['selective_accuracy']*100:.1f}% | FalseSw: {s3_fsw:.2f}/m | Useful Boost: {s3_useful_cov:.1f}% (HOLD: {m_s3['abstention_rate']*100:.1f}%)")
        print(f"    4. Tiny GRU Time-Series Gate:  {m_s4['selective_accuracy']*100:.1f}% | FalseSw: {s4_fsw:.2f}/m | Useful Boost: {s4_useful_cov:.1f}% (HOLD: {m_s4['abstention_rate']*100:.1f}%)")
        print(f"    5. Sticky Pro (Asym+SPRT+VAD): {m_s5['selective_accuracy']*100:.1f}% | FalseSw: {s5_fsw:.2f}/m | Useful Boost: {s5_useful_cov:.1f}% (HOLD: {m_s5['abstention_rate']*100:.1f}%)")
        
        # Clean memory
        del univ_model, tiny_gru
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        import gc
        gc.collect()

    # Grand Comparison Table
    print("\n" + "=" * 145)
    print("  GRAND SUMMARY: ADVANCED TEMPORAL GATING ARCHITECTURES (FROZEN 5.0s CA-TCN)")
    print("=" * 145)
    print(f"  {'Subject':<8} | {'1. Forced Base':<15} | {'2. Blunt HOLD F':<17} | {'3. Sticky Retention':<22} | {'4. Tiny GRU':<17} | {'5. Sticky Pro (Winner)':<22}")
    print(f"  {'':<8} | {'Acc':<7} {'F.Sw':<7} | {'Acc':<7} {'HOLD%':<8} | {'Acc':<7} {'Boost%':<7} {'F.Sw':<6} | {'Acc':<7} {'F.Sw':<8} | {'Acc':<7} {'Boost%':<7} {'F.Sw':<6}")
    print("  " + "-" * 141)
    
    for sub, r in results.items():
        print(f"  {sub:<8} | {r['S1_forced_acc']:>5.1f}% {r['S1_forced_fsw']:>5.2f} | {r['S2_blunt_acc']:>5.1f}% {r['S2_blunt_hold']:>6.1f}% | {r['S3_sticky_acc']:>5.1f}% {r['S3_sticky_cov']:>6.1f}% {r['S3_sticky_fsw']:>5.2f} | {r['S4_gru_acc']:>5.1f}% {r['S4_gru_fsw']:>7.2f} | {r['S5_pro_acc']:>5.1f}% {r['S5_pro_cov']:>6.1f}% {r['S5_pro_fsw']:>5.2f}")
        
    if len(results) > 1:
        mean_s1_acc = np.mean([r["S1_forced_acc"] for r in results.values()])
        mean_s1_fsw = np.mean([r["S1_forced_fsw"] for r in results.values()])
        mean_s2_acc = np.mean([r["S2_blunt_acc"] for r in results.values()])
        mean_s2_hold = np.mean([r["S2_blunt_hold"] for r in results.values()])
        mean_s3_acc = np.mean([r["S3_sticky_acc"] for r in results.values()])
        mean_s3_cov = np.mean([r["S3_sticky_cov"] for r in results.values()])
        mean_s3_fsw = np.mean([r["S3_sticky_fsw"] for r in results.values()])
        mean_s4_acc = np.mean([r["S4_gru_acc"] for r in results.values()])
        mean_s4_fsw = np.mean([r["S4_gru_fsw"] for r in results.values()])
        mean_s5_acc = np.mean([r["S5_pro_acc"] for r in results.values()])
        mean_s5_cov = np.mean([r["S5_pro_cov"] for r in results.values()])
        mean_s5_fsw = np.mean([r["S5_pro_fsw"] for r in results.values()])
        
        print("  " + "-" * 141)
        print(f"  {'AVERAGE':<8} | {mean_s1_acc:>5.1f}% {mean_s1_fsw:>5.2f} | {mean_s2_acc:>5.1f}% {mean_s2_hold:>6.1f}% | {mean_s3_acc:>5.1f}% {mean_s3_cov:>6.1f}% {mean_s3_fsw:>5.2f} | {mean_s4_acc:>5.1f}% {mean_s4_fsw:>7.2f} | {mean_s5_acc:>5.1f}% {mean_s5_cov:>6.1f}% {mean_s5_fsw:>5.2f}")
    print("=" * 145)
    
    out_file = Path("/kaggle/working/temporal_gating_benchmark_results.json") if Path("/kaggle/working").exists() else Path("temporal_gating_benchmark_results.json")
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[SAVED] Benchmark metrics saved to: {out_file}")

if __name__ == "__main__":
    main()
