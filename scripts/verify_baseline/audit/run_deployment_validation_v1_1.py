import argparse
import sys
import os
import json
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from pathlib import Path
from scipy import signal
from copy import deepcopy

REPO_ROOT = Path(__file__).resolve().parents[3]
VERIFY_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(VERIFY_ROOT) not in sys.path:
    sys.path.insert(0, str(VERIFY_ROOT))

from models.catcn import CATCNDirectDecoder
from src.streaming.causal_filters import StreamingCausalEEGFilter
from src.streaming.pipeline import StreamingAADPipeline
from src.deployment.engine import StreamingCATCNEngine
from src.deployment.decision_smoother import EMAHysteresisDecisionLayer
from baselines.ridge_aad import load_subject_examples, subject_files
from training.train_matchnet_wavlm import FS, TRAIN_WINDOW_SEC, prepare_dataset, get_mapping_data
from training.montages import MONTAGES, DTU_CHANNELS

def butter_bandpass_filtfilt(data: np.ndarray, lowcut: float, highcut: float, fs: float, order: int = 2) -> np.ndarray:
    """Original zero-phase offline filter."""
    b, a = signal.butter(order, [lowcut, highcut], btype='bandpass', fs=fs)
    return signal.filtfilt(b, a, data, axis=0)

def run_v1_1_validation(args):
    fs = FS
    montage_channels = MONTAGES[args.montage]
    n_ch = len(montage_channels)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("=" * 90)
    print("  DEPLOYMENT VALIDATION PROTOCOL v1.1")
    print(f"  Target: {args.subject} | Montage: {args.montage} ({n_ch} channels) | Device: {device}")
    print("=" * 90)
    
    # 1. Locate Subject Files
    all_paths = subject_files()
    held_out_path = None
    train_paths = []
    for p in all_paths:
        if p.stem == args.subject:
            held_out_path = p
        else:
            train_paths.append(p)
            
    is_real_data = held_out_path is not None
    
    # 2. Setup Model
    torch.manual_seed(42)
    np.random.seed(42)
    model = CATCNDirectDecoder(eeg_channels=n_ch, audio_channels=1, hidden_dim=64, max_lag_samples=8).to(device)
    
    if not args.checkpoint and Path("/kaggle/working/catcn_deployment_weights.pt").exists():
        args.checkpoint = "/kaggle/working/catcn_deployment_weights.pt"
        print(f"[MODEL] Found previously trained weights: {args.checkpoint}")

    if args.checkpoint and Path(args.checkpoint).exists():
        print(f"[MODEL] Loading trained checkpoint from: {args.checkpoint}")
        state = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(state.get("model_state_dict", state.get("state_dict", state)))
        print("[MODEL] Checkpoint loaded successfully!")
    elif is_real_data and args.train_epochs > 0:
        print(f"\n[MODEL TRAINING]: Training CA-TCN for {args.train_epochs} epoch(s) on {len(train_paths)} subjects to establish real weights...")
        mapping, envelopes = get_mapping_data("gammatone")
        win_samples = int(TRAIN_WINDOW_SEC * fs)
        hop_samples = int(2.5 * fs)
        
        X_tr_list, YA_tr_list, YB_tr_list = [], [], []
        for p in train_paths:
            sub_name = p.stem
            exs = list(load_subject_examples(p))
            X_sub, YA_sub, YB_sub = prepare_dataset(exs, montage_channels, 1.0, 6.0, sub_name, mapping, envelopes)
            YA_sub = [ya.mean(axis=0, keepdims=True).astype(np.float32) if ya.shape[0] > 1 else ya.astype(np.float32) for ya in YA_sub]
            YB_sub = [yb.mean(axis=0, keepdims=True).astype(np.float32) if yb.shape[0] > 1 else yb.astype(np.float32) for yb in YB_sub]
            
            for idx in range(len(X_sub)):
                x, ya, yb = X_sub[idx], YA_sub[idx], YB_sub[idx]
                t_len = x.shape[1]
                start = 0
                while start + win_samples <= t_len:
                    end = start + win_samples
                    X_tr_list.append(x[:, start:end])
                    YA_tr_list.append(ya[:, start:end])
                    YB_tr_list.append(yb[:, start:end])
                    start += hop_samples
                    
        train_ds = TensorDataset(
            torch.from_numpy(np.stack(X_tr_list, axis=0)),
            torch.from_numpy(np.stack(YA_tr_list, axis=0)),
            torch.from_numpy(np.stack(YB_tr_list, axis=0))
        )
        train_loader = DataLoader(train_ds, batch_size=256, shuffle=True, pin_memory=True)
        optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        
        model.train()
        for epoch in range(args.train_epochs):
            total_loss = 0.0
            for bx, bya, byb in train_loader:
                bx, bya, byb = bx.to(device), bya.to(device), byb.to(device)
                optimizer.zero_grad()
                delta, (la, lb), _ = model(bx, bya, byb)
                target = torch.ones_like(delta)
                loss = torch.clamp(0.5 - (la - lb), min=0.0).mean()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            print(f"  * Epoch {epoch+1}/{args.train_epochs} Loss: {total_loss/len(train_loader):.4f}")
        print("[MODEL TRAINING] Complete. Proceeding to deployment audit.")
    else:
        print("[MODEL] No checkpoint or training requested. Running with initialized weights.")
        
    model.eval()
    
    # 3. Load Evaluation Data
    trials_raw_eeg = []
    trials_ya = []
    trials_yb = []
    
    if is_real_data:
        print(f"\n[DATA]: Loading genuine DTU patient recording for {held_out_path.name}...")
        mapping, envelopes = get_mapping_data("gammatone")
        test_exs = list(load_subject_examples(held_out_path))
        ch_indices = montage_channels
        
        # Prepare candidate audio envelopes
        _, YA_te, YB_te = prepare_dataset(test_exs, montage_channels, 1.0, 6.0, args.subject, mapping, envelopes)
        YA_te = [ya.mean(axis=0, keepdims=True).squeeze(0).astype(np.float32) if ya.shape[0] > 1 else ya.squeeze(0).astype(np.float32) for ya in YA_te]
        YB_te = [yb.mean(axis=0, keepdims=True).squeeze(0).astype(np.float32) if yb.shape[0] > 1 else yb.squeeze(0).astype(np.float32) for yb in YB_te]
        
        for idx in range(min(args.trials, len(test_exs), len(YA_te))):
            # Extract raw unprocessed EEG channels for exact filter comparison
            raw_eeg_ch = test_exs[idx].eeg[:, ch_indices].astype(np.float32) # [T, C]
            min_len = min(len(raw_eeg_ch), len(YA_te[idx]), len(YB_te[idx]))
            trials_raw_eeg.append(raw_eeg_ch[:min_len])
            trials_ya.append(YA_te[idx][:min_len])
            trials_yb.append(YB_te[idx][:min_len])
        print(f"[DATA] Successfully loaded {len(trials_raw_eeg)} genuine DTU trials for {args.subject}.")
    else:
        print(f"[DATA INFO] No local DTU data found for {args.subject}. Generating synthetic continuous benchmark data.")
        n_samples_trial = int(60.0 * fs)
        for _ in range(args.trials):
            trials_raw_eeg.append(np.random.randn(n_samples_trial, n_ch).astype(np.float32))
            trials_ya.append(np.random.randn(n_samples_trial).astype(np.float32))
            trials_yb.append(np.random.randn(n_samples_trial).astype(np.float32))

    # -------------------------------------------------------------
    # TEST 1 & 2: Offline (filtfilt) vs. Causal Streaming vs. Delay Compensated
    # -------------------------------------------------------------
    print("\n" + "-" * 90)
    print("  TEST 1 & 2: OFFLINE (filtfilt) vs. CAUSAL STREAMING vs. DELAY COMPENSATED")
    print("-" * 90)
    
    window_sec = 5.0
    w_samples = int(window_sec * fs)
    group_delay_samples = 6 # ~94 ms at 64 Hz
    
    deltas_filtfilt = []
    deltas_causal = []
    deltas_compensated = []
    
    causal_filter = StreamingCausalEEGFilter(lowcut=1.0, highcut=6.0, fs=fs, order=2, n_channels=n_ch)
    
    for raw_eeg, ya, yb in zip(trials_raw_eeg, trials_ya, trials_yb):
        # 1. Zero-phase filtfilt (Offline reference)
        filt_ff = butter_bandpass_filtfilt(raw_eeg, 1.0, 6.0, fs, order=2)
        
        # 2. Causal sosfilt (Deployment streaming)
        causal_filter.reset()
        filt_causal = causal_filter.process_chunk(raw_eeg)
        
        # 3. Causal delay-compensated (Shifted forward by 6 samples)
        filt_comp = np.zeros_like(filt_causal)
        filt_comp[:-group_delay_samples] = filt_causal[group_delay_samples:]
        
        # Evaluate non-overlapping 5-second windows
        t_len = len(raw_eeg)
        for start in range(0, t_len - w_samples + 1, w_samples):
            end = start + w_samples
            
            w_ff = torch.from_numpy(filt_ff[start:end].T.copy()).unsqueeze(0).float().to(device)
            w_causal = torch.from_numpy(filt_causal[start:end].T.copy()).unsqueeze(0).float().to(device)
            w_comp = torch.from_numpy(filt_comp[start:end].T.copy()).unsqueeze(0).float().to(device)
            
            w_ya = torch.from_numpy(ya[start:end].copy()).unsqueeze(0).unsqueeze(0).float().to(device)
            w_yb = torch.from_numpy(yb[start:end].copy()).unsqueeze(0).unsqueeze(0).float().to(device)
            
            with torch.no_grad():
                d_ff, _, _ = model(w_ff, w_ya, w_yb)
                d_causal, _, _ = model(w_causal, w_ya, w_yb)
                d_comp, _, _ = model(w_comp, w_ya, w_yb)
                
            deltas_filtfilt.append(d_ff.item())
            deltas_causal.append(d_causal.item())
            deltas_compensated.append(d_comp.item())
            
    deltas_filtfilt = np.array(deltas_filtfilt)
    deltas_causal = np.array(deltas_causal)
    deltas_compensated = np.array(deltas_compensated)
    
    # Classification accuracy (Target is Stream A: delta > 0)
    acc_ff = np.mean(deltas_filtfilt > 0) * 100.0
    acc_causal = np.mean(deltas_causal > 0) * 100.0
    acc_comp = np.mean(deltas_compensated > 0) * 100.0
    
    corr_ff_causal = np.corrcoef(deltas_filtfilt, deltas_causal)[0, 1]
    corr_causal_comp = np.corrcoef(deltas_causal, deltas_compensated)[0, 1]
    
    print(f"  Evaluated 5.0s Windows: {len(deltas_filtfilt)}")
    print(f"  [A] Offline Zero-Phase (filtfilt) Accuracy:      {acc_ff:.2f}%")
    print(f"  [B] Causal Streaming (sosfilt) Accuracy:         {acc_causal:.2f}% (Delta vs. filtfilt: {acc_causal - acc_ff:+.2f} pp)")
    print(f"  [C] Causal + 6-Sample Delay Compensated:         {acc_comp:.2f}% (Delta vs. uncompensated: {acc_comp - acc_causal:+.2f} pp)")
    print(f"  Correlation r(filtfilt, causal):                 {corr_ff_causal:.4f}")
    print(f"  Correlation r(causal, delay_compensated):        {corr_causal_comp:.4f}")
    
    # -------------------------------------------------------------
    # TEST 3: Precision & Layer-Specific Quantization Profiling
    # -------------------------------------------------------------
    print("\n" + "-" * 90)
    print("  TEST 3: PRECISION & LAYER-SPECIFIC QUANTIZATION PROFILING (CPU)")
    print("-" * 90)
    
    cpu_model = deepcopy(model).cpu().eval()
    engine_fp32 = StreamingCATCNEngine(cpu_model, mode="pytorch_fp32", device="cpu", num_threads=4)
    engine_ts = StreamingCATCNEngine(cpu_model, mode="torchscript", device="cpu", num_threads=4)
    engine_int8 = StreamingCATCNEngine(cpu_model, mode="pytorch_int8", device="cpu", num_threads=4)
    
    bench_fp32 = engine_fp32.benchmark(n_iters=50, window_samples=w_samples)
    bench_ts = engine_ts.benchmark(n_iters=50, window_samples=w_samples)
    bench_int8 = engine_int8.benchmark(n_iters=50, window_samples=w_samples)
    
    print(f"  {'Configuration':<25} | {'Mean Latency':<14} | {'P95 Latency':<14} | {'Quantized Layers'}")
    print("  " + "-" * 80)
    print(f"  {'PyTorch Native FP32':<25} | {bench_fp32['mean_ms']:>8.2f} ms     | {bench_fp32['p95_ms']:>8.2f} ms     | None (All FP32)")
    print(f"  {'TorchScript JIT FP32':<25} | {bench_ts['mean_ms']:>8.2f} ms     | {bench_ts['p95_ms']:>8.2f} ms     | Graph Fusion (All FP32)")
    print(f"  {'Dynamic Linear-INT8':<25}  | {bench_int8['mean_ms']:>8.2f} ms     | {bench_int8['p95_ms']:>8.2f} ms     | nn.Linear (Convs remain FP32)")
    
    # -------------------------------------------------------------
    # TEST 4: Decision Layer Dynamic Stability & Switching Characterization
    # -------------------------------------------------------------
    print("\n" + "-" * 90)
    print("  TEST 4: DECISION LAYER DYNAMIC STABILITY (EMA + HYSTERESIS)")
    print("-" * 90)
    
    decision_layer = EMAHysteresisDecisionLayer(alpha=0.7, threshold=0.25, n_confirm=2, boost_db=6.0)
    
    total_steps = 0
    switches = 0
    time_in_uncertain = 0
    first_lock_times = []
    
    step_duration_sec = 0.5
    for raw_eeg, ya, yb in zip(trials_raw_eeg, trials_ya, trials_yb):
        decision_layer.reset()
        c_filter = StreamingCausalEEGFilter(1.0, 6.0, fs, order=2, n_channels=n_ch)
        f_eeg = c_filter.process_chunk(raw_eeg)
        locked = False
        
        for curr_end in range(w_samples, len(raw_eeg), int(step_duration_sec * fs)):
            curr_start = curr_end - w_samples
            w_e = torch.from_numpy(f_eeg[curr_start:curr_end].T.copy()).unsqueeze(0).float().to(device)
            w_a = torch.from_numpy(ya[curr_start:curr_end].copy()).unsqueeze(0).unsqueeze(0).float().to(device)
            w_b = torch.from_numpy(yb[curr_start:curr_end].copy()).unsqueeze(0).unsqueeze(0).float().to(device)
            with torch.no_grad():
                d, _, _ = model(w_e, w_a, w_b)
                
            res = decision_layer.update(d.item())
            total_steps += 1
            if res["switched"]:
                switches += 1
            if res["attended_stream"] == "UNCERTAIN":
                time_in_uncertain += 1
            elif not locked:
                first_lock_times.append(curr_end / fs)
                locked = True
                
    total_sim_minutes = (total_steps * step_duration_sec) / 60.0
    switches_per_min = switches / max(0.01, total_sim_minutes)
    pct_uncertain = (time_in_uncertain / max(1, total_steps)) * 100.0
    median_lock_sec = float(np.median(first_lock_times)) if first_lock_times else 0.0
    
    print(f"  Total Simulated Time:          {total_sim_minutes:.2f} minutes")
    print(f"  Observed Switches per Minute:  {switches_per_min:.2f} switches/min")
    print(f"  Time in UNCERTAIN State:       {pct_uncertain:.1f}%")
    print(f"  Median Time to First Lock:     {median_lock_sec:.2f} s")
    
    # -------------------------------------------------------------
    # TEST 5: Decision Latency Pareto Sweep (W in [2s, 3s, 5s, 10s, 20s])
    # -------------------------------------------------------------
    print("\n" + "-" * 90)
    print("  TEST 5: DECISION LATENCY PARETO SWEEP (Accuracy vs. Window Length)")
    print("-" * 90)
    
    sweep_windows = [2.0, 3.0, 5.0, 10.0, 20.0]
    print(f"  {'Window W (s)':<15} | {'Window Samples':<15} | {'Causal Accuracy':<18} | {'Decision Latency'}")
    print("  " + "-" * 80)
    
    for w_sec in sweep_windows:
        w_samp = int(w_sec * fs)
        w_deltas = []
        c_filt = StreamingCausalEEGFilter(1.0, 6.0, fs, order=2, n_channels=n_ch)
        
        for raw_eeg, ya, yb in zip(trials_raw_eeg, trials_ya, trials_yb):
            c_filt.reset()
            f_e = c_filt.process_chunk(raw_eeg)
            for s in range(0, len(raw_eeg) - w_samp + 1, w_samp):
                e = s + w_samp
                w_e = torch.from_numpy(f_e[s:e].T.copy()).unsqueeze(0).float().to(device)
                w_a = torch.from_numpy(ya[s:e].copy()).unsqueeze(0).unsqueeze(0).float().to(device)
                w_b = torch.from_numpy(yb[s:e].copy()).unsqueeze(0).unsqueeze(0).float().to(device)
                with torch.no_grad():
                    d, _, _ = model(w_e, w_a, w_b)
                w_deltas.append(d.item())
                
        w_acc = np.mean(np.array(w_deltas) > 0) * 100.0 if w_deltas else 50.0
        print(f"  {w_sec:<15.1f} | {w_samp:<15} | {w_acc:>12.2f}%       | {w_sec:.1f} s")
        
    print("=" * 90)
    print("  DEPLOYMENT VALIDATION PROTOCOL v1.1 COMPLETE")
    print("=" * 90)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deployment Validation Protocol v1.1")
    parser.add_argument("--subject", type=str, default="S1_data_preproc", help="Subject name")
    parser.add_argument("--montage", type=str, default="near_ear_expanded", help="Montage name")
    parser.add_argument("--trials", type=int, default=10, help="Number of trials")
    parser.add_argument("--train_epochs", type=int, default=1, help="Number of epochs to train CA-TCN on training fold (default: 1)")
    parser.add_argument("--checkpoint", type=str, default="", help="Path to trained model checkpoint")
    args = parser.parse_args()
    
    run_v1_1_validation(args)
