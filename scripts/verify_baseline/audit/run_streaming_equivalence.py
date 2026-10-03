import argparse
import sys
import os
import json
import time
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from scipy.stats import pearsonr

REPO_ROOT = Path(__file__).resolve().parents[3]
VERIFY_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(VERIFY_ROOT) not in sys.path:
    sys.path.insert(0, str(VERIFY_ROOT))

from models.catcn import CATCNDirectDecoder
from src.streaming.causal_filters import StreamingCausalEEGFilter
from src.streaming.pipeline import StreamingAADPipeline
from training.montages import MONTAGES, DTU_CHANNELS

def run_streaming_equivalence_audit(
    subject_name: str = "S1_data_preproc",
    montage_name: str = "near_ear_expanded",
    window_sec: float = 5.0,
    step_sec: float = 0.5,
    chunk_samples: int = 16, # 250 ms chunks at 64 Hz
    n_synthetic_trials: int = 10,
    trial_duration_sec: float = 60.0
):
    """
    Rigorously compares offline batch AAD decoding with real-time causal streaming AAD decoding.
    """
    fs = 64.0
    channels = MONTAGES[montage_name]
    n_ch = len(channels)
    
    print("=" * 85)
    print("  CA-TCN OFFLINE vs. STREAMING EQUIVALENCE AUDIT")
    print(f"  Target Subject: {subject_name} | Montage: {montage_name} ({n_ch} channels)")
    print(f"  Window: {window_sec:.1f} s ({int(window_sec*fs)} samples) | Step: {step_sec:.1f} s ({int(step_sec*fs)} samples)")
    print(f"  Streaming Chunk Size: {chunk_samples} samples ({chunk_samples/fs*1000:.1f} ms)")
    print("=" * 85)
    
    # 1. Instantiate reference CA-TCN model
    torch.manual_seed(42)
    np.random.seed(42)
    model = CATCNDirectDecoder(eeg_channels=n_ch, audio_channels=1, hidden_dim=64, max_lag_samples=8)
    model.eval()
    
    # 2. Check if real DTU data is accessible locally
    # Otherwise generate realistic calibrated synthetic trials
    n_samples_per_trial = int(trial_duration_sec * fs)
    trials_eeg = []
    trials_ya = []
    trials_yb = []
    
    try:
        from training.train_matchnet_wavlm import get_mapping_data, prepare_dataset
        from baselines.ridge_aad import load_subject_examples, subject_files
        
        mapping, envelopes = get_mapping_data("gammatone")
        files = subject_files()
        target_file = [f for f in files if subject_name in f.name]
        if target_file:
            print(f"[DATA] Loading real DTU recording: {target_file[0].name}...")
            examples = list(load_subject_examples(target_file[0]))
            X_all, YA_all, YB_all = prepare_dataset(
                examples, channels, 1.0, 6.0, subject_name, mapping, envelopes
            )
            for i in range(min(5, len(X_all))):
                trials_eeg.append(X_all[i].T) # [T, C]
                ya_v = YA_all[i].mean(axis=0).squeeze() if YA_all[i].ndim > 1 else YA_all[i].squeeze()
                yb_v = YB_all[i].mean(axis=0).squeeze() if YB_all[i].ndim > 1 else YB_all[i].squeeze()
                trials_ya.append(ya_v) # [T]
                trials_yb.append(yb_v) # [T]
            print(f"[DATA] Loaded {len(trials_eeg)} real DTU trials.")
        else:
            raise FileNotFoundError(f"No DTU data file found for {subject_name}")
    except Exception as e:
        print(f"[DATA ERROR] Could not load genuine DTU data: {e}")
        raise e
            
    # 3. Setup Streaming Pipeline
    pipeline = StreamingAADPipeline(
        model=model,
        n_eeg_channels=n_ch,
        fs=fs,
        raw_audio_input=False,
        window_sec=window_sec,
        step_sec=step_sec,
        engine_mode="torchscript",
        decision_alpha=0.7,
        decision_threshold=0.25,
        n_confirm=2
    )
    
    # 4. Run Both Modes Across All Trials
    offline_deltas_all = []
    streaming_deltas_all = []
    streaming_latencies = []
    window_samples = int(window_sec * fs)
    step_samples = int(step_sec * fs)
    
    # Batch filter for Mode A
    batch_filter = StreamingCausalEEGFilter(lowcut=1.0, highcut=6.0, fs=fs, order=2, n_channels=n_ch)
    
    for t_idx, (eeg_trial, ya_trial, yb_trial) in enumerate(zip(trials_eeg, trials_ya, trials_yb)):
        pipeline.reset()
        batch_filter.reset()
        t_len = len(eeg_trial)
        
        # Mode A: Apply same causal filter in batch, then evaluate at identical window positions
        filt_eeg_batch = batch_filter.process_chunk(eeg_trial)
        
        offline_trial_deltas = {}
        curr_end = window_samples
        while curr_end <= t_len:
            curr_start = curr_end - window_samples
            
            # Extract offline slices from batch-filtered EEG
            win_eeg = torch.from_numpy(filt_eeg_batch[curr_start:curr_end].T).unsqueeze(0).float() # [1, C, W]
            win_ya = torch.from_numpy(ya_trial[curr_start:curr_end]).unsqueeze(0).unsqueeze(0).float() # [1, 1, W]
            win_yb = torch.from_numpy(yb_trial[curr_start:curr_end]).unsqueeze(0).unsqueeze(0).float() # [1, 1, W]
            
            with torch.no_grad():
                d_off, _, _ = model(win_eeg, win_ya, win_yb)
            offline_trial_deltas[curr_end] = float(d_off.item())
            curr_end += step_samples
            
        # --- Mode B: Streaming Causal Chunk-by-Chunk Feeding ---
        streaming_trial_deltas = {}
        for c_start in range(0, t_len, chunk_samples):
            c_end = min(c_start + chunk_samples, t_len)
            c_e = eeg_trial[c_start:c_end]
            c_a = ya_trial[c_start:c_end]
            c_b = yb_trial[c_start:c_end]
            
            telemetry = pipeline.feed_sample_block(c_e, c_a, c_b)
            if telemetry is not None:
                # Samples processed up to this point
                sample_pos = int(telemetry["timestamp_sec"] * fs)
                streaming_trial_deltas[sample_pos] = telemetry["raw_delta"]
                streaming_latencies.append(telemetry["compute_ms"])
                
        # Align timepoints present in both
        common_timepoints = sorted(set(offline_trial_deltas.keys()) & set(streaming_trial_deltas.keys()))
        for tp in common_timepoints:
            offline_deltas_all.append(offline_trial_deltas[tp])
            streaming_deltas_all.append(streaming_trial_deltas[tp])
            
    offline_deltas_all = np.array(offline_deltas_all)
    streaming_deltas_all = np.array(streaming_deltas_all)
    streaming_latencies = np.array(streaming_latencies)
    
    # 5. Statistical Equivalence Metrics
    r_val, p_val = pearsonr(offline_deltas_all, streaming_deltas_all)
    
    # Directional Agreement (percentage of times both sign(Delta) agree)
    agree_mask = (offline_deltas_all > 0) == (streaming_deltas_all > 0)
    agreement_pct = np.mean(agree_mask) * 100.0
    
    # Delta differences
    mae = np.mean(np.abs(offline_deltas_all - streaming_deltas_all))
    
    print("\n" + "=" * 85)
    print("  EQUIVALENCE AUDIT RESULTS")
    print("=" * 85)
    print(f"  Total Evaluated Rolling Windows: {len(offline_deltas_all)}")
    print(f"  Pearson Correlation r(Offline, Streaming): {r_val:.4f} (p = {p_val:.2e})")
    print(f"  Directional Decision Agreement:           {agreement_pct:.2f}%")
    print(f"  Mean Absolute Error (|Delta_off - Delta_str|): {mae:.4f}")
    print(f"  Inference Latency T_compute (Mean):       {np.mean(streaming_latencies):.2f} ms")
    print(f"  Inference Latency T_compute (P95):        {np.percentile(streaming_latencies, 95):.2f} ms")
    print(f"  Inference Latency T_compute (Max):        {np.max(streaming_latencies):.2f} ms")
    print("=" * 85)
    
    # Verify Acceptance Thresholds
    pass_r = r_val > 0.90
    pass_agree = agreement_pct > 90.0
    pass_latency = np.percentile(streaming_latencies, 95) < 35.0
    
    print("\n  AUDIT CHECKLIST:")
    print(f"  [{'PASS' if pass_r else 'FAIL'}] Pearson Correlation r > 0.90 (Got: {r_val:.4f})")
    print(f"  [{'PASS' if pass_agree else 'FAIL'}] Decision Agreement > 90% (Got: {agreement_pct:.1f}%)")
    print(f"  [{'PASS' if pass_latency else 'FAIL'}] P95 Latency < 35 ms (Got: {np.percentile(streaming_latencies, 95):.2f} ms)")
    
    if pass_r and pass_agree and pass_latency:
        print("\n>>> ALL STREAMING EQUIVALENCE CHECKS PASSED SUCCESSFULLY. <<<")
    else:
        print("\n>>> WARNING: One or more equivalence checks did not meet threshold. <<<")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CA-TCN Offline vs Streaming Equivalence Audit")
    parser.add_argument("--subject", type=str, default="S1_data_preproc", help="Subject to audit")
    parser.add_argument("--montage", type=str, default="near_ear_expanded", help="Montage name")
    parser.add_argument("--window_sec", type=float, default=5.0, help="Window size in seconds")
    parser.add_argument("--step_sec", type=float, default=0.5, help="Step size in seconds")
    parser.add_argument("--trials", type=int, default=10, help="Number of trials to simulate if local data absent")
    args = parser.parse_args()
    
    run_streaming_equivalence_audit(
        subject_name=args.subject,
        montage_name=args.montage,
        window_sec=args.window_sec,
        step_sec=args.step_sec,
        n_synthetic_trials=args.trials
    )
