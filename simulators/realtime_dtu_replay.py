import time
import argparse
import sys
from pathlib import Path
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.streaming.pipeline import StreamingAADPipeline
from scripts.verify_baseline.models.catcn import CATCNDirectDecoder

def simulate_realtime_stream(
    eeg_stream: np.ndarray,
    audio_a_stream: np.ndarray,
    audio_b_stream: np.ndarray,
    fs: float = 64.0,
    chunk_samples: int = 16, # 16 samples @ 64 Hz = 250 ms chunk
    speed_factor: float = 1.0,
    window_sec: float = 5.0,
    step_sec: float = 0.5,
    verbose: bool = True
):
    """
    Simulates real-time arrival of multichannel EEG and candidate audio envelopes.
    
    Parameters:
        eeg_stream: [n_total_samples, n_channels]
        audio_a_stream: [n_total_samples]
        audio_b_stream: [n_total_samples]
        fs: Sampling rate (64 Hz)
        chunk_samples: Chunk size arriving per clock tick (e.g. 16 samples = 250 ms)
        speed_factor: 1.0 = true wall-clock real time; 0.0 = as fast as possible; >1.0 = accelerated
    """
    n_total, n_channels = eeg_stream.shape
    assert len(audio_a_stream) == n_total
    assert len(audio_b_stream) == n_total
    
    # Initialize CA-TCN model (randomly initialized or loaded)
    model = CATCNDirectDecoder(eeg_channels=n_channels, audio_channels=1, hidden_dim=64, max_lag_samples=8)
    model.eval()
    
    # Initialize streaming pipeline
    pipeline = StreamingAADPipeline(
        model=model,
        n_eeg_channels=n_channels,
        fs=fs,
        raw_audio_input=False,
        window_sec=window_sec,
        step_sec=step_sec,
        engine_mode="torchscript",
        decision_alpha=0.7,
        decision_threshold=0.25,
        n_confirm=2,
        boost_db=6.0
    )
    
    chunk_duration_sec = chunk_samples / fs
    print("=" * 85)
    print(f"  REAL-TIME CA-TCN AAD REPLAY SIMULATOR (Decoupled Loop)")
    print(f"  EEG Channels: {n_channels} | Rate: {fs} Hz | Window: {window_sec}s | Step: {step_sec}s")
    print(f"  Chunk: {chunk_samples} samples ({chunk_duration_sec*1000:.1f} ms) | Speed Factor: {speed_factor}x")
    print("=" * 85)
    
    start_wall_time = time.perf_counter()
    sim_time_sec = 0.0
    idx = 0
    step_count = 0
    switches = 0
    total_compute_ms = 0.0
    
    while idx < n_total:
        t_chunk_start = time.perf_counter()
        
        # Slice current incoming hardware block
        end_idx = min(idx + chunk_samples, n_total)
        chunk_e = eeg_stream[idx:end_idx]
        chunk_a = audio_a_stream[idx:end_idx]
        chunk_b = audio_b_stream[idx:end_idx]
        actual_samples = end_idx - idx
        
        # 1. Fast Acoustic Pipeline Simulation (< 10 ms requirement)
        t_audio_start = time.perf_counter()
        # Simulated instant linear mixing of current audio sample block using active gains
        active_ga = pipeline.decision_layer.gain_a
        active_gb = pipeline.decision_layer.gain_b
        _simulated_acoustic_mix = active_ga * chunk_a + active_gb * chunk_b
        t_audio_end = time.perf_counter()
        acoustic_delay_ms = (t_audio_end - t_audio_start) * 1000.0
        
        # 2. Lagged BCI Control Loop
        telemetry = pipeline.feed_sample_block(chunk_e, chunk_a, chunk_b)
        
        if telemetry is not None:
            step_count += 1
            total_compute_ms += telemetry["compute_ms"]
            if telemetry["switched"]:
                switches += 1
                
            if verbose:
                flag = " [SWITCHED!]" if telemetry["switched"] else ""
                print(
                    f"[{telemetry['timestamp_sec']:5.1f}s] "
                    f"Stream: {telemetry['attended_stream']:<9} | "
                    f"Conf: {telemetry['confidence']*100:4.1f}% | "
                    f"Gains: [A:{telemetry['gain_a']:.2f}, B:{telemetry['gain_b']:.2f}] | "
                    f"T_comp: {telemetry['compute_ms']:4.1f} ms | "
                    f"Acoustic Delay: {acoustic_delay_ms:.3f} ms{flag}"
                )
                
        idx = end_idx
        sim_time_sec += actual_samples / fs
        
        # Real-time clock synchronization
        if speed_factor > 0:
            elapsed_wall = time.perf_counter() - start_wall_time
            expected_wall = sim_time_sec / speed_factor
            sleep_time = expected_wall - elapsed_wall
            if sleep_time > 0.001:
                time.sleep(sleep_time)
                
    total_wall_sec = time.perf_counter() - start_wall_time
    mean_compute_ms = total_compute_ms / max(1, step_count)
    print("=" * 85)
    print("  SIMULATION COMPLETE")
    print(f"  Total Stream Time: {sim_time_sec:.1f} s | Wall-Clock Time: {total_wall_sec:.2f} s")
    print(f"  Inference Steps: {step_count} | Total Speaker Switches: {switches}")
    print(f"  Mean Compute Latency (T_compute): {mean_compute_ms:.2f} ms (< 500 ms step budget)")
    print(f"  Instantaneous Audio Pipeline Latency: {acoustic_delay_ms:.3f} ms (< 10 ms budget)")
    print("=" * 85)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Real-Time CA-TCN Replay Simulator")
    parser.add_argument("--duration_sec", type=float, default=20.0, help="Duration of stream to simulate in seconds")
    parser.add_argument("--speed_factor", type=float, default=2.0, help="Clock speedup factor (e.g. 1.0=realtime, 2.0=2x speed, 0=max)")
    parser.add_argument("--channels", type=int, default=8, help="Number of EEG channels")
    parser.add_argument("--window_sec", type=float, default=5.0, help="Rolling window size in seconds")
    parser.add_argument("--step_sec", type=float, default=0.5, help="Rolling step size in seconds")
    args = parser.parse_args()
    
    # Generate synthetic streaming multichannel trial
    n_samples = int(args.duration_sec * 64.0)
    np.random.seed(42)
    syn_eeg = np.random.randn(n_samples, args.channels).astype(np.float32)
    syn_a = np.random.randn(n_samples).astype(np.float32)
    syn_b = np.random.randn(n_samples).astype(np.float32)
    
    simulate_realtime_stream(
        eeg_stream=syn_eeg,
        audio_a_stream=syn_a,
        audio_b_stream=syn_b,
        fs=64.0,
        chunk_samples=16,
        speed_factor=args.speed_factor,
        window_sec=args.window_sec,
        step_sec=args.step_sec
    )
