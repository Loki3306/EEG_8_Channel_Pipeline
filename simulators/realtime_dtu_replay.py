import time
import argparse
import sys
from pathlib import Path
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

VERIFY_ROOT = REPO_ROOT / "scripts" / "verify_baseline"
if str(VERIFY_ROOT) not in sys.path:
    sys.path.insert(0, str(VERIFY_ROOT))

from src.streaming.pipeline import StreamingAADPipeline
from scripts.verify_baseline.models.catcn import CATCNDirectDecoder
from scripts.verify_baseline.training.montages import MONTAGES
from scripts.verify_baseline.training.train_matchnet_wavlm import get_mapping_data, prepare_dataset, FS
from scripts.verify_baseline.baselines.ridge_aad import load_subject_examples, subject_files

def format_bipolar_meter(score: float, width: int = 7) -> str:
    """Renders a visual spatial steering meter between Stream A (left) and Stream B (right)."""
    clamped = max(-1.0, min(1.0, score))
    # score > 0 -> Stream A (left), score < 0 -> Stream B (right)
    pos = int(round((1.0 - clamped) / 2.0 * (2 * width)))
    pos = max(0, min(2 * width, pos))
    bar = ["-"] * (2 * width + 1)
    bar[width] = "|"
    if pos == width:
        bar[pos] = "●"
    elif pos < width:
        bar[pos] = "◄"
    else:
        bar[pos] = "►"
    return f"[A] <{''.join(bar)}> [B]"

def simulate_realtime_stream(
    eeg_stream: np.ndarray,
    audio_a_stream: np.ndarray,
    audio_b_stream: np.ndarray,
    model: torch.nn.Module = None,
    fs: float = 64.0,
    chunk_samples: int = 16, # 16 samples @ 64 Hz = 250 ms chunk
    speed_factor: float = 1.0,
    window_sec: float = 5.0,
    step_sec: float = 0.5,
    verbose: bool = True
):
    """
    Simulates real-time arrival of multichannel EEG and candidate audio envelopes from genuine DTU recordings.
    
    Parameters:
        eeg_stream: [n_total_samples, n_channels]
        audio_a_stream: [n_total_samples]
        audio_b_stream: [n_total_samples]
        model: Trained CATCNDirectDecoder
        fs: Sampling rate (64 Hz)
        chunk_samples: Chunk size arriving per clock tick (e.g. 16 samples = 250 ms)
        speed_factor: 1.0 = true wall-clock real time; 0.0 = as fast as possible; >1.0 = accelerated
    """
    n_total, n_channels = eeg_stream.shape
    assert len(audio_a_stream) == n_total, f"Audio A length mismatch: {len(audio_a_stream)} vs {n_total}"
    assert len(audio_b_stream) == n_total, f"Audio B length mismatch: {len(audio_b_stream)} vs {n_total}"
    
    if model is None:
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
    acoustic_delay_ms = 0.0
    
    while idx < n_total:
        # Slice current incoming hardware block
        end_idx = min(idx + chunk_samples, n_total)
        chunk_e = eeg_stream[idx:end_idx]
        chunk_a = audio_a_stream[idx:end_idx]
        chunk_b = audio_b_stream[idx:end_idx]
        actual_samples = end_idx - idx
        
        # 1. Fast Acoustic Pipeline Simulation (< 10 ms requirement)
        t_audio_start = time.perf_counter()
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
                la = telemetry["logit_a"]
                lb = telemetry["logit_b"]
                delta = telemetry["raw_delta"]
                smooth = telemetry["smoothed_score"]
                stream = telemetry["attended_stream"]
                ga = telemetry["gain_a"]
                gb = telemetry["gain_b"]
                ga_db = 20.0 * np.log10(max(1e-3, ga))
                gb_db = 20.0 * np.log10(max(1e-3, gb))
                conf = telemetry["confidence"] * 100.0
                meter = format_bipolar_meter(smooth)
                flag = " [SWITCHED!]" if telemetry["switched"] else ""
                
                print(
                    f"[{telemetry['timestamp_sec']:5.1f}s] "
                    f"Logits:[A:{la:+.2f}, B:{lb:+.2f}] "
                    f"Δ:{delta:+.2f} "
                    f"EMA:{smooth:+.2f} | "
                    f"{meter} | "
                    f"Lock: {stream:<9} ({conf:4.1f}%) | "
                    f"Gains:[A:{ga_db:+4.1f}dB, B:{gb_db:+4.1f}dB] | "
                    f"{telemetry['compute_ms']:4.1f}ms{flag}"
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
    print(f"  Mean BCI Compute Latency (T_compute): {mean_compute_ms:.2f} ms (< 500 ms step budget)")
    print(f"  Digital Software Mixing Computation Time: {acoustic_delay_ms:.4f} ms (Excludes physical DAC/OS latency)")
    print("=" * 85)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Real-Time CA-TCN DTU Replay Simulator")
    parser.add_argument("--subject", type=str, default="S1_data_preproc", help="DTU Subject name")
    parser.add_argument("--montage", type=str, default="near_ear_expanded", help="Electrode montage")
    parser.add_argument("--trial_idx", type=int, default=0, help="Trial index to replay")
    parser.add_argument("--checkpoint", type=str, default="", help="Path to trained model checkpoint")
    parser.add_argument("--speed_factor", type=float, default=2.0, help="Clock speedup factor (1.0=realtime, 2.0=2x speed, 0=max)")
    parser.add_argument("--window_sec", type=float, default=5.0, help="Rolling window size in seconds")
    parser.add_argument("--step_sec", type=float, default=0.5, help="Rolling step size in seconds")
    args = parser.parse_args()
    
    montage_channels = MONTAGES[args.montage]
    n_ch = len(montage_channels)
    model = CATCNDirectDecoder(eeg_channels=n_ch, audio_channels=1, hidden_dim=64, max_lag_samples=8)
    
    ckpt_path = args.checkpoint
    if not ckpt_path and Path("/kaggle/working/catcn_deployment_weights.pt").exists():
        ckpt_path = "/kaggle/working/catcn_deployment_weights.pt"
        
    if ckpt_path and Path(ckpt_path).exists():
        print(f"[MODEL] Loading trained checkpoint from: {ckpt_path}")
        state = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(state.get("model_state_dict", state.get("state_dict", state)))
        print("[MODEL] Checkpoint loaded successfully!")
    else:
        print("[MODEL WARNING] No checkpoint found. Running with initialized weights.")
        
    model.eval()

    files = subject_files()
    target_files = [f for f in files if args.subject in f.name]
    if not target_files:
        raise FileNotFoundError(f"Could not find DTU subject file for {args.subject} in DATA_DIR. Provide genuine DTU data.")
        
    print(f"[DATA] Loading genuine DTU recording: {target_files[0].name}...")
    mapping, envelopes = get_mapping_data("gammatone")
    test_exs = list(load_subject_examples(target_files[0]))
    
    _, YA_all, YB_all = prepare_dataset(test_exs, montage_channels, 1.0, 6.0, args.subject, mapping, envelopes)
    
    trial_idx = min(args.trial_idx, len(test_exs) - 1, len(YA_all) - 1)
    raw_eeg = test_exs[trial_idx].eeg[:, montage_channels].astype(np.float32)
    ya = YA_all[trial_idx].mean(axis=0).squeeze() if YA_all[trial_idx].ndim > 1 else YA_all[trial_idx].squeeze()
    yb = YB_all[trial_idx].mean(axis=0).squeeze() if YB_all[trial_idx].ndim > 1 else YB_all[trial_idx].squeeze()
    
    min_len = min(len(raw_eeg), len(ya), len(yb))
    raw_eeg = raw_eeg[:min_len]
    ya = ya[:min_len]
    yb = yb[:min_len]
    
    print(f"[DATA] Loaded Trial {trial_idx}: {min_len} samples ({min_len / FS:.1f} seconds) of genuine 8-channel EEG & speech.")
    
    simulate_realtime_stream(
        eeg_stream=raw_eeg,
        audio_a_stream=ya,
        audio_b_stream=yb,
        model=model,
        fs=FS,
        chunk_samples=16,
        speed_factor=args.speed_factor,
        window_sec=args.window_sec,
        step_sec=args.step_sec
    )
