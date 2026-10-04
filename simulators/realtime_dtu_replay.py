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
    ground_truth: str = "A",
    subject_id: str = "S1",
    trial_idx: int = 0,
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
    Displays side-by-side Ground Truth vs Live Prediction.
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
    print("=" * 102)
    print(f"  REAL-TIME CA-TCN AAD REPLAY SIMULATOR (Decoupled Loop)")
    print(f"  Subject: {subject_id} (Trial {trial_idx}) | Ground Truth Attended: Speaker {ground_truth}")
    print(f"  EEG Channels: {n_channels} | Rate: {fs} Hz | Window: {window_sec}s | Step: {step_sec}s | Speed: {speed_factor}x")
    print("=" * 102)
    
    start_wall_time = time.perf_counter()
    sim_time_sec = 0.0
    idx = 0
    step_count = 0
    correct_steps = 0
    switches = 0
    total_compute_ms = 0.0
    acoustic_delay_ms = 0.0
    deltas = []
    
    while idx < n_total:
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
            deltas.append(telemetry["raw_delta"])
            if telemetry["switched"]:
                switches += 1
                
            stream = telemetry["attended_stream"]
            if stream == ground_truth:
                correct_steps += 1
                match_badge = "[✓ MATCH]"
            elif stream == "UNCERTAIN":
                match_badge = "[? SEARCH]"
            else:
                match_badge = "[✗ WRONG]"
                
            if verbose:
                la = telemetry["logit_a"]
                lb = telemetry["logit_b"]
                delta = telemetry["raw_delta"]
                smooth = telemetry["smoothed_score"]
                ga = telemetry["gain_a"]
                gb = telemetry["gain_b"]
                ga_db = 20.0 * np.log10(max(1e-3, ga))
                gb_db = 20.0 * np.log10(max(1e-3, gb))
                conf = telemetry["confidence"] * 100.0
                meter = format_bipolar_meter(smooth)
                flag = " [SWITCHED!]" if telemetry["switched"] else ""
                
                print(
                    f"[{telemetry['timestamp_sec']:5.1f}s] "
                    f"GT:[{ground_truth}] vs Pred:[{stream:<1}] {match_badge} | "
                    f"Logits:[A:{la:+.2f}, B:{lb:+.2f}] "
                    f"Δ:{delta:+.2f} "
                    f"EMA:{smooth:+.2f} | "
                    f"{meter} | "
                    f"Conf:{conf:4.1f}% | "
                    f"Gains:[A:{ga_db:+4.1f}dB, B:{gb_db:+4.1f}dB] | "
                    f"{telemetry['compute_ms']:4.1f}ms{flag}"
                )
                
        idx = end_idx
        sim_time_sec += actual_samples / fs
        
        if speed_factor > 0:
            elapsed_wall = time.perf_counter() - start_wall_time
            expected_wall = sim_time_sec / speed_factor
            sleep_time = expected_wall - elapsed_wall
            if sleep_time > 0.001:
                time.sleep(sleep_time)
                
    total_wall_sec = time.perf_counter() - start_wall_time
    mean_compute_ms = total_compute_ms / max(1, step_count)
    accuracy_pct = (correct_steps / max(1, step_count)) * 100.0
    
    print("=" * 102)
    print(f"  TRIAL SIMULATION COMPLETE: Subject {subject_id} (Trial {trial_idx})")
    print(f"  Ground Truth Attended Speaker: Stream {ground_truth}")
    print(f"  Model Real-Time Lock Accuracy: {accuracy_pct:.1f}% ({correct_steps}/{step_count} steps on Ground Truth)")
    print(f"  Final Decision Lock: Stream {stream} | Mean Neural Margin (Δ): {float(np.mean(deltas)):+.2f}")
    print(f"  Total Speaker Switches: {switches} | Mean BCI Latency: {mean_compute_ms:.2f} ms")
    print("=" * 102)
    
    return {
        "subject": subject_id,
        "trial_idx": trial_idx,
        "ground_truth": ground_truth,
        "final_lock": stream,
        "accuracy_pct": accuracy_pct,
        "correct_steps": correct_steps,
        "total_steps": step_count,
        "switches": switches,
        "mean_delta": float(np.mean(deltas)) if deltas else 0.0,
        "mean_latency": mean_compute_ms
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Real-Time CA-TCN DTU Replay Simulator")
    parser.add_argument("--subject", type=str, default="S1", help="DTU Subject name (e.g. S1, S2, S7)")
    parser.add_argument("--montage", type=str, default="near_ear_expanded", help="Electrode montage")
    parser.add_argument("--trial_idx", type=int, default=0, help="Trial index to replay")
    parser.add_argument("--checkpoint", type=str, default="", help="Path to trained model checkpoint")
    parser.add_argument("--speed_factor", type=float, default=2.0, help="Clock speedup factor (1.0=realtime, 2.0=2x speed, 0=max)")
    parser.add_argument("--window_sec", type=float, default=5.0, help="Rolling window size in seconds")
    parser.add_argument("--step_sec", type=float, default=0.5, help="Rolling step size in seconds")
    parser.add_argument("--compare_subjects", type=str, default="", help="Comma-separated subjects to compare (e.g. S1,S2,S7,S8,S15)")
    parser.add_argument("--compare_trials", type=str, default="", help="Comma-separated trials to compare (e.g. 0,1,2,3)")
    parser.add_argument("--swap_streams", action="store_true", help="Swap A and B streams so Ground Truth is Stream B")
    args = parser.parse_args()
    
    montage_channels = MONTAGES[args.montage]
    n_ch = len(montage_channels)
    model = CATCNDirectDecoder(eeg_channels=n_ch, audio_channels=1, hidden_dim=64, max_lag_samples=8)
    
    ckpt_path = args.checkpoint
    if not ckpt_path:
        for candidate in ["/kaggle/working/catcn_deployment_weights.pt", "/kaggle/working/catcn_universal_model.pt"]:
            if Path(candidate).exists():
                ckpt_path = candidate
                break
        
    if ckpt_path and Path(ckpt_path).exists():
        print(f"[MODEL] Loading trained checkpoint from: {ckpt_path}")
        state = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(state.get("model_state_dict", state.get("state_dict", state)))
        print("[MODEL] Checkpoint loaded successfully!")
    else:
        print("[MODEL WARNING] No checkpoint found. Running with initialized weights.")
        
    model.eval()
    files = subject_files()
    mapping, envelopes = get_mapping_data("gammatone")

    # Determine subjects to run
    if args.compare_subjects:
        subjects_to_run = [s.strip() for s in args.compare_subjects.split(",") if s.strip()]
    else:
        subjects_to_run = [args.subject]
        
    # Determine trials to run
    if args.compare_trials:
        trials_to_run = [int(t.strip()) for t in args.compare_trials.split(",") if t.strip()]
    else:
        trials_to_run = [args.trial_idx]

    results_summary = []

    for sub in subjects_to_run:
        target_files = [f for f in files if f.stem == sub or f.stem.split("_")[0] == sub.split("_")[0]]
        if not target_files:
            print(f"[WARNING] Could not find DTU subject file for {sub}. Skipping.")
            continue
            
        test_exs = list(load_subject_examples(target_files[0]))
        _, YA_all, YB_all = prepare_dataset(test_exs, montage_channels, 1.0, 6.0, sub, mapping, envelopes)
        
        for t_idx in trials_to_run:
            actual_t_idx = min(t_idx, len(test_exs) - 1, len(YA_all) - 1)
            raw_eeg = test_exs[actual_t_idx].eeg[:, montage_channels].astype(np.float32)
            ya = YA_all[actual_t_idx].mean(axis=0).squeeze() if YA_all[actual_t_idx].ndim > 1 else YA_all[actual_t_idx].squeeze()
            yb = YB_all[actual_t_idx].mean(axis=0).squeeze() if YB_all[actual_t_idx].ndim > 1 else YB_all[actual_t_idx].squeeze()
            
            min_len = min(len(raw_eeg), len(ya), len(yb))
            raw_eeg = raw_eeg[:min_len]
            ya = ya[:min_len]
            yb = yb[:min_len]
            
            if args.swap_streams:
                # Candidate 1 is unattended, Candidate 2 is attended
                feed_a = yb
                feed_b = ya
                gt = "B"
            else:
                feed_a = ya
                feed_b = yb
                gt = "A"
                
            res = simulate_realtime_stream(
                eeg_stream=raw_eeg,
                audio_a_stream=feed_a,
                audio_b_stream=feed_b,
                ground_truth=gt,
                subject_id=sub,
                trial_idx=actual_t_idx,
                model=model,
                fs=FS,
                chunk_samples=16,
                speed_factor=args.speed_factor,
                window_sec=args.window_sec,
                step_sec=args.step_sec,
                verbose=True
            )
            results_summary.append(res)

    if len(results_summary) > 1:
        print("\n" + "=" * 98)
        print("  MULTI-SUBJECT GROUND TRUTH vs REAL-TIME LIVESTREAM BENCHMARK")
        print("=" * 98)
        print(f"  {'Subject':<10} | {'Trial':<6} | {'Ground Truth':<14} | {'Final Lock':<12} | {'% On Ground Truth':<18} | {'Switches':<9} | {'Mean Margin (Δ)'}")
        print("  " + "-" * 94)
        for r in results_summary:
            status_symbol = "✓" if r["final_lock"] == r["ground_truth"] else "✗"
            print(f"  {r['subject']:<10} | {r['trial_idx']:<6} | Stream {r['ground_truth']:<7} | Stream {r['final_lock']} {status_symbol:<4} | {r['accuracy_pct']:>6.1f}% ({r['correct_steps']}/{r['total_steps']})    | {r['switches']:<9} | {r['mean_delta']:+6.2f}")
        print("=" * 98)

