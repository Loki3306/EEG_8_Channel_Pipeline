"""
Joint End-to-End Real-Time Streaming Benchmark: Concurrent Raw EEG + Raw Audio Ingestion.

Benchmarks full hardware-simulated streaming:
1. Ingests raw continuous 512 Hz multi-channel EEG (S<id>.mat).
2. Ingests raw audio candidate streams (44.1 kHz / 16 kHz WAV) for Talker A and Talker B.
3. Synchronizes through DualStreamIngestionEngine (31.25 ms lock-step chunks).
4. Evaluates CA-TCN direct match decoder + Spatial Adapter + Sticky Hysteresis Gate on GPU.
5. Executes AudioSteeringDSP (soft 100 ms crossfade) and benchmarks total CPU/GPU load and RTF.
6. Supports single-trial detailed telemetry as well as multi-trial / all-trial aggregate benchmark.
"""

import argparse
import sys
import json
import time
from pathlib import Path
import numpy as np
import scipy.io as sio
from scipy.io import wavfile
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
VERIFY_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(VERIFY_ROOT) not in sys.path:
    sys.path.insert(0, str(VERIFY_ROOT))

from src.streaming.raw_eeg_loader import load_raw_dtu_file
from src.streaming.dual_stream_ingestor import DualStreamIngestionEngine
from src.audio.steering_engine import AudioSteeringDSP
from models.catcn import CATCNDirectDecoder
from src.models.spatial_adapter import SpatialEEGAdapter
from src.selective_aad.temporal_gate import StickyHysteresisGate, SignalQualityMonitor


def resolve_candidate_path(candidates):
    for c in candidates:
        p = Path(c)
        if p.exists():
            return p
    return None


def find_audio_file(audio_dir: Path, target_filename: str):
    cand = audio_dir / target_filename
    if cand.exists():
        return cand
    matches = list(audio_dir.rglob(target_filename))
    if matches:
        return matches[0]
    for p in audio_dir.rglob("*.wav"):
        if p.name.lower() == target_filename.lower():
            return p
    return None


def find_model_checkpoint(subject: str, explicit_path: str = None):
    if explicit_path and Path(explicit_path).exists():
        return Path(explicit_path)
        
    candidates = [
        Path(f"catcn_loso_{subject}.pt"),
        Path(f"checkpoints/catcn_loso_{subject}.pt"),
        Path("/kaggle/working/loso_checkpoints") / f"catcn_loso_{subject}.pt",
        Path("/kaggle/working/ISEF_Project") / f"catcn_loso_{subject}.pt",
        Path("/kaggle/working") / f"catcn_loso_{subject}.pt",
        Path("/kaggle/input") / f"catcn_loso_{subject}.pt",
        Path("checkpoints/best_catcn_model.pt"),
        REPO_ROOT / f"catcn_loso_{subject}.pt",
    ]
    for c in candidates:
        if c.exists():
            return c
            
    # Search recursively in /kaggle/input or /kaggle/working if available
    for search_dir in ["/kaggle/working", "/kaggle/input"]:
        if Path(search_dir).exists():
            pts = list(Path(search_dir).rglob("*.pt"))
            for p in pts:
                if subject.lower() in p.name.lower():
                    return p
            if pts:
                return pts[0]
            
    return None


def run_trial_streaming(
    raw_sub,
    trial_idx: int,
    mapping: dict,
    audio_dir: Path,
    model,
    adapter,
    device,
    args,
    verbose: bool = True
):
    sub_key = args.subject.split("_")[0]
    trial_key = f"trial_{trial_idx}"
    if sub_key not in mapping or trial_key not in mapping[sub_key]:
        if verbose:
            print(f"[ERROR] Mapping missing for {sub_key} {trial_key}")
        return None

    wav_a_name = mapping[sub_key][trial_key]["wavA"]["filename"]
    wav_b_name = mapping[sub_key][trial_key]["wavB"]["filename"]
    
    wav_a_path = find_audio_file(audio_dir, wav_a_name)
    wav_b_path = find_audio_file(audio_dir, wav_b_name)
    if wav_a_path is None or wav_b_path is None:
        if verbose:
            print(f"[ERROR] Missing audio files for Trial {trial_idx}: wavA={wav_a_path}, wavB={wav_b_path}")
        return None

    fs_a, raw_audio_a = wavfile.read(str(wav_a_path))
    fs_b, raw_audio_b = wavfile.read(str(wav_b_path))
    if fs_a != fs_b:
        if verbose:
            print(f"[ERROR] Sample rates differ: A={fs_a} Hz, B={fs_b} Hz")
        return None
    audio_fs = float(fs_a)

    # Initialize DSP & Streaming Engines
    gate = StickyHysteresisGate(
        alpha=0.82,
        threshold_switch=0.35,
        threshold_maintain=0.20,
        n_confirm=2,
        temperature=0.69
    )
    steering_dsp = AudioSteeringDSP(fs=int(audio_fs), tau_ms=100.0, threshold_switch=0.35)
    
    dual_engine = DualStreamIngestionEngine(
        raw_eeg_fs=raw_sub.fs,
        audio_fs=audio_fs,
        target_fs=64.0,
        window_sec=args.window_sec,
        hop_sec=args.hop_sec,
        power_exponent=args.power_exponent
    )

    # Extract Trial Data
    trial_meta = raw_sub.trials[trial_idx] if trial_idx < len(raw_sub.trials) else None
    if trial_meta:
        start_s = trial_meta.start_sample
        end_s = trial_meta.end_sample
    else:
        start_s = 0
        end_s = int(args.max_seconds * raw_sub.fs)
        
    max_samples = int(args.max_seconds * raw_sub.fs)
    end_s = min(end_s, start_s + max_samples)
    
    raw_scalp = raw_sub.eeg_raw[start_s:end_s]
    raw_veog = raw_sub.veog_raw[start_s:end_s]
    raw_heog = raw_sub.heog_raw[start_s:end_s]
    
    eeg_block_samples = 16 # 31.25 ms @ 512 Hz
    audio_block_samples = int(round(audio_fs * (eeg_block_samples / raw_sub.fs))) # ~1378 samples
    
    n_ticks = min(
        len(raw_scalp) // eeg_block_samples,
        len(raw_audio_a) // audio_block_samples,
        len(raw_audio_b) // audio_block_samples
    )

    if verbose:
        print(f"[TRIAL] Attended Stream (wavA):   {wav_a_name}")
        print(f"[TRIAL] Unattended Stream (wavB): {wav_b_name}")
        print(f"[INPUT] Loaded candidate audios at {audio_fs} Hz: wavA={len(raw_audio_a)} smp, wavB={len(raw_audio_b)} smp")
        print("\n" + "-" * 115)
        print(f" {'TIME':<7} | {'CHUNK':<10} | {'EEG PRE (us)':<12} | {'AUD A (us)':<10} | {'AUD B (us)':<10} | {'GPU INF (ms)':<12} | {'MARGIN s_t':<10} | {'STATE':<7} | {'RTF':<8}")
        print("-" * 115)

    # Streaming Hardware Simulation Loop
    total_eeg_us = 0.0
    total_aud_a_us = 0.0
    total_aud_b_us = 0.0
    total_gpu_ms = 0.0
    total_evals = 0
    correct_evals = 0
    state_counts = {"A": 0, "B": 0, "HOLD": 0}
    steered_audio_chunks = []
    recorded_margins = []

    for t_idx in range(n_ticks):
        e_start = t_idx * eeg_block_samples
        e_end = e_start + eeg_block_samples
        e_chunk = raw_scalp[e_start:e_end]
        v_chunk = raw_veog[e_start:e_end]
        h_chunk = raw_heog[e_start:e_end]
        
        a_start = t_idx * audio_block_samples
        a_end = a_start + audio_block_samples
        chunk_a = raw_audio_a[a_start:a_end]
        chunk_b = raw_audio_b[a_start:a_end]
        
        frame = dual_engine.step(e_chunk, chunk_a, chunk_b, v_chunk, h_chunk)
        
        total_eeg_us += frame.dsp_timing_us.get("eeg_us", 0.0)
        total_aud_a_us += frame.dsp_timing_us.get("audio_a_us", 0.0)
        total_aud_b_us += frame.dsp_timing_us.get("audio_b_us", 0.0)

        if frame.ready_for_eval:
            total_evals += 1
            if device.type == "cuda":
                torch.cuda.synchronize()
            t_gpu_start = time.perf_counter()
            
            with torch.no_grad():
                t_eeg = torch.from_numpy(frame.eeg_window).unsqueeze(0).to(device)
                t_ya = torch.from_numpy(frame.audio_a_window).unsqueeze(0).to(device)
                t_yb = torch.from_numpy(frame.audio_b_window).unsqueeze(0).to(device)
                
                t_eeg_adapted = adapter(t_eeg)
                delta, _, _ = model(t_eeg_adapted, t_ya, t_yb)
                raw_margin = delta.item()
                
            if device.type == "cuda":
                torch.cuda.synchronize()
            gpu_lat_ms = (time.perf_counter() - t_gpu_start) * 1000.0
            total_gpu_ms += gpu_lat_ms
            
            recorded_margins.append(raw_margin)
            if raw_margin > 0:
                correct_evals += 1
                
            gate_out = gate.update(raw_margin)
            state = gate_out["decision"]
            smooth_margin = gate_out["smoothed_margin"]
            is_switching = gate_out["switched"]
            state_counts[state] += 1
            
            if args.save_audio and verbose:
                out_chunk = steering_dsp.process_frame(chunk_a, chunk_b, state, smooth_margin)
                steered_audio_chunks.append(out_chunk)
                
            if verbose:
                cur_sec = (t_idx * eeg_block_samples) / raw_sub.fs
                tick_dsp_time_s = (frame.dsp_timing_us.get("eeg_us", 0.0) + frame.dsp_timing_us.get("audio_a_us", 0.0) + frame.dsp_timing_us.get("audio_b_us", 0.0)) * 1e-6
                total_cycle_time_s = tick_dsp_time_s + (gpu_lat_ms * 1e-3)
                window_rtf = total_cycle_time_s / args.hop_sec
                print(f" {cur_sec:5.1f}s | Block {t_idx:<6} | {frame.dsp_timing_us.get('eeg_us', 0.0):10.1f} us | {frame.dsp_timing_us.get('audio_a_us', 0.0):8.1f} us | {frame.dsp_timing_us.get('audio_b_us', 0.0):8.1f} us | {gpu_lat_ms:10.2f} ms | s={smooth_margin:+6.2f}   | {state:<7} | {window_rtf:6.3f}x")
        else:
            if args.save_audio and verbose:
                decision_label = "A" if "A" in gate.current_state else ("B" if "B" in gate.current_state else "HOLD")
                out_chunk = steering_dsp.process_frame(chunk_a, chunk_b, decision_label, gate.smoothed_margin)
                steered_audio_chunks.append(out_chunk)

    total_stream_sec = n_ticks * (eeg_block_samples / raw_sub.fs)
    eeg_cpu_pct = (total_eeg_us * 1e-6 / max(1e-6, total_stream_sec)) * 100.0
    aud_a_cpu_pct = (total_aud_a_us * 1e-6 / max(1e-6, total_stream_sec)) * 100.0
    aud_b_cpu_pct = (total_aud_b_us * 1e-6 / max(1e-6, total_stream_sec)) * 100.0
    total_dsp_pct = eeg_cpu_pct + aud_a_cpu_pct + aud_b_cpu_pct
    avg_gpu_latency = (total_gpu_ms / max(1, total_evals))
    total_compute_sec = (total_eeg_us + total_aud_a_us + total_aud_b_us) * 1e-6 + (total_gpu_ms * 1e-3)
    overall_rtf = total_compute_sec / max(1e-6, total_stream_sec)

    margins_arr = np.array(recorded_margins) if recorded_margins else np.array([])
    acc_window = (correct_evals / max(1, total_evals)) * 100.0
    
    step_10s = int(round(10.0 / args.hop_sec))
    corr_10s, total_10s = 0, 0
    for i in range(0, len(margins_arr) - step_10s + 1, step_10s):
        if np.sum(margins_arr[i:i + step_10s]) > 0:
            corr_10s += 1
        total_10s += 1
    acc_10s = (corr_10s / max(1, total_10s)) * 100.0 if total_10s > 0 else 0.0
    
    step_20s = int(round(20.0 / args.hop_sec))
    corr_20s, total_20s = 0, 0
    for i in range(0, len(margins_arr) - step_20s + 1, step_20s):
        if np.sum(margins_arr[i:i + step_20s]) > 0:
            corr_20s += 1
        total_20s += 1
    acc_20s = (corr_20s / max(1, total_20s)) * 100.0 if total_20s > 0 else 0.0
    
    cum_margin = float(np.sum(margins_arr)) if len(margins_arr) > 0 else 0.0
    mean_margin = float(np.mean(margins_arr)) if len(margins_arr) > 0 else 0.0
    trial_winner_cum = cum_margin > 0
    trial_winner_maj = correct_evals > (total_evals / 2)
    
    return {
        "trial": trial_idx,
        "n_ticks": n_ticks,
        "total_stream_sec": total_stream_sec,
        "total_evals": total_evals,
        "correct_evals": correct_evals,
        "acc_window": acc_window,
        "corr_10s": corr_10s,
        "total_10s": total_10s,
        "acc_10s": acc_10s,
        "corr_20s": corr_20s,
        "total_20s": total_20s,
        "acc_20s": acc_20s,
        "cum_margin": cum_margin,
        "mean_margin": mean_margin,
        "trial_winner_cum": trial_winner_cum,
        "trial_winner_maj": trial_winner_maj,
        "state_counts": state_counts,
        "eeg_cpu_pct": eeg_cpu_pct,
        "aud_a_cpu_pct": aud_a_cpu_pct,
        "aud_b_cpu_pct": aud_b_cpu_pct,
        "total_dsp_pct": total_dsp_pct,
        "avg_gpu_latency": avg_gpu_latency,
        "overall_rtf": overall_rtf,
        "steered_audio_chunks": steered_audio_chunks,
        "audio_fs": audio_fs
    }


def main():
    parser = argparse.ArgumentParser(description="End-to-End Dual-Stream (Raw EEG + Raw Audio) Streaming Benchmark")
    parser.add_argument("--raw_mat", type=str, default="/kaggle/input/datasets/lokeshgile/raw-s1-dtu/S1.mat")
    parser.add_argument("--audio_dir", type=str, default="/kaggle/input/datasets/lokeshgile/eeg-audio")
    parser.add_argument("--mapping_file", type=str, default="scripts/verify_baseline/data/audio_mapping.json")
    parser.add_argument("--checkpoint_path", type=str, default=None)
    parser.add_argument("--subject", type=str, default="S1")
    parser.add_argument("--trial", type=int, default=15, help="Single trial index to benchmark")
    parser.add_argument("--trials", type=str, default="", help="Multi-trial range: 'all', '0-59', or comma-separated '0,1,2,5,10,15'")
    parser.add_argument("--window_sec", type=float, default=5.0, help="Sliding evaluation window in seconds (e.g. 5.0, 10.0, 20.0)")
    parser.add_argument("--hop_sec", type=float, default=0.5, help="Evaluation hop cadence in seconds")
    parser.add_argument("--power_exponent", type=float, default=0.3)
    parser.add_argument("--max_seconds", type=float, default=50.0)
    parser.add_argument("--save_audio", action="store_true")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    
    device = torch.device(args.device)
    
    # 1. Resolve Raw EEG File
    raw_mat_cand = [
        Path(args.raw_mat),
        Path(f"/kaggle/input/datasets/lokeshgile/raw-s1-dtu/{args.subject}.mat"),
        Path(f"/kaggle/input/raw-s1-dtu/{args.subject}.mat"),
        Path(r"C:\Users\lokes\Downloads") / f"{args.subject}.mat"
    ]
    raw_mat_path = resolve_candidate_path(raw_mat_cand)
    if raw_mat_path is None:
        print(f"[ERROR] Could not locate raw EEG file: {args.raw_mat}")
        sys.exit(1)
        
    print(f"[INPUT] Loading raw DTU BioSemi ActiveTwo file: {raw_mat_path}")
    raw_sub = load_raw_dtu_file(raw_mat_path)
    print(f"        Subject: {raw_sub.subject_id} | Sample Rate: {raw_sub.fs} Hz | Channels: {len(raw_sub.channel_names)} | Total Trials: {len(raw_sub.trials)}")
    
    # 2. Resolve Audio Mapping & Directory
    map_cand = [
        REPO_ROOT / args.mapping_file,
        Path(args.mapping_file),
        Path("/kaggle/working/ISEF_Project/scripts/verify_baseline/data/audio_mapping.json")
    ]
    map_path = resolve_candidate_path(map_cand)
    if map_path is None:
        print(f"[ERROR] Could not locate audio_mapping.json")
        sys.exit(1)
        
    with open(map_path, "r", encoding="utf-8") as f:
        mapping = json.load(f)
        
    audio_dir_cand = [
        Path(args.audio_dir),
        Path("/kaggle/input/eeg-audio"),
        Path("/kaggle/input/datasets/lokeshgile/eeg-audio"),
        REPO_ROOT / "data" / "audio"
    ]
    audio_dir = resolve_candidate_path(audio_dir_cand)
    if audio_dir is None:
        print(f"[ERROR] Could not locate raw audio directory")
        sys.exit(1)

    # 3. Model Architecture & Checkpoint Setup
    print("\n[MODEL] Initializing CA-TCN Direct Decoder & 8-Channel Spatial Adapter...")
    adapter = SpatialEEGAdapter(channels=8).to(device)
    model = CATCNDirectDecoder(eeg_channels=8, audio_channels=1, hidden_dim=64, max_lag_samples=8).to(device)
    
    ckpt_path = find_model_checkpoint(args.subject, args.checkpoint_path)
    if ckpt_path:
        print(f"[MODEL] Loading pre-trained checkpoint: {ckpt_path}")
        try:
            ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
        except TypeError:
            ckpt = torch.load(str(ckpt_path), map_location=device)
        if "model_state_dict" in ckpt:
            model.load_state_dict(ckpt["model_state_dict"])
        elif "model" in ckpt:
            model.load_state_dict(ckpt["model"])
        else:
            model.load_state_dict(ckpt)
        if "adapter_state_dict" in ckpt and ckpt["adapter_state_dict"] is not None:
            adapter.load_state_dict(ckpt["adapter_state_dict"])
    else:
        print(f"[WARN] No pre-trained checkpoint found for {args.subject}. Initializing architecture for latency profiling.")
        
    model.eval()
    adapter.eval()

    # Determine trials to run
    if args.trials:
        if args.trials.lower() == "all":
            trials_to_run = list(range(len(raw_sub.trials)))
        elif "-" in args.trials:
            s_str, e_str = args.trials.split("-")
            trials_to_run = list(range(int(s_str), int(e_str) + 1))
        else:
            trials_to_run = [int(x.strip()) for x in args.trials.split(",") if x.strip()]
    else:
        trials_to_run = [args.trial]

    is_multi_trial = len(trials_to_run) > 1

    print("=" * 115)
    print("  AUDITORY ATTENTION DECODING: DUAL-STREAM (RAW EEG + RAW AUDIO) REAL-TIME BENCHMARK")
    print(f"  Device: {device} | Subject: {args.subject} | Trials: {len(trials_to_run)} trials | Hop: {args.hop_sec*1e3:.1f} ms | Window: {args.window_sec:.1f} s")
    print("=" * 115)

    if not is_multi_trial:
        # Single-trial detailed execution
        res = run_trial_streaming(
            raw_sub, trials_to_run[0], mapping, audio_dir, model, adapter, device, args, verbose=True
        )
        if res is None:
            sys.exit(1)
            
        print("\n" + "=" * 115)
        print("  DUAL-STREAM REAL-TIME STREAMING BENCHMARK RESULTS")
        print(f"  Total Simulated Audio/EEG Streamed: {res['total_stream_sec']:.1f} seconds ({res['n_ticks']} hardware packets)")
        print(f"  EEG Preprocessing CPU Load:         {res['eeg_cpu_pct']:6.2f}% of single CPU core")
        print(f"  Audio A Gammatone CPU Load:         {res['aud_a_cpu_pct']:6.2f}% of single CPU core")
        print(f"  Audio B Gammatone CPU Load:         {res['aud_b_cpu_pct']:6.2f}% of single CPU core")
        print(f"  Combined Total DSP CPU Load:        {res['total_dsp_pct']:6.2f}% of single CPU core (Headroom: {100.0 - res['total_dsp_pct']:.1f}% idle)")
        print(f"  Average GPU CA-TCN Latency:         {res['avg_gpu_latency']:6.2f} milliseconds per evaluation")
        print(f"  Total Real-Time Factor (RTF):       {res['overall_rtf']:6.4f}x (Budget < 1.0x, Speedup = {1.0/max(1e-6, res['overall_rtf']):.1f}x)")
        print("-" * 115)
        print(f"  ATTENTION DECODING TELEMETRY (Subject {args.subject} | Trial {res['trial']}):")
        print(f"    - Instantaneous {args.window_sec:.1f}s Window 2AFC: {res['acc_window']:5.1f}% ({res['correct_evals']}/{res['total_evals']} windows)")
        print(f"    - Integrated 10.0s Window 2AFC:   {res['acc_10s']:5.1f}% ({res['corr_10s']}/{res['total_10s']} windows)")
        print(f"    - Integrated 20.0s Window 2AFC:   {res['acc_20s']:5.1f}% ({res['corr_20s']}/{res['total_20s']} windows)")
        print(f"    - Cumulative Neural Margin:       {res['cum_margin']:+7.2f} (Mean: {res['mean_margin']:+0.3f})")
        trial_win_str = "Talker A (Attended - CORRECT)" if res['trial_winner_cum'] else "Talker B (Unattended - WRONG)"
        print(f"    - Full Trial Winner Decision:     {trial_win_str}")
        print(f"    - Acoustic Gating Distribution:   A={res['state_counts'].get('A', 0)} frames, B={res['state_counts'].get('B', 0)} frames, HOLD={res['state_counts'].get('HOLD', 0)} frames")
        print("=" * 115)
        
        if args.save_audio and res['steered_audio_chunks']:
            out_wav = Path(f"audit_results/steered_{args.subject}_trial{res['trial']}.wav")
            out_wav.parent.mkdir(parents=True, exist_ok=True)
            steered_audio_full = np.concatenate(res['steered_audio_chunks'], axis=0)
            wavfile.write(str(out_wav), int(res['audio_fs']), (np.clip(steered_audio_full, -1.0, 1.0) * 32767).astype(np.int16))
            print(f"\n[ARTIFACT] Saved steered stereo audio output to: {out_wav}")

    else:
        # Multi-trial aggregate execution
        print(f"\n[STREAMING] Commencing multi-trial continuous streaming across {len(trials_to_run)} trials...\n")
        all_results = []
        
        for idx, t_num in enumerate(trials_to_run):
            res = run_trial_streaming(
                raw_sub, t_num, mapping, audio_dir, model, adapter, device, args, verbose=False
            )
            if res is None:
                continue
            all_results.append(res)
            
            win_tag = "CORRECT" if res['trial_winner_cum'] else "WRONG"
            print(f"  [TRIAL {t_num:02d} ({idx+1:02d}/{len(trials_to_run):02d})] "
                  f"Margin: {res['cum_margin']:+6.2f} (mean {res['mean_margin']:+0.3f}) | "
                  f"{args.window_sec:.0f}s: {res['acc_window']:5.1f}% | "
                  f"10s: {res['acc_10s']:5.1f}% | "
                  f"20s: {res['acc_20s']:5.1f}% | "
                  f"Winner: {win_tag:<7} | "
                  f"RTF: {res['overall_rtf']:0.3f}x")

        if not all_results:
            print("[ERROR] No trials successfully streamed.")
            sys.exit(1)

        # Master Aggregations
        tot_streamed_sec = sum(r['total_stream_sec'] for r in all_results)
        tot_hardware_packets = sum(r['n_ticks'] for r in all_results)
        tot_evals = sum(r['total_evals'] for r in all_results)
        tot_corr = sum(r['correct_evals'] for r in all_results)
        tot_10s = sum(r['total_10s'] for r in all_results)
        corr_10s = sum(r['corr_10s'] for r in all_results)
        tot_20s = sum(r['total_20s'] for r in all_results)
        corr_20s = sum(r['corr_20s'] for r in all_results)
        
        cum_winners = sum(1 for r in all_results if r['trial_winner_cum'])
        maj_winners = sum(1 for r in all_results if r['trial_winner_maj'])
        n_trials = len(all_results)
        
        mean_dsp_cpu = np.mean([r['total_dsp_pct'] for r in all_results])
        mean_gpu_lat = np.mean([r['avg_gpu_latency'] for r in all_results])
        mean_rtf = np.mean([r['overall_rtf'] for r in all_results])
        grand_mean_margin = np.mean([r['cum_margin'] for r in all_results])
        
        acc_win_overall = (tot_corr / max(1, tot_evals)) * 100.0
        acc_10s_overall = (corr_10s / max(1, tot_10s)) * 100.0 if tot_10s > 0 else 0.0
        acc_20s_overall = (corr_20s / max(1, tot_20s)) * 100.0 if tot_20s > 0 else 0.0
        maj_acc = (maj_winners / max(1, n_trials)) * 100.0
        cum_acc = (cum_winners / max(1, n_trials)) * 100.0

        print("\n" + "=" * 115)
        print(f"  MULTI-TRIAL REAL-TIME STREAMING BENCHMARK RESULTS (Subject {args.subject}: N={n_trials} Trials)")
        print(f"  Total Streamed Audio/EEG:           {tot_streamed_sec:.1f} seconds ({tot_streamed_sec/60.0:.1f} minutes)")
        print(f"  Total Hardware Packets Processed:   {tot_hardware_packets} packets (31.25 ms each)")
        print(f"  Combined Total DSP CPU Load:        {mean_dsp_cpu:6.2f}% of single CPU core (Headroom: {100.0 - mean_dsp_cpu:.1f}% idle)")
        print(f"  Average GPU CA-TCN Latency:         {mean_gpu_lat:6.2f} milliseconds per evaluation")
        print(f"  Overall Real-Time Factor (RTF):     {mean_rtf:6.4f}x (Speedup: {1.0/max(1e-6, mean_rtf):.1f}x faster than real-time)")
        print("-" * 115)
        print(f"  MULTI-SCALE ATTENTION DECODING ACCURACY (Subject {args.subject}):")
        print(f"    - Instantaneous {args.window_sec:.1f}s Window 2AFC: {acc_win_overall:5.2f}% ({tot_corr}/{tot_evals} windows)")
        print(f"    - Integrated 10.0s Window 2AFC:   {acc_10s_overall:5.2f}% ({corr_10s}/{tot_10s} windows)")
        print(f"    - Integrated 20.0s Window 2AFC:   {acc_20s_overall:5.2f}% ({corr_20s}/{tot_20s} windows)")
        print(f"    - Trial Majority Vote Accuracy:   {maj_acc:5.2f}% ({maj_winners}/{n_trials} trials)")
        print(f"    - Cumulative Margin Winner:       {cum_acc:5.2f}% ({cum_winners}/{n_trials} trials)")
        print(f"    - Grand Mean Trial Neural Margin: {grand_mean_margin:+7.2f}")
        print("=" * 115)


if __name__ == "__main__":
    main()
