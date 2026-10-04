"""
Joint End-to-End Real-Time Streaming Benchmark: Concurrent Raw EEG + Raw Audio Ingestion.

Benchmarks full hardware-simulated streaming:
1. Ingests raw continuous 512 Hz multi-channel EEG (S<id>.mat).
2. Ingests raw audio candidate streams (44.1 kHz / 16 kHz WAV) for Talker A and Talker B.
3. Synchronizes through DualStreamIngestionEngine (31.25 ms lock-step chunks).
4. Evaluates CA-TCN direct match decoder + Spatial Adapter + Sticky Hysteresis Gate on GPU.
5. Executes AudioSteeringDSP (soft 100 ms crossfade) and benchmarks total CPU/GPU load and RTF.
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
        Path("/kaggle/working/ISEF_Project") / f"catcn_loso_{subject}.pt",
        Path("/kaggle/working") / f"catcn_loso_{subject}.pt",
        Path("/kaggle/input") / f"catcn_loso_{subject}.pt",
        Path("checkpoints/best_catcn_model.pt"),
        REPO_ROOT / f"catcn_loso_{subject}.pt",
    ]
    for c in candidates:
        if c.exists():
            return c
            
    # Search recursively in /kaggle/input if available
    if Path("/kaggle/input").exists():
        pts = list(Path("/kaggle/input").rglob("*.pt"))
        for p in pts:
            if subject.lower() in p.name.lower():
                return p
        if pts:
            return pts[0]
            
    return None


def main():
    parser = argparse.ArgumentParser(description="End-to-End Dual-Stream (Raw EEG + Raw Audio) Streaming Benchmark")
    parser.add_argument("--raw_mat", type=str, default="/kaggle/input/datasets/lokeshgile/raw-s1-dtu/S1.mat")
    parser.add_argument("--audio_dir", type=str, default="/kaggle/input/datasets/lokeshgile/eeg-audio")
    parser.add_argument("--mapping_file", type=str, default="scripts/verify_baseline/data/audio_mapping.json")
    parser.add_argument("--checkpoint_path", type=str, default=None)
    parser.add_argument("--subject", type=str, default="S1")
    parser.add_argument("--trial", type=int, default=15)
    parser.add_argument("--window_sec", type=float, default=5.0)
    parser.add_argument("--hop_sec", type=float, default=0.5)
    parser.add_argument("--power_exponent", type=float, default=0.3)
    parser.add_argument("--max_seconds", type=float, default=50.0)
    parser.add_argument("--save_audio", action="store_true")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    
    device = torch.device(args.device)
    
    print("=" * 115)
    print("  AUDITORY ATTENTION DECODING: DUAL-STREAM (RAW EEG + RAW AUDIO) REAL-TIME BENCHMARK")
    print(f"  Device: {device} | Subject: {args.subject} (Trial {args.trial}) | Hop: {args.hop_sec*1e3:.1f} ms | Window: {args.window_sec:.1f} s")
    print("=" * 115)
    
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
    print(f"        Subject: {raw_sub.subject_id} | Sample Rate: {raw_sub.fs} Hz | Channels: {len(raw_sub.channel_names)}")
    
    # 2. Resolve Audio Mapping & WAV Files
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
        
    sub_key = args.subject.split("_")[0]
    trial_key = f"trial_{args.trial}"
    if sub_key not in mapping or trial_key not in mapping[sub_key]:
        print(f"[ERROR] Mapping missing for {sub_key} {trial_key}")
        sys.exit(1)
        
    wav_a_name = mapping[sub_key][trial_key]["wavA"]["filename"]
    wav_b_name = mapping[sub_key][trial_key]["wavB"]["filename"]
    print(f"[TRIAL] Attended Stream (wavA):   {wav_a_name}")
    print(f"[TRIAL] Unattended Stream (wavB): {wav_b_name}")
    
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
        
    wav_a_path = find_audio_file(audio_dir, wav_a_name)
    wav_b_path = find_audio_file(audio_dir, wav_b_name)
    if wav_a_path is None or wav_b_path is None:
        print(f"[ERROR] Missing audio files: wavA={wav_a_path}, wavB={wav_b_path}")
        sys.exit(1)
        
    fs_a, raw_audio_a = wavfile.read(str(wav_a_path))
    fs_b, raw_audio_b = wavfile.read(str(wav_b_path))
    if fs_a != fs_b:
        print(f"[ERROR] Sample rates differ: A={fs_a} Hz, B={fs_b} Hz")
        sys.exit(1)
    audio_fs = float(fs_a)
    print(f"[INPUT] Loaded candidate audios at {audio_fs} Hz: wavA={len(raw_audio_a)} smp, wavB={len(raw_audio_b)} smp")
    
    # 3. Model Architecture & Checkpoint Setup
    print("\n[MODEL] Initializing CA-TCN Direct Decoder & 8-Channel Spatial Adapter...")
    adapter = SpatialEEGAdapter(channels=8).to(device)
    model = CATCNDirectDecoder(eeg_channels=8, audio_channels=1, hidden_dim=64, max_lag_samples=8).to(device)
    
    ckpt_path = find_model_checkpoint(args.subject, args.checkpoint_path)
    if ckpt_path:
        print(f"[MODEL] Loading pre-trained checkpoint: {ckpt_path}")
        ckpt = torch.load(str(ckpt_path), map_location=device)
        if "model_state_dict" in ckpt:
            model.load_state_dict(ckpt["model_state_dict"])
        elif "model" in ckpt:
            model.load_state_dict(ckpt["model"])
        else:
            model.load_state_dict(ckpt)
        if "adapter_state_dict" in ckpt:
            adapter.load_state_dict(ckpt["adapter_state_dict"])
    else:
        print(f"[WARN] No pre-trained checkpoint found for {args.subject}. Initializing architecture for latency profiling.")
        
    model.eval()
    adapter.eval()
    
    # 4. Initialize DSP & Streaming Engines
    gate = StickyHysteresisGate(theta=0.08, tau=3)
    sq_monitor = SignalQualityMonitor(fs=64.0)
    steering_dsp = AudioSteeringDSP(sr=audio_fs, crossfade_ms=100.0)
    
    dual_engine = DualStreamIngestionEngine(
        raw_eeg_fs=raw_sub.fs,
        audio_fs=audio_fs,
        target_fs=64.0,
        window_sec=args.window_sec,
        hop_sec=args.hop_sec,
        power_exponent=args.power_exponent
    )
    
    # 5. Extract Trial Data
    trial_meta = raw_sub.trials[args.trial] if args.trial < len(raw_sub.trials) else None
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
    
    print("\n" + "-" * 115)
    print(f" {'TIME':<7} | {'CHUNK':<10} | {'EEG PRE (us)':<12} | {'AUD A (us)':<10} | {'AUD B (us)':<10} | {'GPU INF (ms)':<12} | {'MARGIN s_t':<10} | {'STATE':<7} | {'RTF':<8}")
    print("-" * 115)
    
    # 6. Real-Time Streaming Hardware Simulation Loop
    total_eeg_us = 0.0
    total_aud_a_us = 0.0
    total_aud_b_us = 0.0
    total_gpu_ms = 0.0
    total_evals = 0
    correct_evals = 0
    state_counts = {"ATTEND_A": 0, "ATTEND_B": 0, "HOLD": 0}
    steered_audio_chunks = []
    
    for t_idx in range(n_ticks):
        # Slice hardware chunks
        e_start = t_idx * eeg_block_samples
        e_end = e_start + eeg_block_samples
        e_chunk = raw_scalp[e_start:e_end]
        v_chunk = raw_veog[e_start:e_end]
        h_chunk = raw_heog[e_start:e_end]
        
        a_start = t_idx * audio_block_samples
        a_end = a_start + audio_block_samples
        chunk_a = raw_audio_a[a_start:a_end]
        chunk_b = raw_audio_b[a_start:a_end]
        
        # Ingest through Dual-Stream Engine
        frame = dual_engine.step(e_chunk, chunk_a, chunk_b, v_chunk, h_chunk)
        
        total_eeg_us += frame.dsp_timing_us.get("eeg_us", 0.0)
        total_aud_a_us += frame.dsp_timing_us.get("audio_a_us", 0.0)
        total_aud_b_us += frame.dsp_timing_us.get("audio_b_us", 0.0)
        
        # When 0.5s evaluation cadence triggers
        if frame.ready_for_eval:
            total_evals += 1
            
            # CUDA Inference Timing
            if device.type == "cuda":
                torch.cuda.synchronize()
            t_gpu0 = time.perf_counter()
            
            with torch.no_grad():
                eeg_t = torch.from_numpy(frame.eeg_window).unsqueeze(0).to(device)       # [1, 8, 320]
                ad_eeg = adapter(eeg_t)                                                    # [1, 8, 320]
                ya_t = torch.from_numpy(frame.audio_a_window).unsqueeze(0).to(device)    # [1, 1, 320]
                yb_t = torch.from_numpy(frame.audio_b_window).unsqueeze(0).to(device)    # [1, 1, 320]
                
                delta, _, _ = model(ad_eeg, ya_t, yb_t)
                s_t = float(delta.item())
                
            if device.type == "cuda":
                torch.cuda.synchronize()
            t_gpu_ms = (time.perf_counter() - t_gpu0) * 1e3
            total_gpu_ms += t_gpu_ms
            
            # Signal Quality Check & Hysteresis Decision
            sq = sq_monitor.check_eeg_window(frame.eeg_window.T)
            gate_out = gate.update(s_t, is_artifact=not sq["is_valid"])
            dec = gate_out["decision"]
            state_counts[dec] = state_counts.get(dec, 0) + 1
            
            if s_t > 0:
                correct_evals += 1
                
            # Real-Time Audio Steering
            steered_chunk = steering_dsp.process_frame(chunk_a, chunk_b, decision=dec)
            if args.save_audio:
                steered_audio_chunks.append(steered_chunk)
                
            sim_time = (e_end / raw_sub.fs)
            # RTF = (Total Compute Time during this 0.5s window) / (0.5s window duration)
            dsp_time_window_sec = (frame.dsp_timing_us.get("total_dsp_us", 0.0) * 16.0) / 1e6 # 16 ticks in 0.5s
            rtf = (dsp_time_window_sec + (t_gpu_ms / 1e3)) / args.hop_sec
            
            eeg_us = frame.dsp_timing_us.get("eeg_us", 0.0)
            aud_a_us = frame.dsp_timing_us.get("audio_a_us", 0.0)
            aud_b_us = frame.dsp_timing_us.get("audio_b_us", 0.0)
            print(f" {sim_time:5.1f}s | Block {t_idx:<5} | {eeg_us:10.1f} us | {aud_a_us:8.1f} us | {aud_b_us:8.1f} us | {t_gpu_ms:10.2f} ms | s={gate_out['smoothed_margin']:+6.2f}   | {dec:<7} | {rtf:6.3f}x")
            
    print("-" * 115)
    
    # 7. Summary Benchmark Report
    total_stream_sec = (n_ticks * eeg_block_samples) / raw_sub.fs
    eeg_cpu_pct = (total_eeg_us / 1e6) / total_stream_sec * 100.0
    aud_a_cpu_pct = (total_aud_a_us / 1e6) / total_stream_sec * 100.0
    aud_b_cpu_pct = (total_aud_b_us / 1e6) / total_stream_sec * 100.0
    total_dsp_pct = eeg_cpu_pct + aud_a_cpu_pct + aud_b_cpu_pct
    avg_gpu_latency = total_gpu_ms / max(1, total_evals)
    
    total_compute_sec = ((total_eeg_us + total_aud_a_us + total_aud_b_us) / 1e6) + (total_gpu_ms / 1e3)
    overall_rtf = total_compute_sec / total_stream_sec
    
    print("\n" + "=" * 115)
    print("  DUAL-STREAM REAL-TIME STREAMING BENCHMARK RESULTS")
    print(f"  Total Simulated Audio/EEG Streamed: {total_stream_sec:.1f} seconds ({n_ticks} hardware packets)")
    print(f"  EEG Preprocessing CPU Load:         {eeg_cpu_pct:6.2f}% of single CPU core")
    print(f"  Audio A Gammatone CPU Load:         {aud_a_cpu_pct:6.2f}% of single CPU core")
    print(f"  Audio B Gammatone CPU Load:         {aud_b_cpu_pct:6.2f}% of single CPU core")
    print(f"  Combined Total DSP CPU Load:        {total_dsp_pct:6.2f}% of single CPU core")
    print(f"  Average GPU CA-TCN Latency:         {avg_gpu_latency:6.2f} milliseconds per evaluation")
    print(f"  Total Real-Time Factor (RTF):       {overall_rtf:6.4f}x (Budget < 1.0x, Speedup = {1.0/max(1e-6, overall_rtf):.1f}x)")
    print(f"  Decision Distribution:              ATTEND_A={state_counts['ATTEND_A']}, ATTEND_B={state_counts['ATTEND_B']}, HOLD={state_counts['HOLD']}")
    if total_evals > 0:
        print(f"  Raw Window Decision Accuracy:       {(correct_evals/total_evals)*100.0:.1f}% ({correct_evals}/{total_evals})")
    print("=" * 115)
    
    if args.save_audio and steered_audio_chunks:
        out_wav = Path(f"audit_results/steered_{args.subject}_trial{args.trial}.wav")
        out_wav.parent.mkdir(parents=True, exist_ok=True)
        steered_audio_full = np.concatenate(steered_audio_chunks, axis=0)
        wavfile.write(str(out_wav), int(audio_fs), (np.clip(steered_audio_full, -1.0, 1.0) * 32767).astype(np.int16))
        print(f"\n[ARTIFACT] Saved steered audio output to: {out_wav}")


if __name__ == "__main__":
    main()
