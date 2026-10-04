"""
Audio Parity Verification Script: Causal Real-Time Gammatone Extractor vs DTU MATLAB Baseline.

Streams raw Danish story WAV files through StreamingCausalAudioGammatoneExtractor in 31.25 ms chunks
and evaluates cross-correlation parity against data.wavA and data.wavB from S1_data_preproc.mat.
"""

import argparse
import sys
import json
import time
from pathlib import Path
import numpy as np
import scipy.io as sio
from scipy.io import wavfile
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.audio.causal_gammatone import StreamingCausalAudioGammatoneExtractor


def resolve_candidate_path(candidates):
    for c in candidates:
        p = Path(c)
        if p.exists():
            return p
    return None


def find_audio_file(audio_dir: Path, target_filename: str) -> Optional[Path]:
    cand = audio_dir / target_filename
    if cand.exists():
        return cand
    matches = list(audio_dir.rglob(target_filename))
    if matches:
        return matches[0]
    # Case-insensitive search fallback
    for p in audio_dir.rglob("*.wav"):
        if p.name.lower() == target_filename.lower():
            return p
    return None


def compute_parity_metrics(extracted: np.ndarray, reference: np.ndarray, fs: float = 64.0):
    min_len = min(len(extracted), len(reference))
    y_ext = extracted[:min_len].astype(np.float64)
    y_ref = reference[:min_len].astype(np.float64)
    
    # 1. Pearson Correlation
    ext_cent = y_ext - np.mean(y_ext)
    ref_cent = y_ref - np.mean(y_ref)
    denom = (np.linalg.norm(ext_cent) * np.linalg.norm(ref_cent)) + 1e-12
    r = float(np.dot(ext_cent, ref_cent) / denom)
    
    # 2. Normalized Mean Squared Error
    norm_ref = np.linalg.norm(ref_cent) + 1e-12
    nmse = float(np.mean((ext_cent - ref_cent) ** 2) / (norm_ref ** 2 / min_len))
    
    # 3. Empirical Group Delay (Cross-correlation lag)
    xcorr = np.correlate(ref_cent, ext_cent, mode='full')
    lags = np.arange(-len(ext_cent) + 1, len(ref_cent))
    best_lag_samples = lags[np.argmax(xcorr)]
    group_delay_ms = float((best_lag_samples / fs) * 1000.0)
    
    return {
        "r": r,
        "nmse": nmse,
        "group_delay_ms": group_delay_ms,
        "samples": min_len,
        "duration_sec": min_len / fs
    }


def main():
    parser = argparse.ArgumentParser(description="Audit Causal Audio Gammatone Extraction against DTU Baseline")
    parser.add_argument("--audio_dir", type=str, default="/kaggle/input/datasets/lokeshgile/eeg-audio")
    parser.add_argument("--preproc_mat", type=str, default="/kaggle/input/datasets/lokeshgile/dataset-eeg/S1_data_preproc.mat")
    parser.add_argument("--mapping_file", type=str, default="scripts/verify_baseline/data/audio_mapping.json")
    parser.add_argument("--subject", type=str, default="S1")
    parser.add_argument("--trial", type=int, default=0)
    parser.add_argument("--power_exponent", type=float, default=0.3, help="Power compression exponent (0.3 for DTU MATLAB, 0.6 for MatchNet pkl)")
    parser.add_argument("--chunk_ms", type=float, default=31.25, help="Simulated streaming packet duration in ms")
    parser.add_argument("--out_dir", type=str, default="audit_results")
    args = parser.parse_args()
    
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 100)
    print("  RAW AUDIO STREAMING GAMMATONE EXTRACTION: GROUND-TRUTH PARITY AUDIT")
    print(f"  Target Subject:   {args.subject} | Target Trial: Trial {args.trial}")
    print(f"  Power Exponent:   p = {args.power_exponent}")
    print(f"  Streaming Chunk:  {args.chunk_ms:.2f} ms")
    print("=" * 100)
    
    # 1. Resolve Audio Directory
    audio_dir_cand = [
        Path(args.audio_dir),
        Path("/kaggle/input/eeg-audio"),
        Path("/kaggle/input/datasets/lokeshgile/eeg-audio"),
        Path(r"C:\Users\lokes\Downloads\audio"),
        REPO_ROOT / "data" / "audio"
    ]
    audio_dir = resolve_candidate_path(audio_dir_cand)
    if audio_dir is None:
        print(f"[ERROR] Could not locate raw audio directory. Checked candidates: {[str(c) for c in audio_dir_cand]}")
        sys.exit(1)
    print(f"[INPUT] Located raw audio directory: {audio_dir}")
    
    # 2. Resolve Preprocessed MATLAB Reference File
    mat_cand = [
        Path(args.preproc_mat),
        Path("/kaggle/input/dataset-eeg/S1_data_preproc.mat"),
        Path("/kaggle/input/datasets/lokeshgile/dataset-eeg/S1_data_preproc.mat"),
        Path(r"C:\Users\lokes\Downloads\S1_data_preproc.mat"),
        Path(r"C:\Users\lokes\Downloads\archive (2)\DATA_preproc\S1_data_preproc.mat")
    ]
    preproc_mat_path = resolve_candidate_path(mat_cand)
    if preproc_mat_path is None:
        print(f"[ERROR] Could not locate preprocessed reference MAT file: {args.preproc_mat}")
        sys.exit(1)
    print(f"[INPUT] Located MATLAB ground truth MAT: {preproc_mat_path}")
    
    # 3. Load Audio Mapping
    mapping_cand = [
        REPO_ROOT / args.mapping_file,
        Path(args.mapping_file),
        Path("/kaggle/working/ISEF_Project/scripts/verify_baseline/data/audio_mapping.json")
    ]
    map_path = resolve_candidate_path(mapping_cand)
    if map_path is None or not map_path.exists():
        print(f"[ERROR] Could not locate audio_mapping.json")
        sys.exit(1)
        
    with open(map_path, "r", encoding="utf-8") as f:
        mapping = json.load(f)
        
    sub_key = args.subject.split("_")[0]
    trial_key = f"trial_{args.trial}"
    if sub_key not in mapping or trial_key not in mapping[sub_key]:
        print(f"[ERROR] Mapping not found for {sub_key} {trial_key}")
        sys.exit(1)
        
    trial_map = mapping[sub_key][trial_key]
    wav_a_name = trial_map["wavA"]["filename"]
    wav_b_name = trial_map["wavB"]["filename"]
    print(f"\n[TRIAL MAP] Attended Stream (wavA):   {wav_a_name}")
    print(f"[TRIAL MAP] Unattended Stream (wavB): {wav_b_name}")
    
    # 4. Load MATLAB Ground Truth Envelopes
    mat_data = sio.loadmat(str(preproc_mat_path), struct_as_record=False, squeeze_me=False)
    data_struct = mat_data["data"][0, 0]
    ref_wav_a = np.asarray(data_struct.wavA[0, args.trial], dtype=np.float64).ravel()
    ref_wav_b = np.asarray(data_struct.wavB[0, args.trial], dtype=np.float64).ravel()
    ref_fs = float(data_struct.fsample[0, 0].eeg[0, 0]) if hasattr(data_struct, "fsample") else 64.0
    print(f"\n[GROUND TRUTH] Loaded MATLAB envelopes: wavA shape={ref_wav_a.shape}, wavB shape={ref_wav_b.shape} @ {ref_fs} Hz")
    
    # 5. Locate & Load Raw WAV Files
    wav_a_file = find_audio_file(audio_dir, wav_a_name)
    wav_b_file = find_audio_file(audio_dir, wav_b_name)
    if wav_a_file is None or wav_b_file is None:
        print(f"[ERROR] Could not locate one or both WAV files in {audio_dir}: wavA={wav_a_file}, wavB={wav_b_file}")
        sys.exit(1)
        
    fs_a, raw_audio_a = wavfile.read(str(wav_a_file))
    fs_b, raw_audio_b = wavfile.read(str(wav_b_file))
    print(f"[INPUT] Loaded Raw Audio A: {wav_a_file.name} | fs={fs_a} Hz | samples={len(raw_audio_a)} ({len(raw_audio_a)/fs_a:.1f}s)")
    print(f"[INPUT] Loaded Raw Audio B: {wav_b_file.name} | fs={fs_b} Hz | samples={len(raw_audio_b)} ({len(raw_audio_b)/fs_b:.1f}s)")
    
    # 6. Stream Raw Audio through Causal Gammatone Extractor
    print(f"\n[PROCESSING] Streaming audio through Causal 28-Band Gammatone Filterbank (Chunk = {args.chunk_ms:.2f} ms)...")
    extractor_a = StreamingCausalAudioGammatoneExtractor(audio_fs=fs_a, target_fs=ref_fs, power_exponent=args.power_exponent)
    extractor_b = StreamingCausalAudioGammatoneExtractor(audio_fs=fs_b, target_fs=ref_fs, power_exponent=args.power_exponent)
    
    chunk_samples_a = int(round(fs_a * (args.chunk_ms / 1000.0)))
    chunk_samples_b = int(round(fs_b * (args.chunk_ms / 1000.0)))
    
    streamed_env_a = []
    streamed_env_b = []
    
    t_start = time.perf_counter()
    n_chunks_a = len(raw_audio_a) // chunk_samples_a
    for i in range(n_chunks_a):
        c_a = raw_audio_a[i * chunk_samples_a : (i + 1) * chunk_samples_a]
        out_a = extractor_a.process_audio_chunk(c_a)
        if len(out_a) > 0:
            streamed_env_a.append(out_a)
            
    n_chunks_b = len(raw_audio_b) // chunk_samples_b
    for i in range(n_chunks_b):
        c_b = raw_audio_b[i * chunk_samples_b : (i + 1) * chunk_samples_b]
        out_b = extractor_b.process_audio_chunk(c_b)
        if len(out_b) > 0:
            streamed_env_b.append(out_b)
            
    t_proc = time.perf_counter() - t_start
    env_a_extracted = np.concatenate(streamed_env_a, axis=0) if streamed_env_a else np.empty(0)
    env_b_extracted = np.concatenate(streamed_env_b, axis=0) if streamed_env_b else np.empty(0)
    print(f"  Streaming complete in {t_proc:.2f}s: Stream A = {len(env_a_extracted)} samples, Stream B = {len(env_b_extracted)} samples")
    
    # 7. Compute Parity Metrics
    print("\n" + "-" * 100)
    print(f" {'STREAM':<10} | {'AUDIO FILE':<32} | {'PEARSON r':<12} | {'NMSE':<10} | {'GROUP DELAY':<14} | {'STATUS':<8} ")
    print("-" * 100)
    
    metrics_a = compute_parity_metrics(env_a_extracted, ref_wav_a, fs=ref_fs)
    status_a = "PASS" if metrics_a["r"] >= 0.70 else "WARN"
    print(f" {'Talker A':<10} | {wav_a_name:<32} | r = {metrics_a['r']:+0.4f}   | {metrics_a['nmse']:<10.4f} | {metrics_a['group_delay_ms']:+8.1f} ms    | {status_a:<8} ")
    
    metrics_b = compute_parity_metrics(env_b_extracted, ref_wav_b, fs=ref_fs)
    status_b = "PASS" if metrics_b["r"] >= 0.70 else "WARN"
    print(f" {'Talker B':<10} | {wav_b_name:<32} | r = {metrics_b['r']:+0.4f}   | {metrics_b['nmse']:<10.4f} | {metrics_b['group_delay_ms']:+8.1f} ms    | {status_b:<8} ")
    print("-" * 100)
    
    # 8. Save Verification Plot
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    t_axis = np.arange(metrics_a["samples"]) / ref_fs
    
    axes[0].plot(t_axis[:640], ref_wav_a[:640], label="MATLAB Ground Truth (preproc_data.m)", color="black", alpha=0.8, lw=1.5)
    axes[0].plot(t_axis[:640], env_a_extracted[:640], label=f"Python Causal Streaming (r = {metrics_a['r']:.3f})", color="royalblue", alpha=0.8, lw=1.2)
    axes[0].set_title(f"Talker A ({wav_a_name}) - First 10 Seconds Envelope Overlay @ 64 Hz", fontsize=11, fontweight="bold")
    axes[0].set_ylabel("Broadband Envelope")
    axes[0].legend(loc="upper right")
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(t_axis[:640], ref_wav_b[:640], label="MATLAB Ground Truth (preproc_data.m)", color="black", alpha=0.8, lw=1.5)
    axes[1].plot(t_axis[:640], env_b_extracted[:640], label=f"Python Causal Streaming (r = {metrics_b['r']:.3f})", color="crimson", alpha=0.8, lw=1.2)
    axes[1].set_title(f"Talker B ({wav_b_name}) - First 10 Seconds Envelope Overlay @ 64 Hz", fontsize=11, fontweight="bold")
    axes[1].set_ylabel("Broadband Envelope")
    axes[1].set_xlabel("Time (seconds)")
    axes[1].legend(loc="upper right")
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_file = out_dir / f"raw_audio_parity_{sub_key}_trial{args.trial}.png"
    plt.savefig(plot_file, dpi=150)
    plt.close()
    print(f"\n[ARTIFACT] Saved parity verification plot: {plot_file}")
    print("=" * 100)
    print("  AUDIO PARITY AUDIT COMPLETED")
    print("=" * 100)


if __name__ == "__main__":
    main()
