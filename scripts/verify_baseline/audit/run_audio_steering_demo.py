import argparse
import sys
import os
import json
import base64
import time
from typing import Tuple, List, Dict, Any, Optional
from pathlib import Path
from copy import deepcopy
import numpy as np
import scipy.io.wavfile as wavfile
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[3]
VERIFY_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(VERIFY_ROOT) not in sys.path:
    sys.path.insert(0, str(VERIFY_ROOT))

from models.catcn import CATCNDirectDecoder
from src.models.spatial_adapter import SpatialEEGAdapter
from src.streaming.causal_filters import StreamingCausalEEGFilter
from training.montages import MONTAGES
from training.train_matchnet_wavlm import get_mapping_data, prepare_dataset, FS
from baselines.ridge_aad import load_subject_examples, subject_files

from src.selective_aad.core import TemperatureCalibrator
from src.selective_aad.evaluator import SelectiveAADEvaluator
from src.selective_aad.temporal_gate import SignalQualityMonitor, StickyHysteresisGate
from src.audio.steering_engine import AudioSteeringDSP
from src.audio.metrics import evaluate_audio_steering_trial

# Helper: Synthesize speech-like modulated acoustic carrier if raw WAV is missing
def synthesize_acoustic_speech(envelope_64hz: np.ndarray, target_fs: int = 16000) -> np.ndarray:
    """
    Synthesizes natural-sounding speech-like acoustic audio from 64 Hz envelope
    using a multi-formant shaped noise carrier. Guarantees listening capability
    even if external raw WAV files are unmounted on Kaggle.
    """
    total_sec = len(envelope_64hz) / 64.0
    N = int(total_sec * target_fs)
    
    # Upsample envelope to 16 kHz using linear interpolation
    t_env = np.linspace(0, total_sec, len(envelope_64hz))
    t_audio = np.linspace(0, total_sec, N)
    env_upsampled = np.interp(t_audio, t_env, envelope_64hz)
    env_upsampled = np.maximum(0.0, env_upsampled)
    
    # Formant carrier (vowel-like resonance at 500 Hz, 1500 Hz, 2500 Hz)
    np.random.seed(42)
    noise = np.random.randn(N).astype(np.float32)
    t = np.arange(N) / float(target_fs)
    carrier = (
        0.5 * np.sin(2 * np.pi * 130 * t) + # Fundamental pitch
        0.3 * np.sin(2 * np.pi * 500 * t) + # F1
        0.2 * np.sin(2 * np.pi * 1500 * t) + # F2
        0.3 * noise                         # Fricatives / breath
    )
    # Bandpass filter carrier into telephone / speech band
    audio = carrier * env_upsampled
    audio = audio / (np.max(np.abs(audio)) + 1e-6) * 0.4
    return audio.astype(np.float32)

def load_or_synthesize_trial_audio(
    sub_key: str,
    trial_idx: int,
    mapping: dict,
    audio_dir: Path,
    env_a: np.ndarray,
    env_b: np.ndarray,
    target_fs: int = 16000
) -> Tuple[np.ndarray, np.ndarray, str, str]:
    """
    Attempts to load genuine 16 kHz raw WAV files for Stream A and Stream B.
    If files are unavailable or unmounted, seamlessly falls back to envelope-driven synthesis.
    """
    trial_key = f"trial_{trial_idx}"
    fname_a = None
    fname_b = None
    if sub_key in mapping and trial_key in mapping[sub_key]:
        fname_a = mapping[sub_key][trial_key]["wavA"]["filename"]
        fname_b = mapping[sub_key][trial_key]["wavB"]["filename"]
        
    wav_a_path = None
    wav_b_path = None
    if audio_dir and audio_dir.exists():
        if fname_a:
            cands_a = list(audio_dir.rglob(fname_a))
            if cands_a:
                wav_a_path = cands_a[0]
        if fname_b:
            cands_b = list(audio_dir.rglob(fname_b))
            if cands_b:
                wav_b_path = cands_b[0]
                
    # Load A
    if wav_a_path and wav_a_path.exists():
        fs_a, raw_a = wavfile.read(str(wav_a_path))
        if raw_a.ndim > 1:
            raw_a = raw_a.mean(axis=-1)
        audio_a = (raw_a / (np.max(np.abs(raw_a)) + 1e-6) * 0.4).astype(np.float32)
        source_a = f"WAV ({wav_a_path.name})"
    else:
        audio_a = synthesize_acoustic_speech(env_a, target_fs=target_fs)
        source_a = "Synthesized from Envelope"
        
    # Load B
    if wav_b_path and wav_b_path.exists():
        fs_b, raw_b = wavfile.read(str(wav_b_path))
        if raw_b.ndim > 1:
            raw_b = raw_b.mean(axis=-1)
        audio_b = (raw_b / (np.max(np.abs(raw_b)) + 1e-6) * 0.4).astype(np.float32)
        source_b = f"WAV ({wav_b_path.name})"
    else:
        audio_b = synthesize_acoustic_speech(env_b, target_fs=target_fs)
        source_b = "Synthesized from Envelope"
        
    min_len = min(len(audio_a), len(audio_b))
    return audio_a[:min_len], audio_b[:min_len], source_a, source_b

def train_adapter_for_subject(univ_model, montage_channels, eeg_calib, ya_calib, yb_calib, window_sec, step_sec, device):
    """Calibrates the 64-parameter spatial adapter on 12 calibration trials."""
    from scripts.verify_baseline.audit.run_spatial_adapter_benchmark import extract_windows_dataset, train_spatial_adapter
    calib_x, calib_ya, calib_yb = extract_windows_dataset(eeg_calib, ya_calib, yb_calib, window_sec, step_sec, FS)
    adapter = SpatialEEGAdapter(channels=len(montage_channels)).to(device)
    adapter = train_spatial_adapter(adapter, univ_model, calib_x, calib_ya, calib_yb, epochs=15, lr=1e-3, l2_identity=0.05, device=device)
    return adapter

def generate_html_player(demo_results: list, output_html_path: Path):
    """
    Generates a standalone, beautiful HTML5 Interactive Audio Player with embedded base64 audio.
    Playable directly in Kaggle notebooks or any browser.
    """
    cards_html = []
    
    for r in demo_results:
        sub = r["subject"]
        trial_idx = r["trial_idx"]
        gt = r["ground_truth"]
        m = r["metrics"]
        
        # Read WAVs and base64 encode
        def get_b64(path):
            with open(path, "rb") as f:
                return base64.b64encode(f.read()).decode("utf-8")
                
        b64_steered = get_b64(r["steered_wav_path"])
        b64_mixture = get_b64(r["mixture_wav_path"])
        b64_ref = get_b64(r["ref_wav_path"])
        
        card = f"""
        <div class="trial-card">
            <div class="card-header">
                <span class="subject-badge">{sub} — Trial {trial_idx}</span>
                <span class="gt-badge">Attended Target: Speaker {gt}</span>
                <span class="acc-badge">Model Accuracy: {m['decision_accuracy_pct']:.1f}%</span>
            </div>
            
            <div class="metrics-grid">
                <div class="metric-box">
                    <div class="m-val" style="color: #10b981;">{m['delta_sir_db']:+.1f} dB</div>
                    <div class="m-lbl">SIR Separation Gain</div>
                </div>
                <div class="metric-box">
                    <div class="m-val" style="color: #6366f1;">{m['mean_contrast_db']:+.1f} dB</div>
                    <div class="m-lbl">Mean Contrast</div>
                </div>
                <div class="metric-box">
                    <div class="m-val" style="color: #f59e0b;">{m['stoi_steered']:.2f}</div>
                    <div class="m-lbl">STOI Intelligibility</div>
                </div>
                <div class="metric-box">
                    <div class="m-val" style="color: #ec4899;">{m['false_switches_per_min']:.2f}/m</div>
                    <div class="m-lbl">False Switches</div>
                </div>
                <div class="metric-box">
                    <div class="m-val" style="color: #06b6d4;">{m['boost_coverage_pct']:.1f}%</div>
                    <div class="m-lbl">Active Boost Time</div>
                </div>
                <div class="metric-box">
                    <div class="m-val" style="color: #8b5cf6;">{m['headroom_db']:.1f} dB</div>
                    <div class="m-lbl">Dynamic Headroom</div>
                </div>
            </div>
            
            <div class="audio-controls">
                <div class="player-group">
                    <div class="player-title">1. Live AAD Steered Binaural Output (Attended Amplified + Unattended Suppressed)</div>
                    <audio controls style="width: 100%;">
                        <source src="data:audio/wav;base64,{b64_steered}" type="audio/wav">
                    </audio>
                </div>
                
                <div class="player-group">
                    <div class="player-title">2. Raw Unsteered Mixture (Baseline 50/50 Cocktail Party)</div>
                    <audio controls style="width: 100%;">
                        <source src="data:audio/wav;base64,{b64_mixture}" type="audio/wav">
                    </audio>
                </div>
                
                <div class="player-group">
                    <div class="player-title">3. Clean Attended Speaker Reference (Isolated Target)</div>
                    <audio controls style="width: 100%;">
                        <source src="data:audio/wav;base64,{b64_ref}" type="audio/wav">
                    </audio>
                </div>
            </div>
            
            <div style="margin-top: 15px;">
                <img src="{r['plot_filename']}" style="width: 100%; border-radius: 8px; border: 1px solid #334155;" alt="Gain Timeline Plot">
            </div>
        </div>
        """
        cards_html.append(card)
        
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AAD Audio Amplification & Suppression Listening Suite</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            background-color: #0f172a;
            color: #f8fafc;
            margin: 0;
            padding: 24px;
        }}
        .header {{
            text-align: center;
            margin-bottom: 30px;
        }}
        h1 {{
            font-size: 26px;
            color: #38bdf8;
            margin-bottom: 8px;
        }}
        .subtitle {{
            color: #94a3b8;
            font-size: 14px;
        }}
        .container {{
            max-width: 1100px;
            margin: 0 auto;
        }}
        .trial-card {{
            background: #1e293b;
            border: 1px solid #334155;
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 30px;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
        }}
        .card-header {{
            display: flex;
            align-items: center;
            gap: 12px;
            margin-bottom: 16px;
        }}
        .subject-badge {{
            background: #3b82f6;
            color: #fff;
            padding: 4px 10px;
            border-radius: 6px;
            font-weight: 600;
            font-size: 14px;
        }}
        .gt-badge {{
            background: #059669;
            color: #fff;
            padding: 4px 10px;
            border-radius: 6px;
            font-size: 13px;
        }}
        .acc-badge {{
            background: #7c3aed;
            color: #fff;
            padding: 4px 10px;
            border-radius: 6px;
            font-size: 13px;
        }}
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
            gap: 12px;
            margin-bottom: 20px;
        }}
        .metric-box {{
            background: #0f172a;
            border: 1px solid #334155;
            border-radius: 8px;
            padding: 12px;
            text-align: center;
        }}
        .m-val {{
            font-size: 20px;
            font-weight: 700;
            margin-bottom: 4px;
        }}
        .m-lbl {{
            font-size: 11px;
            color: #94a3b8;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        .audio-controls {{
            display: flex;
            flex-direction: column;
            gap: 14px;
            background: #0f172a;
            border: 1px solid #334155;
            padding: 16px;
            border-radius: 8px;
        }}
        .player-group {{
            display: flex;
            flex-direction: column;
            gap: 6px;
        }}
        .player-title {{
            font-size: 13px;
            font-weight: 600;
            color: #e2e8f0;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Auditory Attention Decoding (AAD) Audio Steering Suite</h1>
            <div class="subtitle">Real-time Causal Brain-Steered Amplification & Suppression (Frozen 5.0s CA-TCN + 8x8 Adapter + Sticky Gate)</div>
        </div>
        {''.join(cards_html)}
    </div>
</body>
</html>
"""
    with open(output_html_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"[HTML DASHBOARD] Generated interactive player at: {output_html_path}")

def run_single_demo_trial(
    target_sub: str,
    target_trial_idx: int,
    univ_model,
    montage_channels,
    target_exs,
    mapping,
    envelopes,
    causal_filter,
    audio_dir,
    out_dir,
    dsp,
    device
):
    print(f"\n>>> Processing {target_sub} — Trial {target_trial_idx}...")
    from scripts.verify_baseline.audit.run_spatial_adapter_benchmark import process_subject_trials, evaluate_trials
    
    eeg_all, ya_all, yb_all = process_subject_trials(target_exs, montage_channels, target_sub, mapping, envelopes, causal_filter, FS)
    
    K = 12
    eeg_calib, ya_calib, yb_calib = eeg_all[:K], ya_all[:K], yb_all[:K]
    
    # 1. Train 8x8 Adapter
    adapter = train_adapter_for_subject(univ_model, montage_channels, eeg_calib, ya_calib, yb_calib, 5.0, 0.5, device)
    
    # 2. Fit Temperature & Hysteresis on calibration set
    ad_calib_margins, ad_calib_labels, _ = evaluate_trials(univ_model, adapter, eeg_calib, ya_calib, yb_calib, 5.0, 0.5, FS, device)
    flat_calib_m = np.concatenate(ad_calib_margins)
    flat_calib_l = np.concatenate(ad_calib_labels)
    calibrator = TemperatureCalibrator()
    fitted_t = calibrator.fit(flat_calib_m, flat_calib_l, bounds=(0.05, 10.0))
    
    hyst_sweep = SelectiveAADEvaluator.sweep_hysteresis_parameters(
        ad_calib_margins, ad_calib_labels, alpha=0.82,
        switch_candidates=[0.30, 0.35], confirm_candidates=[2], step_sec=0.5
    )
    best_hyst = hyst_sweep["best_config"]
    
    # 3. Stream Inference on Target Trial
    target_eeg = [eeg_all[target_trial_idx]]
    target_ya = [ya_all[target_trial_idx]]
    target_yb = [yb_all[target_trial_idx]]
    target_m_list, _, raw_e_list = evaluate_trials(univ_model, adapter, target_eeg, target_ya, target_yb, 5.0, 0.5, FS, device)
    
    trial_margins = target_m_list[0]
    trial_raw_eeg = raw_e_list[0]
    
    # Ground truth: in DTU, ya is ALWAYS attended (Stream A)
    gt = "A"
    
    # Run Sticky Gate
    gate = StickyHysteresisGate(
        alpha=0.82,
        threshold_switch=best_hyst["threshold_switch"],
        threshold_maintain=best_hyst["threshold_maintain"],
        n_confirm=best_hyst["n_confirm"],
        temperature=fitted_t
    )
    sq_monitor = SignalQualityMonitor()
    
    control_times = []
    control_states = []
    control_margins = []
    
    for step_i, (m_val, w_eeg) in enumerate(zip(trial_margins, trial_raw_eeg)):
        t_sec = (step_i + 1) * 0.5
        sq = sq_monitor.check_eeg_window(w_eeg)
        out = gate.update(m_val, is_artifact=not sq["is_valid"])
        control_times.append(t_sec)
        control_states.append(out["decision"])
        control_margins.append(out["smoothed_margin"])
        
    # 4. Load or synthesize audio
    audio_a, audio_b, src_a, src_b = load_or_synthesize_trial_audio(
        target_sub, target_trial_idx, mapping, audio_dir, target_ya[0], target_yb[0], target_fs=dsp.fs
    )
    print(f"  [AUDIO SOURCE] A: {src_a} | B: {src_b} ({len(audio_a)/dsp.fs:.1f}s)")
    
    # 5. Render full trial audio
    render_dict = dsp.render_full_trial(
        audio_a=audio_a,
        audio_b=audio_b,
        control_timestamps_sec=np.array(control_times),
        control_states=control_states,
        control_margins=np.array(control_margins),
        ground_truth_attended=gt
    )
    
    # 6. Evaluate metrics
    metrics = evaluate_audio_steering_trial(
        audio_a=audio_a,
        audio_b=audio_b,
        render_dict=render_dict,
        ground_truth=gt,
        decisions=control_states,
        step_sec=0.5
    )
    
    # 7. Save WAV files
    out_dir.mkdir(parents=True, exist_ok=True)
    p_steered = out_dir / f"{target_sub}_trial_{target_trial_idx}_steered.wav"
    p_mixture = out_dir / f"{target_sub}_trial_{target_trial_idx}_mixture.wav"
    p_ref = out_dir / f"{target_sub}_trial_{target_trial_idx}_reference.wav"
    
    def save_wav(path, arr2d, fs):
        # arr2d: [2, N] float32 in [-1, 1]
        pcm = np.clip(arr2d.T * 32767.0, -32768, 32767).astype(np.int16)
        wavfile.write(str(path), fs, pcm)
        
    save_wav(p_steered, render_dict["steered_binaural"], dsp.fs)
    save_wav(p_mixture, render_dict["raw_mixture"], dsp.fs)
    save_wav(p_ref, render_dict["clean_attended_reference"], dsp.fs)
    
    # 8. Generate High-Res Diagnostic Plot
    p_plot = out_dir / f"{target_sub}_trial_{target_trial_idx}_plot.png"
    plt.figure(figsize=(12, 8))
    
    t_audio = np.linspace(0, len(audio_a) / dsp.fs, len(audio_a))
    
    # Subplot 1: Raw speech waveforms
    plt.subplot(3, 1, 1)
    plt.plot(t_audio, audio_a, color="#38bdf8", alpha=0.7, label="Stream A (Attended Talker)")
    plt.plot(t_audio, -audio_b, color="#f43f5e", alpha=0.5, label="Stream B (Unattended Talker)")
    plt.title(f"Acoustic Speech Signals ({target_sub} — Trial {target_trial_idx})", fontsize=11, fontweight="bold")
    plt.ylabel("Amplitude")
    plt.legend(loc="upper right")
    plt.grid(True, alpha=0.2)
    
    # Subplot 2: Dynamic Gain Trajectory
    plt.subplot(3, 1, 2)
    plt.plot(t_audio, render_dict["gain_db_a"], color="#10b981", linewidth=2.0, label="Stream A Gain (dB)")
    plt.plot(t_audio, render_dict["gain_db_b"], color="#f59e0b", linewidth=1.5, linestyle="--", label="Stream B Gain (dB)")
    plt.axhline(0.0, color="gray", linestyle=":", alpha=0.5)
    plt.ylabel("Applied Gain (dB)")
    plt.title(f"Dynamic Steering Trajectory (Separation: {metrics['delta_sir_db']:+.1f} dB)", fontsize=11, fontweight="bold")
    plt.legend(loc="upper right")
    plt.grid(True, alpha=0.2)
    
    # Subplot 3: Margin and Decisions
    plt.subplot(3, 1, 3)
    plt.plot(control_times, control_margins, color="#8b5cf6", marker="o", markersize=3, label="Smoothed Margin s_t")
    plt.axhline(best_hyst["threshold_switch"], color="#10b981", linestyle="--", alpha=0.6, label="Switch Threshold (+)")
    plt.axhline(-best_hyst["threshold_switch"], color="#f43f5e", linestyle="--", alpha=0.6, label="Switch Threshold (-)")
    plt.axhline(0.0, color="black", linestyle="-", alpha=0.3)
    plt.ylabel("Confidence Margin")
    plt.xlabel("Trial Time (seconds)")
    plt.title(f"EEG Direct Decoder Decisions (Accuracy: {metrics['decision_accuracy_pct']:.1f}% | Flips: {metrics['false_switches_per_min']:.2f}/m)", fontsize=11, fontweight="bold")
    plt.legend(loc="upper right")
    plt.grid(True, alpha=0.2)
    
    plt.tight_layout()
    plt.savefig(str(p_plot), dpi=150)
    plt.close()
    
    print(f"  [METRICS] ΔSIR: {metrics['delta_sir_db']:+.1f} dB | Contrast: {metrics['mean_contrast_db']:+.1f} dB | STOI: {metrics['stoi_steered']:.2f} | Acc: {metrics['decision_accuracy_pct']:.1f}%")
    print(f"  [SAVED] Audio: {p_steered.name} | Plot: {p_plot.name}")
    
    return {
        "subject": target_sub,
        "trial_idx": target_trial_idx,
        "ground_truth": gt,
        "metrics": metrics,
        "steered_wav_path": p_steered,
        "mixture_wav_path": p_mixture,
        "ref_wav_path": p_ref,
        "plot_path": p_plot,
        "plot_filename": p_plot.name
    }

def main():
    parser = argparse.ArgumentParser(description="AAD Audio Amplification & Unattended Suppression Demo Suite")
    parser.add_argument("--checkpoint_dir", type=str, default="/kaggle/working/loso_checkpoints", help="Path to checkpoints")
    parser.add_argument("--audio_dir", type=str, default="/kaggle/input/datasets/lokeshgile/eeg-audio", help="Path to raw audio WAVs")
    parser.add_argument("--out_dir", type=str, default="/kaggle/working/audio_demo_output", help="Output directory")
    parser.add_argument("--showcase", action="store_true", help="Run 4 curated showcase trials (S8, S16, S4, S10)")
    parser.add_argument("--subject", type=str, default="", help="Single target subject (e.g. S8)")
    parser.add_argument("--trial", type=int, default=15, help="Single target trial index (test trial >= 12)")
    parser.add_argument("--max_boost", type=float, default=9.0, help="Maximum boost in dB (default +9 dB)")
    parser.add_argument("--max_suppress", type=float, default=18.0, help="Maximum suppression in dB (default -18 dB)")
    parser.add_argument("--tau_ms", type=float, default=60.0, help="Slew rate time constant in ms")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 115)
    print("  AUDITORY ATTENTION DECODING: AUDIO AMPLIFICATION & SUPPRESSION LISTENING SUITE")
    print(f"  Device: {device} | Max Boost: +{args.max_boost} dB | Max Suppress: -{args.max_suppress} dB | Slew: {args.tau_ms} ms")
    print("=" * 115)
    
    out_dir = Path(args.out_dir)
    audio_dir = Path(args.audio_dir)
    montage_channels = MONTAGES["near_ear_expanded"]
    mapping, envelopes = get_mapping_data("gammatone")
    all_paths = subject_files()
    
    dsp = AudioSteeringDSP(
        fs=16000,
        max_boost_db=args.max_boost,
        max_suppress_db=args.max_suppress,
        tau_ms=args.tau_ms,
        threshold_switch=0.35
    )
    causal_filter = StreamingCausalEEGFilter(fs=FS, lowcut=1.0, highcut=6.0, order=2, n_channels=len(montage_channels))
    
    # Determine which trials to run
    if args.showcase:
        run_specs = [
            ("S8", 15, "Clinical Peak Performer (93.1% Acc, 0.22 flips/min)"),
            ("S16", 16, "Realignment Star (+5.8 pp gain, 81.0% Acc)"),
            ("S4", 15, "Typical Subject (77.2% Acc)"),
            ("S10", 15, "Challenging Subject (70.9% Acc)"),
        ]
    elif args.subject:
        run_specs = [(args.subject, args.trial, f"Subject {args.subject} Trial {args.trial}")]
    else:
        run_specs = [("S8", 15, "Default: S8 Trial 15")]
        
    demo_results = []
    
    for sub, trial_idx, desc in run_specs:
        print(f"\n==========================================================================")
        print(f"  TARGET: {sub} (Trial {trial_idx}) — {desc}")
        print(f"==========================================================================")
        
        target_path = next((p for p in all_paths if p.stem.split("_")[0] == sub), None)
        if not target_path:
            print(f"[ERROR] No DTU file found for {sub}")
            continue
            
        # Load universal checkpoint
        backbone_candidates = [
            Path(args.checkpoint_dir) / f"catcn_loso_{sub}.pt",
            Path(args.checkpoint_dir) / f"catcn_univ_heldout_{sub}.pt",
            Path(f"/kaggle/working/loso_checkpoints/catcn_loso_{sub}.pt"),
            Path(f"/kaggle/working/checkpoints/catcn_univ_heldout_{sub}.pt"),
            Path(f"checkpoints/loso/catcn_loso_{sub}.pt"),
        ]
        found_ckpt = next((p for p in backbone_candidates if p.exists()), None)
        if not found_ckpt:
            print(f"[ERROR] Checkpoint not found for {sub}")
            continue
            
        univ_model = CATCNDirectDecoder(eeg_channels=len(montage_channels), audio_channels=1, hidden_dim=64, max_lag_samples=8).to(device)
        univ_model.load_state_dict(torch.load(found_ckpt, map_location=device))
        univ_model.eval()
        for p in univ_model.parameters():
            p.requires_grad = False
            
        target_exs = load_subject_examples(target_path)
        
        res = run_single_demo_trial(
            target_sub=sub,
            target_trial_idx=trial_idx,
            univ_model=univ_model,
            montage_channels=montage_channels,
            target_exs=target_exs,
            mapping=mapping,
            envelopes=envelopes,
            causal_filter=causal_filter,
            audio_dir=audio_dir,
            out_dir=out_dir,
            dsp=dsp,
            device=device
        )
        demo_results.append(res)
        
        del univ_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
    if demo_results:
        # Build HTML dashboard
        html_path = out_dir / "aad_audio_steering_dashboard.html"
        generate_html_player(demo_results, html_path)
        
        print("\n" + "=" * 115)
        print("  DEMO EXECUTION COMPLETE!")
        print(f"  All WAV files, PNG timeline plots, and HTML5 dashboard saved to: {out_dir}")
        print("  To play interactive audio directly in Kaggle, run:")
        print("    from IPython.display import HTML; HTML(open('/kaggle/working/audio_demo_output/aad_audio_steering_dashboard.html').read())")
        print("=" * 115)

if __name__ == "__main__":
    main()
