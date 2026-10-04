import argparse
import sys
import time
import json
from pathlib import Path
import numpy as np
import scipy.io.wavfile as wavfile
import matplotlib.pyplot as plt
import torch
from typing import Optional

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
from training.train_matchnet_wavlm import get_mapping_data, FS
from baselines.ridge_aad import load_subject_examples, subject_files

from src.selective_aad.core import TemperatureCalibrator
from src.selective_aad.evaluator import SelectiveAADEvaluator
from src.selective_aad.temporal_gate import SignalQualityMonitor, StickyHysteresisGate
from src.audio.steering_engine import AudioSteeringDSP
from src.audio.metrics import (
    evaluate_audio_steering_trial,
    compute_sir_metrics,
    compute_stoi_intelligibility,
    compute_headroom_metrics
)
from src.audio.live_visualizer import save_live_streaming_dashboard
from scripts.verify_baseline.audit.run_audio_steering_demo import (
    load_or_synthesize_trial_audio,
    train_adapter_for_subject
)
from scripts.verify_baseline.audit.run_spatial_adapter_benchmark import (
    process_subject_trials,
    evaluate_trials
)

def run_live_streaming_demo(
    target_sub: str = "S8",
    target_trial_idx: int = 15,
    checkpoint_dir: str = "/kaggle/working/loso_checkpoints",
    audio_dir_path: str = "/kaggle/input/datasets/lokeshgile/eeg-audio",
    out_dir_path: str = "/kaggle/working/audio_demo_output",
    max_boost_db: float = 9.0,
    max_suppress_db: float = 18.0,
    tau_ms: float = 60.0,
    duration_sec: Optional[float] = 45.0,
    realtime_pacing: bool = False,
    device_str: str = "auto"
):
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)
        
    print("=" * 115)
    print("  AUDITORY ATTENTION DECODING: LIVE STREAMING NEURAL AUDIO & TELEMETRY SUITE")
    print(f"  Device: {device} | Target: {target_sub} (Trial {target_trial_idx}) | Realtime: {realtime_pacing}")
    print("=" * 115)
    
    out_dir = Path(out_dir_path)
    audio_dir = Path(audio_dir_path)
    montage_channels = MONTAGES["near_ear_expanded"]
    mapping, envelopes = get_mapping_data("gammatone")
    all_paths = subject_files()
    
    target_path = next((p for p in all_paths if p.stem.split("_")[0] == target_sub), None)
    if not target_path:
        raise FileNotFoundError(f"No DTU data file found for subject {target_sub}")
        
    # 1. Locate and Load Frozen Universal Backbone
    backbone_candidates = [
        Path(checkpoint_dir) / f"catcn_loso_{target_sub}.pt",
        Path(checkpoint_dir) / f"catcn_univ_heldout_{target_sub}.pt",
        Path(f"/kaggle/working/loso_checkpoints/catcn_loso_{target_sub}.pt"),
        Path(f"/kaggle/working/loso_checkpoints/catcn_univ_heldout_{target_sub}.pt"),
        Path(f"/kaggle/working/checkpoints/catcn_univ_heldout_{target_sub}.pt"),
        Path(f"/kaggle/working/checkpoints/catcn_loso_{target_sub}.pt"),
        Path(f"checkpoints/loso/catcn_loso_{target_sub}.pt"),
        Path(f"checkpoints/adaptation/catcn_univ_heldout_{target_sub}.pt"),
    ]
    found_ckpt = next((p for p in backbone_candidates if p.exists()), None)
    if not found_ckpt:
        # Fallback: recursive search in checkpoint_dir, /kaggle/working, /kaggle/input, and ./checkpoints
        search_roots = [Path(checkpoint_dir), Path("/kaggle/working"), Path("/kaggle/input"), Path("checkpoints")]
        for s_root in search_roots:
            if s_root.exists():
                cands = list(s_root.rglob(f"*{target_sub}*.pt"))
                if cands:
                    found_ckpt = cands[0]
                    break
                    
    if not found_ckpt:
        all_pts = []
        for s_root in [Path(checkpoint_dir), Path("/kaggle/working"), Path("/kaggle/input")]:
            if s_root.exists():
                all_pts.extend([str(p) for p in s_root.rglob("*.pt")])
        avail_str = "\n".join(all_pts[:10]) if all_pts else "No .pt files found"
        raise FileNotFoundError(
            f"Checkpoint not found for {target_sub}.\nSearched candidates in: {checkpoint_dir}, /kaggle/working, /kaggle/input\nAvailable files on disk:\n{avail_str}"
        )
        
    print(f"[CHECKPOINT] Loaded frozen backbone from: {found_ckpt}")
    univ_model = CATCNDirectDecoder(
        eeg_channels=len(montage_channels), audio_channels=1, hidden_dim=64, max_lag_samples=8
    ).to(device)
    univ_model.load_state_dict(torch.load(found_ckpt, map_location=device))
    univ_model.eval()
    for p in univ_model.parameters():
        p.requires_grad = False
        
    # 2. Partition Subject Trials & Calibrate 8x8 Adapter
    target_exs = load_subject_examples(target_path)
    causal_filter = StreamingCausalEEGFilter(fs=FS, lowcut=1.0, highcut=6.0, order=2, n_channels=len(montage_channels))
    eeg_all, ya_all, yb_all = process_subject_trials(target_exs, montage_channels, target_sub, mapping, envelopes, causal_filter, FS)
    
    K = 12
    eeg_calib, ya_calib, yb_calib = eeg_all[:K], ya_all[:K], yb_all[:K]
    print(f"[CALIB] Calibrating 8x8 Spatial Adapter on {K} calibration trials...")
    adapter = train_adapter_for_subject(univ_model, montage_channels, eeg_calib, ya_calib, yb_calib, 5.0, 0.5, device)
    
    # 3. Fit Temperature & Hysteresis
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
    th_switch = float(best_hyst["threshold_switch"])
    th_maintain = float(best_hyst["threshold_maintain"])
    
    print(f"[GATE] Calibrated Gate: Temp={fitted_t:.2f} | Switch={th_switch:.2f} | Maintain={th_maintain:.2f}")
    
    # 4. Stream Target Trial Causal Neural Inference
    target_eeg = [eeg_all[target_trial_idx]]
    target_ya = [ya_all[target_trial_idx]]
    target_yb = [yb_all[target_trial_idx]]
    target_m_list, _, raw_e_list = evaluate_trials(univ_model, adapter, target_eeg, target_ya, target_yb, 5.0, 0.5, FS, device)
    
    trial_margins = target_m_list[0]
    trial_raw_eeg = raw_e_list[0]
    gt = "A" # Ground truth attended talker in DTU
    
    # 5. Load Native Audio (44.1 kHz CD Quality)
    audio_a, audio_b, actual_fs, src_a, src_b = load_or_synthesize_trial_audio(
        target_sub, target_trial_idx, mapping, audio_dir, target_ya[0], target_yb[0], target_fs=44100
    )
    if duration_sec and duration_sec > 0:
        target_samples = int(duration_sec * actual_fs)
        if target_samples < len(audio_a):
            audio_a = audio_a[:target_samples]
            audio_b = audio_b[:target_samples]
            max_steps = max(1, int((len(audio_a) / float(actual_fs) - 5.0) / 0.5))
            trial_margins = trial_margins[:max_steps]
            trial_raw_eeg = trial_raw_eeg[:max_steps]
    total_audio_sec = len(audio_a) / float(actual_fs)
    print(f"[AUDIO] Stream A: {src_a} | Stream B: {src_b} ({total_audio_sec:.1f}s @ {actual_fs} Hz)")
    
    # 6. Step-by-Step Causal Live Streaming Execution
    print("\n" + "=" * 115)
    print("  LIVE STREAMING MODEL OUTPUTS & NEURAL TELEMETRY (STEP = 0.5s)")
    print("=" * 115)
    print(f" {'TIME':<7} | {'TALKER A GAIN':<14} | {'NEURAL MARGIN (s_t)':<22} | {'TALKER B GAIN':<14} | {'DECISION':<9} | {'STATUS':<8} | {'ACC':<6}")
    print("-" * 115)
    
    gate = StickyHysteresisGate(
        alpha=0.82,
        threshold_switch=th_switch,
        threshold_maintain=th_maintain,
        n_confirm=best_hyst["n_confirm"],
        temperature=fitted_t
    )
    sq_monitor = SignalQualityMonitor()
    dsp = AudioSteeringDSP(
        fs=actual_fs,
        max_boost_db=max_boost_db,
        max_suppress_db=max_suppress_db,
        tau_ms=tau_ms,
        threshold_switch=th_switch
    )
    
    control_times = []
    control_states = []
    control_margins = []
    
    telemetry_frames = []
    n_correct = 0
    total_decisions = 0
    switches_count = 0
    prev_decision = "HOLD"
    
    for step_i, (m_val, w_eeg) in enumerate(zip(trial_margins, trial_raw_eeg)):
        t_sec = 5.0 + step_i * 0.5
        sq = sq_monitor.check_eeg_window(w_eeg)
        out = gate.update(m_val, is_artifact=not sq["is_valid"])
        
        dec = out["decision"]
        s_t = out["smoothed_margin"]
        conf = float(np.clip(abs(s_t) / (th_switch + 1e-8), 0.0, 1.0))
        
        if dec != prev_decision and step_i > 0 and dec in ["A", "B"] and prev_decision in ["A", "B"]:
            switches_count += 1
        prev_decision = dec
        
        control_times.append(t_sec)
        control_states.append(dec)
        control_margins.append(s_t)
        
        g_a_db, g_b_db = dsp.compute_target_gains_db(dec, s_t)
        
        is_correct = (dec == gt)
        if dec in ["A", "B"]:
            total_decisions += 1
            if is_correct:
                n_correct += 1
        running_acc = (n_correct / max(1, total_decisions)) * 100.0
        
        # Audio voice energy window around t_sec
        sample_idx = int(t_sec * actual_fs)
        half_win = int(0.25 * actual_fs) # 250 ms
        w_start = max(0, sample_idx - half_win)
        w_end = min(len(audio_a), sample_idx + half_win)
        rms_a = float(np.sqrt(np.mean(np.square(audio_a[w_start:w_end])))) if w_end > w_start else 0.0
        rms_b = float(np.sqrt(np.mean(np.square(audio_b[w_start:w_end])))) if w_end > w_start else 0.0
        
        # Telemetry Frame for 60 FPS Browser Dashboard
        telemetry_frames.append({
            "time_sec": round(t_sec, 2),
            "raw_margin": round(float(m_val), 3),
            "smoothed_margin": round(float(s_t), 3),
            "confidence": round(conf, 3),
            "decision": dec,
            "ground_truth": gt,
            "is_correct": is_correct,
            "gain_a_db": round(float(g_a_db), 1),
            "gain_b_db": round(float(g_b_db), 1),
            "rms_a": round(rms_a, 4),
            "rms_b": round(rms_b, 4),
            "cumulative_accuracy_pct": round(running_acc, 1),
            "running_delta_sir_db": round(float(g_a_db - g_b_db), 1),
            "running_stoi": 0.96, # Computed globally on rendered audio
            "switch_count": switches_count,
        })
        
        # Colorized Console Status
        status_tag = "[MATCH]" if is_correct else ("[HOLD]" if dec == "HOLD" else "[ERR!]")
        
        # Visual Bipolar Margin Bar: -1.0 to +1.0
        margin_clamped = np.clip(s_t, -1.0, 1.0)
        pos = int((margin_clamped + 1.0) / 2.0 * 10)
        bar = list("----------")
        if 0 <= pos < 10:
            bar[pos] = "●"
        bar_str = "[" + "".join(bar) + "]"
        
        print(f" {t_sec:5.1f}s | A: {g_a_db:+5.1f} dB    | {bar_str} s={s_t:+5.2f}    | B: {g_b_db:+5.1f} dB    | State: {dec:<4} | {status_tag:<8} | {running_acc:5.1f}%")
        
        if realtime_pacing:
            time.sleep(0.5)
            
    print("-" * 115)
    
    # 7. Render High-Fidelity Audio Tracks
    print("[DSP] Rendering 44.1 kHz continuous slew-limited audio tracks...")
    render_dict = dsp.render_full_trial(
        audio_a=audio_a,
        audio_b=audio_b,
        control_timestamps_sec=np.array(control_times),
        control_states=control_states,
        control_margins=np.array(control_margins),
        ground_truth_attended=gt
    )
    
    metrics = evaluate_audio_steering_trial(
        audio_a=audio_a,
        audio_b=audio_b,
        render_dict=render_dict,
        ground_truth=gt,
        decisions=control_states,
        step_sec=0.5
    )
    
    # Update running STOI in telemetry frames
    for f in telemetry_frames:
        f["running_stoi"] = round(float(metrics["stoi_steered"]), 2)
        
    out_dir.mkdir(parents=True, exist_ok=True)
    p_steered = out_dir / f"{target_sub}_trial_{target_trial_idx}_steered.wav"
    p_mixture = out_dir / f"{target_sub}_trial_{target_trial_idx}_mixture.wav"
    p_ref = out_dir / f"{target_sub}_trial_{target_trial_idx}_reference.wav"
    
    def save_wav(path, arr2d, fs):
        pcm = np.clip(arr2d.T * 32767.0, -32768, 32767).astype(np.int16)
        wavfile.write(str(path), fs, pcm)
        
    save_wav(p_steered, render_dict["steered_binaural"], actual_fs)
    save_wav(p_mixture, render_dict["raw_mixture"], actual_fs)
    save_wav(p_ref, render_dict["clean_attended_reference"], actual_fs)
    
    # 8. Export Full Synchronized Telemetry Packet
    telemetry_packet = {
        "subject": target_sub,
        "trial_idx": target_trial_idx,
        "duration_sec": round(total_audio_sec, 2),
        "sample_rate": actual_fs,
        "ground_truth": gt,
        "threshold_switch": th_switch,
        "threshold_maintain": th_maintain,
        "max_boost_db": max_boost_db,
        "max_suppress_db": max_suppress_db,
        "tau_ms": tau_ms,
        "headroom_db": round(float(metrics["headroom_db"]), 1),
        "steered_wav_filename": p_steered.name,
        "mixture_wav_filename": p_mixture.name,
        "ref_wav_filename": p_ref.name,
        "frames": telemetry_frames
    }
    
    telemetry_json_path = out_dir / f"{target_sub}_trial_{target_trial_idx}_telemetry.json"
    with open(telemetry_json_path, "w", encoding="utf-8") as f:
        json.dump(telemetry_packet, f, indent=2)
    print(f"[TELEMETRY] Saved telemetry JSON to: {telemetry_json_path.name}")
    
    # 9. Generate Interactive Standalone HTML5 Dashboard
    html_player_path = out_dir / "aad_live_streaming_player.html"
    save_live_streaming_dashboard(telemetry_packet, html_player_path)
    print(f"[HTML PLAYER] Generated live interactive player at: {html_player_path}")
    
    # 10. Generate High-Res Diagnostic Plot
    p_plot = out_dir / f"{target_sub}_trial_{target_trial_idx}_plot.png"
    plt.figure(figsize=(12, 8))
    t_audio = np.linspace(0, total_audio_sec, len(audio_a))
    
    plt.subplot(3, 1, 1)
    plt.plot(t_audio, audio_a, color="#38bdf8", alpha=0.7, label="Stream A (Attended Talker)")
    plt.plot(t_audio, -audio_b, color="#f43f5e", alpha=0.5, label="Stream B (Unattended Talker)")
    plt.xlim(0, total_audio_sec)
    plt.title(f"Acoustic Speech Signals ({target_sub} — Trial {target_trial_idx})", fontsize=11, fontweight="bold")
    plt.ylabel("Amplitude")
    plt.legend(loc="upper right")
    plt.grid(True, alpha=0.2)
    
    plt.subplot(3, 1, 2)
    plt.plot(t_audio, render_dict["gain_db_a"], color="#10b981", linewidth=2.0, label="Stream A Gain (dB)")
    plt.plot(t_audio, render_dict["gain_db_b"], color="#f59e0b", linewidth=1.5, linestyle="--", label="Stream B Gain (dB)")
    plt.axhline(0.0, color="gray", linestyle=":", alpha=0.5)
    plt.xlim(0, total_audio_sec)
    plt.ylabel("Applied Gain (dB)")
    plt.title(f"Dynamic Steering Trajectory (Separation: {metrics['delta_sir_db']:+.1f} dB)", fontsize=11, fontweight="bold")
    plt.legend(loc="upper right")
    plt.grid(True, alpha=0.2)
    
    plt.subplot(3, 1, 3)
    plt.plot(control_times, control_margins, color="#8b5cf6", marker="o", markersize=3, label="Smoothed Margin s_t")
    plt.axhline(th_switch, color="#10b981", linestyle="--", alpha=0.6, label="Switch Threshold (+)")
    plt.axhline(-th_switch, color="#f43f5e", linestyle="--", alpha=0.6, label="Switch Threshold (-)")
    plt.axhline(0.0, color="black", linestyle="-", alpha=0.3)
    plt.xlim(0, total_audio_sec)
    plt.ylabel("Confidence Margin")
    plt.xlabel("Trial Time (seconds)")
    plt.title(f"EEG Direct Decoder Decisions (Accuracy: {metrics['decision_accuracy_pct']:.1f}% | Flips: {metrics['false_switches_per_min']:.2f}/m)", fontsize=11, fontweight="bold")
    plt.legend(loc="upper right")
    plt.grid(True, alpha=0.2)
    
    plt.tight_layout()
    plt.savefig(str(p_plot), dpi=150)
    plt.close()
    
    print("\n" + "=" * 115)
    print("  LIVE STREAMING AUDIT & TELEMETRY GENERATION COMPLETE!")
    print(f"  Final Decision Accuracy: {metrics['decision_accuracy_pct']:.1f}%")
    print(f"  Acoustic SIR Separation: {metrics['delta_sir_db']:+.1f} dB")
    print(f"  Speech Intelligibility:  STOI = {metrics['stoi_steered']:.2f}")
    print(f"  Dynamic Headroom:        {metrics['headroom_db']:.1f} dB")
    print("-" * 115)
    print("  To launch the live interactive player in Kaggle, run in the next cell:")
    print("    from IPython.display import HTML")
    print(f"    HTML(open('{html_player_path}', encoding='utf-8').read())")
    print("\n  Or listen to high-res uncompressed 44.1 kHz WAV directly in notebook:")
    print("    import IPython.display as ipd")
    print(f"    ipd.Audio('{p_steered}')")
    print("=" * 115)
    
    return {
        "metrics": metrics,
        "html_player_path": html_player_path,
        "telemetry_json_path": telemetry_json_path,
        "steered_wav": p_steered,
        "mixture_wav": p_mixture,
        "ref_wav": p_ref
    }

def main():
    parser = argparse.ArgumentParser(description="AAD Live Streaming Neural Audio & Telemetry Suite")
    parser.add_argument("--checkpoint_dir", type=str, default="/kaggle/working/loso_checkpoints", help="Checkpoints directory")
    parser.add_argument("--audio_dir", type=str, default="/kaggle/input/datasets/lokeshgile/eeg-audio", help="Raw audio WAV directory")
    parser.add_argument("--out_dir", type=str, default="/kaggle/working/audio_demo_output", help="Output directory")
    parser.add_argument("--subject", type=str, default="S8", help="Target subject (default: S8)")
    parser.add_argument("--trial", type=int, default=15, help="Target test trial index (default: 15)")
    parser.add_argument("--max_boost", type=float, default=9.0, help="Max boost in dB (default +9 dB)")
    parser.add_argument("--max_suppress", type=float, default=18.0, help="Max suppression in dB (default -18 dB)")
    parser.add_argument("--tau_ms", type=float, default=60.0, help="Slew time constant in ms (default 60 ms)")
    parser.add_argument("--duration", type=float, default=45.0, help="Demo duration in seconds (default: 45.0s, pass 0 for full trial)")
    parser.add_argument("--realtime", action="store_true", help="Paces console output with 500ms delay to simulate real-time clock")
    args = parser.parse_args()
    
    run_live_streaming_demo(
        target_sub=args.subject,
        target_trial_idx=args.trial,
        checkpoint_dir=args.checkpoint_dir,
        audio_dir_path=args.audio_dir,
        out_dir_path=args.out_dir,
        max_boost_db=args.max_boost,
        max_suppress_db=args.max_suppress,
        tau_ms=args.tau_ms,
        duration_sec=args.duration,
        realtime_pacing=args.realtime
    )

if __name__ == "__main__":
    main()
