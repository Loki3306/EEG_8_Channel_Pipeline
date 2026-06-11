from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import hilbert

from _common import (
    PLOTS_DIR,
    SUMMARY_DIR,
    append_readme_update,
    ensure_output_dirs,
    load_subject_data,
    save_json,
    subject_files,
)


def moving_average(x: np.ndarray, window: int = 64) -> np.ndarray:
    kernel = np.ones(window, dtype=float) / float(window)
    return np.convolve(np.asarray(x, dtype=float), kernel, mode="same")


def analyze_subject(path: Path) -> dict[str, object]:
    data = load_subject_data(path)
    eeg = np.asarray(data.eeg[0, 0], dtype=float)
    wav_a = np.asarray(data.wavA[0, 0], dtype=float).ravel()
    wav_b = np.asarray(data.wavB[0, 0], dtype=float).ravel()

    eeg_two = eeg[:, :2]
    eeg_stats = {
        "min": float(np.min(eeg_two)),
        "max": float(np.max(eeg_two)),
        "mean": float(np.mean(eeg_two)),
        "std": float(np.std(eeg_two)),
    }
    wav_a_stats = {
        "min": float(np.min(wav_a)),
        "max": float(np.max(wav_a)),
        "mean": float(np.mean(wav_a)),
        "std": float(np.std(wav_a)),
    }
    wav_b_stats = {
        "min": float(np.min(wav_b)),
        "max": float(np.max(wav_b)),
        "mean": float(np.mean(wav_b)),
        "std": float(np.std(wav_b)),
    }

    eeg_time = np.arange(eeg_two.shape[0])
    audio_time = np.arange(wav_a.size)
    wav_a_envelope = np.abs(hilbert(wav_a))
    wav_b_envelope = np.abs(hilbert(wav_b))
    wav_a_smooth = moving_average(wav_a, window=64)
    wav_b_smooth = moving_average(wav_b, window=64)

    return {
        "file": path.name,
        "eeg_stats": eeg_stats,
        "wavA_stats": wav_a_stats,
        "wavB_stats": wav_b_stats,
        "eeg_time": eeg_time.tolist(),
        "eeg_two": eeg_two.tolist(),
        "audio_time": audio_time.tolist(),
        "wavA": wav_a.tolist(),
        "wavB": wav_b.tolist(),
        "wavA_envelope": wav_a_envelope.tolist(),
        "wavB_envelope": wav_b_envelope.tolist(),
        "wavA_smooth": wav_a_smooth.tolist(),
        "wavB_smooth": wav_b_smooth.tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate basic EEG and audio signal checks.")
    parser.add_argument("--subject", type=str, default="S1_data_preproc.mat")
    parser.add_argument("--json-out", type=Path, default=SUMMARY_DIR / "signal_checks.json")
    parser.add_argument("--plots-dir", type=Path, default=PLOTS_DIR)
    parser.add_argument("--no-readme-update", action="store_true")
    args = parser.parse_args()

    ensure_output_dirs()
    args.plots_dir.mkdir(parents=True, exist_ok=True)

    subject_path = next((path for path in subject_files() if path.name == args.subject), None)
    if subject_path is None:
        raise SystemExit(f"Subject file not found: {args.subject}")

    summary = analyze_subject(subject_path)
    save_json(args.json_out, summary)

    eeg_two = np.asarray(summary["eeg_two"], dtype=float)
    wav_a = np.asarray(summary["wavA"], dtype=float)
    wav_b = np.asarray(summary["wavB"], dtype=float)
    wav_a_smooth = np.asarray(summary["wavA_smooth"], dtype=float)
    wav_b_smooth = np.asarray(summary["wavB_smooth"], dtype=float)
    wav_a_envelope = np.asarray(summary["wavA_envelope"], dtype=float)
    wav_b_envelope = np.asarray(summary["wavB_envelope"], dtype=float)

    eeg_plot = args.plots_dir / f"{subject_path.stem}_eeg_channels.png"
    audio_plot = args.plots_dir / f"{subject_path.stem}_audio_signals.png"

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    axes[0].plot(eeg_two[:, 0], linewidth=0.9, color="#143D59")
    axes[0].set_title(f"{subject_path.name} - EEG channel 1")
    axes[0].set_ylabel("Amplitude")
    axes[1].plot(eeg_two[:, 1], linewidth=0.9, color="#7D4E57")
    axes[1].set_title(f"{subject_path.name} - EEG channel 2")
    axes[1].set_ylabel("Amplitude")
    axes[1].set_xlabel("Samples")
    fig.tight_layout()
    fig.savefig(eeg_plot, dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    axes[0].plot(wav_a, linewidth=0.8, color="#C44536", alpha=0.75, label="wavA raw")
    axes[0].plot(wav_a_smooth, linewidth=1.4, color="#102542", label="wavA smooth")
    axes[0].plot(wav_a_envelope, linewidth=1.1, color="#F4B942", label="wavA envelope")
    axes[0].legend(loc="upper right")
    axes[0].set_title(f"{subject_path.name} - wavA raw, smooth envelope, and Hilbert envelope")
    axes[0].set_ylabel("Amplitude")

    axes[1].plot(wav_b, linewidth=0.8, color="#4C78A8", alpha=0.75, label="wavB raw")
    axes[1].plot(wav_b_smooth, linewidth=1.4, color="#2F4858", label="wavB smooth")
    axes[1].plot(wav_b_envelope, linewidth=1.1, color="#8FCB9B", label="wavB envelope")
    axes[1].legend(loc="upper right")
    axes[1].set_title(f"{subject_path.name} - wavB raw, smooth envelope, and Hilbert envelope")
    axes[1].set_ylabel("Amplitude")
    axes[1].set_xlabel("Samples")
    fig.tight_layout()
    fig.savefig(audio_plot, dpi=160)
    plt.close(fig)

    print(json.dumps({"eeg_plot": str(eeg_plot), "audio_plot": str(audio_plot), "summary": summary["eeg_stats"]}, indent=2))

    if not args.no_readme_update:
        append_readme_update(
            [
                f"Generated signal-check plots for {subject_path.name}: {eeg_plot.name}, {audio_plot.name}.",
                f"EEG channel 1/2 range: min={summary['eeg_stats']['min']:.3f}, max={summary['eeg_stats']['max']:.3f}.",
                f"wavA range: min={summary['wavA_stats']['min']:.6f}, max={summary['wavA_stats']['max']:.6f}; wavB range: min={summary['wavB_stats']['min']:.6f}, max={summary['wavB_stats']['max']:.6f}.",
            ],
            title="signal_checks.py completed",
        )


if __name__ == "__main__":
    main()
