#!/usr/bin/env python3
"""
CLI Tool: Separate Two Speakers from a Single Phone Recording.

Usage:
    python scripts/separate_phone_audio.py --input phone_recording.wav --output_dir separated/
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.audio.neural_speech_separator import SingleChannelNeuralSeparator


def main():
    parser = argparse.ArgumentParser(description="Separate Two Overlapping Speakers from a Single Phone Audio Track")
    parser.add_argument("--input", "-i", type=str, required=True, help="Path to input phone audio file (.wav, .mp3, .m4a)")
    parser.add_argument("--output_dir", "-o", type=str, default="results/phone_separation", help="Directory to save separated tracks")
    parser.add_argument("--device", type=str, default=None, help="Device ('cuda' or 'cpu')")
    parser.add_argument("--plot", action="store_true", default=True, help="Generate visual comparison plot")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"[ERROR] Input audio file not found: {input_path}")
        sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 80)
    print("  SINGLE-CHANNEL NEURAL SPEECH SEPARATION (PHONE RECORDING)")
    print(f"  Input: {input_path.name} | Output Dir: {output_dir}")
    print("=" * 80)

    print("  [1/3] Loading pre-trained Conv-TasNet model...")
    separator = SingleChannelNeuralSeparator(device=args.device)

    print("  [2/3] Performing neural speech separation...")
    s1, s2, p1, p2 = separator.separate_file(input_path, output_dir=output_dir)

    print(f"  [SAVED] Speaker 1 -> {p1}")
    print(f"  [SAVED] Speaker 2 -> {p2}")

    print("  [3/3] Extracting 64 Hz causal Gammatone neural envelopes...")
    env1, env2 = separator.extract_neural_envelopes(s1, s2)
    print(f"  Generated {len(env1)} envelope samples ready for CA-TCN AAD ingestion.")

    if args.plot:
        fig_path = output_dir / f"{input_path.stem}_separation_waveform.png"
        fig, axes = plt.subplots(3, 1, figsize=(10, 6), sharex=True, dpi=300)

        t_audio = np.linspace(0, len(s1) / 16000.0, len(s1))
        # Mixture proxy: sum of s1 and s2
        axes[0].plot(t_audio[:16000*5], (s1 + s2)[:16000*5], color="#555555", alpha=0.8)
        axes[0].set_title(f"Input Phone Mixture ({input_path.name})", fontweight="bold")
        axes[0].set_ylabel("Amplitude")

        axes[1].plot(t_audio[:16000*5], s1[:16000*5], color="#2b5c8f", alpha=0.85)
        axes[1].set_title("Separated Candidate: Speaker 1", fontweight="bold")
        axes[1].set_ylabel("Amplitude")

        axes[2].plot(t_audio[:16000*5], s2[:16000*5], color="#1b7837", alpha=0.85)
        axes[2].set_title("Separated Candidate: Speaker 2", fontweight="bold")
        axes[2].set_ylabel("Amplitude")
        axes[2].set_xlabel("Time (seconds)", fontweight="bold")

        plt.tight_layout()
        plt.savefig(fig_path, dpi=300)
        plt.close()
        print(f"  [SAVED] Separation visualization -> {fig_path}")

    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
