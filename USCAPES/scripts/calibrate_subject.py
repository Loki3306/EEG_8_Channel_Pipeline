"""
USCAPES: Universal Subject-Calibrated Auditory Processing & EEG Steering
Subject Few-Shot Calibration CLI Script.

Fine-tunes the 8-electrode spatial adapter on Trials 1-3 (~3 min) for any DTU subject.
Usage:
    python scripts/calibrate_subject.py --subject S2
    python scripts/calibrate_subject.py --subject S5 --calib-trials 3 --epochs 25
"""

import argparse
import sys
from pathlib import Path

# Add USCAPES root to Python path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from uscapes.pipeline.calibrator import calibrate_subject


def main():
    parser = argparse.ArgumentParser(
        description="USCAPES 3-Trial Few-Shot Subject Calibration"
    )
    parser.add_argument(
        "--subject", type=str, default="S1",
        help="Target subject identifier (e.g. S1, S2, ..., S18)"
    )
    parser.add_argument(
        "--calib-trials", type=int, default=3,
        help="Number of initial trials to use for calibration (default: 3)"
    )
    parser.add_argument(
        "--epochs", type=int, default=25,
        help="Optimization epochs for spatial adapter (default: 25)"
    )
    parser.add_argument(
        "--lr", type=float, default=1e-3,
        help="Learning rate for AdamW optimizer (default: 0.001)"
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Compute device ('cuda' or 'cpu', default: auto-detect)"
    )
    parser.add_argument(
        "--backbone", type=str, default=None,
        help="Path to universal base model checkpoint"
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Directory to save adapted weights"
    )
    args = parser.parse_args()

    sub = args.subject.upper()
    if not sub.startswith("S"):
        sub = f"S{sub}"

    backbone_path = Path(args.backbone) if args.backbone else None
    output_dir = Path(args.output_dir) if args.output_dir else None

    result = calibrate_subject(
        subject_id=sub,
        backbone_path=backbone_path,
        output_dir=output_dir,
        calib_trials=args.calib_trials,
        epochs=args.epochs,
        lr=args.lr,
        device=args.device
    )

    print("\n[CALIBRATION COMPLETE]")
    print(f"  • Subject:         {result['subject_id']}")
    print(f"  • Trials Used:     1 to {result['calib_trials']}")
    print(f"  • Final Loss:      {result['final_loss']}")
    print(f"  • Training Time:   {result['duration_sec']}s")
    print(f"  • Saved Weight:    {result['checkpoint_path']}")
    print("\nYou can now launch the dashboard to stream test trials:")
    print(f"    python run_uscapes.py --subject {result['subject_id']}")


if __name__ == "__main__":
    main()
