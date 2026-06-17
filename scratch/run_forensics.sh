#!/bin/bash
# run_forensics.sh
# Master script to execute all 7 phases of the Raw EEG Forensic Analysis on Kaggle.

set -e

RAW_DIR="/kaggle/input/datasets/lokeshgile/raw-eeg"
OUT_DIR="/kaggle/working/reports"

echo "=========================================="
echo " RAW EEG FORENSIC ANALYSIS PROJECT"
echo "=========================================="

echo "[Phase 1] Dataset Forensics..."
python scratch/raw_eeg_forensics.py --raw_dir "$RAW_DIR" --out_dir "$OUT_DIR"

echo "[Phase 2] Frequency Content Analysis..."
python scratch/eeg_band_energy.py --raw_dir "$RAW_DIR" --out_dir "$OUT_DIR"

echo "[Phase 3] Channel Importance Analysis..."
python scratch/channel_importance.py --raw_dir "$RAW_DIR" --out_dir "$OUT_DIR"

echo "[Phase 4] High Frequency Feasibility..."
python scratch/high_frequency_feasibility.py --raw_dir "$RAW_DIR" --out_dir "$OUT_DIR"

echo "[Phase 5] Temporal Resolution Audit..."
python scratch/temporal_resolution_audit.py --out_dir "$OUT_DIR"

echo "[Phase 6] EOG / Eye Movement Analysis..."
python scratch/eog_analysis.py --raw_dir "$RAW_DIR" --out_dir "$OUT_DIR"

echo "[Phase 7] Compiling Opportunity Report..."
python scratch/compile_opportunity_report.py --out_dir "$OUT_DIR"

echo "=========================================="
echo " ALL PHASES COMPLETE."
echo " Reports generated in: $OUT_DIR"
echo " Please download the reports directory or copy the markdown outputs."
echo "=========================================="
