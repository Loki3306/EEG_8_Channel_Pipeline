# Phase 11 Project State Checkpoint

## Introduction
This document serves as the reproducible checkpoint and comprehensive record of all achievements in the EEG-AAD pipeline prior to beginning Phase 12 (Sequential Decision System). It documents the evolution of the datasets, models, and confidence systems, culminating in the root cause analysis of the selective prediction mismatch.

## 1. Dataset Work
### DTU Dataset
- Served as the primary development dataset for the project.
- Discovered and resolved evaluation protocol issues (canonical 60s correlation evaluation).
- Conducted deep audio bias studies, latency audits, and preprocessing normalization verifications.

### KUL Dataset
- Introduced for zero-shot cross-dataset generalization.
- Built `kul_cached_dataset.py` to support efficient 64 Hz data loading matching the DTU schema.
- Verified input equivalence (spatial alignment, scaling) between DTU and KUL.
- Final cross-dataset evaluation proved zero-shot capability across datasets.

## 2. Model Evolution
### Linear Decoders (Neural Ridge)
- Baseline decoder to verify EEG-to-Audio mapping.
- Validated that spatial and temporal information can decode the attended speaker above chance.

### Baseline Neural Networks
- **Kuruvila CNN-LSTM**: Implemented and benchmarked, though it demonstrated limited robustness.
- **MatchNet**: Established as a stronger baseline, learning non-linear mappings. Augmented with confidence scoring for comparative evaluation.

### AAD-Conformer
- The state-of-the-art core architecture for this project.
- Uses spatio-temporal self-attention.
- Extensive hyperparameter ablations, channel pruning, and layer inspections conducted.
- The Conformer backbone is now **FROZEN** for Phase 12.

## 3. Confidence Research
### 3.1. Original Confidence Head
- Added a parallel output head to the AAD-Conformer to estimate the probability of correctness.
- Trained using Binary Cross Entropy (BCE) loss on the similarity margins.

### 3.2. Verification & Validation
- **Scientific Falsification**: Subjected the confidence head to rigorous falsification against random audio, zeroed EEG, and mismatched subjects.
- **Negative Controls**: Confirmed confidence plummets (AUC ~ 0.5) when presented with noise or permuted data.
- **Calibration (ECE)**: Measured Expected Calibration Error.
- **Discrimination (AUROC)**: Achieved ~0.73 AUROC for separating correct from incorrect predictions.
- **Cross-Dataset Verification**: Proved that the learned confidence generalizes to KUL data without retraining.

### 3.3. Ablation Studies
- **Latent-only Ablation**: Investigated if confidence could be estimated from internal representations without needing the audio references.
- **XGBoost Leakage Study**: Trained independent gradient boosters on the extracted features to prove that the confidence head was not simply memorizing confounding variables (RMS energy, volume, etc.).

### 3.4. Selective AAD & Root Cause Analysis
- **Majority Vote vs Accumulated Pearson**: Investigated why Trial-level Selective AAD (using majority vote over 10s windows) yielded 0.0% accuracy on KUL despite high Window Accuracy (~53%). 
- **Root Cause**: Phase 7 Trial Accuracy (72%) was based on 60s Pearson correlation. Phase 11 implemented a new paradigm using window majority voting. Since S1's window accuracy was ~53%, trials hovered around 50/50 correct/incorrect windows. Because the threshold logic was misaligned with the majority vote mechanics, trials consistently fell on the incorrect side of the vote.
- **Resolution**: Identified the need for formal temporal memory (Window Buffer) and evidence accumulation instead of naive window voting.

## 4. Current Limitations
- The model lacks temporal context across sequential windows (each window is treated independently).
- No mechanism to maintain prediction stability during momentary attention lapses or signal artifacts.
- No adaptive thresholding for dynamic environments.
- Decision policy is hardcoded rather than context-aware.

## 5. Next Steps
Phase 12 will implement a production-grade Confidence-Aware Sequential Decision System to bridge the gap between window-level predictions and robust real-world performance, starting with a model-agnostic Window Buffer.
