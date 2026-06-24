# 00 Executive Summary

## Project Goal
To develop a robust, continuous Auditory Attention Decoding (AAD) system capable of identifying which speaker a listener is attending to from single-trial EEG data, and to augment this system with an introspective confidence framework that allows for selective prediction (rejecting uncertain windows).

## Problem Statement
Standard AAD models are forced to output a prediction for every time window, even when the EEG signal is heavily corrupted by noise, artifacts, or momentary lapses in user attention. This results in unpredictable errors in real-time applications like smart hearing aids. Furthermore, transferring these models across datasets (e.g., DTU to KUL) often reveals severe domain shifts and hidden dependencies on exact preprocessing pipelines.

## Final System Architecture
The final system is a two-stage pipeline:
1. **ContrastiveMatchNet**: A deep learning architecture comprising an EEGNet encoder and a 1D-CNN Audio encoder. It projects both EEG and competing audio streams into a shared latent space optimized via InfoNCE loss, computing cosine similarities to identify the attended stream.
2. **XGBoost Confidence Runtime**: A secondary model that extracts temporal dynamics (margin magnitude, rolling standard deviation of the margin, and trial consistency) from the MatchNet outputs. It generates a calibrated confidence score (Probability of Correctness) to accept or reject predictions in real-time.

## Key Findings
- **Margin is the Signal**: The absolute difference in latent similarity between the attended and unattended streams (`margin`) is the strongest predictor of model correctness.
- **Temporal Consistency**: Errors cluster in time. A rolling standard deviation of the margin highly correlates with localized signal corruption.
- **Selective AAD is Viable**: By rejecting the bottom 20-30% of windows based on confidence scores, the system's effective accuracy on accepted windows jumps from ~71% to over 80-85%.
- **Cross-Dataset Fragility**: Transferring from DTU to KUL revealed severe domain shifts. Proper translation required replicating the exact ERB-spaced 28-band Gammatone filterbank and power compression (`^0.3`) used in DTU.

## Current Status
The DTU-based MatchNet and Confidence System are fully trained, validated, and audited. The framework supports runtime simulation. The project has successfully bridged the preprocessing gap to the KUL dataset, isolating domain shift variables for future zero-shot or few-shot transfer learning.
