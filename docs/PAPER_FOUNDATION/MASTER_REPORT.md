# MASTER REPORT: EEG Auditory Attention Decoding Confidence Framework

## Table of Contents
1. [Executive Summary](./00_EXECUTIVE_SUMMARY.md)
2. [Project History](./01_PROJECT_HISTORY.md)
3. [Dataset and Preprocessing](./02_DATASET_AND_PREPROCESSING.md)
4. [MatchNet Architecture](./03_MATCHNET_ARCHITECTURE.md)
5. [Training Pipeline](./04_TRAINING_PIPELINE.md)
6. [Evaluation Protocol](./05_EVALUATION_PROTOCOL.md)
7. [Baseline Results](./06_BASELINE_RESULTS.md)
8. [Confidence Framework](./07_CONFIDENCE_FRAMEWORK.md)
9. [Confidence Audits](./08_CONFIDENCE_AUDITS.md)
10. [Failure Analysis](./09_FAILURE_ANALYSIS.md)
11. [Runtime System](./10_RUNTIME_SYSTEM.md)
12. [KUL Transfer Study](./11_KUL_TRANSFER_STUDY.md)
13. [Limitations](./14_LIMITATIONS.md)

---

## I. System Overview
This project presents an end-to-end framework for Selective Auditory Attention Decoding (AAD). Unlike traditional continuous AAD systems that produce mandatory predictions at every time step, our framework introduces a second-order introspective model (the Confidence Framework) that evaluates the reliability of the primary AAD model (ContrastiveMatchNet). This allows the system to reject highly uncertain predictions, dramatically increasing effective accuracy and stabilizing downstream applications like smart hearing aids.

```text
[Raw EEG & Audio] ---> [28-Band Preprocessing] ---> [ContrastiveMatchNet]
                                                           |
                                                      (Latent Space)
                                                           |
                                                   [Feature Extraction]
                                                   (margin, std_margin)
                                                           |
                                                   [XGBoost Confidence]
                                                           |
                                                   [Accept or Reject?]
```

## II. The Core Architecture: ContrastiveMatchNet
The primary model projects 8-channel EEG (`Fp1`, `Fp2`, `F7`, `F8`, `T7`, `T8`, `P7`, `P8`) and 28-band Gammatone audio envelopes into a shared 64-dimensional latent space using InfoNCE loss.
- **EEG Encoder**: Convolutional topology extracting spatial and temporal features.
- **Audio Encoder**: 1D-CNN compressing the 28 acoustic bands.
- **Matching Mechanism**: Computes Pearson Correlation between `z_eeg` and `{z_a, z_b}`. The model predicts the attended stream by picking `max(sim_a, sim_b)`.

## III. The Breakthrough: The Margin
During Phase 3, we discovered that raw EEG does not need to be fed into the confidence model. The geometry of the latent space itself encodes certainty.
The **Margin** is defined as `abs(sim_a - sim_b)`.
- Correct predictions possess large, stable margins.
- Incorrect predictions possess near-zero margins.
By calculating the trailing standard deviation of the margin (`rolling_std_margin`), the system detects localized bursts of artifact corruption.

## IV. Experimental Results (DTU Dataset)
- **Base MatchNet Accuracy**: ~71% Window Accuracy.
- **Confidence Calibration**: The XGBoost model successfully correlates its output probability with empirical accuracy (AUROC ~0.78).
- **Selective Prediction Performance**: Rejecting the bottom 30% of windows boosts accuracy from 71% to >85%.

## V. Zero-Shot Cross-Dataset Transfer (KUL)
Transferring the model from the DTU dataset to the KUL dataset initially failed. Deep audits (Phase 4.5 and 6) revealed that the "domain shift" was entirely due to mechanical preprocessing discrepancies. 
When the precise 28-band ERB Gammatone filterbank and `^0.3` power compression were replicated for KUL, the DTU-trained MatchNet demonstrated **Strong Transfer** (100% Trial Accuracy on KUL S1), proving the robustness of the latent space geometry.

## VI. Conclusion
This framework shifts the AAD paradigm from "always-on" continuous decoding to a probabilistic, stateful runtime system. It establishes that robust selective prediction can be achieved purely through latent similarity analysis, avoiding the need for complex deep-ensemble uncertainty quantification.

*For full results, ablation tables, and reproducibility commands, see the associated sub-documents in this directory.*
