# ARCHIVE: KUL TRANSFER EXPERIMENTS

*Note: This section documents exploratory experiments testing the zero-shot generalization of the DTU-trained MatchNet on the KUL dataset. These experiments are currently out-of-scope for the primary DTU-based Selective AAD framework and confidence validation.*

## 1. KUL Dataset Structure
The KUL dataset was evaluated to test cross-dataset transfer.
- **Subjects**: 16 subjects (Focus on `S1`).
- **Trials**: 20 trials per subject.
- **Duration**: ~389 seconds (6.5 minutes).
- **Audio**: Dutch audiobooks presented dichotically.
- **Metadata**:
  - **`attended_ear`**: 'L' or 'R'.
  - **`stimuli`**: 1x2 cell array of filenames.
  - **Mapping Logic**: If `attended_ear == 'L'`, attended stream is `stimuli[0]`. If `'R'`, it is `stimuli[1]`.

## 2. Preprocessing Alignment
To feed KUL data into the DTU-trained MatchNet, the DTU MATLAB preprocessing was rebuilt in Python:
1. **Channel Selection**: BioSemi 64-channel mapped to DTU 8-channel subset.
2. **EEG Resampling**: 128 Hz downsampled to 64 Hz.
3. **The 28-Band Reconstruction**: 28 ERB-spaced gammatone filterbank (50Hz to 8000Hz) with `^0.3` power compression, downsampled to 64 Hz.
4. **Global Normalization**: Trials normalized using global means and standard deviations.

## 3. Forward Pass Validation (Phase 4.5 Audits)
Before retraining, a distribution audit (`step_6_9_kul_vs_dtu_distribution_audit.py`) was run. 
- Proved the reconstructed KUL envelopes possessed the exact same statistical mean and standard deviation as the DTU envelopes. 
- Passing both through the frozen MatchNet revealed that the L2 norms of the latent embeddings (`z_eeg`, `z_a`) aligned perfectly.

## 4. KUL S1 Transfer Ablation Results (Zero-Shot)
The zero-shot transfer was executed on all 20 trials of KUL Subject 1 (`step_6_8_kul_ablation_and_confidence.py`).

| Window Length | Window Accuracy | Trial Accuracy | Confidence AUROC |
|---------------|-----------------|----------------|------------------|
| 30s           | 75.8%           | 100.0%         | NaN              |
| 20s           | 71.4%           | 95.0%          | 0.952            |
| 15s           | 69.1%           | 95.0%          | 0.814            |
| 10s           | 66.8%           | 90.0%          | 0.812            |
| 5s            | 60.1%           | 90.0%          | 0.729            |
| 2s            | 54.3%           | 75.0%          | 0.612            |

## 5. Lessons Learned from Transfer Failure
Initial transfer attempts failed entirely (accuracy near 50%). We originally assumed MatchNet was brittle or the confidence network was overfitting. We proved the failure was actually just improper acoustic preprocessing. The DTU MatchNet expects 28 distinct, frequency-localized amplitude modulations. Feeding raw or single-band audio collapsed the latent space geometry. Once mechanically aligned, transfer was strong.
