# Repository Model Lineage Audit

## Executive Summary
This archaeological audit traces the exact neural architecture that produced the dataset (`subject_distance_predictions.csv`) which powers our downstream Confidence System.

**CRITICAL FINDING:** The true model in production is **`ContrastiveMatchNet`**, not a generic `MatchNet`. Crucially, its Audio Encoder is hardcoded to accept a **28-band Gammatone filterbank** (`audio_channels=28`), not a flat 1D envelope. 

Our KUL-2 tensor conversion generated `(192,)` audio envelopes. If fed into the production model, it will crash instantly due to a channel mismatch (`1` vs `28`). **KUL preprocessing must be upgraded to generate 28-band Gammatone envelopes before inference can proceed.**

---

## TASK 1: Model Inventory
A scan of `models/` reveals the following neural architectures:
- `models/atcnet.py`: `ATCNet` (EEG-specific CNN-Attention architecture)
- `models/eegnet.py`: `EEGNet` (Standard EEG Convolutional architecture)
- `models/eegnet_tcn.py`: `EEGNetTCN` (EEGNet combined with Temporal Convolutional Network)
- `models/temporal_cnn.py`: `TemporalCNN`
- `models/vlaai_lite.py`: `VLAAI_Lite`
- `models/matchnet.py`: `ContrastiveMatchNet` and `AudioEncoder`

## TASK 2: Training Pipeline Lineage
Lineage for `subject_distance_predictions.csv`:
1. **Creation Script**: `training/export_subject_distance.py`
2. **Model Loaded**: `ContrastiveMatchNet(eeg_model_type="eegnet", eeg_channels=8, audio_channels=28, latent_dim=64)`
3. **Checkpoint Used**: Subject-specific LOSO weights (`checkpoints/matchnet_fold_{subject_id}_best.pth`)
4. **Preprocessing**: Handled by `prepare_dataset` (imported from `training/train_matchnet_loso.py`)
5. **EEG Channels**: 8 channels (Indices: `[13, 46, 43, 23, 50, 0, 52, 14]`)
6. **Audio Representation**: 28-channel frequency-decomposed envelopes (`audio_channels=28`)

## TASK 3: MatchNet Identity Audit
Inspection of `models/matchnet.py` confirms:
- **Does a class named MatchNet exist?** NO.
- **True Model Name**: `ContrastiveMatchNet`
- **Internal Structure**: Contains an `eeg_encoder` (which wraps `EEGNet`) and an `audio_encoder` (`AudioEncoder` class).
- **Exported Symbols**: `ContrastiveMatchNet`, `AudioEncoder`, `contrastive_loss`, `infonce_loss`.

## TASK 4: Checkpoint Audit
Checkpoints are saved under the `checkpoints/` directory following the naming convention:
- `matchnet_fold_{subject_id}_best.pth`
These are produced by the LOSO (Leave-One-Subject-Out) training script `training/train_matchnet_loso.py`.

## TASK 5: Confidence Pipeline Dependency Audit
Trace of inputs:
- The Confidence system trains on `margin`, `sim_A`, and `sim_B` from the CSV.
- `training/export_subject_distance.py` generates these CSV metrics.
- The metrics are directly derived from the cosine similarities output by `ContrastiveMatchNet`'s latent space mappings:
  ```python
  sim_a = pearson_corr(z_eeg, z_a, dim=1).mean().item()
  sim_b = pearson_corr(z_eeg, z_b, dim=1).mean().item()
  ```
- **Conclusion**: The entire Phase 4 and Phase 5 confidence architecture is implicitly modeling the exact latent space behaviors of `ContrastiveMatchNet`.

## TASK 6: Current Production Model
- **Production AAD Model**: `ContrastiveMatchNet` (using `EEGNet` backend)
- **Training Script**: `training/train_matchnet_loso.py`
- **Checkpoint**: `checkpoints/matchnet_fold_*_best.pth`
- **Input**: EEG `[B, 8, T]`, Audio `[B, 28, T]`
- **Output**: Latent embeddings `z_eeg`, `z_a`, `z_b` (Shape: `[B, 64, T]`)
- **Used in Confidence System?** YES.

## TASK 7: KUL Compatibility Verdict
**Verdict: C. Major Preprocessing Mismatch.**

*Evidence:*
The production `ContrastiveMatchNet` explicitly constructs its audio encoder as:
```python
self.audio_encoder = AudioEncoder(in_channels=28, latent_dim=64)
```
The DTU audio preprocessing pipeline generated 28 distinct subbands per audio stream. Our previous KUL tensor conversion proof (Phase KUL-2) extracted a simplistic 1-dimensional absolute-value envelope (Shape: `(192,)`). 

If we feed the `(192,)` KUL audio tensor into `ContrastiveMatchNet`, it will crash due to a channel dimensionality mismatch. We cannot use KUL for inference until we implement a 28-band Gammatone filterbank preprocessing module for the KUL audio streams.
