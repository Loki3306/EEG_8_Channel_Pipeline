# Current End-to-End System Architecture

This document serves as the complete, reverse-engineered technical design document for the current EEG Auditory Attention Decoding (AAD) system. It outlines the pipeline precisely as it exists in the codebase today.

## Section 1: Repository Pipeline Overview
See `docs/CURRENT_SYSTEM_FLOW_DIAGRAM.md` for the visual flow chart. 

The system trains a Siamese MatchNet to align EEG and Audio into a shared temporal latent space using contrastive loss, validated using Leave-One-Subject-Out (LOSO) cross-validation.

## Section 2: Dataset Documentation

- **Dataset**: Auditory Attention Dataset (KUL Dataset format, typically `S*_data_preproc.mat` files).
- **Number of Subjects**: Historically up to 18 (`S1` to `S18`), though `subject_files()` dynamically globs all available `S*.mat`.
- **Trials per Subject**: ~20 trials (dynamically loaded via `load_subject_examples`).
- **Sampling Rate**: `FS = 64` Hz.
- **Trial Duration**: Variable, typically ~60 seconds.
- **EEG Channel Count**: 8 active channels (`[13, 46, 43, 23, 50, 0, 52, 14]`).
- **Audio Representation**: 28-band Gammatone subband envelopes.
- **Label Format**: Integer (1 or 2, denoting whether `wav_a` or `wav_b` was attended).
- **Mapping File Format**: 
  - `audio_mapping.json`: Maps `{subject_id: {trial_index: {wavA: {filename}, wavB: {filename}}}}`
  - `gammatone_envelopes.pkl`: A serialized dictionary mapping string filenames to `[28, T]` numpy arrays.

**Single Example Contents:**
An example contains raw matrices that are synchronized.
- **EEG**: `[8, T]`
- **Audio A**: `[28, T]`
- **Audio B**: `[28, T]`
- **Label**: Integer tracking which audio stream the subject was instructed to focus on.

## Section 3: EEG Preprocessing Pipeline
Implemented primarily in `training/train_matchnet_loso.py -> prepare_dataset()`:

1. **Channel Selection**: EEG matrix `[C, T]` is sliced to 8 channels: `[13, 46, 43, 23, 50, 0, 52, 14]`.
2. **Bandpass Filtering**: 
   - Uses `scipy.signal.butter` & `filtfilt`.
   - Lowcut: 1.0 Hz, Highcut: 6.0 Hz. Order: 2.
3. **Normalization**: 
   - `normalize_array()`: Z-score normalization independently for each channel (`arr - mean / (std + 1e-12)`).
4. **Length Truncation**:
   - Audio envelopes and EEG arrays are truncated to `min(eeg.shape[1], env.shape[1])` to guarantee perfect alignment.
5. **Windowing & Chunk Generation**:
   - `chunk_trial(x, ya, yb, window_sec, hop_sec)` slices the full continuous trial into `[8, win_samples]` chunks.
   - E.g., Training chunks use `window_sec=5.0`, `hop_sec=2.0`.

## Section 4: Audio Processing Pipeline
- **Raw Audio**: Originally .wav files, externally preprocessed into `.pkl` envelopes.
- **Representation**: 28-subband Gammatone filterbank envelopes (`[28, T]`).
- **Processing**: 
  - Subbands are loaded dynamically via `audio_mapping.json` matching in `prepare_dataset()`.
  - Normalization: Z-score normalized independently per subband using `normalize_array()`.
- **Final Tensor Shape**: `[28, T]` continuous array prior to chunking.

## Section 5: MatchNet Architecture
`models/matchnet.py` -> `ContrastiveMatchNet`

**AudioEncoder (Standard)**:
- **Input**: `[B, 140, T]` (28 channels * 5 temporal lags)
- **Conv1D**: `in_channels=140`, `out=32`, `kernel=15`, `padding=7`
- **BatchNorm1D**: `32`
- **GELU**
- **Dropout**: `0.2`
- **Conv1D**: `in=32`, `out=64`, `kernel=15`, `padding=7`
- **BatchNorm1D**: `64`
- **GELU**
- **Dropout**: `0.2`
- **Conv1D**: `in=64`, `out=64` (Latent Dim), `kernel=1`

**Audio Lag Modeling**:
- Native `lags=[3, 6, 10, 13, 16]` (approx 50ms - 250ms at 64Hz).
- Concatenated along the channel dimension prior to the `AudioEncoder` via `create_lagged_audio()`.

## Section 6: EEG Encoder Documentation

### 1. EEGNet (Current Baseline Default)
- **Input**: `[Batch, 1, 8, T]`
- **Block 1**: 
  - `Conv2D(1, 8, kernel=(1, 64), padding=(0, 32), bias=False)` (Temporal Convolution)
  - `BatchNorm2D(8)`
  - `Conv2D(8, 16, kernel=(8, 1), groups=8, bias=False)` (Spatial Depthwise)
  - `BatchNorm2D(16)`
  - `GELU`
  - `Dropout(0.25)`
- **Block 2**:
  - `Conv2D(16, 16, kernel=(1, 16), padding=(0, 8), groups=16, bias=False)` (Separable)
  - `Conv2D(16, 16, kernel=(1, 1), bias=False)` (Pointwise)
  - `BatchNorm2D(16)`
  - `GELU`
  - `Dropout(0.25)`
- **Output Projection**: Squeezed to `[B, 16, T]`, then `Conv1D(16, 64, kernel=1)` maps to latent dimension.
- **Output Shape**: `[B, 64, T]`

### 2. MultiScaleEEGNet
- **Architecture**: Replaces the single `kernel=(1, 64)` with 4 parallel temporal convolutions:
  - `Conv2D(1, 8, (1, 8))`
  - `Conv2D(1, 8, (1, 16))`
  - `Conv2D(1, 8, (1, 32))`
  - `Conv2D(1, 8, (1, 64))`
- Outputs are concatenated along channel dim, passing through spatial depthwise and pointwise layers. Output shape is `[B, 64, T]`.

### 3. EEGNetTCN
- Uses the EEGNet frontend (Temporal + Spatial) but replaces Block 2 with a Temporal Convolutional Network (TCN) featuring dilated causal convolutions for wider receptive fields.

### 4. ATCNet
- Highly complex model leveraging a CNN frontend, Multi-head Self-Attention, and a Temporal Convolutional Network. Output shape differs natively, causing compatibility issues requiring adapter layers in MatchNet.

**Currently Used**: `EEGNet` is the hardcoded baseline in `train_matchnet_loso.py`.

## Section 7: Loss Functions

**contrastive_loss(z_eeg, z_a, z_b, margin=0.1)**
- Mathematically computes the point-wise cosine similarity over the temporal dimension `[B, T]` for `(eeg, a)` and `(eeg, b)`.
- `sim_a = cosine_similarity(z_eeg, z_a, dim=1)`
- `sim_b = cosine_similarity(z_eeg, z_b, dim=1)`
- Averages similarities across time: `sim_a = sim_a.mean()`, `sim_b = sim_b.mean()`
- **Target**: Ensure `sim_a` exceeds `sim_b` by `margin`.
- **Loss Equation**: `loss = max(0, margin - sim_a + sim_b)`
- Summed over the batch.

## Section 8: Training Pipeline
`train_matchnet_loso.py`

1. **Subject Split**: Outer loop iterates over test subjects (`S1` to `S18`).
2. **Train/Val Split**: Remaining subjects are shuffled. 80% assigned to Train, 20% to Validation.
3. **Data Prepping**: Continual trials are chunked (Train: 5s window / 2s hop. Val: 10s window / 5s hop).
4. **Optimizer**: `AdamW(lr=1e-3, weight_decay=1e-4)`.
5. **Scheduler**: `CosineAnnealingLR(T_max=epochs)`.
6. **AMP**: `torch.amp.autocast` and `GradScaler` used.
7. **Batch Size**: Default 128.
8. **Validation Metric**: After each epoch, `evaluate_model` measures Validation Accuracy on chunks.
9. **Checkpointing**: If Val Acc > Best Val Acc, `matchnet_fold_S{x}_best.pth` is updated. 
10. **Early Stopping**: Halts if no improvement for 10 epochs.
11. **Best Model Definition**: Model with highest chunk-level accuracy on the 20% validation pool.

## Section 9: LOSO Evaluation Pipeline
After an inner split finishes, the "Best Model" is loaded from disk.
- Evaluated on the held-out **Test Subject**.
- The Test Subject's full 60s trials are chunked into **10s non-overlapping windows** (`window_sec=10`, `hop_sec=10`).
- Accuracy is computed across all 10s chunks.
- The average test accuracy is reported and aggregated across all subjects to yield the final LOSO mean.

## Section 10: Evaluation Logic
`evaluate_model(model, eeg_loader, device)`

- Model encodes `eeg`, `audio_a`, `audio_b` into `z_eeg`, `z_a`, `z_b` `[B, 64, T]`.
- **Cosine Similarity**: `sim_a = F.cosine_similarity(z_eeg, z_a, dim=1).mean(dim=1)`. (Calculates similarity at every time step, then averages across time to yield a single scalar per chunk).
- **Decision**: The model chooses the stream with higher similarity: `correct = (sim_a > sim_b).sum()`.
- **Accuracy**: `correct / total` over all chunks in the DataLoader.

## Section 11: Window Scaling Study
`analysis/window_scaling_study.py` & `training/evidence_accumulation_study.py`

Evaluates the exact same pre-trained `checkpoints/` but dynamically changes the inference window.
- **Sizes evaluated**: 0.5s, 1s, 2s, 5s, 10s, 20s, 30s, 60s.
- **Aggregation**: A 60s trial generates multiple decisions (e.g., thirty 2s decisions). 
- **Latency**: Shorter windows reduce wait time but severely drop accuracy (e.g., 20s ≈ 75%, 2s ≈ 58%, 0.5s ≈ 53%).
- **Evidence Accumulation**: Summing log-odds (`sim_a - sim_b`) of dense overlapping 2s hops mathematically recovers long-window performance, proving feature extraction is sound but latency is bottlenecked by the requirement for multiple observations.

## Section 12: Current Experimental Branches

| Experiment | Status | Result | Notes |
| :--- | :--- | :--- | :--- |
| **Baseline EEGNet** | Standard | LOSO ~77% (10s) | Serves as the primary checkpoint target. |
| **InfoNCE Loss** | Investigating | N/A | Exists in `matchnet.py` but unused natively by LOSO pipeline. |
| **Lag Modeling** | Active | Beneficial | 5 native lags natively embedded in `ContrastiveMatchNet`. |
| **Inception Audio** | Kill | Degradation | Multi-kernel audio encoder failed to improve correlation. |
| **Temporal Attention**| Kill | Degradation | Pooling temporal dimension natively destroyed similarity alignment. |
| **MultiScale EEGNet** | Promising | +2-3% | Replaces temporal kernel with 8,16,32,64 parallel scales. |

## Section 13: Parameter Counts

Exact trainable parameter counts for current modules:
- **EEGNet**: 1,249
- **MultiScaleEEGNet**: 3,761
- **EEGNetTCN**: 4,402
- **ATCNet**: 11,969
- **AudioEncoder** (Standard, 140-in): 48,608
- **Full MatchNet** (EEGNet + Audio + Latent Proj): 104,688

## Section 14: Tensor Shape Audit

**Tracking a Single Batch (`B=128`, `Window=5s` @ 64Hz = 320 samples):**

**EEG Path:**
1. `Input EEG`: `[128, 8, 320]`
2. `Unsqueeze`: `[128, 1, 8, 320]`
3. `Temporal Conv (1x64)`: `[128, 8, 8, 321]`
4. `Spatial Depthwise`: `[128, 16, 1, 321]`
5. `Separable Block`: `[128, 16, 1, 322]`
6. `Squeeze`: `[128, 16, 322]`
7. `Latent Conv1D (16->64)`: `[128, 64, 322]`
8. `Truncate to orig_len`: `[128, 64, 320]`

**Audio Path:**
1. `Input Audio`: `[128, 28, 320]`
2. `Lags Extraction (5 lags)`: `[128, 140, 320]`
3. `Conv1D (140->32)`: `[128, 32, 320]`
4. `Conv1D (32->64)`: `[128, 64, 320]`
5. `Latent Conv1D (64->64)`: `[128, 64, 320]`

**Similarity Path:**
1. `Cosine Similarity`: `[128, 64, 320] x [128, 64, 320]` -> `[128, 320]`
2. `Temporal Mean`: `[128]` scalar scores.

## Section 15: Current Known Limitations

- **Short-Window Degradation**: Feature extraction is too weak at ≤ 2s to reliably overcome noise on a single chunk. Accuracy drops from 75% at 20s down to ~55% at 0.5s.
- **Subject Variability**: S3 and S7 perform severely below the LOSO mean due to label noise or physical electrode misplacement (audited as corrupted distributions).
- **Temporal Pooling Failures**: Attempting to pool the temporal dimension via Attention before calculating similarity aggressively destroys accuracy, proving that maintaining `T` throughout the model is strictly necessary for Contrastive cosine distance.
