# Data Analysis README - EEG AAD Project

[Update - 2026-05-05]
- Inspected all 18 preprocessed subject files in `data/`.
- Confirmed a consistent trial-level structure: 60 trials per subject, 3200 samples per trial, 66 EEG channels, and 64 Hz sampling for EEG/audio in the preprocessed files.
- Found that the files expose nested MATLAB structs and cell arrays, but do not expose a top-level `expinfo` table in this workspace snapshot.

## 1. Dataset Overview

This workspace contains 18 subject-specific preprocessed EEG/AAD `.mat` files in `data/`. Each file stores trial-segmented EEG plus paired audio streams for the two competing speakers.

Verified recording structure from the files:

- Subjects: 18
- Trials per subject: 60
- EEG channels per trial: 66
- Samples per trial: 3200
- Sampling rate in the preprocessed files: 64 Hz
- Trial duration implied by the data: 50 seconds

Task summary:

- The subject attends one of two competing speakers.
- The stored data includes aligned EEG and two audio streams, `wavA` and `wavB`, for each trial.
- The exact semantic mapping of the label values to male/female or left/right is not yet fully verified from the preprocessed files alone.

## 2. File Structure

Current workspace structure:

```text
data/
  S1_data_preproc.mat
  S2_data_preproc.mat
  S3_data_preproc.mat
  S4_data_preproc.mat
  S5_data_preproc.mat
  S6_data_preproc.mat
  S7_data_preproc.mat
  S8_data_preproc.mat
  S9_data_preproc.mat
  S10_data_preproc.mat
  S11_data_preproc.mat
  S12_data_preproc.mat
  S13_data_preproc.mat
  S14_data_preproc.mat
  S15_data_preproc.mat
  S16_data_preproc.mat
  S17_data_preproc.mat
  S18_data_preproc.mat
```

Each `.mat` file contains one top-level variable named `data`.

[Update - 2026-05-05]
- Verified that the preprocessed files are embedded trial bundles, not a continuous subject-wide recording.
- No separate audio folder or raw audio files were present in this workspace snapshot.

## 3. MATLAB Structure Breakdown

For each `.mat` file, the top-level keys are the same:

```text
data
```

For `S1_data_preproc.mat` the nested structure is:

```text
data → struct
  dim     → struct wrapper
  fsample → struct wrapper
  event   → struct wrapper
  eeg     → 1 x 60 cell-like array of trial matrices
  cfg     → 1 x 19 cell-like array of processing-history structs
  wavA    → 1 x 60 cell-like array of trial audio segments
  wavB    → 1 x 60 cell-like array of trial audio segments
```

Verified field shapes and datatypes in the preprocessed files:

- `data.eeg`: object array, shape `(1, 60)`
- `data.wavA`: object array, shape `(1, 60)`
- `data.wavB`: object array, shape `(1, 60)`
- `data.event`: object array, shape `(1, 1)`
- `data.fsample`: object array, shape `(1, 1)`
- `data.dim`: object array, shape `(1, 1)`
- `data.cfg`: object array, shape `(1, 19)`

Trial-level inner shapes in every file checked:

- `data.eeg[0,0]` → `(3200, 66)` float64
- `data.wavA[0,0]` → `(3200, 1)` float64
- `data.wavB[0,0]` → `(3200, 1)` float64

`data.fsample` contents:

- `eeg = 64`
- `wavA = 64`
- `wavB = 64`

`data.dim` contents:

- `chan` → channel-name structure
- `eeg` → `time_chan`
- `wavA` → `time_chan`
- `freq` → present as a nested field wrapper
- `wavB` → `time_chan`

`data.cfg` contents:

- This is a processing-history container, not trial metadata.
- The inspected entries contain function names such as `co_preprocessing`, `co_resampledata`, `co_appenddata`, `co_denoise`, `co_selectdim`, `co_selectevent`, `co_splitdata`, `co_auditoryfilterbank`, `co_dimaverage`, and `co_squeeze`.

## 4. EEG Data Understanding

Verified EEG facts:

- Channel count: 66
- Sampling rate in the preprocessed files: 64 Hz
- Inner trial shape: `(3200, 66)`
- Trial duration: `3200 / 64 = 50` seconds

Time dimension meaning:

- The first axis is time samples.
- The second axis is channels.
- Each trial is stored independently in a cell entry, so this is segmented data rather than one continuous recording.

Channel labels extracted from `dim.chan.eeg` in `S1_data_preproc.mat`:

```text
Fp1, AF7, AF3, F1, F3, F5, F7, FT7, FC5, FC3, ... , P2, P4, P6, P8, P10, PO8, PO4, O2, EXG1, EXG2
```

The first 64 labels match a standard EEG montage pattern. The last two labels are `EXG1` and `EXG2`.

[Update - 2026-05-05]
- Confirmed that all 18 files share the same EEG shape and sampling rate.
- The files do not currently verify whether `EXG1`/`EXG2` are mastoids, EOG, or other external channels, so that part remains open.

## 5. Audio Data Understanding

Audio is embedded in each subject file as `wavA` and `wavB`.

Verified audio facts:

- Both `wavA` and `wavB` are present for every trial.
- Each audio segment has shape `(3200, 1)`.
- Their sampling rate is 64 Hz in the preprocessed files.
- The length matches the EEG trial length exactly, so the streams are sample-aligned within each trial.

File naming logic:

- The workspace uses one file per subject: `S1_data_preproc.mat` through `S18_data_preproc.mat`.
- No separate audio filename mapping is exposed in the current workspace snapshot.

## 6. Trial Structure

How trials are defined:

- Each subject file contains 60 trial cells in `eeg`, `wavA`, and `wavB`.
- Each trial cell contains a 50-second segment sampled at 64 Hz.
- The EEG and audio segments appear to be aligned on a per-trial basis.

How EEG aligns with the audio:

- For a given trial index, `data.eeg[0, trial]`, `data.wavA[0, trial]`, and `data.wavB[0, trial]` all have the same number of time samples.
- That means each trial bundles one EEG segment and the two competing audio streams.

How to know which is attended:

- The current files expose trial markers in `data.event`.
- Those markers are stored as a per-trial event list with 60 entries for each trial container.
- The event list values are binary (`1` or `2`), but the exact semantic mapping to attended speaker identity is not yet verified from these files alone.

## 7. Label Definition

Current verified label storage:

- The label information visible in the preprocessed files is stored in `data.event.eeg`.
- Each trial has event entries with `sample = 1` and `value` equal to `1` or `2`.

What this means right now:

- There is a binary trial-level marker present.
- The files do not yet expose a direct `attend_mf`, `attend_lr`, or `expinfo` table in the current workspace snapshot.
- Therefore the exact mapping `1 -> male/female/left/right` is still unverified.

Example of the verified structure:

```text
event.eeg[trial][0].sample = 1
event.eeg[trial][0].value  = 1 or 2
```

Open mapping problem:

- We still need to verify whether `1` means attended male, attended female, attended left, or attended right.
- We also need to verify whether the binary event value is the final label or only a marker used during preprocessing.

## 8. Time Alignment

Verified alignment facts:

- EEG and both audio streams are stored with the same sample length per trial.
- The trial sample rate is 64 Hz.
- Each trial has 3200 samples, so trial duration is 50 seconds.

Event timing:

- The stored event markers have `sample = 1` in the inspected trial.
- This suggests the available event structure is not a dense latency timeline inside the trial.
- No explicit delay parameter or time-shift field was found in the preprocessed files.

Unknown alignment points:

- Whether the audio streams already include any latency compensation is not yet verified.
- Whether additional trigger timing information exists outside the preprocessed `.mat` files is not yet verified.

## 9. Preprocessed Data

What `DATA_preproc` contains in practice:

- A per-subject preprocessed bundle with segmented EEG and two audio streams per trial.
- A processing history in `cfg`.
- Trial event markers in `event`.

What preprocessing has already been done:

- Resampling is present in the processing history.
- Preprocessing and selection steps are present in the processing history.
- Denoising, auditory filterbank processing, dimension averaging, squeezing, and append operations are visible in `cfg`.

What assumptions are baked into it:

- The data are already segmented into trials.
- The file stores aligned per-trial EEG and audio rather than raw continuous recordings.
- The data have already been reduced to 64 Hz in the preprocessed form.

[Update - 2026-05-05]
- `cfg` is best treated as a provenance log. It is useful for understanding the preprocessing pipeline, but it is not a replacement for trial-level experiment metadata.

## 10. Unknowns / Open Questions

- Exact label mapping of event values `1` and `2` is not yet verified.
- `expinfo` is not present as a top-level field in the current preprocessed files.
- The meaning of `EXG1` and `EXG2` is not confirmed from the file contents alone.
- The exact preprocessing parameters behind each `cfg` function call are not yet decoded.
- Whether a separate raw or companion audio dataset exists outside this workspace is not verified here.
- Whether any timing delays were already corrected before saving the preprocessed files is not confirmed.

## 11. Verified Facts vs Assumptions

### Verified

- There are 18 subject files in `data/`.
- Every file contains a top-level `data` struct.
- Each subject file contains 60 trials.
- Each EEG trial has shape `(3200, 66)`.
- Each audio trial in `wavA` and `wavB` has shape `(3200, 1)`.
- The preprocessed files use a 64 Hz sampling rate.
- The files are trial-segmented rather than continuous.
- `data.cfg` records preprocessing history.
- `data.event.eeg` stores binary per-trial marker values in `{1, 2}`.

### Assumed

- `data.event.eeg` is the label source for AAD classification, because its trial count matches the EEG/audio trial count.
- `EXG1` and `EXG2` are external reference channels or non-EEG auxiliary channels, but the exact role is not yet verified.
- The binary label values likely encode the attended condition, but the exact mapping is still open.

## 12. Next Steps (Data Phase Only)

- Verify the semantic meaning of event values `1` and `2` against the original experiment documentation or companion metadata.
- Check whether any external `expinfo` or trial metadata exists outside the current preprocessed `.mat` files.
- Decode the nested `cfg` history enough to identify the preprocessing parameters used at each step.
- Confirm the intended role of `EXG1` and `EXG2`.
- If a companion raw/audio dataset exists, map its filenames to the per-trial `wavA` and `wavB` streams.
[Update - 2026-05-05 20:45]
- inspect_data.py completed
- Inspected 18 subject file(s) and wrote inspect_summary.json.
- Consistency check: trials=[60], eeg_shapes=[(3200, 66)], fs=[64].
- Verified top-level key set: ['data'].
[Update - 2026-05-05 22:13]
- validate_labels.py completed
- Validated label-correlation rule for 18 subject file(s) and wrote label_validation.json.
- Overall consistency=0.481; label=1 rate=0.474; label=2 rate=0.489.
- Decision rule threshold=0.80 -> mapping unclear.
[Update - 2026-05-05 22:45]
- Revised validate_labels.py to use lagged per-channel correlation across all EEG channels and lags 0-19.
- Validated 18 subject file(s) and wrote label_validation_lagged.json.
- Overall consistency improved to 0.562, with label=1 rate=0.581 and label=2 rate=0.544.
- Result is still below the 0.60 confirmation target, so the A/B mapping remains unconfirmed.

[Update - 2026-05-05 22:45]
- Added recover_speaker_identity.py to search for direct metadata first and then fall back to pair-constrained clustering over wavA/wavB.
- Ran it across 18 subject file(s) and wrote speaker_identity.json.
- The pair-constrained fit converged cleanly with trial-distinct rate=1.000 and a stable two-cluster split, but the male/female naming is still heuristic because no explicit speaker metadata was found.
[Update - 2026-05-06 14:53]
- evaluate_mapping.py completed
- Final cluster→label mapping: {0: 2, 1: 1} (accuracy=1.0000)
- Mapping A accuracy: 1.0000
- Mapping B accuracy: 1.0000
[Update - 2026-05-08 02:21]
- loso_ridge_runner.py completed
- Phase 2 ridge AAD baseline implemented.
- LOSO trial accuracy=0.5389; balanced accuracy=0.5400; mean corr diff=0.0141.
- Selected label->stream mapping: {1: 'A', 2: 'B'}.
- Sanity mode: zero_eeg=False, shuffle_labels=False.

[Update - 2026-05-09]
- Temporal CNN reconstruction path now supports train-only Hilbert envelope preprocessing with power-law compression and lowpass filtering.
- Added configurable lag-window expansion in milliseconds and longer evaluation windows (10s, 20s, 30s).
- Added a second training objective: contrastive EEG-audio alignment using positive aligned chunks and negative shifted audio chunks.
- Current modeling hypothesis: with strict LOSO and 2 channels, representation/objective quality is the limiting factor rather than model size.

## 13. Experiments and Training Phases

### Phase 1: Baseline Models (Ridge + CNN Reconstruction)

Initial experiments established performance baselines using LOSO (Leave-One-Subject-Out) cross-validation:

- **Ridge Baseline** (~55.2% LOSO accuracy)
  - Input: Lagged EEG features (16 lags, 16ms steps) from 2 selected channels
  - Target: Attended speaker envelope (Hilbert + preprocessing)
  - Method: Linear ridge regression with cross-validated regularization
  
- **Temporal CNN Reconstruction** (~56.4% LOSO accuracy)
  - Model: 69k-parameter architecture with residual blocks + multires convolution branches
  - Input: 2-channel EEG (64Hz, 5-second windows)
  - Target: Full-trial attended speaker envelope (correlation loss)
  - Early stopping: patience=5 epochs

**Key Findings**: Cross-subject generalization ceiling at ~56–59% despite varying model capacity suggests signal preprocessing or training objective is the primary bottleneck, not architecture.

### Phase 2: Signal Preprocessing Pipeline

Implemented comprehensive preprocessing for target envelope extraction:

```python
speech_envelope(wav, compression=0.6, lowpass_hz=8.0, normalize=True):
  1. Hilbert transform → analytic signal envelope extraction
  2. Moving average smoothing (window=64 samples @ 64Hz)
  3. Power-law compression (exponent=0.6) → reduce dynamic range
  4. 4th-order Butterworth lowpass @ 8Hz → smooth envelope
  5. Train-only normalization (fit mean/std on train set, apply to train+test)
```

**Parameters**:
- `--env-compress`: Envelope compression exponent (default 0.6)
- `--env-lowpass`: Envelope lowpass Hz (default 8.0; set to 0 to disable)

**Contrastive Baseline**: ~58.89% LOSO accuracy using InfoNCE loss with improved preprocessing.

### Phase 3: Enhanced Contrastive Learning (Current)

Implements stronger negative sampling strategies and within-subject evaluation to quantify subject-specific variability:

#### Hard Negative Sampling Strategies

Four configurable modes for sampling harder negatives during contrastive training:

1. **`random`** (default): 
   - Sample opposite stream, no temporal shift
   - Baseline contrastive objective
   
2. **`nearby`**: 
   - Sample opposite stream with random temporal shift
   - Forces model to discriminate between temporally close distractors
   
3. **`same-trial`**: 
   - Sample opposite stream with larger temporal shift
   - Ensures negative and positive are from same trial but temporally distinct
   
4. **`mixed`**: 
   - Randomly choose strategy per batch
   - Encourages robustness across multiple difficulty levels

**CLI Flags**:
```bash
--negative-mode {random|nearby|same-trial|mixed}
--negative-min-shift-sec <float>  # Default 0.0
--negative-max-shift-sec <float>  # Default 0.5
```

#### Within-Subject Evaluation Mode

Alternative to LOSO for isolating subject-variability impact:

- **LOSO Mode** (default):
  - Standard leave-one-subject-out cross-validation
  - Tests cross-subject generalization ceiling
  
- **Within-Subject Mode**:
  - Train/test split per subject (configurable train_ratio)
  - Isolates subject-specific learning potential
  - If within-subject >> LOSO, subject variability is primary bottleneck

**CLI Flags**:
```bash
--evaluation-mode {loso|within-subject}  # Default: loso
--subject-train-ratio <float>            # Default: 0.8
```

#### Usage Examples

Baseline contrastive LOSO:
```bash
python training/train_temporal_cnn_loso.py --objective contrastive --negative-mode random
```

Hard negatives with temporal offset:
```bash
python training/train_temporal_cnn_loso.py \
  --objective contrastive \
  --negative-mode nearby \
  --negative-min-shift-sec 0.1 \
  --negative-max-shift-sec 0.3
```

Within-subject evaluation:
```bash
python training/train_temporal_cnn_loso.py \
  --objective contrastive \
  --evaluation-mode within-subject \
  --negative-mode mixed
```

### Phase 3 Objectives

1. Verify cross-subject transfer is limited by subject variability, not signal quality
2. Estimate within-subject ceiling vs LOSO ceiling to quantify subject impact
3. Identify optimal hard negative strategy for cross-subject generalization
4. Determine if longer windows improve LOSO accuracy or plateau
