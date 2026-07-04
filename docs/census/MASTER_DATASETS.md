# MASTER DATASETS — Complete Dataset Reference

> Every detail about every dataset used in this project.

---

## 1. DTU Dataset (Primary Development)

### 1.1 Provenance
- **Full Name**: Technical University of Denmark Auditory Attention Decoding Dataset
- **Citation**: Fuglsang et al., 2017
- **Storage**: Kaggle input datasets, originally `.mat` files

### 1.2 Structure

| Property | Value |
|----------|-------|
| Subjects | 18 (S1–S18), normal hearing |
| Trials per subject | 60 |
| Trial duration | ~50 seconds |
| Total EEG channels | 66 (64 BioSemi + EXG1, EXG2) |
| EEG sampling rate (raw) | 512 Hz |
| EEG sampling rate (preprocessed) | 64 Hz |
| Audio | Two competing Danish audiobooks |
| Presentation | Dichotic (one speaker per ear) |
| Task | Attend to designated speaker |
| Total data volume | 18 × 60 × ~50s × 64 Hz ≈ 3,456,000 samples/channel |

### 1.3 File Format

```
S{n}_data_preproc.mat
├── data.eeg      : (1, 60) cell array → each cell: (T_trial, 66) float64
├── data.wavA     : (1, 60) cell array → each cell: (T_trial, 1) float64
├── data.wavB     : (1, 60) cell array → each cell: (T_trial, 1) float64
├── data.fsample  : {eeg: 64, wavA: 64, wavB: 64}
└── data.event    : trial labels (1 or 2 = speaker GENDER, NOT stream)
```

### 1.4 Label Convention (CRITICAL)

> **Label 1** = Male speaker attended
> **Label 2** = Female speaker attended
> **`wavA`** = ALWAYS the attended stream (in preprocessed data)
> **`wavB`** = ALWAYS the unattended stream

The MATLAB preprocessing script ([preproc_data.m](file:///C:/Users/lokes/OneDrive/Documents/GitHub/EEG_Training_New/preproc_data.m), Lines 113–114) loads the attended audio into `wavA` based on `expinfo.attend_mf` and the speaker mapping. Labels do NOT indicate A/B assignment.

### 1.5 MATLAB Preprocessing Pipeline (COCOHA v0.5.0)

| Step | Operation | Detail |
|------|-----------|--------|
| 1 | Line Noise Removal | 50 Hz moving average (window = fs/50) |
| 2 | Downsampling | 512 Hz → 64 Hz (anti-aliased polyphase) |
| 3 | High-Pass Filter | 0.1 Hz, 2nd-order Butterworth, one-pass |
| 4 | EOG Artifact Removal | Bipolar VEOG/HEOG regression (co_denoise) |
| 5 | Average Re-Referencing | Common Average Reference (CAR) |
| 6 | Trial Segmentation | Split at event markers, load wavA/wavB |

### 1.6 Python-Side Additional Preprocessing

| Step | Operation | Code |
|------|-----------|------|
| 1 | Channel Selection | 8 peripheral: indices [13, 46, 43, 23, 50, 0, 52, 14] |
| 2 | Bandpass Filter | 1–6 Hz, 2nd-order Butterworth, zero-phase (filtfilt) |
| 3 | Z-Score Normalization | Per-channel mean subtraction + std division (ε=1e-12) |

### 1.7 Audio Preprocessing (28-Band Gammatone)

| Step | Operation | Detail |
|------|-----------|--------|
| 1 | Gammatone Filterbank | 28 bands, ERB-spaced, 50–8000 Hz |
| 2 | Envelope Extraction | `abs()` rectification |
| 3 | Compression | `^0.3` power-law (mimics Stevens' loudness) |
| 4 | Downsampling | → 64 Hz |
| 5 | Normalization | Per-band Z-score |

**Output**: (28, T) per audio stream

### 1.8 Channel Map (8 Peripheral)

| Channel | 10-20 Name | Hardware Index | Wearable Rationale |
|---------|-----------|----------------|-------------------|
| 0 | Fp1 | 13 | Forehead band |
| 1 | Fp2 | 46 | Forehead band |
| 2 | F7 | 43 | Near left ear |
| 3 | F8 | 23 | Near right ear |
| 4 | T7 | 50 | In-ear/around-ear left |
| 5 | T8 | 0 | In-ear/around-ear right |
| 6 | P7 | 52 | Behind left ear |
| 7 | P8 | 14 | Behind right ear |

### 1.9 Full 66-Channel Order

```
Fp1, AF7, AF3, F1, F3, F5, F7, FT7, FC5, FC3, FC1, C1, C3, C5, T7, 
TP7, CP5, CP3, CP1, P1, P3, P5, P7, P9, PO7, PO3, O1, Iz, Oz, POz, 
Pz, CPz, Fpz, Fp2, AF8, AF4, AFz, Fz, F2, F4, F6, F8, FT8, FC6, 
FC4, FC2, FCz, Cz, C2, C4, C6, T8, TP8, CP6, CP4, CP2, P2, P4, P6, 
P8, P10, PO8, PO4, O2, EXG1, EXG2
```

### 1.10 Known Limitations
1. **100% stimulus overlap**: All test stories heard during training by other subjects
2. **Single language**: Danish only
3. **Normal hearing only**: No hearing-impaired subjects
4. **Lab conditions**: High-impedance wet electrodes, controlled acoustic environment

---

## 2. KUL Dataset (Cross-Dataset Validation)

### 2.1 Provenance
- **Full Name**: KU Leuven Auditory Attention Dataset
- **Source**: Kaggle (`/kaggle/input/datasets/lowk1ee/s1-klu`)

### 2.2 Structure

| Property | Value |
|----------|-------|
| Subjects | 16 (S1–S16), normal hearing |
| Trials per subject | 20 |
| Trial duration | ~389 seconds |
| EEG channels | 64 (BioSemi64) |
| EEG sampling rate (raw) | 128 Hz |
| Audio | Raw `.wav` stereo files, Dutch audiobooks |
| Presentation | Dichotic (HRTF or dry condition) |

### 2.3 File Format
- `S1_KLU.mat` (subject data files)
- Raw `.wav` audio: `part{X}_track{Y}_{condition}.wav`

### 2.4 Trial Metadata Fields
| Field | Description |
|-------|------------|
| TrialID | Unique identifier (1–20) |
| attended_ear | `L` or `R` |
| attended_track | Track number (e.g., 1 or 2) |
| condition | `hrtf` or `dry` |
| experiment | Session identifier |
| part | Segment of experiment |
| repetition | Repetition indicator |
| subject | Subject ID |
| stimuli | Array of audio filenames [left_ear, right_ear] |

### 2.5 Label Convention (CRITICAL)

> If `attended_ear == 'L'`: attended = `stimuli[0]`
> If `attended_ear == 'R'`: attended = `stimuli[1]`
> **Track number is NOT fixed to a specific ear. Tracks swap across trials.**

Example (Trial 0):
- LEFT (`stimuli[0]`): `part1_track2_hrtf.wav`
- RIGHT (`stimuli[1]`): `part1_track1_hrtf.wav`
- `attended_ear`: `R` → attended = `part1_track1_hrtf.wav`

### 2.6 Conversion Pipeline (KUL → DTU Format)

```
Raw EEG (128 Hz, 64 Ch) 
    → Select 8 channels (Fp1, Fp2, F7, F8, T7, T8, P7, P8)
    → Downsample to 64 Hz (scipy.signal.resample_poly)
    → Bandpass 1–6 Hz
    → Z-score normalize

Raw Audio (.wav files)
    → 28-band Gammatone filterbank (ERB-spaced, 50–8000 Hz)
    → Absolute envelope extraction
    → ^0.3 power compression
    → Downsample to 64 Hz
    → Z-score normalize

Alignment
    → Slice into windows (3s = 192 samples, or 5s = 320 samples)
```

### 2.7 Output Tensor Geometries (Validated)
- **EEG**: `(192, 8)` or `(B, 8, T)`
- **Attended Audio**: `(192,)` → needs conversion to `(28, 192)` for MatchNet
- **Unattended Audio**: `(192,)` → needs conversion to `(28, 192)` for MatchNet

---

## 3. DTU vs KUL Comparison

| Property | DTU | KUL |
|----------|-----|-----|
| Channels | 8 (selected from 66) | 8 (selected from 64) |
| EEG Fs (preprocessed) | 64 Hz | 64 Hz (from 128 Hz) |
| Trial Duration | ~50 seconds | ~389 seconds |
| Trials/Subject | 60 | 20 |
| Audio Format | Preprocessed envelopes in .mat | Raw `.wav` stereo files |
| Audio Bands | 28 Gammatone (in .mat) | Must reconstruct 28-band |
| Labels | `wavA` always attended | `attended_ear` + `stimuli` |
| Language | Danish | Dutch |
| Conditions | Single | HRTF + Dry |
| Electrode System | BioSemi64 + EXG | BioSemi64 |

### Compatibility Verdict
**Major Preprocessing Mismatch (resolved)**: KUL audio requires manual reconstruction of 28-band Gammatone envelopes to match DTU tensor format. Once reconstructed, zero-shot transfer succeeds (75.8% at 30s windows on KUL S1).

---

## 4. Synthetic Dataset Schema

For synthetic data generation purposes:

| Column | Type | Allowed Values |
|--------|------|---------------|
| subject_id | int | 1–18 |
| trial_id | int | 0–59 |
| label | int | 1 or 2 |
| fsample_eeg | int | 64 |
| eeg | float matrix | 3200 × 66 |
| wavA | float matrix | 3200 × 1 |
| wavB | float matrix | 3200 × 1 |
| channel_names | string list | 66 names (see Section 1.9) |

**Rules**:
- Keep trial length: 50 seconds per record
- Keep EEG and audio aligned sample-by-sample
- Use label values only as 1 or 2
- wavA/wavB are NOT ordered by label
