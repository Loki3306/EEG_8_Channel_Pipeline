# KUL Dataset Audit

## Section 1 — Dataset Overview
- **Dataset name**: KUL Auditory Attention Dataset
- **Source**: Kaggle (`/kaggle/input/datasets/lowk1ee/s1-klu`)
- **Subject analyzed**: S1
- **Files used**: `S1_KLU.mat`, corresponding `.wav` audio files (e.g. `part1_track1_hrtf.wav`)

## Section 2 — Raw EEG Structure
- **Trials**: 20 trials per subject
- **Channels**: 64 channels (BioSemi64 system)
- **Sampling Rate**: 128 Hz
- **Duration**: ~389 seconds per trial
- **Raw EEG shape**: `(49792, 64)`

## Section 3 — Trial Metadata
- `TrialID`: Unique identifier for the trial (1-20).
- `attended_ear`: The ear the subject was instructed to attend to (`L` or `R`).
- `attended_track`: Identifies the attended audio track number (e.g., `1` or `2`).
- `condition`: Experimental condition (e.g., `hrtf` or `dry`).
- `experiment`: Experiment session identifier.
- `part`: Segment or part of the experiment.
- `repetition`: Indicates if the trial was a repetition.
- `subject`: Subject ID (e.g., `S1`).
- `stimuli`: Array of strings containing the exact filenames of the audio streams presented to the left and right ears.

## Section 4 — Stimulus Mapping
Trial 0 mapping example:
- **LEFT (`stimuli[0]`)**: `part1_track2_hrtf.wav`
- **RIGHT (`stimuli[1]`)**: `part1_track1_hrtf.wav`
- **`attended_ear`**: `R`

Therefore:
- **Attended Audio**: `part1_track1_hrtf.wav` (Right stream)
- **Unattended Audio**: `part1_track2_hrtf.wav` (Left stream)

*Critical Finding*: Track number is NOT fixed to a specific ear. Tracks swap across trials. The `attended_ear` field definitively determines which stream is the ground truth attended audio.

## Section 5 — DTU vs KUL Comparison

| Property | DTU | KUL |
| -------- | --- | --- |
| Channels | 8 (Selected) | 64 (BioSemi) |
| EEG FS | 64 Hz | 128 Hz |
| Trial Duration | ~50 seconds | ~389 seconds |
| Audio | Preprocessed envelopes | Raw `.wav` stereo files |
| Labels | Embedded in targets | `attended_ear` + `stimuli` |
| Preprocessing | Ready for MatchNet | Requires downsampling & filtering |

## Section 6 — Conversion Pipeline
Raw EEG (128 Hz, 64 Ch) → Select 8 channels → Downsample to 64 Hz
Raw Audio → Extract Envelope (Absolute/Lowpass) → Downsample to 64 Hz
Alignment → Slice into 3s windows (192 samples) with 1.5s stride

Resulting tensor geometries (Validated):
- **EEG**: `(192, 8)`
- **Attended Audio**: `(192,)`
- **Unattended Audio**: `(192,)`

## Section 7 — Proven Facts
- ✓ KUL contains all required signals.
- ✓ KUL contains explicit attention labels.
- ✓ KUL can be successfully converted into DTU tensor format dynamically.
- ✓ No MatchNet architecture changes appear necessary regarding dimensionality.

## Section 8 — Open Questions
- **Cross-dataset generalization**: Will a DTU-trained model work seamlessly on KUL without fine-tuning, or is there domain shift?
- **Subject variability**: Do KUL subjects differ behaviorally or physiologically from DTU subjects?
- **Audio synchronization robustness**: Are the KUL EEG and Audio streams perfectly zero-aligned (delay = 0)?
- **Envelope matching**: Does the simple Hilbert/Absolute lowpass envelope perfectly match the exact DTU acoustic representation feature scale?
