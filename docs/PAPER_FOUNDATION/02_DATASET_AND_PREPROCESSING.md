# 02 Dataset and Preprocessing

## DTU Dataset
### Structure
- **Subjects**: 18 Subjects (e.g., S1-S18)
- **Trials**: ~24-30 trials per subject
- **Trial Duration**: ~50 seconds per trial
- **Channels**: 64 channels available, but **8 specific channels** were selected for optimal runtime performance (`Fp1`, `Fp2`, `F7`, `F8`, `T7`, `T8`, `P7`, `P8`).
- **Sampling Rate**: 64 Hz (downsampled from original)
- **Labels**: Embedded directly into the target tensors (Label `1` or `2`).

### Preprocessing Pipeline
1. **EEG Bandpass**: Filtered to standard EEG bands (typically 1-8 Hz or 1-32 Hz depending on the specific baseline run, but production focuses on low-frequency tracking).
2. **Audio Extraction**: 28-band Gammatone filterbank applied to raw `.wav` files.
3. **Envelope Power Compression**: Absolute values of the gammatone outputs were raised to the power of `0.3` (`x ^ 0.3`) to match human auditory perception models.
4. **Alignment**: Audio and EEG streams were aligned and normalized globally (`normalize_array` using overall mean and std).
5. **Windowing**: Chopped into 3-second windows (192 samples at 64Hz) with a 1.5-second stride.

---

## KUL Dataset
### Structure
- **Subjects**: 16 Subjects (Focus on `S1` for transfer studies)
- **Trials**: 20 trials per subject
- **Trial Duration**: ~389 seconds per trial (~6.5 minutes)
- **Channels**: 64 channels (BioSemi system).
- **Sampling Rate**: 128 Hz (Raw).

### Metadata Discoveries
- **`attended_ear`**: 'L' or 'R'. Indicates which ear to attend to.
- **`stimuli`**: Array containing `[left_stream_wav, right_stream_wav]`.
- **Label Mapping Logic**: The attended audio is determined purely by `attended_ear`. If `L`, then `stimuli[0]`. If `R`, then `stimuli[1]`. Tracks swap ears between trials, so fixing to "Track 1" is incorrect.

### Preprocessing Pipeline
To match the DTU format, the KUL data required:
1. **Channel Selection**: Mapping BioSemi channel names to the 8 DTU target channels.
2. **EEG Resampling**: Bandpass filtering (1-6 Hz typical) and downsampling from 128 Hz to 64 Hz.
3. **Audio Reconstruction**: Re-implementing the 28-band ERB Gammatone filterbank in Python:
   - 28 ERB-spaced frequencies from 50Hz to 8000Hz.
   - Butterworth bandpass filtering.
   - Absolute value extraction and `^ 0.3` power compression.
   - Resampling from raw audio FS (e.g., 44.1kHz) to 64 Hz.
4. **Windowing**: 3-second windows with 1.5-second stride (192 samples).
