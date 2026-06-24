# 11 KUL Transfer Study

## Objective
To determine if a ContrastiveMatchNet trained entirely on the DTU dataset could successfully decode attention (zero-shot transfer) on the KUL dataset.

## Tensor Conversion and Pipeline Matching
Transfer learning failed in early tests because KUL and DTU use different preprocessing paradigms. DTU provided pre-extracted 28-band Gammatone envelopes, while KUL provided raw stereo `.wav` files.

### 28-Band Replication
A precise Python replication of the DTU MATLAB preprocessing was built (`analysis/step_6_5_kul_audio_28_band_proof.py`):
1. Generated 28 ERB-spaced frequencies (50Hz - 8000Hz).
2. Extracted absolute Hilbert envelopes via Butterworth bandpass filters.
3. Applied power compression (`^0.3`).
4. Downsampled to 64Hz.

## Phase 4.5 Distribution Audits
Before running zero-shot inference, a formal statistical distribution audit (`analysis/step_6_9_kul_vs_dtu_distribution_audit.py`) was performed comparing DTU S1 and KUL S1.
- **EEG Distribution**: Demonstrated minor numerical scaling shifts. PSD analysis confirmed overlapping Alpha/Theta content.
- **Audio Distribution**: The replicated Gammatone envelopes matched the scaling magnitude of DTU's offline envelopes.
- **Margin Analysis**: KUL margins were slightly compressed compared to DTU, but maintained a distinct positive shift.

## Cross-Dataset Results
With the preprocessing mathematically aligned, the DTU-trained MatchNet was applied to all 20 trials of KUL Subject 1 (`analysis/step_6_8_kul_ablation_and_confidence.py`).

| Metric | DTU Baseline | KUL Zero-Shot (30s Window) |
|--------|--------------|----------------------------|
| Trial Accuracy | ~85% | 100.0% |
| Window Accuracy| ~71% | ~75% |
| Confidence AUROC| ~0.78| NaN (All correct at 30s)|

### Why Initial Transfer Failed
Domain shift in AAD is often incorrectly blamed on the neural network architecture. Our audits proved the shift was entirely mechanical: feeding `(1, 192)` single-band envelopes into an `AudioEncoder(in_channels=28)` collapses the geometry. Once the feature space was correctly mapped, the model demonstrated **Strong Transfer**.
