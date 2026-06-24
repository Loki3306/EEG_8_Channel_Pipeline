# 09 Failure Analysis

## Information Gap Audit
**Hypothesis**: The model fails because the raw EEG contains less "attention information" during certain windows (the Information Gap).
**Method**: Analyzed the relationship between confidence scores and raw EEG spectrum power (Alpha/Theta ratios).
**Findings**: Low-confidence windows frequently correlated with bursts of high-frequency noise or suppressed Alpha band activity, supporting the hypothesis that the model fails when the biological signal is corrupted, not merely due to mathematical instability.

## Decision Path Audit
**Hypothesis**: Errors occur when the latent spaces `z_eeg`, `z_a`, and `z_b` collapse into a single cluster.
**Method**: Analyzed the L2 norms and pairwise cosine distances of the embeddings during incorrect predictions.
**Findings**: During failures, `sim_A` and `sim_B` often approach zero, and `margin` drops near zero. The embeddings do not blow up to infinity; rather, `z_eeg` fails to correlate with either audio stream.

## Root Cause Audit
**Hypothesis**: The XGBoost model is exploiting a subtle form of data leakage or artifact detection (e.g., recognizing eye blinks) rather than true attention tracking.
**Method**: Hostile reviews, including `check_processed_norm_leak.py` and temporal shuffling tests.
**Findings**: Negative. The confidence model relies strictly on the mathematical interactions between the audio and EEG embeddings. It cannot identify the subject or trial index from the features provided.

## Subject-Specific Failures (Archetypes)
- **The Noisy Subject**: Subjects with high baseline EEG variance (muscle tension, artifacts). Characterized by consistently low margins and high `rolling_std_margin`.
- **The Flatline Subject**: Subjects with very clean EEG but poor attention tracking. Margins are stable but near zero.
- **Conclusion**: Different failure archetypes require different interventions (e.g., better artifact rejection vs. longer window lengths).
