# 07 Confidence Framework

## Motivation
Standard AAD models produce a binary prediction for every time step, regardless of signal quality. In realistic environments, EEG is frequently contaminated by motion artifacts, and attention wavers. A real-time hearing aid needs to know *when to act* and *when to do nothing*. The Confidence Framework was designed to sit on top of the AAD model and predict the probability that a given window's AAD prediction is correct.

## Feature Engineering
Rather than feeding raw EEG into the confidence model (which proved susceptible to spatial leakage and over-fitting), the system extracts metadata from the MatchNet's latent representations.

### Temporal Dynamic Features
1. **`margin`**: The absolute difference in similarity scores.
   - `margin = abs(sim_A - sim_B)`
   - Causality: A larger margin implies the model has cleanly separated the attended and unattended streams in the latent space.
2. **`sim_chosen`**: The maximum similarity score.
   - `sim_chosen = max(sim_A, sim_B)`
3. **`sim_unchosen`**: The minimum similarity score.
   - `sim_unchosen = min(sim_A, sim_B)`
4. **`rolling_std_margin`**: The standard deviation of the margin over a trailing rolling window (e.g., 5 steps).
   - Causality: High variance in the margin suggests temporal instability, often caused by artifacts or attention shifts.
5. **`trial_consistency`**: The fraction of previous windows in the current trial that agree with the current window's prediction.
   - Causality: If a prediction flips rapidly back and forth, consistency drops, flagging uncertainty.

## XGBoost Runtime Implementation
- The features are passed to a lightweight `XGBClassifier`.
- **Target**: Binary classification (1 = MatchNet Correct, 0 = MatchNet Incorrect).
- **Output**: A calibrated probability between 0 and 1.
- **Deployment**: The runtime system uses this probability against a predefined threshold (e.g., 0.6) to decide whether to switch the audio tracking or maintain the current state.
