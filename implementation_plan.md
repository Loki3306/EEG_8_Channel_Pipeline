# Goal: Definitively test if the Oracle Decoder Drift is observable from Unsupervised Statistics

## User Review Required
> [!IMPORTANT]
> The failure of Phase 163 (Unsupervised Covariance Alignment) confirms that the distribution shift in this EEG dataset is far more complex than simple covariance drift (impedance changes). 
>
> GPT provided a brilliant final step to close this research loop permanently: **If the optimal decoder drifts, is that drift even observable without ground-truth labels?** 
> 
> Instead of guessing another unsupervised tracking algorithm, we will directly test if the observable unsupervised statistics (e.g., spatial covariance) contain enough information to predict the Oracle Decoder weights ($W_t$). 

## Proposed Strategy: Phase 164 Oracle Regression
1. Run a 50-minute chronological session.
2. For each 2-minute block, compute the **Oracle Decoder** ($W_t$) using the ground-truth labels.
3. For the same block, compute the **Unsupervised Statistics** (e.g., the spatial covariance matrix $C_t$).
4. Train a multivariate regression model: $f(C_t) \rightarrow W_t$.
5. Evaluate the $R^2$ of the regression on held-out subjects or time blocks.

### The Decisive Outcome
- **If $R^2$ is high**: The decoder drift is a deterministic function of the observable background covariance. We can track it purely by observing the covariance shift.
- **If $R^2 \approx 0$**: The optimal neural decoder rotates entirely independently of the observable background statistics. This mathematically proves that **Unsupervised Online Tracking is impossible for this dataset**, and any real-world AAD hearing aid built on this data MUST use periodic recalibration or semi-supervised feedback.

## Proposed Changes
### [NEW] `analysis/phase164_oracle_drift_observability.py`
This script will:
- Iterate through subjects in chronological blocks (e.g., 2 minutes).
- Extract $W_t$ (Ridge weights) and $C_t$ (Covariance matrices).
- Flatten and normalize the features.
- Train a cross-validated Multi-Output Ridge Regressor to predict $W_t$ from $C_t$.
- Report the mean $R^2$ scores.

## Open Questions
- Since you mentioned you are tired, this is the **final definitive experiment**. Should I proceed with implementing Phase 164 to write the conclusive research summary for this repository?
