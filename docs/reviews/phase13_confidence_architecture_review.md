# Phase 13: Next-Generation Confidence Estimation Architecture Review

## 1. Research Review: Diagnostics of the Current System

Based on the empirical evidence gathered in the Phase 12.1.5 Scientific Audit, we observe the following critical failures in the baseline Confidence Head:

### Why is our confidence compressed? (Dynamic Range: [0.35, 0.52])
- **Dead Neurons & Capacity Collapse**: The Stage 2 diagnostic revealed that **46.8%** of the neurons in `layer_1_ReLU` are dead. The MLP lacks the representational capacity to express high-variance predictions.
- **Mean-Reversion due to BCE**: In highly noisy domains like EEG, Binary Cross-Entropy (BCE) heavily penalizes extreme predictions (0.0 or 1.0) when the network is unsure. The optimal strategy for a weak network to minimize BCE is to collapse its outputs toward the mean empirical accuracy (~65-70%), preventing true instance-level confidence scaling.

### Why does the latent vector (`z_pool`) fail?
- **Shortcut Learning**: The `z_pool` is a 64-dimensional feature vector. The network concatenates it with highly determinative 1D scalars (`margin`, `corr_a`). The gradients take the path of least resistance: optimizing the weights for the 1D scalars drops the loss instantly, while the 64-D latent weights are ignored and decay.
- **Misaligned Objective**: The Conformer was optimized for continuous audio regression (MatchNet/InfoNCE). The latent space organizes around auditory features, not epistemic uncertainty. Without explicit regularizers, the MLP cannot extract uncertainty bounds from regression latents easily.

### Why does Pearson dominate?
- **Deterministic Proxy**: The prediction itself is exactly `margin > 0`. Therefore, the margin is a perfect proxy for the decision boundary. The network naturally learns to just apply a smooth Sigmoid-like scaling over the margin. This makes the confidence head an *engineering scaling function*, not a *scientific uncertainty estimator*.

### Modern Approaches in AAD, BCI, and Speech AI
- **Evidential Deep Learning (EDL)**: Instead of predicting a point-estimate probability, the network predicts the parameters of a Dirichlet distribution (Evidence), cleanly separating data noise (aleatoric) from model ignorance (epistemic).
- **Conformal Prediction**: Provides rigorous, distribution-free statistical guarantees on predictions (e.g., "We are 90% sure the speaker is A"). Extremely popular in medical AI.
- **Monte Carlo Dropout / Ensembles**: The gold standard for Bayesian deep learning, but computationally prohibitive for real-time edge devices (hearing aids).
- **Sequential Accumulation (Drift-Diffusion Models)**: Decision-making in continuous BCI relies on accumulating evidence over time until a threshold is reached, rather than relying on independent windows.

---

## 2. Candidate Architecture Matrix

| Candidate | Mechanism | Inference Cost | Expected AUROC | OOD Robustness | Hearing Aid Feasibility |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Current Baseline** | MLP on `z_pool` + `margin` | Very Low | Low (~0.72) | Poor (Shortcut) | Excellent |
| **1. Margin-Free MLP** | MLP on `z_pool` only | Very Low | Moderate | Moderate | Excellent |
| **2. MC Dropout** | Multiple stochastic forward passes | **Very High** (Nx) | High | High | **Unfeasible** (Battery/Latency) |
| **3. Deep Ensembles** | Train N independent models | **Very High** (Nx) | Very High | Very High | **Unfeasible** (Memory/Compute) |
| **4. Temperature Scaling** | Post-hoc logits scaling | None | Same as Base | Poor | Excellent (Only calibrates) |
| **5. Mahalanobis Dist.** | Distance to class centroids | Moderate | Moderate | High | Good (Requires memory) |
| **6. Evidential DL** | Predicts Dirichlet Evidence | Low | High | **Very High** | **Excellent** |
| **7. Conformal Predict.** | Calibration set bounded intervals | Low | Moderate | High | Good (Requires calibration set) |
| **8. Temporal HMM / Accumulation** | Smooths confidences over time | Very Low | **Very High** | Moderate | **Excellent** (Crucial for streaming) |

---

## 3. Production Architecture Recommendation

We require an architecture that operates on **real hearing aids** (strict power, memory, and latency constraints) and processes **continuous, streaming EEG** (where predictions must be stable across time).

### The Chosen Architecture: Hybrid Evidential-Temporal Confidence (HETC)

**1. Spatial/Latent Stage: Evidential Deep Learning (EDL) Head**
- **How it works**: We discard `corr_a`, `corr_b`, and `margin` from the inputs to strictly prevent shortcut learning. The confidence head is attached *exclusively* to `z_pool`. Instead of outputting a scalar via Sigmoid+BCE, it outputs **Evidence ($e \ge 0$)** via a Softplus activation. 
- **Loss**: Trained using the Evidential MSE Loss (or Type II Maximum Likelihood). 
- **Why it outperforms**: It allows the network to explicitly say "I don't know" (Epistemic Uncertainty) when it encounters noisy or OOD EEG data, rather than being forced to output a 50/50 probability. It restores the dynamic range completely.

**2. Temporal Stage: Sequential Evidence Accumulation**
- **How it works**: Hearing aids operate continuously. A 0.9 confidence in Window $t$ is useless if Window $t+1$ is 0.1. We introduce a lightweight Temporal Accumulator (e.g., an Exponential Moving Average of the Dirichlet parameters, or a Drift-Diffusion hidden state). 
- **Why it outperforms**: It enforces decision stability. The threshold for switching auditory attention requires the accumulated *Evidence* to surpass a dynamic threshold, preventing erratic channel hopping.

### Failure Modes & Risks
- **Risk**: The latent vector (`z_pool`) alone may not contain sufficient linearly separable information for the EDL head to converge.
- **Mitigation**: If latent-only fails, we inject a *detached* margin (no gradients flowing back) strictly as a weak prior, or use intermediate Transformer tokens.
- **Expected Calibration**: EDL naturally calibrates by pushing low-evidence predictions toward uniform uncertainty.

---

## 4. Scientific Rationale & Next Steps

This architecture achieves the highest ROI:
1. **Scientific Novelty**: Combining Evidential Deep Learning with Temporal Accumulation for AAD is state-of-the-art.
2. **Computational Feasibility**: It requires exactly one forward pass per window and minimal memory (O(1) state tracking).
3. **Robustness**: Naturally flags OOD data (e.g., motion artifacts, loose electrodes) by generating low Epistemic Evidence.

### Validation Plan (Phase 13 Implementation)
If approved, I will:
1. Implement the `EvidentialConfidenceHead` in `models/aad_conformer.py`.
2. Implement the `EvidentialLoss` function.
3. Add the `TemporalEvidenceAccumulator` to the sequential decision logic.
4. Execute a training run and validate against the 11-stage Phase 12.1.5 diagnostic pipeline to prove dynamic range recovery and AUROC improvements.
