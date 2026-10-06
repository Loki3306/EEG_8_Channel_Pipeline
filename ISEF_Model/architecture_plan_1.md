# MSCA-AAD Architecture Plan 1

> Proposed multi-scale, attention-augmented, domain-adversarial EEG encoder architecture for Auditory Attention Decoding (AAD), with critical engineering critiques and adaptation strategies.

---

# 1. High-Level Architecture Overview

```text
                    ┌──────────────────────┐
                    │      EEG 8 × T       │
                    │       @ 64 Hz        │
                    └──────────┬───────────┘
                               │
                     Per-subject normalization
                               │
                ┌──────────────┴──────────────┐
                │                             │
                ▼                             ▼
       MULTI-SCALE TEMPORAL           FREQUENCY BRANCH
             BRANCH                      SincConv
                │                             │
       k=3,7,15,31                       8 filters
                │                             │
                └──────────────┬──────────────┘
                               │
                         concatenate
                               │
                               ▼
                     Spatial feature mixing
                               │
                               ▼
                       SE Channel Attention
                               │
                               ▼
                       Downsample ×4
                               │
                               ▼
                    Lightweight Conformer ×2
                               │
                               ▼
                       Global average pool
                               │
                               ▼
                        EEG embedding
                            64-D
                               │
                 ┌─────────────┴─────────────┐
                 │                           │
                 ▼                           ▼
          Similarity branch            Classification
                 │                           │
           EEG ↔ Audio A               EEG + A + B
           EEG ↔ Audio B                     │
                 │                           │
                 └─────────────┬─────────────┘
                               │
                         learned fusion
                               │
                               ▼
                         P(A), P(B)
```

There is also a separate **domain-adversarial branch** originating from the 64-D EEG embedding:

```text
                    EEG embedding
                         │
                         ▼
                 Gradient Reversal
                         │
                         ▼
                 Subject classifier
```

**Core Principle:**
> Make the EEG representation discriminative for identifying the attended speaker while rendering it invariant to individual subject identity via adversarial training, directly targeting cross-subject LOSO generalization.

---

# 2. Input Specifications

For a 1-second evaluation window:

```text
EEG = 8 electrodes × 64 samples (@ 64 Hz)
```

Electrodes:
```text
             time
        ─────────────────>
Fp1     x x x x x x x ...
Fp2     x x x x x x x ...
F7      x x x x x x x ...
F8      x x x x x x x ...
P7      x x x x x x x ...
P8      x x x x x x x ...
TP7     x x x x x x x ...
TP8     x x x x x x x ...
```

The network must extract robust discriminative features from only **64 measurements per electrode**.

---

# 3. First Stage: Subject Normalization

EEG amplitudes vary significantly across subjects due to skull thickness, skin impedance, and contact resistances:

```text
Subject 1: mean = 14 μV, std = 7 μV
Subject 2: mean = 31 μV, std = 15 μV
```

The model must not interpret absolute amplitude offsets as attention dynamics:

$$
x' = \frac{x - \mu}{\sigma}
$$

* **LOSO Guardrail:** Avoid calculating normalization statistics across the test subject's entire continuous recording in an unconstrained manner that causes temporal leakage. Practical implementation relies on per-window or running calibration window statistics.

---

# 4. Multi-Scale Temporal Branch

Replaces single-scale temporal convolutions with parallel multi-resolution filters:

```text
                    EEG (8 × T)
                         │
          ┌──────────────┼──────────────┬──────────────┐
          │              │              │              │
        Conv 3         Conv 7        Conv 15        Conv 31
      (16 filters)   (16 filters)   (16 filters)   (16 filters)
          │              │              │              │
          └──────────────┼──────────────┴──────────────┘
                         │
                       concat
                         │
                     64 channels × T
```

Kernel spans at 64 Hz:
* `k = 3`: ~47 ms (fast transients / onsets)
* `k = 7`: ~109 ms (phonemic / high gamma envelopes)
* `k = 15`: ~234 ms (syllabic envelope tracking)
* `k = 31`: ~484 ms (phrase / prosodic dynamics)

Produces $4 \times 16 = 64$ temporal feature channels.

---

# 5. Frequency Branch (Learnable SincConv)

In parallel to the temporal branch, raw EEG is filtered via parameterized bandpass filters (SincNet paradigm):

```text
EEG (8 × T)
 │
 ▼
SincConv (8 bandpass filters parameterized by learnable [f_low, f_high])
 │
 ▼
1×1 Conv across electrodes
 │
 ▼
Frequency representation (8 channels × T)
```

The network dynamically learns optimal passband cutoffs rather than locking into rigid classical bands ($\delta, \theta, \alpha, \beta$).

---

# 6. Temporal-Spectral Feature Concatenation

Combine temporal and frequency representations:

```text
Temporal branch:  64 channels × T
Frequency branch:  8 channels × T
─────────────────────────────────
Concatenated:     72 channels × T
```

Each time step now contains joint information regarding both temporal waveforms and spectral energy distribution.

---

# 7. Spatial Feature Mixing

Mix the combined 72 feature channels into 64 channels:

```text
72 × T ───► Conv1d(72, 64, kernel_size=1) ───► 64 × T
```

*(Note: While termed "spatial mixing", this operates across feature channels; true electrode topological modeling is addressed in Section 20).*

---

# 8. Squeeze-and-Excitation (SE) Channel Attention

Dynamically recalibrate channel weights based on global temporal context:

```text
64 × T ──► Global Average Pooling ──► 64-D vector ──► MLP (Reduction & Expansion) ──► Sigmoid Weights (w_1 ... w_64)
                                                                                            │
64 × T ◄───────────────────────── Scaled by w ◄─────────────────────────────────────────────┘
```

Allows the network to emphasize transient vs. semantic channels on a per-window basis.

---

# 9. Temporal Downsampling

```text
64 × T ──► Average Pooling / Conv Stride 4 ──► 64 × (T / 4)
```

For a 1-second window (64 samples):
* Reduces sequence length from 64 to 16 temporal tokens.
* Drastically reduces attention quadratic complexity and focuses the Transformer on macro-temporal transitions.

---

# 10. Lightweight Conformer Modules ($\times 2$)

Input shape: `16 tokens × 64 dimensions`.

Two cascaded Conformer blocks combining:
1. **Feed-Forward Module (FFN):** Nonlinear point-wise transformations.
2. **Multi-Head Self-Attention (MHSA):** Long-range temporal interactions across the 16 tokens (4 attention heads, $d_{model} = 64$).
3. **Convolution Module:** Depthwise separable local temporal convolutions capturing local phase alignment.
4. **Macaron-style Half-Step Residuals:** LayerNorm + FFN half-step wrappers.

Hyperparameters:
* Blocks: 2
* Attention heads: 4
* $d_{model}$: 64
* FFN expansion: 128

---

# 11. Global EEG Embedding

```text
16 tokens × 64 dimensions ──► Temporal Pooling (Global Average Pool) ──► z_E ∈ ℝ^64
```

$z_E$ serves as the concise latent state describing attentional tracking during the window.

---

# 12. Audio Encoders (A and B)

Both candidate acoustic streams (attended and unattended) pass through a weight-shared 1D convolutional pipeline:

```text
Audio Waveform ──► Gammatone Filterbank (28 bands) ──► Conv1D ──► Conv1D ──► 64-D Latents (z_A, z_B ∈ ℝ^64)
```

---

# 13. Decision Mechanism 1: Cosine Similarity Branch

Computes geometric alignment between EEG and audio latents (MatchNet paradigm):

$$
s_A = \frac{z_E \cdot z_A}{\|z_E\| \|z_A\|}, \quad s_B = \frac{z_E \cdot z_B}{\|z_E\| \|z_B\|}
$$

Decision logits: $[s_A, s_B]$.

---

# 14. Decision Mechanism 2: Direct Classification Head

Feeds the concatenated multimodal representations into an MLP to directly infer attention:

```text
[z_E (64-D), z_A (64-D), z_B (64-D)] = 192-D
                 │
                 ▼
          MLP Classifier
                 │
                 ▼
     Direct Logits: [c_A, c_B]
```

Captures higher-order nonlinear interactions that inner-product similarity might omit.

---

# 15. Learned Decision Gating & Fusion

A learnable scalar gate $g = \sigma(\alpha)$ dynamically blends the similarity and classification logits:

$$
\text{Logits}_{\text{fused}} = g \cdot \text{Logits}_{\text{sim}} + (1 - g) \cdot \text{Logits}_{\text{cls}}
$$

$$
P(A), P(B) = \text{Softmax}(\text{Logits}_{\text{fused}})
$$

---

# 16. Domain-Adversarial Branch (GRL for LOSO)

Pushes the EEG encoder to learn subject-invariant representations:

```text
64-D EEG Embedding (z_E)
          │
          ▼
Gradient Reversal Layer (GRL, scale λ_GRL)
          │
          ▼
Subject Classifier (Linear → ReLU → Linear → N_subjects)
          │
          ▼
Loss: CrossEntropy(y_subject_hat, y_subject_true)
```

During backpropagation, gradients from the subject classifier are reversed ($-\lambda \nabla$), penalizing the encoder if $z_E$ encodes subject-identifying characteristics.

---

# 17. Multi-Task Training Objective

$$
\mathcal{L}_{\text{total}} = 0.5 \mathcal{L}_{\text{CE}} + 0.2 \mathcal{L}_{\text{margin}} + 0.3 \mathcal{L}_{\text{consistency}} + 0.1 \mathcal{L}_{\text{domain}}
$$

1. **Classification Loss ($\mathcal{L}_{\text{CE}}$):** Cross-entropy on fused logits $[P(A), P(B)]$.
2. **Contrastive Margin Loss ($\mathcal{L}_{\text{margin}}$):** $\max(0, m - (s_{\text{attended}} - s_{\text{unattended}}))$.
3. **Augmentation Consistency Loss ($\mathcal{L}_{\text{consistency}}$):** Mean squared error / cosine similarity between original EEG embedding $z_E$ and augmented EEG embedding $z_E'$.
4. **Domain-Adversarial Loss ($\mathcal{L}_{\text{domain}}$):** Cross-entropy loss of the subject classifier under GRL.

---

# 18. Multi-Stage Window Curriculum

Instead of training solely on static window lengths:

```text
Stage 1 (15 epochs): Window = 5.0 s  (Establish robust macro-tracking)
      ↓
Stage 2 (15 epochs): Window = 2.0 s  (Compress temporal context)
      ↓
Stage 3 (15 epochs): Window = 1.0 s  (Production hearing-aid target)
      ↓
Stage 4 (20 epochs): Window = 0.5 s  (Ultra-low-latency optimization)
```

Evaluation benchmark grid:
* $0.10 \text{ s}, 0.25 \text{ s}, 0.50 \text{ s}, 1.0 \text{ s}, 2.0 \text{ s}, 5.0 \text{ s}, 10.0 \text{ s}$.

---

# 19. Integration with Selective Confidence Layer

The neural network acts purely as a probabilistic inference engine producing $[P(A), P(B)]$. The selective decision engine sits downstream:

```text
EEG + Audio A/B ──► MSCA-AAD Model ──► P(A), P(B) ──► Reliability / Confidence Evaluator
                                                                 │
                                          ┌──────────────────────┼──────────────────────┐
                                          ▼                      ▼                      ▼
                                      ACCEPT A               ACCEPT B                  HOLD
                                  (P(A) ≥ τ_high)        (P(B) ≥ τ_high)        (Uncertain / Margin < τ)
```

---

# 20. Critical Engineering Audits & Identified Flaws

Before concrete implementation, five primary architectural concerns must be resolved:

### Flaw 1: The 31-Sample Kernel on Ultra-Short Windows
* At 64 Hz, a $0.1 \text{ s}$ window comprises only $\approx 6.4$ samples.
* A kernel size of $k = 31$ ($484 \text{ ms}$) exceeds the entire window by nearly $5\times$. Severe zero-padding creates synthetic boundary artifacts.
* **Proposed Solution:** Window-adaptive dilation or dynamic kernel masking (e.g. disabling $k=15, 31$ branches when window $< 250 \text{ ms}$).

### Flaw 2: Pseudo-Spatial Mixing vs. True Electrode Topology
* A `Conv1d(72, 64, 1)` operates purely across channel combinations after spatial collapse. It ignores bilateral ear/scalp symmetries (`Fp1-Fp2`, `F7-F8`, `T7-T8`, `P7-P8`).
* **Proposed Solution:** Introduce a dedicated spatial graph convolutional layer (GCN) or cross-electrode bilateral difference/attention layer before temporal flattening.

### Flaw 3: SincConv Filterbank Initialization Inconsistency
* The initial design calls for 8 bands but initializes cutoffs with 6 boundaries: `[1, 4, 8, 13, 20, 30] Hz`, which only creates 5 passbands.
* **Proposed Solution:** Explicitly initialize 8 biologically meaningful auditory passbands:
  1. Low-Delta: 0.5–2.0 Hz
  2. High-Delta: 2.0–4.0 Hz
  3. Theta-1: 4.0–6.0 Hz
  4. Theta-2: 6.0–8.0 Hz
  5. Alpha-1: 8.0–10.5 Hz
  6. Alpha-2: 10.5–13.0 Hz
  7. Low-Beta: 13.0–20.0 Hz
  8. Mid-Beta: 20.0–30.0 Hz

### Flaw 4: Information Loss in Global Average Pooling
* Averaging across the 16 Conformer tokens discards *when* auditory tracking occurred in the window, ignoring onset vs. sustained tracking.
* **Proposed Solution:** Benchmark Global Average Pooling against Multi-Head Attention Pooling (learned query vector) and concatenated $[Mean, Max]$ pooling.

### Flaw 5: Risk of Representation Collapse via GRL
* Forcing strict domain invariance in Ear-EEG can inadvertently penalize biological signal strength (e.g. individual dipole orientation).
* **Proposed Solution:** Implement $\lambda_{\text{GRL}}$ annealing and include an explicit ablation condition ($\lambda_{\text{GRL}} = 0$) to measure whether domain invariance assists or impairs LOSO generalization.

---

# 21. Summary of Architectural Direction

```text
                                 EEG Input (8 × T @ 64Hz)
                                            │
                             ┌──────────────┴──────────────┐
                             │                             │
                   TEMPORAL MULTI-SCALE             SPECTRAL SINC-CONV
                     (k = 3, 7, 15, 31)             (8 Learnable Bands)
                             │                             │
                             └──────────────┬──────────────┘
                                            │
                                  True Spatial Topology
                                            │
                                  SE Channel Attention
                                            │
                                    Temporal Pool /4
                                            │
                                Lightweight Conformer ×2
                                            │
                                    Temporal Pooling
                                            │
                                   z_E (64-D Latent)
                                            │
                    ┌───────────────────────┼───────────────────────┐
                    │                       │                       │
                    ▼                       ▼                       ▼
            MatchNet Head           Direct Classifier       GRL Subject Head
          (Cosine Similarity)             (MLP)            (Domain Invariant)
                    │                       │                       │
                    └───────────┬───────────┘                Loss: L_domain
                                │
                        Learned Gate Fusion
                                │
                         P(A), P(B) Logits
                                │
                        Loss: L_CE, L_margin
```
