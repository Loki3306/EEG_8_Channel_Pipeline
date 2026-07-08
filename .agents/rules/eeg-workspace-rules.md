---
trigger: always_on
---

# EEG Workspace Rules

## Mission

This workspace is an active EEG Auditory Attention Decoding (AAD) research repository.

Preserve research correctness, reproducibility, and methodology over implementation speed.

---

## Mandatory Workflow

For every non-trivial implementation:

1. Understand the requested task.
2. Search the repository for related implementations before creating new code.
3. Search previous experiments and project memory for similar work.
4. Reuse existing implementations whenever possible.
5. Produce a short implementation plan before modifying code.
6. Only then implement.

---

## Before Finishing

Every code change must pass the autonomous engineering pipeline.

Never consider work complete until:

- Static validation passes.
- EEG validation passes.
- Autonomous review passes.
- Independent verification passes.

---

## Repository Rules

Do not duplicate existing models.

Do not introduce new architectures if an existing one can be extended.

Preserve repository structure.

Prefer modifying existing training pipelines rather than creating parallel versions.

---

## Research Rules

Treat methodological correctness as the highest priority.

Always look for:

- data leakage
- subject leakage
- validation leakage
- incorrect evaluation methodology
- reproducibility issues

If uncertain, investigate before implementing.

---

## External Research

When repository information is insufficient:

- Search official documentation.
- Search relevant papers.
- Search GitHub implementations.
- Compare approaches before implementing.

Do not blindly copy external code.

---

## Completion Criteria

A task is complete only after:

- implementation
- validation
- review
- verification

If any stage fails, continue improving the implementation until it passes or clearly explain why it cannot.

---

## Ear-EEG Architectural & Optimization Rules

**1. Normalization:**
Never use `GroupNorm(1, C)` or `LayerNorm` across EEG channels. Normalizing across channels destroys the relative amplitude ratios that represent physical electrode impedances and spatial geometries, causing catastrophic performance collapse. Always use `BatchNorm1d` to explicitly learn per-channel physical scaling and shifts.

**2. The Anti-Correlation Trap (Universal Pre-Training):**
Do not use cross-subject Universal Pre-Training for spatial models. The physical geometry of Ear-EEG electrodes varies too significantly across subjects. Universal spatial filters actively trap the model in anti-correlated local minima (Zero-Shot AUROC < 0.50).

**3. Personalization Protocol:**
When calibrating an Ear-EEG model for a specific subject, you must initialize a random network and **train completely from scratch**. Do not attempt to fine-tune a pre-trained backbone, as it will consistently underperform a from-scratch initialization.

**4. The Dimensionality Collapse Trap (SSL Features):**
When using self-supervised speech representations (e.g., WavLM, HuBERT) on Ear-EEG, do not pass massive 768-dimensional manifolds directly to the network. While these features hold valuable latent information (e.g., boosting 'weak' subjects like S11 to >0.60 AUROC), the massive modality imbalance overwhelms the optimizer on noisier subjects (collapsing S05 to <0.50 AUROC). You must strictly perform extreme dimensionality reduction (e.g., projecting to 16-64 dims) or use intermediate biologically-grounded representations (e.g., Multi-band Cochlear Envelopes) to prevent catastrophic network collapse.

---

## Product Deployment Rules (Real-Time Architecture)

When building simulators or deployment pipelines for the active hearing aid, adhere to the **Decoupled Control Loop** architecture:

1. **Instantaneous Audio Pipeline:** Blind source separation (e.g. beamforming) and audio mixing must run continuously in real-time (<10ms latency). Never delay the acoustic output by the AAD window size.
2. **Lagged BCI Control Loop:** The `SequenceAADModel` runs in a slower background thread, buffering the last 3.5s of data.
3. **EMA Smoothing:** The AAD model's predictions must update the audio mixer's Gain Multipliers asynchronously, using an Exponential Moving Average (EMA) to prevent volume jitter.

**5. The Recurrent Sequence Trap (No LSTMs/Mamba):**
Do not use continuous sequence models (LSTM, GRU, RNN, Mamba) for spatial or temporal modeling of Ear-EEG data. These models suffer from severe dimensionality collapse and overfit catastrophically to the training set (e.g., dropping loss to ~0.00 while test AUROC drops to ~0.27). The optimal temporal architecture for Ear-EEG phase-alignment is the **Temporal Convolutional Network (TCN)** with rigid, dilated receptive fields.