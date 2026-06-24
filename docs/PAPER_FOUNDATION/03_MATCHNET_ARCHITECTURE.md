# 03 MatchNet Architecture

## ContrastiveMatchNet
The core deep learning model is `ContrastiveMatchNet`, which maps EEG and Audio inputs into a shared 64-dimensional latent space.

### EEG Encoder (EEGNet)
- **Input Shape**: `(Batch, Channels, Time)` e.g., `(B, 8, 192)`
- **Structure**: Based on the standard EEGNet topology.
  - **Temporal Convolution**: 1D convolution over time to extract frequency features.
  - **Depthwise Convolution**: Spatial convolution across the 8 channels to learn spatial filters.
  - **Separable Convolution**: To summarize feature maps.
  - **Projection Layer**: Flattens and projects the output to the `latent_dim` (64).
- **Output**: `z_eeg` of shape `(Batch, 64, Time_Encoded)` or `(Batch, 64)` depending on aggregation.

### Audio Encoder (1D-CNN)
- **Input Shape**: `(Batch, Audio_Channels, Time)` e.g., `(B, 28, 192)`
- **Structure**: 
  - Multiple layers of 1D Convolutions with Batch Normalization and ReLU.
  - Designed to compress the 28 gammatone bands into robust acoustic representations.
  - **Projection Layer**: Projects to the same `latent_dim` (64).
- **Output**: `z_audio` of shape `(Batch, 64)`.

### Contrastive Matching (InfoNCE)
- **Inputs**: `z_eeg`, `z_a` (attended audio), `z_b` (unattended audio).
- **Similarities**: Computed using Pearson Correlation (or Cosine Similarity) along the feature dimension.
  - `sim_a = pearson_corr(z_eeg, z_a)`
  - `sim_b = pearson_corr(z_eeg, z_b)`
- **Objective (Loss)**:
  - The network is optimized using an InfoNCE-style loss to maximize `sim_a` and minimize `sim_b`.
  - Margin is defined as `abs(sim_a - sim_b)`. A larger margin indicates the model is successfully separating the attended stream from the distractor.

### Architecture Diagram (Text)
```text
[EEG Input] (8, 192)      [Audio A] (28, 192)     [Audio B] (28, 192)
       |                          |                       |
   [EEGNet]                 [1D-CNN Audio]          [1D-CNN Audio]
       |                          |                       |
       v                          v                       v
 [z_eeg] (64,)               [z_a] (64,)             [z_b] (64,)
       |                          |                       |
       +--------------------------+-----------------------+
                     |                          |
               [sim_a = corr]             [sim_b = corr]
                     |                          |
                     +---->[  Margin ]<---------+
```
