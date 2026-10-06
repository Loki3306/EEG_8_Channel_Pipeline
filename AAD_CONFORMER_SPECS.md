# AAD-Conformer Master Specification

> **Model Name**: AAD-Conformer  
> **Dataset**: KUL (Katholieke Universiteit Leuven)  
> **Total Parameters**: ~2,083,000  

---

## 1. Architecture Hyperparameters

The AAD-Conformer (`AADConformer`) combines a spatial-temporal CNN stem (inspired by EEGNet) with Convolution-Augmented Transformer (Conformer) blocks.

| Hyperparameter | Value | Description |
|----------------|-------|-------------|
| `in_channels` | 8 | Number of EEG input channels |
| `temporal_filters` | 32 | Number of temporal convolution filters |
| `spatial_filters` | 64 | Number of spatial depthwise filters |
| `embed_dim` | 64 | Embedding dimension for the Conformer blocks |
| `num_heads` | 4 | Number of attention heads in MHA |
| `num_layers` | 2 | Number of Conformer blocks |
| `dropout` | 0.3 | Dropout probability across all layers |
| `stride` | 4 | Downsampling stride for tokenization |

---

## 2. Component Details

### 2.1 Stem (EEGNet-Style)
The stem processes the raw EEG data into spatial-temporal features:
- **Temporal Convolution**: `Conv2d(1, 32, kernel_size=(1, 33), padding=(0, 16))` — acts as a trainable temporal bandpass filter.
- **Spatial Convolution**: `Conv2d(32, 64, kernel_size=(8, 1), groups=32)` — depthwise convolution across the 8 EEG channels to learn spatial filters per temporal band.
- **Activation**: `SiLU()` + `BatchNorm2d`

### 2.2 Tokenization & Positional Encoding
- **Tokenization**: `Conv1d(64, 64, kernel_size=4, stride=4)` — strided convolution to reduce temporal resolution and project to `embed_dim`.
- **Positional Encoding**: Standard sinusoidal positional embeddings added to the tokens.

### 2.3 Conformer Blocks (x2)
Each block uses a "Macaron" structure:
1. **FFN 1**: Half-step Feed-Forward Network.
2. **MHA**: Multi-Head Attention (`num_heads=4`).
3. **Convolution Module**: Depthwise 1D Conv (`kernel_size=15`) sandwiched by pointwise convolutions.
4. **FFN 2**: Half-step Feed-Forward Network.

### 2.4 Regression Head
- **Upsampling**: `ConvTranspose1d(stride=4)` to return to the original temporal resolution.
- **Output Projection**: `Conv1d(64, 1)` to output a single continuous envelope representing the attended audio stream.

### 2.5 Auxiliary Evidential Confidence Head
- **Inputs**: `z_pool` (the mean-pooled 64-D latent representation from the Conformer blocks).
- **Structure**: 2-layer MLP (`Linear(64, 64) -> ReLU -> Dropout(0.3) -> Linear(64, 2)`).
- **Function**: Outputs evidential logits for 2 classes (Correct/Incorrect) to model predictive uncertainty.

---

## 3. Training & Preprocessing Specifications

### 3.1 Loss Function (Hybrid)
The network is optimized using a combined loss function that penalizes both mean squared error and negative Pearson correlation:
```python
def custom_loss(pred, target, mse_weight=0.5, corr_weight=0.5):
    mse = nn.functional.mse_loss(pred, target)
    corr = safe_corr_torch(pred, target)
    corr_loss = 1.0 - corr.mean()
    return mse_weight * mse + corr_weight * corr_loss
```

### 3.2 Preprocessing
- **EEG**: Downsampled to 64 Hz, bandpass filtered.
- **Audio**: The 28-band Gammatone envelopes are averaged across all 28 subbands (`audio_a.mean(dim=0)`) to produce a **single broadband envelope** (1 channel) for the model to reconstruct.
- **Windowing**: Trained on 10-second windows with a 2-second hop size.

### 3.3 Hardware Optimization Peak
While the model is configured for `in_channels=8`, feature permutation audits revealed the architecture can be truncated to **5 channels** in a deployed hardware setting while maintaining peak performance.

---

## 4. Performance Summary (LOSO)
- **Grand Mean Accuracy**: 77.12% ± 9.99%
- **Selective Prediction**: 94.94% accuracy at a 0.70 confidence threshold (retaining 12.32% of data).
- **OOD Detection**: Confidence successfully collapses from ~0.54 (Clean EEG) to ~0.13 (Random/Zero Noise).
