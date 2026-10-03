import torch
import torch.nn as nn
import torch.nn.functional as F

class DirectionalDepthwiseConv1d(nn.Module):
    """
    Depthwise 1D Convolution with strict directional padding.
    - 'causal': Receptive field extends strictly into past (t, t-1, t-2, ...).
    - 'anticausal': Receptive field extends strictly into future (t, t+1, t+2, ...).
    """
    def __init__(self, channels, kernel_size=3, dilation=1, direction='causal'):
        super().__init__()
        assert direction in ['causal', 'anticausal'], f"Invalid direction: {direction}"
        self.direction = direction
        self.pad_len = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            channels, channels, kernel_size, 
            dilation=dilation, padding=0, groups=channels, bias=False
        )
        
    def forward(self, x):
        if self.direction == 'causal':
            # Pad past (left side) by pad_len, future (right side) by 0
            x_padded = F.pad(x, (self.pad_len, 0))
        else:
            # Pad future (right side) by pad_len, past (left side) by 0
            x_padded = F.pad(x, (0, self.pad_len))
        return self.conv(x_padded)

class DepthwiseSeparableTCNBlock(nn.Module):
    """
    CA-TCN Depthwise-Separable Temporal Convolutional Block.
    Strictly unidirectional (either causal or anticausal).
    Structure:
      Directional Depthwise Conv1d -> BatchNorm1d -> GELU -> Pointwise Conv1d (1x1) -> BatchNorm1d -> Dropout
      + Residual Connection
    """
    def __init__(self, channels, kernel_size=3, dilation=1, direction='causal', dropout=0.2):
        super().__init__()
        self.depthwise = DirectionalDepthwiseConv1d(
            channels, kernel_size=kernel_size, dilation=dilation, direction=direction
        )
        self.bn1 = nn.BatchNorm1d(channels)
        self.pointwise = nn.Conv1d(channels, channels, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm1d(channels)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        res = x
        out = self.depthwise(x)
        out = F.elu(self.bn1(out))
        out = self.pointwise(out)
        out = self.bn2(out)
        out = self.dropout(out)
        return F.elu(out + res)

class CATCN_AudioEncoder(nn.Module):
    """
    CA-TCN Audio/Stimulus Encoder:
    Strictly CAUSAL temporal processing with ~984 ms receptive field.
    5 Depthwise-Separable TCN layers with dilations [1, 2, 4, 8, 16], K=3.
    Receptive Field = 1 + 2 * (1 + 2 + 4 + 8 + 16) = 63 samples (984.4 ms at 64 Hz).
    """
    def __init__(self, in_channels=1, hidden_dim=64, dilations=[1, 2, 4, 8, 16], dropout=0.2):
        super().__init__()
        self.proj = nn.Conv1d(in_channels, hidden_dim, kernel_size=1, bias=False)
        self.bn_proj = nn.BatchNorm1d(hidden_dim)
        
        self.blocks = nn.ModuleList([
            DepthwiseSeparableTCNBlock(
                channels=hidden_dim, kernel_size=3, dilation=d, direction='causal', dropout=dropout
            )
            for d in dilations
        ])
        
    def forward(self, x):
        feat = F.elu(self.bn_proj(self.proj(x)))
        for block in self.blocks:
            feat = block(feat)
        return feat

class CATCN_EEGEncoder(nn.Module):
    """
    CA-TCN EEG Encoder:
    Strictly ANTICAUSAL temporal processing with ~234 ms receptive field.
    Spatial Depthwise Channel-Mixing + 3 Depthwise-Separable TCN layers with dilations [1, 2, 4], K=3.
    Receptive Field = 1 + 2 * (1 + 2 + 4) = 15 samples (234.4 ms at 64 Hz).
    """
    def __init__(self, in_channels=64, hidden_dim=64, dilations=[1, 2, 4], dropout=0.2):
        super().__init__()
        # 1. Spatial channel-mixing projection (1x1 across electrodes)
        self.spatial_proj = nn.Conv1d(in_channels, hidden_dim, kernel_size=1, bias=False)
        self.bn_spatial = nn.BatchNorm1d(hidden_dim)
        
        # 2. Anticausal TCN blocks
        self.blocks = nn.ModuleList([
            DepthwiseSeparableTCNBlock(
                channels=hidden_dim, kernel_size=3, dilation=d, direction='anticausal', dropout=dropout
            )
            for d in dilations
        ])
        
    def forward(self, x):
        # x: [B, C_eeg, T]
        feat = F.elu(self.bn_spatial(self.spatial_proj(x)))
        for block in self.blocks:
            feat = block(feat)
        return feat

class CrossCorrelationClassificationHead(nn.Module):
    """
    Cross-Correlation + Linear Classifier Head.
    Computes normalized cross-correlation between EEG sequence and candidate audio sequence
    across a temporal lag window [-max_lag_samples, +max_lag_samples], then passes
    the resulting correlation vector to a linear classifier.
    """
    def __init__(self, hidden_dim=64, max_lag_samples=8):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_lag = max_lag_samples
        self.num_lags = 2 * max_lag_samples + 1
        
        # Linear classifier from concatenated cross-correlation coefficients
        self.classifier = nn.Linear(hidden_dim * self.num_lags, 1, bias=False)
        
    def compute_cross_correlation(self, z_eeg, z_audio):
        """
        Computes normalized cross-correlation across lags tau in [-max_lag, +max_lag].
        z_eeg: [B, D, T]
        z_audio: [B, D, T]
        Returns: [B, D * num_lags]
        """
        B, D, T = z_eeg.shape
        
        # Standardize along temporal dimension (zero mean, unit variance per channel)
        ze_mean = z_eeg.mean(dim=-1, keepdim=True)
        ze_std = z_eeg.std(dim=-1, keepdim=True) + 1e-8
        ze_norm = (z_eeg - ze_mean) / ze_std
        
        za_mean = z_audio.mean(dim=-1, keepdim=True)
        za_std = z_audio.std(dim=-1, keepdim=True) + 1e-8
        za_norm = (z_audio - za_mean) / za_std
        
        corrs = []
        for tau in range(-self.max_lag, self.max_lag + 1):
            if tau > 0:
                # EEG at t + tau matches Audio at t
                ze_slice = ze_norm[:, :, tau:]
                za_slice = za_norm[:, :, :-tau]
            elif tau < 0:
                abs_tau = abs(tau)
                ze_slice = ze_norm[:, :, :-abs_tau]
                za_slice = za_norm[:, :, abs_tau:]
            else:
                ze_slice = ze_norm
                za_slice = za_norm
                
            r_tau = (ze_slice * za_slice).mean(dim=-1) # [B, D]
            corrs.append(r_tau)
            
        # Stack: [B, D, num_lags] -> flatten to [B, D * num_lags]
        r_all = torch.stack(corrs, dim=-1).view(B, -1)
        return r_all
        
    def forward(self, z_eeg, z_a, z_b):
        """
        Returns:
          delta: logit_A - logit_B
          (logit_a, logit_b)
        """
        r_a = self.compute_cross_correlation(z_eeg, z_a)
        r_b = self.compute_cross_correlation(z_eeg, z_b)
        
        logit_a = self.classifier(r_a).squeeze(-1) # [B]
        logit_b = self.classifier(r_b).squeeze(-1) # [B]
        
        delta = logit_a - logit_b
        return delta, (logit_a, logit_b)

class CATCNDirectDecoder(nn.Module):
    """
    CA-TCN: Causal-Anticausal Temporal Convolutional Network for Direct Auditory Attention Decoding.
    Reference: arXiv:2603.26394
    
    Structure:
      - Audio Stream: Causal Depthwise-Separable TCN (5 layers, d=[1,2,4,8,16], RF ≈ 984 ms past)
      - EEG Stream: Anticausal Depthwise-Separable TCN (3 layers, d=[1,2,4], RF ≈ 234 ms future)
      - Head: Multi-lag Cross-Correlation + Linear Classifier (Strict A/B Anti-Symmetry)
    """
    def __init__(self, eeg_channels=64, audio_channels=1, hidden_dim=64, max_lag_samples=8, dropout=0.2):
        super().__init__()
        self.audio_encoder = CATCN_AudioEncoder(
            in_channels=audio_channels, hidden_dim=hidden_dim, dilations=[1, 2, 4, 8, 16], dropout=dropout
        )
        self.eeg_encoder = CATCN_EEGEncoder(
            in_channels=eeg_channels, hidden_dim=hidden_dim, dilations=[1, 2, 4], dropout=dropout
        )
        self.classifier_head = CrossCorrelationClassificationHead(
            hidden_dim=hidden_dim, max_lag_samples=max_lag_samples
        )
        
    def forward(self, eeg, audio_a, audio_b):
        z_eeg = self.eeg_encoder(eeg)
        z_a = self.audio_encoder(audio_a)
        z_b = self.audio_encoder(audio_b)
        
        delta, (logit_a, logit_b) = self.classifier_head(z_eeg, z_a, z_b)
        return delta, (logit_a, logit_b), (z_eeg, z_a, z_b)

def print_summary():
    model = CATCNDirectDecoder(eeg_channels=64, audio_channels=1, hidden_dim=64, max_lag_samples=8)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Faithful CA-TCN (64ch) Parameter Count: {params:,}")
    dummy_eeg = torch.randn(2, 64, 320)
    dummy_a = torch.randn(2, 1, 320)
    dummy_b = torch.randn(2, 1, 320)
    delta, (la, lb), (ze, za, zb) = model(dummy_eeg, dummy_a, dummy_b)
    print(f"Output shapes: delta={delta.shape}, la={la.shape}, ze={ze.shape}, za={za.shape}")

if __name__ == "__main__":
    print_summary()
