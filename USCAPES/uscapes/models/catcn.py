"""
CA-TCN: Causal-Anticausal Temporal Convolutional Network for Direct Auditory Attention Decoding.
Reference: arXiv:2603.26394

Structure:
  - Audio Stream: Causal Depthwise-Separable TCN (5 layers, d=[1,2,4,8,16], RF ≈ 984 ms past)
  - EEG Stream: Anticausal Depthwise-Separable TCN (3 layers, d=[1,2,4], RF ≈ 234 ms future)
  - Head: Multi-lag Cross-Correlation + Linear Classifier (Strict A/B Anti-Symmetry)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DirectionalDepthwiseConv1d(nn.Module):
    """
    Depthwise 1D Convolution with strict directional padding.
    - 'causal': Receptive field extends strictly into past (t, t-1, t-2, ...).
    - 'anticausal': Receptive field extends strictly into future (t, t+1, t+2, ...).
    """
    def __init__(self, channels: int, kernel_size: int = 3, dilation: int = 1, direction: str = 'causal'):
        super().__init__()
        assert direction in ['causal', 'anticausal'], f"Invalid direction: {direction}"
        self.direction = direction
        self.pad_len = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            channels, channels, kernel_size, 
            dilation=dilation, padding=0, groups=channels, bias=False
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.direction == 'causal':
            x_padded = F.pad(x, (self.pad_len, 0))
        else:
            x_padded = F.pad(x, (0, self.pad_len))
        return self.conv(x_padded)


class DepthwiseSeparableTCNBlock(nn.Module):
    """
    CA-TCN Depthwise-Separable Temporal Convolutional Block.
    Strictly unidirectional (either causal or anticausal).
    """
    def __init__(self, channels: int, kernel_size: int = 3, dilation: int = 1, direction: str = 'causal', dropout: float = 0.2):
        super().__init__()
        self.depthwise = DirectionalDepthwiseConv1d(
            channels, kernel_size=kernel_size, dilation=dilation, direction=direction
        )
        self.bn1 = nn.BatchNorm1d(channels)
        self.pointwise = nn.Conv1d(channels, channels, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm1d(channels)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
    """
    def __init__(self, in_channels: int = 1, hidden_dim: int = 64, dilations: list = None, dropout: float = 0.2):
        super().__init__()
        if dilations is None:
            dilations = [1, 2, 4, 8, 16]
        self.proj = nn.Conv1d(in_channels, hidden_dim, kernel_size=1, bias=False)
        self.bn_proj = nn.BatchNorm1d(hidden_dim)
        
        self.blocks = nn.ModuleList([
            DepthwiseSeparableTCNBlock(
                channels=hidden_dim, kernel_size=3, dilation=d, direction='causal', dropout=dropout
            )
            for d in dilations
        ])
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = F.elu(self.bn_proj(self.proj(x)))
        for block in self.blocks:
            feat = block(feat)
        return feat


class CATCN_EEGEncoder(nn.Module):
    """
    CA-TCN EEG Encoder:
    Strictly ANTICAUSAL temporal processing with ~234 ms receptive field.
    Spatial Depthwise Channel-Mixing + 3 Depthwise-Separable TCN layers with dilations [1, 2, 4], K=3.
    """
    def __init__(self, in_channels: int = 8, hidden_dim: int = 64, dilations: list = None, dropout: float = 0.2):
        super().__init__()
        if dilations is None:
            dilations = [1, 2, 4]
        # Spatial channel-mixing projection (1x1 across electrodes)
        self.spatial_proj = nn.Conv1d(in_channels, hidden_dim, kernel_size=1, bias=False)
        self.bn_spatial = nn.BatchNorm1d(hidden_dim)
        
        # Anticausal TCN blocks
        self.blocks = nn.ModuleList([
            DepthwiseSeparableTCNBlock(
                channels=hidden_dim, kernel_size=3, dilation=d, direction='anticausal', dropout=dropout
            )
            for d in dilations
        ])
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = F.elu(self.bn_spatial(self.spatial_proj(x)))
        for block in self.blocks:
            feat = block(feat)
        return feat


class CrossCorrelationClassificationHead(nn.Module):
    """
    Cross-Correlation + Linear Classifier Head.
    Computes normalized cross-correlation between EEG sequence and candidate audio sequence
    across a temporal lag window [-max_lag_samples, +max_lag_samples].
    """
    def __init__(self, hidden_dim: int = 64, max_lag_samples: int = 8):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_lag = max_lag_samples
        self.num_lags = 2 * max_lag_samples + 1
        self.classifier = nn.Linear(hidden_dim * self.num_lags, 1, bias=False)
        
    def compute_cross_correlation(self, z_eeg: torch.Tensor, z_audio: torch.Tensor) -> torch.Tensor:
        B, D, T = z_eeg.shape
        ze_mean = z_eeg.mean(dim=-1, keepdim=True)
        ze_std = z_eeg.std(dim=-1, keepdim=True) + 1e-8
        ze_norm = (z_eeg - ze_mean) / ze_std
        
        za_mean = z_audio.mean(dim=-1, keepdim=True)
        za_std = z_audio.std(dim=-1, keepdim=True) + 1e-8
        za_norm = (z_audio - za_mean) / za_std
        
        corrs = []
        for tau in range(-self.max_lag, self.max_lag + 1):
            if tau > 0:
                ze_slice = ze_norm[:, :, tau:]
                za_slice = za_norm[:, :, :-tau]
            elif tau < 0:
                abs_tau = abs(tau)
                ze_slice = ze_norm[:, :, :-abs_tau]
                za_slice = za_norm[:, :, abs_tau:]
            else:
                ze_slice = ze_norm
                za_slice = za_norm
                
            r_tau = (ze_slice * za_slice).mean(dim=-1)
            corrs.append(r_tau)
            
        return torch.stack(corrs, dim=-1).view(B, -1)
        
    def forward(self, z_eeg: torch.Tensor, z_a: torch.Tensor, z_b: torch.Tensor):
        r_a = self.compute_cross_correlation(z_eeg, z_a)
        r_b = self.compute_cross_correlation(z_eeg, z_b)
        
        logit_a = self.classifier(r_a).squeeze(-1)
        logit_b = self.classifier(r_b).squeeze(-1)
        delta = logit_a - logit_b
        return delta, (logit_a, logit_b)


class CATCNDirectDecoder(nn.Module):
    """
    CA-TCN Neural Decoder for Direct 2-Speaker Auditory Attention Decoding.
    """
    def __init__(self, eeg_channels: int = 8, audio_channels: int = 1, hidden_dim: int = 64, max_lag_samples: int = 8, dropout: float = 0.2):
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
        
    def forward(self, eeg: torch.Tensor, audio_a: torch.Tensor, audio_b: torch.Tensor):
        z_eeg = self.eeg_encoder(eeg)
        z_a = self.audio_encoder(audio_a)
        z_b = self.audio_encoder(audio_b)
        delta, (logit_a, logit_b) = self.classifier_head(z_eeg, z_a, z_b)
        return delta, (logit_a, logit_b), (z_eeg, z_a, z_b)


class CATCN_MultiBandAudioEncoder(nn.Module):
    """
    Causal Multi-Band Cochlear Stimulus Encoder for 8 Gammatone subbands.
    """
    def __init__(self, in_channels: int = 8, hidden_dim: int = 64, dilations: list[int] = [1, 2, 4, 8, 16], dropout: float = 0.2):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.spectral_proj = nn.Conv1d(in_channels, hidden_dim, kernel_size=1, bias=False)
        self.bn_proj = nn.BatchNorm1d(hidden_dim)
        self.blocks = nn.ModuleList([
            DepthwiseSeparableTCNBlock(
                channels=hidden_dim, kernel_size=3, dilation=d, direction='causal', dropout=dropout
            )
            for d in dilations
        ])
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = F.elu(self.bn_proj(self.spectral_proj(x)))
        for block in self.blocks:
            feat = block(feat)
        return feat


class CausalERPCrossAttentionHead(nn.Module):
    """
    Causal ERP Cross-Attention Head with Cortical Latency Masking + Residual Skip.
    """
    def __init__(
        self,
        hidden_dim: int = 64,
        num_heads: int = 4,
        max_erp_samples: int = 22,
        max_lag_samples: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        self.max_erp_samples = max_erp_samples
        self.max_lag_samples = max_lag_samples
        self.num_lags = 2 * max_lag_samples + 1
        
        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.attn_dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_dim)
        
        self.attn_classifier = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ELU(),
            nn.Linear(32, 1, bias=False)
        )
        self.xcorr_classifier = nn.Linear(hidden_dim * self.num_lags, 1, bias=False)
        self.register_buffer("cached_erp_mask", None, persistent=False)
        self.cached_seq_len = 0

    def get_erp_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        if self.cached_erp_mask is not None and self.cached_seq_len == seq_len and self.cached_erp_mask.device == device:
            return self.cached_erp_mask
        t_e = torch.arange(seq_len, device=device).unsqueeze(1)
        t_a = torch.arange(seq_len, device=device).unsqueeze(0)
        lag = t_e - t_a
        valid = (lag >= 0) & (lag <= self.max_erp_samples)
        mask = torch.full((seq_len, seq_len), float("-inf"), device=device)
        mask[valid] = 0.0
        self.cached_erp_mask = mask
        self.cached_seq_len = seq_len
        return mask

    def compute_cross_correlation(self, z_eeg: torch.Tensor, z_audio: torch.Tensor) -> torch.Tensor:
        B, D, T = z_eeg.shape
        ze_mean = z_eeg.mean(dim=-1, keepdim=True)
        ze_std = z_eeg.std(dim=-1, keepdim=True) + 1e-8
        ze_norm = (z_eeg - ze_mean) / ze_std
        za_mean = z_audio.mean(dim=-1, keepdim=True)
        za_std = z_audio.std(dim=-1, keepdim=True) + 1e-8
        za_norm = (z_audio - za_mean) / za_std
        corrs = []
        for tau in range(-self.max_lag_samples, self.max_lag_samples + 1):
            if tau > 0:
                ze_slice = ze_norm[:, :, tau:]
                za_slice = za_norm[:, :, :-tau]
            elif tau < 0:
                abs_tau = abs(tau)
                ze_slice = ze_norm[:, :, :-abs_tau]
                za_slice = za_norm[:, :, abs_tau:]
            else:
                ze_slice = ze_norm
                za_slice = za_norm
            r_tau = (ze_slice * za_slice).mean(dim=-1)
            corrs.append(r_tau)
        return torch.stack(corrs, dim=-1).view(B, -1)

    def forward_single_stream(self, z_eeg: torch.Tensor, z_audio: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        import math
        B, D, T = z_eeg.shape
        x_e = z_eeg.transpose(1, 2)
        x_a = z_audio.transpose(1, 2)
        q = self.q_proj(x_e).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x_a).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x_a).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        erp_mask = self.get_erp_mask(T, z_eeg.device)
        scores = scores + erp_mask.unsqueeze(0).unsqueeze(0)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)
        attn_weights = self.attn_dropout(attn_weights)
        context = torch.matmul(attn_weights, v).transpose(1, 2).contiguous().view(B, T, D)
        context = self.norm(self.out_proj(context) + x_e)
        s_attn = self.attn_classifier(context.mean(dim=1)).squeeze(-1)
        s_xcorr = self.xcorr_classifier(self.compute_cross_correlation(z_eeg, z_audio)).squeeze(-1)
        return s_attn + s_xcorr, attn_weights

    def forward(self, z_eeg: torch.Tensor, z_a: torch.Tensor, z_b: torch.Tensor):
        logit_a, attn_a = self.forward_single_stream(z_eeg, z_a)
        logit_b, attn_b = self.forward_single_stream(z_eeg, z_b)
        delta = logit_a - logit_b
        return delta, (logit_a, logit_b), (attn_a, attn_b)


class MultiBandCATCNDecoder(nn.Module):
    """
    Multi-Band Cochlear Gammatone (8 subbands) + Causal ERP Cross-Attention CA-TCN Decoder.
    """
    def __init__(
        self,
        eeg_channels: int = 8,
        audio_bands: int = 8,
        hidden_dim: int = 64,
        num_heads: int = 4,
        max_erp_samples: int = 22,
        max_lag_samples: int = 8,
        dropout: float = 0.2
    ):
        super().__init__()
        self.eeg_channels = eeg_channels
        self.audio_bands = audio_bands
        self.hidden_dim = hidden_dim
        self.audio_encoder = CATCN_MultiBandAudioEncoder(
            in_channels=audio_bands, hidden_dim=hidden_dim, dilations=[1, 2, 4, 8, 16], dropout=dropout
        )
        self.eeg_encoder = CATCN_EEGEncoder(
            in_channels=eeg_channels, hidden_dim=hidden_dim, dilations=[1, 2, 4], dropout=dropout
        )
        self.classifier_head = CausalERPCrossAttentionHead(
            hidden_dim=hidden_dim, num_heads=num_heads, max_erp_samples=max_erp_samples,
            max_lag_samples=max_lag_samples, dropout=dropout
        )
        
    def forward(self, eeg: torch.Tensor, audio_a: torch.Tensor, audio_b: torch.Tensor):
        z_eeg = self.eeg_encoder(eeg)
        z_a = self.audio_encoder(audio_a)
        z_b = self.audio_encoder(audio_b)
        delta, (logit_a, logit_b), _ = self.classifier_head(z_eeg, z_a, z_b)
        return delta, (logit_a, logit_b), (z_eeg, z_a, z_b)

