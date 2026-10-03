import torch
import torch.nn as nn
import torch.nn.functional as F

class CausalConv1d(nn.Module):
    """1D convolution with strict past-only receptive field (causal)."""
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, **kwargs):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size, 
            dilation=dilation, padding=0, **kwargs
        )
        
    def forward(self, x):
        # Pad left by (kernel_size - 1) * dilation, pad right by 0
        x_padded = F.pad(x, (self.pad, 0))
        return self.conv(x_padded)

class AnticausalConv1d(nn.Module):
    """1D convolution with strict future-only receptive field (anticausal)."""
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, **kwargs):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size, 
            dilation=dilation, padding=0, **kwargs
        )
        
    def forward(self, x):
        # Pad left by 0, pad right by (kernel_size - 1) * dilation
        x_padded = F.pad(x, (0, self.pad))
        return self.conv(x_padded)

class CausalAnticausalResidualBlock(nn.Module):
    """
    Dual-branch Causal and Anticausal Residual TCN Block.
    Captures forward neural latency (causal) and reverse auditory integration (anticausal).
    """
    def __init__(self, channels, kernel_size=3, dilation=1, dropout=0.2):
        super().__init__()
        # Causal branch
        self.causal_conv1 = CausalConv1d(channels, channels, kernel_size=kernel_size, dilation=dilation)
        self.bn_causal1 = nn.BatchNorm1d(channels)
        self.causal_conv2 = CausalConv1d(channels, channels, kernel_size=kernel_size, dilation=dilation)
        self.bn_causal2 = nn.BatchNorm1d(channels)
        
        # Anticausal branch
        self.anticausal_conv1 = AnticausalConv1d(channels, channels, kernel_size=kernel_size, dilation=dilation)
        self.bn_anticausal1 = nn.BatchNorm1d(channels)
        self.anticausal_conv2 = AnticausalConv1d(channels, channels, kernel_size=kernel_size, dilation=dilation)
        self.bn_anticausal2 = nn.BatchNorm1d(channels)
        
        self.fusion = nn.Conv1d(channels * 2, channels, kernel_size=1)
        self.bn_fusion = nn.BatchNorm1d(channels)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        res = x
        
        # Causal forward
        c = F.gelu(self.bn_causal1(self.causal_conv1(x)))
        c = self.bn_causal2(self.causal_conv2(c))
        
        # Anticausal forward
        a = F.gelu(self.bn_anticausal1(self.anticausal_conv1(x)))
        a = self.bn_anticausal2(self.anticausal_conv2(a))
        
        # Fusion
        fused = torch.cat([c, a], dim=1)
        out = F.gelu(self.bn_fusion(self.fusion(fused)))
        out = self.dropout(out)
        
        return F.gelu(out + res)

class CATCN_EEGEncoder(nn.Module):
    """
    EEG Encoder combining Spatial Depthwise Convolution with Causal-Anticausal TCN.
    """
    def __init__(self, in_channels=64, hidden_dim=64, dilations=[1, 2, 4, 8], dropout=0.2):
        super().__init__()
        # 1. Spatial Filter across physical electrodes
        self.spatial_conv = nn.Conv2d(1, hidden_dim, (in_channels, 1), bias=False)
        self.bn_spatial = nn.BatchNorm2d(hidden_dim)
        
        # 2. Causal-Anticausal TCN stack
        self.blocks = nn.ModuleList([
            CausalAnticausalResidualBlock(hidden_dim, kernel_size=3, dilation=d, dropout=dropout)
            for d in dilations
        ])
        
    def forward(self, x):
        # x: [B, C, T]
        B, C, T = x.shape
        x = x.unsqueeze(1) # [B, 1, C, T]
        feat = F.gelu(self.bn_spatial(self.spatial_conv(x))).squeeze(2) # [B, hidden_dim, T]
        
        for block in self.blocks:
            feat = block(feat)
            
        return feat

class CATCN_AudioEncoder(nn.Module):
    """
    Audio Envelope Encoder with Causal-Anticausal TCN.
    """
    def __init__(self, in_channels=28, hidden_dim=64, dilations=[1, 2, 4, 8], dropout=0.2):
        super().__init__()
        self.proj = nn.Conv1d(in_channels, hidden_dim, kernel_size=1)
        self.bn_proj = nn.BatchNorm1d(hidden_dim)
        
        self.blocks = nn.ModuleList([
            CausalAnticausalResidualBlock(hidden_dim, kernel_size=3, dilation=d, dropout=dropout)
            for d in dilations
        ])
        
    def forward(self, x):
        # x: [B, 28, T]
        feat = F.gelu(self.bn_proj(self.proj(x)))
        for block in self.blocks:
            feat = block(feat)
        return feat

class CATCNDirectDecoder(nn.Module):
    """
    Direct Auditory Attention Decoder (CA-TCN) with dual Causal-Anticausal branches
    and Direct Attended-Speaker Classification.
    """
    def __init__(self, eeg_channels=64, audio_channels=28, hidden_dim=64, dilations=[1, 2, 4, 8], dropout=0.2):
        super().__init__()
        self.eeg_encoder = CATCN_EEGEncoder(
            in_channels=eeg_channels, hidden_dim=hidden_dim, dilations=dilations, dropout=dropout
        )
        self.audio_encoder = CATCN_AudioEncoder(
            in_channels=audio_channels, hidden_dim=hidden_dim, dilations=dilations, dropout=dropout
        )
        
        # Direct Cross-Modal Classification Head
        # Evaluates match quality between EEG and audio streams
        interaction_dim = hidden_dim * 3 # [Z_E, Z_Audio, Z_E * Z_Audio]
        self.classifier = nn.Sequential(
            nn.Linear(interaction_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )
        
    def encode_streams(self, eeg, audio_a, audio_b):
        z_eeg = self.eeg_encoder(eeg)
        z_a = self.audio_encoder(audio_a)
        z_b = self.audio_encoder(audio_b)
        return z_eeg, z_a, z_b
        
    def score_stream(self, z_eeg, z_audio):
        # Temporal pooling of interaction features
        # z_eeg: [B, D, T], z_audio: [B, D, T]
        diff_feat = torch.cat([z_eeg, z_audio, z_eeg * z_audio], dim=1) # [B, 3D, T]
        pooled = diff_feat.mean(dim=-1) # [B, 3D]
        logit = self.classifier(pooled).squeeze(-1) # [B]
        return logit

    def forward(self, eeg, audio_a, audio_b):
        """
        Direct classification forward pass.
        Returns:
          delta_logit: logit_A - logit_B (positive indicates attended stream A)
          (logit_a, logit_b): individual stream logits
          (z_eeg, z_a, z_b): latent representations for optional similarity evaluation
        """
        z_eeg, z_a, z_b = self.encode_streams(eeg, audio_a, audio_b)
        logit_a = self.score_stream(z_eeg, z_a)
        logit_b = self.score_stream(z_eeg, z_b)
        delta_logit = logit_a - logit_b
        return delta_logit, (logit_a, logit_b), (z_eeg, z_a, z_b)

def print_summary():
    model = CATCNDirectDecoder(eeg_channels=64, audio_channels=28, hidden_dim=64)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"CA-TCN Model Parameter Count (64ch): {n_params:,}")
    dummy_eeg = torch.randn(2, 64, 320)
    dummy_a = torch.randn(2, 28, 320)
    dummy_b = torch.randn(2, 28, 320)
    delta, (la, lb), (ze, za, zb) = model(dummy_eeg, dummy_a, dummy_b)
    print(f"Forward output shapes: delta={delta.shape}, la={la.shape}, ze={ze.shape}")

if __name__ == "__main__":
    print_summary()
