import torch
import torch.nn as nn
import torch.nn.functional as F

class AAD_EEGNet(nn.Module):
    """
    EEGNet adapted for Auditory Attention Decoding (regression).
    We use causal padding to ensure no future information leaks into the predictions,
    which is essential for simulating real-world online decoding.
    """
    def __init__(self, in_channels=8, F1=32, D=2, F2=64, temporal_kernel=64, max_lag=24):
        super(AAD_EEGNet, self).__init__()
        
        self.F1 = F1
        self.D = D
        self.F2 = F2
        self.temporal_kernel = temporal_kernel
        self.max_lag = max_lag

        # Block 1: Temporal Conv -> Depthwise Spatial Conv
        self.temporal_conv = nn.Conv2d(1, F1, kernel_size=(1, temporal_kernel), padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(F1)
        
        self.spatial_conv = nn.Conv2d(F1, F1 * D, kernel_size=(in_channels, 1), groups=F1, bias=False)
        self.bn2 = nn.BatchNorm2d(F1 * D)
        self.elu = nn.ELU()
        self.dropout1 = nn.Dropout(0.0)
        
        # Block 2: Separable Conv
        # Depthwise
        self.separable_depth = nn.Conv2d(F1 * D, F1 * D, kernel_size=(1, 16), groups=F1 * D, padding=0, bias=False)
        # Pointwise
        self.separable_point = nn.Conv2d(F1 * D, F2, kernel_size=(1, 1), bias=False)
        self.bn3 = nn.BatchNorm2d(F2)
        self.dropout2 = nn.Dropout(0.0)
        
        # Decoder: Map F2 features to 1D scalar (envelope prediction)
        # We use a 1D Conv as a causal ridge decoder equivalent.
        self.decoder = nn.Conv1d(F2, 1, kernel_size=max_lag + 1, bias=False)
        
    def _causal_pad(self, x, kernel_size):
        # x is [B, C, H, W (Time)]
        # Pad only the left side of the time dimension by kernel_size - 1
        return F.pad(x, (kernel_size - 1, 0, 0, 0))
        
    def extract_features(self, x):
        # x: [B, Channels, Time]
        # Reshape to [B, 1, Channels, Time] for Conv2d
        x = x.unsqueeze(1)
        
        # Block 1
        x_pad1 = self._causal_pad(x, self.temporal_kernel)
        x = self.temporal_conv(x_pad1)
        x = self.bn1(x)
        
        x = self.spatial_conv(x)
        x = self.bn2(x)
        x = self.elu(x)
        x = self.dropout1(x)
        
        # Block 2
        x_pad2 = self._causal_pad(x, 16)
        x = self.separable_depth(x_pad2)
        x = self.separable_point(x)
        x = self.bn3(x)
        x = self.elu(x)
        x = self.dropout2(x)
        
        # Return as [B, F2, Time]
        return x.squeeze(2)

    def forward(self, x):
        z = self.extract_features(x)
        
        # Causal decode using max_lag
        # Pad left by max_lag
        z_pad = F.pad(z, (self.max_lag, 0))
        out = self.decoder(z_pad)
        
        return out.squeeze(1)
