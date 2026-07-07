import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class AAD_EEGNet(nn.Module):
    """
    EEGNet adapted for Auditory Attention Decoding (regression).
    We remove all padding so the network acts as a strict backward decoder.
    pred[t] will use EEG[t : t + receptive_field], which correctly correlates with Audio[t].
    """
    def __init__(self, in_channels=60, F1=32, D=2, F2=64, temporal_kernel=64):
        super(AAD_EEGNet, self).__init__()
        
        self.in_channels = in_channels
        self.F1 = F1
        self.D = D
        self.F2 = F2
        self.temporal_kernel = temporal_kernel

        # Block 1: Temporal Conv -> Depthwise Spatial Conv
        # Output length: T - 63
        self.temporal_conv = nn.Conv2d(1, F1, kernel_size=(1, temporal_kernel), padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(F1)
        
        self.spatial_conv = nn.Conv2d(F1, F1 * D, kernel_size=(in_channels, 1), groups=F1, bias=False)
        self.bn2 = nn.BatchNorm2d(F1 * D)
        self.elu = nn.ELU()
        self.dropout1 = nn.Dropout(0.0)
        
        # Block 2: Separable Conv
        # Output length: (T - 63) - 15 = T - 78
        self.separable_depth = nn.Conv2d(F1 * D, F1 * D, kernel_size=(1, 16), groups=F1 * D, padding=0, bias=False)
        self.separable_point = nn.Conv2d(F1 * D, F2, kernel_size=(1, 1), bias=False)
        self.bn3 = nn.BatchNorm2d(F2)
        self.dropout2 = nn.Dropout(0.0)
        
        # Decoder
        # Output length: (T - 78) - 24 = T - 102
        self.decoder = nn.Conv1d(F2, 1, kernel_size=25, bias=True)
        
    def get_channel_importance(self):
        """
        Computes the absolute sum of the spatial filter weights for each input channel.
        Returns a numpy array of shape (in_channels,)
        """
        weights = self.spatial_conv.weight.detach().cpu().numpy()
        importance = np.sum(np.abs(weights), axis=(0, 1, 3))
        if np.sum(importance) > 0:
            importance = importance / np.sum(importance)
        return importance
        
    def extract_features(self, x):
        # x: [B, Channels, Time]
        x = x.unsqueeze(1)
        
        # Block 1
        x = self.temporal_conv(x)
        x = self.bn1(x)
        x = self.spatial_conv(x)
        x = self.bn2(x)
        x = self.elu(x)
        x = self.dropout1(x)
        
        # Block 2
        x = self.separable_depth(x)
        x = self.separable_point(x)
        x = self.bn3(x)
        x = self.elu(x)
        x = self.dropout2(x)
        
        return x.squeeze(2)

    def forward(self, x):
        z = self.extract_features(x)
        out = self.decoder(z)
        return out.squeeze(1)
