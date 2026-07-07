import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class EEGNetClassifier(nn.Module):
    """
    Standard EEGNet architecture optimized for direct classification.
    Inputs: [Batch, Channels, Time] (e.g., Time = 256 for a 2.0s window at 128Hz)
    Outputs: [Batch] (Raw logits for Binary Cross Entropy Loss)
    """
    def __init__(self, in_channels=60, samples=256, F1=8, D=2, F2=16, temporal_kernel=64):
        super(EEGNetClassifier, self).__init__()
        
        self.F1 = F1
        self.D = D
        self.F2 = F2
        
        # Block 1
        self.temporal_conv = nn.Conv2d(1, F1, kernel_size=(1, temporal_kernel), padding=(0, temporal_kernel//2), bias=False)
        self.bn1 = nn.BatchNorm2d(F1)
        
        self.spatial_conv = nn.Conv2d(F1, F1 * D, kernel_size=(in_channels, 1), groups=F1, bias=False)
        self.bn2 = nn.BatchNorm2d(F1 * D)
        self.elu = nn.ELU()
        
        self.avg_pool1 = nn.AvgPool2d(kernel_size=(1, 4))
        self.dropout1 = nn.Dropout(0.50)
        
        # Block 2
        self.separable_depth = nn.Conv2d(F1 * D, F1 * D, kernel_size=(1, 16), groups=F1 * D, padding=(0, 8), bias=False)
        self.separable_point = nn.Conv2d(F1 * D, F2, kernel_size=(1, 1), bias=False)
        self.bn3 = nn.BatchNorm2d(F2)
        
        self.avg_pool2 = nn.AvgPool2d(kernel_size=(1, 8))
        self.dropout2 = nn.Dropout(0.50)
        
        # Dynamically compute flatten size
        out_time = samples // 32
        self.flatten_size = F2 * out_time
        
        # Output layer for Binary Classification
        self.classifier = nn.Linear(self.flatten_size, 1)

    def forward(self, x):
        # x: [B, C, T] -> [B, 1, C, T]
        x = x.unsqueeze(1)
        
        # Block 1
        x = self.temporal_conv(x)
        x = self.bn1(x)
        x = self.spatial_conv(x)
        x = self.bn2(x)
        x = self.elu(x)
        x = self.avg_pool1(x)
        x = self.dropout1(x)
        
        # Block 2
        x = self.separable_depth(x)
        x = self.separable_point(x)
        x = self.bn3(x)
        x = self.elu(x)
        x = self.avg_pool2(x)
        x = self.dropout2(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Classify
        out = self.classifier(x)
        
        # Output shape [Batch]
        return out.squeeze(1)
