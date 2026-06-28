import torch
import torch.nn as nn

class ResidualBlock1D(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(channels)
        
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.relu(out)
        return out

class NeuralRidgeDecoder(nn.Module):
    def __init__(self, in_channels=8, hidden_channels=32, out_channels=28):
        super().__init__()
        
        # Initial projection
        self.initial = nn.Sequential(
            nn.Conv1d(in_channels, hidden_channels, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU()
        )
        
        # Residual blocks
        self.res1 = ResidualBlock1D(hidden_channels)
        self.res2 = ResidualBlock1D(hidden_channels)
        self.res3 = ResidualBlock1D(hidden_channels)
        
        # Output projection to match envelope dimensions
        self.out = nn.Conv1d(hidden_channels, out_channels, kernel_size=1)
        
    def forward(self, x):
        # x: (Batch, Channels, Time)
        out = self.initial(x)
        out = self.res1(out)
        out = self.res2(out)
        out = self.res3(out)
        out = self.out(out)
        # Output: (Batch, out_channels, Time). No activation (linear regression).
        return out
