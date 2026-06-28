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
    def __init__(self, in_channels=8, out_channels=28):
        super().__init__()
        
        self.initial = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU()
        )
        
        self.up = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )
        
        self.res1 = ResidualBlock1D(64)
        self.res2 = ResidualBlock1D(64)
        
        self.down = nn.Sequential(
            nn.Conv1d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU()
        )
        
        self.out = nn.Conv1d(32, out_channels, kernel_size=1)
        
    def forward(self, x):
        out = self.initial(x)
        out = self.up(out)
        out = self.res1(out)
        out = self.res2(out)
        out = self.down(out)
        out = self.out(out)
        return out
