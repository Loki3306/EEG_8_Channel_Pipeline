import torch
import torch.nn as nn

class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, **kwargs):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation, **kwargs)
        
    def forward(self, x):
        x = nn.functional.pad(x, (self.pad, 0))
        return self.conv(x)

class MultiScaleConv1d(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        c = out_channels // 4
        self.c1 = CausalConv1d(in_channels, c, kernel_size=3)
        self.c2 = CausalConv1d(in_channels, c, kernel_size=7)
        self.c3 = CausalConv1d(in_channels, c, kernel_size=15)
        self.c4 = CausalConv1d(in_channels, out_channels - 3*c, kernel_size=31)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout1d(0.1)
        
    def forward(self, x):
        out = torch.cat([self.c1(x), self.c2(x), self.c3(x), self.c4(x)], dim=1)
        return self.dropout(self.relu(self.bn(out)))

class DilatedResidualBlock(nn.Module):
    def __init__(self, channels, dilation):
        super().__init__()
        self.conv1 = CausalConv1d(channels, channels, kernel_size=3, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(channels)
        self.relu = nn.ReLU()
        self.conv2 = CausalConv1d(channels, channels, kernel_size=3, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(channels)
        self.dropout = nn.Dropout1d(0.1)
        
    def forward(self, x):
        residual = x
        out = self.dropout(self.relu(self.bn1(self.conv1(x))))
        out = self.bn2(self.conv2(out))
        return self.dropout(self.relu(out + residual))

class ResidualNeuralRidgeDecoder(nn.Module):
    def __init__(self, in_channels=8, out_channels=28, lags=16):
        super().__init__()
        
        # Classical Ridge branch (Linear mapping from lagged EEG)
        self.base_ridge = CausalConv1d(in_channels, out_channels, kernel_size=lags+1, bias=True)
        # Freeze Ridge branch by default until explicitly unfrozen
        for p in self.base_ridge.parameters():
            p.requires_grad = False
            
        # Neural residual branch
        self.stem = nn.Sequential(
            CausalConv1d(in_channels, 32, kernel_size=5),
            nn.BatchNorm1d(32),
            nn.ReLU()
        )
        
        self.multi_scale = MultiScaleConv1d(32, 64)
        
        self.res1 = DilatedResidualBlock(64, dilation=1)
        self.res2 = DilatedResidualBlock(64, dilation=2)
        self.res3 = DilatedResidualBlock(64, dilation=4)
        self.res4 = DilatedResidualBlock(64, dilation=8)
        
        self.down = nn.Sequential(
            CausalConv1d(64, 32, kernel_size=3),
            nn.BatchNorm1d(32),
            nn.ReLU()
        )
        
        self.out = CausalConv1d(32, out_channels, kernel_size=1, bias=True)
        self.alpha = nn.Parameter(torch.tensor(0.05))
        
    def forward(self, x):
        # x: (Batch, 8, Time)
        base_env = self.base_ridge(x)
        
        out = self.stem(x)
        out = self.multi_scale(out)
        out = self.res1(out)
        out = self.res2(out)
        out = self.res3(out)
        out = self.res4(out)
        out = self.down(out)
        delta_env = self.alpha * self.out(out)
        
        return base_env, delta_env

    def load_ridge_weights(self, weights_matrix, bias_vector=None):
        """
        Inject analytical Ridge weights into the base_ridge branch.
        weights_matrix: (out_channels, in_channels * kernel_size)
                        where kernel_size = lags + 1
        """
        out_channels = self.base_ridge.conv.weight.shape[0]
        in_channels = self.base_ridge.conv.weight.shape[1]
        kernel_size = self.base_ridge.conv.weight.shape[2]
        
        # PyTorch Conv1d weights are (out_channels, in_channels, kernel_size)
        # Ridge matrix features are typically [ch1_lag0, ..., ch8_lag0, ch1_lag1, ..., ch8_lag1, ...]
        # OR [ch1_lag0, ..., ch1_lag16, ch2_lag0, ..., ch2_lag16]
        # We need to map the analytical Ridge matrix to (out_channels, in_channels, kernel_size).
        
        # We will assume weights_matrix is provided as (out_channels, in_channels, kernel_size)
        # to simplify integration. The training script will handle the reshaping.
        
        with torch.no_grad():
            self.base_ridge.conv.weight.copy_(torch.FloatTensor(weights_matrix))
            if bias_vector is not None:
                self.base_ridge.conv.bias.copy_(torch.FloatTensor(bias_vector))
            else:
                self.base_ridge.conv.bias.zero_()
