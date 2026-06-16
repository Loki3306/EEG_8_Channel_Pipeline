import torch
import torch.nn as nn

class EEGNet(nn.Module):
    """
    EEGNet adapted for continuous auditory envelope reconstruction (regression).
    Standard architecture: Temporal Conv -> Spatial Depthwise -> Separable Conv
    """
    def __init__(self, in_channels=8, F1=8, D=2, F2=16, kernel_length=64):
        super().__init__()
        
        self.block1 = nn.Sequential(
            nn.Conv2d(1, F1, (1, kernel_length), padding=(0, kernel_length//2), bias=False),
            nn.BatchNorm2d(F1),
            nn.Conv2d(F1, F1 * D, (in_channels, 1), groups=F1, bias=False),
            nn.BatchNorm2d(F1 * D),
            nn.GELU(),
            nn.Dropout(0.25)
        )
        
        self.block2 = nn.Sequential(
            nn.Conv2d(F1 * D, F1 * D, (1, 16), padding=(0, 8), groups=F1 * D, bias=False),
            nn.Conv2d(F1 * D, F2, (1, 1), bias=False),
            nn.BatchNorm2d(F2),
            nn.GELU(),
            nn.Dropout(0.25)
        )
        
        self.output_proj = nn.Conv1d(F2, 1, kernel_size=1)

    def forward(self, x):
        orig_len = x.shape[-1]
        # Input: [Batch, Channels, Time]
        x = x.unsqueeze(1) # [Batch, 1, Channels, Time]
        x = self.block1(x) # [Batch, F1*D, 1, Time+1]
        x = self.block2(x) # [Batch, F2, 1, Time+2]
        x = x.squeeze(2)   # [Batch, F2, Time+2]
        x = self.output_proj(x) # [Batch, 1, Time+2]
        return x[..., :orig_len]

class MultiScaleEEGNet(nn.Module):
    """
    Multi-Scale EEGNet adapted for continuous auditory envelope reconstruction.
    Extracts features at parallel temporal scales.
    """
    def __init__(self, in_channels=8, F1=8, D=2, F2=16, p=0.25):
        super().__init__()
        
        self.temp8 = nn.Conv2d(1, F1, (1, 8), padding="same", bias=False)
        self.temp16 = nn.Conv2d(1, F1, (1, 16), padding="same", bias=False)
        self.temp32 = nn.Conv2d(1, F1, (1, 32), padding="same", bias=False)
        self.temp64 = nn.Conv2d(1, F1, (1, 64), padding="same", bias=False)
        
        self.bn1 = nn.BatchNorm2d(F1 * 4)
        
        self.depthwise = nn.Conv2d(F1 * 4, F1 * 4 * D, (in_channels, 1), groups=F1 * 4, bias=False)
        self.bn2 = nn.BatchNorm2d(F1 * 4 * D)
        self.activation = nn.GELU()
        self.dropout1 = nn.Dropout(p)
        
        self.separable = nn.Sequential(
            nn.Conv2d(F1 * 4 * D, F1 * 4 * D, (1, 16), padding="same", groups=F1 * 4 * D, bias=False),
            nn.Conv2d(F1 * 4 * D, F2, (1, 1), bias=False)
        )
        self.bn3 = nn.BatchNorm2d(F2)
        self.dropout2 = nn.Dropout(p)
        
        self.output_proj = nn.Conv1d(F2, 1, kernel_size=1)
        
    def forward(self, x):
        orig_len = x.shape[-1]
        x = x.unsqueeze(1)
        
        # Parallel Temporal
        x_temp = torch.cat([self.temp8(x), self.temp16(x), self.temp32(x), self.temp64(x)], dim=1)
        x_temp = self.bn1(x_temp)
        
        # Depthwise
        x_depth = self.depthwise(x_temp)
        x_depth = self.bn2(x_depth)
        x_depth = self.activation(x_depth)
        x_depth = self.dropout1(x_depth)
        
        # Separable
        x_sep = self.separable(x_depth)
        x_sep = self.bn3(x_sep)
        x_sep = self.activation(x_sep)
        x_sep = self.dropout2(x_sep)
        
        x_sep = x_sep.squeeze(2)
        out = self.output_proj(x_sep)
        return out[..., :orig_len]

def print_summary():
    model = EEGNet()
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"EEGNet Parameter Count: {params:,}")
    
    ms_model = MultiScaleEEGNet()
    ms_params = sum(p.numel() for p in ms_model.parameters() if p.requires_grad)
    print(f"MultiScaleEEGNet Parameter Count: {ms_params:,}")
    
if __name__ == "__main__":
    print_summary()
