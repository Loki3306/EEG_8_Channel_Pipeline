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

def print_summary():
    model = EEGNet()
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"EEGNet Parameter Count: {params:,}")
    
if __name__ == "__main__":
    print_summary()
