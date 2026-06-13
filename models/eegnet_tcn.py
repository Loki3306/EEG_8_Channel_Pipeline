import torch
import torch.nn as nn
from models.eegnet import EEGNet

class EEGNetTCN(nn.Module):
    """
    EEGNet followed by a lightweight TCN block to extend temporal receptive field.
    """
    def __init__(self, in_channels=8, F1=8, D=2, F2=16, kernel_length=64, num_classes=1):
        super().__init__()
        
        # Base EEGNet feature extractor (without final projection)
        self.eegnet = EEGNet(in_channels=in_channels, F1=F1, D=D, F2=F2, kernel_length=kernel_length)
        
        # Lightweight Dilated TCN block
        # F2 = 16 channels. Kernel = 3. Dilations = 1, 2, 4, 8
        self.tcn = nn.Sequential(
            nn.Conv1d(F2, F2, kernel_size=3, padding=1, dilation=1),
            nn.GELU(),
            nn.Conv1d(F2, F2, kernel_size=3, padding=2, dilation=2),
            nn.GELU(),
            nn.Conv1d(F2, F2, kernel_size=3, padding=4, dilation=4),
            nn.GELU(),
            nn.Conv1d(F2, F2, kernel_size=3, padding=8, dilation=8),
            nn.GELU()
        )
        
        self.output_proj = nn.Conv1d(F2, num_classes, kernel_size=1)

    def forward(self, x):
        # x: [B, C, T]
        orig_len = x.shape[-1]
        
        # We need to extract the features right before the output projection in EEGNet
        x = x.unsqueeze(1) # [B, 1, C, T]
        x = self.eegnet.block1(x) # [B, F1*D, 1, T+1]
        x = self.eegnet.block2(x) # [B, F2, 1, T+2]
        x = x.squeeze(2) # [B, F2, T+2]
        
        # Pass through TCN
        x = self.tcn(x)
        
        # Final projection
        x = self.output_proj(x)
        
        return x[..., :orig_len]

def print_summary():
    model = EEGNetTCN()
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"EEGNet+TCN Parameter Count: {params:,}")
    
if __name__ == "__main__":
    print_summary()
