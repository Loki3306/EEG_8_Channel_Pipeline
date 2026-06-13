import torch
import torch.nn as nn
import torch.nn.functional as F

class TCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size=kernel_size,
            padding=(kernel_size - 1) * dilation // 2, dilation=dilation, bias=False
        )
        self.bn = nn.BatchNorm1d(out_channels)
        self.act = nn.GELU()
        
    def forward(self, x):
        res = x
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        if x.shape == res.shape:
            return x + res
        return x

class ATCNet(nn.Module):
    """
    ATCNet adapted for continuous auditory envelope reconstruction (regression).
    Architecture: Spatial/Temporal Conv -> Multi-Head Attention -> TCN
    """
    def __init__(self, in_channels=8, F1=16, D=2, num_heads=2, tcn_depth=2):
        super().__init__()
        
        # Spatial-Temporal feature extraction
        self.conv_block = nn.Sequential(
            nn.Conv2d(1, F1, (1, 64), padding=(0, 32), bias=False),
            nn.BatchNorm2d(F1),
            nn.Conv2d(F1, F1 * D, (in_channels, 1), groups=F1, bias=False),
            nn.BatchNorm2d(F1 * D),
            nn.GELU(),
            nn.Dropout(0.3)
        )
        
        self.pool = nn.AvgPool2d((1, 4))
        hidden_dim = F1 * D
        
        # Multi-Head Attention
        self.mha = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)
        self.mha_dropout = nn.Dropout(0.3)
        self.mha_norm = nn.LayerNorm(hidden_dim)
        
        # TCN
        tcn_layers = []
        for i in range(tcn_depth):
            dilation = 2 ** i
            tcn_layers.append(TCNBlock(hidden_dim, hidden_dim, kernel_size=3, dilation=dilation))
            tcn_layers.append(nn.Dropout(0.3))
        self.tcn = nn.Sequential(*tcn_layers)
        
        self.output_proj = nn.Conv1d(hidden_dim, 1, kernel_size=1)

    def forward(self, x):
        orig_len = x.shape[-1]
        
        # Input: [Batch, Channels, Time]
        x = x.unsqueeze(1) # [B, 1, C, T]
        x = self.conv_block(x) # [B, F1*D, 1, T]
        x = self.pool(x) # [B, F1*D, 1, T//4]
        x = x.squeeze(2) # [B, F1*D, T//4]
        
        # Attention
        x_mha = x.transpose(1, 2) # [B, T//4, F1*D]
        attn_out, _ = self.mha(x_mha, x_mha, x_mha)
        x_mha = self.mha_norm(x_mha + self.mha_dropout(attn_out))
        x = x_mha.transpose(1, 2) # [B, F1*D, T//4]
        
        # TCN
        x = self.tcn(x) # [B, F1*D, T//4]
        
        # Interpolate back to original length
        x = F.interpolate(x, size=orig_len, mode='linear', align_corners=False)
        
        x = self.output_proj(x) # [B, 1, T]
        return x

def print_summary():
    model = ATCNet()
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"ATCNet Parameter Count: {params:,}")
    
if __name__ == "__main__":
    print_summary()
