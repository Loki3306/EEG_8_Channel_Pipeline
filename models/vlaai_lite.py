import torch
import torch.nn as nn
import torch.nn.functional as F

class DepthwiseSeparableConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, dilation=1):
        super().__init__()
        self.depthwise = nn.Conv1d(
            in_channels, in_channels, kernel_size=kernel_size,
            padding=padding, dilation=dilation, groups=in_channels, bias=False
        )
        self.pointwise = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm1d(out_channels)
        self.act = nn.GELU()

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return self.act(x)

class TemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.conv = DepthwiseSeparableConv1d(
            in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2
        )
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        res = x
        x = self.conv(x)
        x = self.dropout(x)
        if x.shape == res.shape:
            return x + res
        return x

class ContextModule(nn.Module):
    def __init__(self, in_channels, out_channels, max_dilation=8):
        super().__init__()
        self.dilations = []
        d = 1
        while d <= max_dilation:
            self.dilations.append(d)
            d *= 2
            
        self.convs = nn.ModuleList([
            DepthwiseSeparableConv1d(
                in_channels, out_channels, kernel_size=5, padding=2 * d, dilation=d
            ) for d in self.dilations
        ])
        self.proj = nn.Conv1d(out_channels * len(self.dilations), out_channels, kernel_size=1)
        self.bn = nn.BatchNorm1d(out_channels)
        self.act = nn.GELU()

    def forward(self, x):
        outs = []
        for conv in self.convs:
            outs.append(conv(x))
        x = torch.cat(outs, dim=1)
        x = self.proj(x)
        x = self.bn(x)
        return self.act(x)

class VLAAILite(nn.Module):
    """
    Lightweight VLAAI-inspired architecture for envelope reconstruction.
    Target parameter count: 50k - 300k.
    """
    def __init__(self, in_channels=8, spatial_dim=32, temporal_dim=64, max_dilation=8):
        super().__init__()
        
        # Spatial Projection Layer
        self.spatial_proj = nn.Sequential(
            nn.Conv1d(in_channels, spatial_dim, kernel_size=1, bias=False),
            nn.BatchNorm1d(spatial_dim),
            nn.GELU()
        )
        
        # Temporal Feature Extractor
        self.temporal_proj = nn.Conv1d(spatial_dim, temporal_dim, kernel_size=1)
        self.temp_block1 = TemporalBlock(temporal_dim, temporal_dim, kernel_size=5)
        self.temp_block2 = TemporalBlock(temporal_dim, temporal_dim, kernel_size=9)
        self.temp_block3 = TemporalBlock(temporal_dim, temporal_dim, kernel_size=15)
        
        # Context Module
        self.context = ContextModule(temporal_dim, temporal_dim, max_dilation=max_dilation)
        
        # Output Projection
        self.output_proj = nn.Conv1d(temporal_dim, 1, kernel_size=1)

    def forward(self, x):
        # x: [Batch, Channels, Time]
        x = self.spatial_proj(x)
        x = self.temporal_proj(x)
        
        x = self.temp_block1(x)
        x = self.temp_block2(x)
        x = self.temp_block3(x)
        
        x = self.context(x)
        
        x = self.output_proj(x)
        return x

def print_summary():
    model = VLAAILite()
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"VLAAI-Lite Parameter Count: {params:,}")
    
    # Estimate receptive field
    x = torch.randn(1, 8, 1024)
    y = model(x)
    print(f"Input shape: {x.shape} -> Output shape: {y.shape}")

if __name__ == "__main__":
    print_summary()
