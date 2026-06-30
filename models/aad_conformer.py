import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(1, 2)  # [1, d_model, max_len]
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is [B, C, T]
        return x + self.pe[:, :, :x.size(2)]

class ConformerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, dropout: float = 0.3):
        super().__init__()
        self.ffn1 = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
            nn.Dropout(dropout)
        )
        
        self.mha_norm = nn.LayerNorm(dim)
        self.mha = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        
        self.conv_norm = nn.LayerNorm(dim)
        self.conv = nn.Sequential(
            nn.Conv1d(dim, dim * 2, kernel_size=1),
            nn.GLU(dim=1),
            nn.Conv1d(dim, dim, kernel_size=15, padding=7, groups=dim), # Depthwise
            nn.BatchNorm1d(dim),
            nn.SiLU(),
            nn.Conv1d(dim, dim, kernel_size=1),
            nn.Dropout(dropout)
        )
        
        self.ffn2 = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
            nn.Dropout(dropout)
        )
        
        self.post_norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T]
        x_trans = x.transpose(1, 2) # [B, T, C] for FFN/MHA
        
        # 1. FFN 1 (Macaron style: half step)
        x_trans = x_trans + 0.5 * self.ffn1(x_trans)
        
        # 2. Multi-Head Attention
        x_mha = self.mha_norm(x_trans)
        attn_out, _ = self.mha(x_mha, x_mha, x_mha)
        x_trans = x_trans + attn_out
        
        # 3. Convolution Module
        x_conv = self.conv_norm(x_trans) # Apply LayerNorm on [B, T, C]
        x_conv = x_conv.transpose(1, 2) # [B, C, T] for Conv1d
        x_conv = x_conv + self.conv(x_conv) # Now apply Conv1d
        x_trans = x_conv.transpose(1, 2) # Back to [B, T, C] for FFN
        
        # 4. FFN 2
        x_trans = x_trans + 0.5 * self.ffn2(x_trans)
        
        x_trans = self.post_norm(x_trans)
        return x_trans.transpose(1, 2) # [B, C, T]

class AADConformer(nn.Module):
    def __init__(
        self,
        in_channels: int = 8,
        temporal_filters: int = 32,
        spatial_filters: int = 64,
        embed_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.3,
        stride: int = 4
    ):
        super().__init__()
        self.in_channels = in_channels
        self.stride = stride
        
        # Stem: Temporal & Spatial Convolution (EEGNet style)
        self.temporal_conv = nn.Conv2d(1, temporal_filters, kernel_size=(1, 33), padding=(0, 16), bias=False)
        self.temporal_norm = nn.BatchNorm2d(temporal_filters)
        
        self.spatial_conv = nn.Conv2d(temporal_filters, spatial_filters, kernel_size=(in_channels, 1), 
                                      groups=temporal_filters, bias=False)
        self.spatial_norm = nn.BatchNorm2d(spatial_filters)
        
        self.stem_act = nn.SiLU()
        self.stem_dropout = nn.Dropout(dropout)
        
        # Tokenization via strided convolution
        self.tokenization = nn.Conv1d(spatial_filters, embed_dim, kernel_size=stride, stride=stride, padding=0)
        self.pos_encoder = PositionalEncoding(embed_dim)
        
        # Conformer Blocks
        self.conformer_blocks = nn.ModuleList([
            ConformerBlock(embed_dim, num_heads, dropout) for _ in range(num_layers)
        ])
        
        # Upsampling back to original sequence length
        self.upsample = nn.ConvTranspose1d(embed_dim, embed_dim, kernel_size=stride, stride=stride, padding=0)
        self.upsample_act = nn.SiLU()
        
        # Final Regression Head
        self.head = nn.Conv1d(embed_dim, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Expected x: [Batch, Channels, Time]
        if x.ndim != 3:
            raise ValueError(f"Expected 3D input, got shape {tuple(x.shape)}")

        if x.shape[-1] == self.in_channels:
            x = x.transpose(1, 2)
        elif x.shape[1] != self.in_channels:
            raise ValueError(
                f"Expected input with {self.in_channels} channels in either the last or second dimension, got shape {tuple(x.shape)}"
            )
            
        B, C, T = x.shape
            
        # Add channel dimension for 2D convolutions: [Batch, 1, Channels, Time]
        x = x.unsqueeze(1)
        
        # Temporal Conv
        x = self.temporal_conv(x)
        x = self.temporal_norm(x)
        
        # Spatial Depthwise Conv
        x = self.spatial_conv(x)
        x = self.spatial_norm(x)
        x = self.stem_act(x)
        x = self.stem_dropout(x)
        
        # Remove spatial dimension: [Batch, Filters, Time]
        x = x.squeeze(2)
        
        # Tokenization
        x = self.tokenization(x)
        
        # Positional Encoding
        x = self.pos_encoder(x)
        
        # Conformer Blocks
        for block in self.conformer_blocks:
            x = block(x)
            
        # Upsampling
        x = self.upsample(x)
        
        # Adjust length if necessary (due to padding/stride mismatch on arbitrary sequence lengths)
        if x.size(-1) != T:
            x = F.interpolate(x, size=T, mode='linear', align_corners=False)
            
        x = self.upsample_act(x)
        
        # Regression Head
        x = self.head(x)
        
        # Squeeze to [Batch, Time] for AAD loss compatibility
        return x.squeeze(1)
