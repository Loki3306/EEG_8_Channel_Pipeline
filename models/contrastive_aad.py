import torch
import torch.nn as nn

class ResidualTCNBlock(nn.Module):
    def __init__(self, channels, dilation):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=dilation, dilation=dilation)
        self.norm1 = nn.BatchNorm1d(channels)
        self.gelu = nn.GELU()
        
    def forward(self, x):
        res = x
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.gelu(x)
        return x + res

class AttentionPooling(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Conv1d(channels, 1, kernel_size=1),
            nn.Softmax(dim=-1)
        )
    def forward(self, x):
        # x: [B, C, T]
        w = self.attn(x) # [B, 1, T]
        return (x * w).sum(dim=-1) # [B, C]

class ProjectionHead(nn.Module):
    def __init__(self, in_features, out_features=128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_features, in_features),
            nn.BatchNorm1d(in_features),
            nn.GELU(),
            nn.Linear(in_features, out_features)
        )
    def forward(self, x):
        x = self.mlp(x)
        return nn.functional.normalize(x, p=2, dim=-1)

class EEGEncoder(nn.Module):
    def __init__(self, in_channels=8, F1=8, D=2, F2=16):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(1, F1, (1, 64), padding=(0, 32), bias=False),
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
        self.tcn = nn.Sequential(
            ResidualTCNBlock(F2, dilation=1),
            ResidualTCNBlock(F2, dilation=2),
            ResidualTCNBlock(F2, dilation=4),
            ResidualTCNBlock(F2, dilation=8)
        )
        self.pool = AttentionPooling(F2)
        self.proj = ProjectionHead(F2, 128)
        
    def forward(self, x):
        orig_len = x.shape[-1]
        x = x.unsqueeze(1)
        x = self.block1(x)
        x = self.block2(x)
        x = x.squeeze(2)[..., :orig_len] # [B, F2, T]
        x = self.tcn(x)
        x = self.pool(x)
        x = self.proj(x)
        return x

class AudioEncoder(nn.Module):
    def __init__(self, F2=16):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(1, F2, kernel_size=33, padding=16, bias=False),
            nn.BatchNorm1d(F2),
            nn.GELU(),
            nn.Dropout(0.25)
        )
        self.tcn = nn.Sequential(
            ResidualTCNBlock(F2, dilation=1),
            ResidualTCNBlock(F2, dilation=2),
            ResidualTCNBlock(F2, dilation=4),
            ResidualTCNBlock(F2, dilation=8)
        )
        self.pool = AttentionPooling(F2)
        self.proj = ProjectionHead(F2, 128)
        
    def forward(self, x):
        orig_len = x.shape[-1]
        x = x.unsqueeze(1) # [B, 1, T]
        x = self.cnn(x)[..., :orig_len]
        x = self.tcn(x)
        x = self.pool(x)
        x = self.proj(x)
        return x

class ContrastiveMatchNet(nn.Module):
    def __init__(self, eeg_channels=8):
        super().__init__()
        self.eeg_enc = EEGEncoder(in_channels=eeg_channels)
        self.aud_enc = AudioEncoder()
        
    def forward(self, eeg, audio_att, audio_unatt):
        z_e = self.eeg_enc(eeg)
        z_a = self.aud_enc(audio_att)
        z_b = self.aud_enc(audio_unatt)
        return z_e, z_a, z_b

def contrastive_loss(z_e, z_att, z_unatt, tau=0.1, margin=0.1, margin_weight=0.5):
    """
    Computes a composite InfoNCE + Margin Ranking Loss.
    z_e: EEG embeddings [B, D]
    z_att: Attended Audio embeddings [B, D]
    z_unatt: Unattended Audio embeddings [B, D]
    """
    # InfoNCE over all available negatives in batch
    sim_matrix_att = torch.matmul(z_e, z_att.T) / tau
    sim_matrix_unatt = torch.matmul(z_e, z_unatt.T) / tau
    
    # logits shape: [B, 2B]
    logits = torch.cat([sim_matrix_att, sim_matrix_unatt], dim=1)
    labels = torch.arange(z_e.size(0), device=z_e.device)
    loss_infonce = nn.functional.cross_entropy(logits, labels)
    
    # Margin Ranking Loss
    s_att = (z_e * z_att).sum(dim=-1)
    s_unatt = (z_e * z_unatt).sum(dim=-1)
    loss_margin = torch.clamp(margin + s_unatt - s_att, min=0.0).mean()
    
    return loss_infonce + margin_weight * loss_margin, loss_infonce, loss_margin
