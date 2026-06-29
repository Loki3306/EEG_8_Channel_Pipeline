import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class EEGEncoder(nn.Module):
    """
    EEGNet-based encoder for mapping EEG signals to a 128-D representation.
    """
    def __init__(self, in_channels=8, F1=8, D=2, F2=16, kernel_length=64, embed_dim=128):
        super().__init__()
        
        self.block1 = nn.Sequential(
            nn.Conv2d(1, F1, (1, kernel_length), padding=(0, kernel_length//2), bias=False),
            nn.BatchNorm2d(F1),
            nn.Conv2d(F1, F1 * D, (in_channels, 1), groups=F1, bias=False),
            nn.BatchNorm2d(F1 * D),
            nn.GELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(0.25)
        )
        
        self.block2 = nn.Sequential(
            nn.Conv2d(F1 * D, F1 * D, (1, 16), padding=(0, 8), groups=F1 * D, bias=False),
            nn.Conv2d(F1 * D, F2, (1, 1), bias=False),
            nn.BatchNorm2d(F2),
            nn.GELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(0.25)
        )
        
        self.flatten = nn.Flatten()
        
        # Adaptive pooling ensures consistent output size before flattening
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 4))
        
        self.proj = nn.Sequential(
            nn.Linear(F2 * 4, 128),
            nn.GELU(),
            nn.Linear(128, embed_dim)
        )

    def forward(self, x):
        # Input: [Batch, Channels, Time]
        x = x.unsqueeze(1) # [Batch, 1, Channels, Time]
        x = self.block1(x)
        x = self.block2(x)
        x = self.adaptive_pool(x)
        x = self.flatten(x)
        x = self.proj(x)
        return F.normalize(x, p=2, dim=1)


class AudioEncoder(nn.Module):
    """
    1D CNN for mapping 28-band audio envelopes to a 128-D representation.
    """
    def __init__(self, in_channels=28, embed_dim=128):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=11, padding=5),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.MaxPool1d(4),
            
            nn.Conv1d(32, 64, kernel_size=11, padding=5),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.MaxPool1d(8),
        )
        
        self.adaptive_pool = nn.AdaptiveAvgPool1d(4)
        self.flatten = nn.Flatten()
        
        self.proj = nn.Sequential(
            nn.Linear(64 * 4, 128),
            nn.GELU(),
            nn.Linear(128, embed_dim)
        )

    def forward(self, x):
        # Input: [Batch, 28, Time]
        x = self.net(x)
        x = self.adaptive_pool(x)
        x = self.flatten(x)
        x = self.proj(x)
        return F.normalize(x, p=2, dim=1)


class InfoNCELoss(nn.Module):
    """
    Temperature-scaled cross entropy loss for symmetric embedding alignment.
    """
    def __init__(self, init_tau=0.07):
        super().__init__()
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / init_tau))
        
    def forward(self, eeg_emb, aud_emb):
        # eeg_emb: [B, D], aud_emb: [B, D]
        # Both are assumed to be L2 normalized
        
        logit_scale = self.logit_scale.exp()
        
        # [B, B] similarity matrix
        sim = logit_scale * (eeg_emb @ aud_emb.T)
        
        labels = torch.arange(sim.size(0), device=sim.device)
        
        loss_eeg = F.cross_entropy(sim, labels)
        loss_aud = F.cross_entropy(sim.T, labels)
        
        return (loss_eeg + loss_aud) / 2.0


class ContrastiveAADModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.eeg_encoder = EEGEncoder()
        self.audio_encoder = AudioEncoder()
        self.criterion = InfoNCELoss()
        
    def forward(self, eeg, audio):
        """
        During training, returns the InfoNCE loss.
        """
        e_emb = self.eeg_encoder(eeg)
        a_emb = self.audio_encoder(audio)
        return self.criterion(e_emb, a_emb)
        
    def get_embeddings(self, eeg, audio):
        """
        Returns normalized embeddings for inference.
        """
        e_emb = self.eeg_encoder(eeg)
        a_emb = self.audio_encoder(audio)
        return e_emb, a_emb
