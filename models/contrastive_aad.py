import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lam):
        ctx.lam = lam
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lam, None

class GradientReversal(nn.Module):
    def __init__(self, lam=1.0):
        super().__init__()
        self.lam = lam

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lam)

class EEGEncoder(nn.Module):
    """
    EEGNet-based base encoder. Output is the raw representation.
    """
    def __init__(self, in_channels=8, F1=8, D=2, F2=16, kernel_length=64, rep_dim=128):
        super().__init__()
        
        # GroupNorm requires num_channels to be divisible by num_groups.
        # F1 = 8 -> GN(8, 8) or GN(4, 8)
        
        self.block1 = nn.Sequential(
            nn.Conv2d(1, F1, (1, kernel_length), padding=(0, kernel_length//2), bias=False),
            nn.GroupNorm(4, F1),
            nn.Conv2d(F1, F1 * D, (in_channels, 1), groups=F1, bias=False),
            nn.GroupNorm(4, F1 * D),
            nn.GELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(0.25)
        )
        
        self.block2 = nn.Sequential(
            nn.Conv2d(F1 * D, F1 * D, (1, 16), padding=(0, 8), groups=F1 * D, bias=False),
            nn.Conv2d(F1 * D, F2, (1, 1), bias=False),
            nn.GroupNorm(4, F2),
            nn.GELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(0.25)
        )
        
        self.flatten = nn.Flatten()
        
        # Adaptive pooling ensures consistent output size before flattening
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 4))
        
        self.rep_proj = nn.Sequential(
            nn.Linear(F2 * 4, rep_dim),
            nn.GELU()
        )

    def forward(self, x):
        # Input: [Batch, Channels, Time]
        x = x.unsqueeze(1) # [Batch, 1, Channels, Time]
        x = self.block1(x)
        x = self.block2(x)
        x = self.adaptive_pool(x)
        x = self.flatten(x)
        return self.rep_proj(x)


class AudioEncoder(nn.Module):
    """
    1D CNN base encoder. Output is the raw representation.
    """
    def __init__(self, in_channels=28, rep_dim=128):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=11, padding=5),
            nn.GroupNorm(4, 32),
            nn.GELU(),
            nn.MaxPool1d(4),
            
            nn.Conv1d(32, 64, kernel_size=11, padding=5),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            nn.MaxPool1d(8),
        )
        
        self.adaptive_pool = nn.AdaptiveAvgPool1d(4)
        self.flatten = nn.Flatten()
        
        self.rep_proj = nn.Sequential(
            nn.Linear(64 * 4, rep_dim),
            nn.GELU()
        )

    def forward(self, x):
        # Input: [Batch, 28, Time]
        x = self.net(x)
        x = self.adaptive_pool(x)
        x = self.flatten(x)
        return self.rep_proj(x)


class ProjectionHead(nn.Module):
    """
    Maps representation to the InfoNCE embedding space.
    """
    def __init__(self, in_dim=128, embed_dim=128):
        super().__init__()
        self.proj = nn.Linear(in_dim, embed_dim)
        
    def forward(self, x):
        return F.normalize(self.proj(x), p=2, dim=1)


class InfoNCELossWithHardNegatives(nn.Module):
    """
    InfoNCE loss including explicit in-trial hard negatives.
    """
    def __init__(self, init_tau=0.07):
        super().__init__()
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / init_tau))
        
    def forward(self, eeg_emb, aud_pos_emb, aud_neg_emb):
        # All embeddings: [B, D] and L2 normalized
        # Clamp temperature to max 100 as in CLIP
        logit_scale = self.logit_scale.exp().clamp(max=100.0)
        
        # Concat positive and negative audio embeddings: [2B, D]
        all_aud_emb = torch.cat([aud_pos_emb, aud_neg_emb], dim=0)
        
        # Sim matrix: [B, 2B]
        # The correct pair for eeg_i is aud_pos_i (which is at index i)
        sim = logit_scale * (eeg_emb @ all_aud_emb.T)
        
        labels = torch.arange(sim.size(0), device=sim.device)
        loss_eeg = F.cross_entropy(sim, labels)
        
        # Audio-to-EEG loss (only for positive audio anchors against all EEGs)
        # We only care if aud_pos_i finds eeg_i.
        sim_aud2eeg = logit_scale * (aud_pos_emb @ eeg_emb.T)
        loss_aud = F.cross_entropy(sim_aud2eeg, labels)
        
        return (loss_eeg + loss_aud) / 2.0


class ContrastiveAADModel(nn.Module):
    def __init__(self, rep_dim=128, embed_dim=128, num_subjects=16):
        super().__init__()
        self.eeg_encoder = EEGEncoder(rep_dim=rep_dim)
        self.audio_encoder = AudioEncoder(rep_dim=rep_dim)
        
        self.eeg_proj = ProjectionHead(rep_dim, embed_dim)
        self.aud_proj = ProjectionHead(rep_dim, embed_dim)
        
        self.criterion = InfoNCELossWithHardNegatives()
        
        # Subject Discriminator branch for adversarial domain adaptation
        self.grl = GradientReversal(lam=1.0)
        self.subject_discriminator = nn.Sequential(
            nn.Linear(rep_dim, 64),
            nn.GELU(),
            nn.Linear(64, num_subjects)
        )
        
    def forward(self, eeg, audio_pos, audio_neg):
        """
        Training pass computing InfoNCE with hard negatives.
        """
        e_rep = self.eeg_encoder(eeg)
        ap_rep = self.audio_encoder(audio_pos)
        an_rep = self.audio_encoder(audio_neg)
        
        e_emb = self.eeg_proj(e_rep)
        ap_emb = self.aud_proj(ap_rep)
        an_emb = self.aud_proj(an_rep)
        
        infonce_loss = self.criterion(e_emb, ap_emb, an_emb)
        
        # Adversarial subject discriminator
        e_rep_grl = self.grl(e_rep)
        subj_logits = self.subject_discriminator(e_rep_grl)
        
        return infonce_loss, subj_logits
        
    def get_representations(self, eeg, audio):
        """
        Extracts raw representations (before projection) for linear probing.
        """
        e_rep = self.eeg_encoder(eeg)
        a_rep = self.audio_encoder(audio)
        return e_rep, a_rep
