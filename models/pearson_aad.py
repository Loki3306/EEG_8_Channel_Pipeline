import torch
import torch.nn as nn
import torch.nn.functional as F
from models.contrastive_aad import EEGEncoder, GradientReversal

class NegativePearsonLoss(nn.Module):
    """
    Computes the negative Pearson correlation over the time dimension.
    Optimizes the network to maximize linear correlation with the envelope.
    """
    def __init__(self):
        super().__init__()
        
    def forward(self, preds, targets):
        # preds, targets: [Batch, Channels, Time]
        
        preds_mean = preds.mean(dim=-1, keepdim=True)
        targets_mean = targets.mean(dim=-1, keepdim=True)
        
        preds_centered = preds - preds_mean
        targets_centered = targets - targets_mean
        
        cov = (preds_centered * targets_centered).sum(dim=-1)
        
        preds_std = torch.sqrt((preds_centered**2).sum(dim=-1) + 1e-8)
        targets_std = torch.sqrt((targets_centered**2).sum(dim=-1) + 1e-8)
        
        corr = cov / (preds_std * targets_std)
        
        # Average correlation across batch and subbands, then negate for minimization
        return 1.0 - corr.mean()

class PearsonAADModel(nn.Module):
    def __init__(self, rep_dim=128, num_subjects=16, num_subbands=28):
        super().__init__()
        self.eeg_encoder = EEGEncoder(rep_dim=rep_dim)
        
        # Up-projection to map the pooled temporal sequence back to the envelope shape
        # block2 of EEGEncoder outputs 16 channels, downsampled by 32 in time.
        self.up_proj = nn.Conv1d(16, num_subbands, kernel_size=1)
        
        # Subject Discriminator using the existing GRL
        self.grl = GradientReversal(lam=1.0)
        self.subject_discriminator = nn.Sequential(
            nn.Linear(rep_dim, 64),
            nn.GELU(),
            nn.Linear(64, num_subjects)
        )
        
    def forward(self, eeg):
        # eeg: [Batch, 8, Time]
        x = eeg.unsqueeze(1)
        
        # Pass through EEGNet spatial and temporal blocks
        x = self.eeg_encoder.block1(x)
        x = self.eeg_encoder.block2(x) # shape: [Batch, 16, 1, Time/32]
        
        # 1. Envelope Prediction Branch (Time-preserving)
        x_1d = x.squeeze(2) # [Batch, 16, Time/32]
        env_lowres = self.up_proj(x_1d) # [Batch, 28, Time/32]
        
        # Interpolate back to original time resolution
        env_pred = F.interpolate(env_lowres, size=eeg.shape[-1], mode='linear', align_corners=False)
        
        # 2. Subject Discriminator Branch (Pooled representation)
        x_pool = self.eeg_encoder.adaptive_pool(x)
        x_flat = self.eeg_encoder.flatten(x_pool)
        rep = self.eeg_encoder.rep_proj(x_flat)
        
        rep_grl = self.grl(rep)
        subj_logits = self.subject_discriminator(rep_grl)
        
        return env_pred, subj_logits

    def predict(self, eeg):
        """ Inference only: returns envelope prediction """
        x = eeg.unsqueeze(1)
        x = self.eeg_encoder.block1(x)
        x = self.eeg_encoder.block2(x)
        x_1d = x.squeeze(2)
        env_lowres = self.up_proj(x_1d)
        return F.interpolate(env_lowres, size=eeg.shape[-1], mode='linear', align_corners=False)
