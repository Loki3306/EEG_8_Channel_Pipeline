import torch
import torch.nn as nn
import torch.nn.functional as F
from models.contrastive_aad import GradientReversal

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


class RegressionEEGEncoder(nn.Module):
    """
    EEGNet-based encoder modified for regression tasks.
    Allows parameterized temporal pooling to preserve high-resolution time dynamics.
    """
    def __init__(self, in_channels=8, F1=8, D=2, F2=16, kernel_length=64, rep_dim=128, temporal_pooling_factors=(4, 8)):
        super().__init__()
        
        # Temporal pooling logic
        pool1_t, pool2_t = temporal_pooling_factors
        
        self.block1 = nn.Sequential(
            nn.Conv2d(1, F1, (1, kernel_length), padding=(0, kernel_length//2), bias=False),
            nn.GroupNorm(4, F1),
            nn.Conv2d(F1, F1 * D, (in_channels, 1), groups=F1, bias=False),
            nn.GroupNorm(4, F1 * D),
            nn.GELU(),
            nn.AvgPool2d((1, pool1_t)) if pool1_t > 1 else nn.Identity(),
            nn.Dropout(0.25)
        )
        
        self.block2 = nn.Sequential(
            nn.Conv2d(F1 * D, F1 * D, (1, 16), padding=(0, 8), groups=F1 * D, bias=False),
            nn.Conv2d(F1 * D, F2, (1, 1), bias=False),
            nn.GroupNorm(4, F2),
            nn.GELU(),
            nn.AvgPool2d((1, pool2_t)) if pool2_t > 1 else nn.Identity(),
            nn.Dropout(0.25)
        )
        
        self.flatten = nn.Flatten()
        
        # Adaptive pooling ensures consistent output size before flattening (only for Subject Discriminator)
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
        return x

    def forward_subject_rep(self, x_features):
        x = self.adaptive_pool(x_features)
        x = self.flatten(x)
        return self.rep_proj(x)


class PearsonAADModel(nn.Module):
    def __init__(self, rep_dim=128, num_subjects=16, num_subbands=28, temporal_pooling_factors=(4, 8)):
        super().__init__()
        self.eeg_encoder = RegressionEEGEncoder(rep_dim=rep_dim, temporal_pooling_factors=temporal_pooling_factors)
        
        # Up-projection to map the pooled temporal sequence back to the envelope shape
        self.up_proj = nn.Conv1d(16, num_subbands, kernel_size=1)
        
        # Subject Discriminator using the existing GRL
        self.grl = GradientReversal(lam=1.0)
        self.subject_discriminator = nn.Sequential(
            nn.Linear(rep_dim, 64),
            nn.GELU(),
            nn.Linear(64, num_subjects)
        )
        
        self.debug_shapes = False

    def print_temporal_resolution(self, stage_name, shape, fs=64):
        if not self.debug_shapes:
            return
            
        time_len = shape[-1]
        
        # In this model, initial input is at fs=64.
        # We don't have the original time_len explicitly passed through every function natively,
        # but since we know the exact temporal poolings applied, we can approximate it.
        # Wait, shape[-1] is the current time_len. We just print shape[-1].
        
        print(f"{stage_name}: shape {list(shape)}, temporal length {time_len}")

    def forward(self, eeg, fs=64):
        # eeg: [Batch, 8, Time]
        orig_time = eeg.shape[-1]
        
        if self.debug_shapes:
            print(f"Input: shape {list(eeg.shape)}, temporal length {orig_time} @ {fs} Hz")
            
        x = eeg.unsqueeze(1)
        
        # Pass through EEGNet spatial and temporal blocks
        x = self.eeg_encoder.block1(x)
        if self.debug_shapes:
            effective_hz = fs * (x.shape[-1] / orig_time)
            print(f"After Block 1 (Pool 1): shape {list(x.shape)}, temporal length {x.shape[-1]} @ {effective_hz:.1f} Hz")
            
        x = self.eeg_encoder.block2(x) # shape: [Batch, 16, 1, Time/N]
        if self.debug_shapes:
            effective_hz = fs * (x.shape[-1] / orig_time)
            print(f"After Block 2 (Pool 2): shape {list(x.shape)}, temporal length {x.shape[-1]} @ {effective_hz:.1f} Hz")
        
        # 1. Envelope Prediction Branch (Time-preserving)
        x_1d = x.squeeze(2) # [Batch, 16, Time/N]
        env_lowres = self.up_proj(x_1d) # [Batch, 28, Time/N]
        
        # Interpolate back to original time resolution
        env_pred = F.interpolate(env_lowres, size=orig_time, mode='linear', align_corners=False)
        
        if self.debug_shapes:
            print(f"Final Interpolated Envelope: shape {list(env_pred.shape)}")
            print("-" * 50)
            
        # 2. Subject Discriminator Branch (Pooled representation)
        rep = self.eeg_encoder.forward_subject_rep(x)
        
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
