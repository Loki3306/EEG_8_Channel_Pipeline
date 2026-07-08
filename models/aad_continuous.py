import torch
import torch.nn as nn
import torch.nn.functional as F

# -------------------------------------------------------------------------
# CONTINUOUS SEQUENCE ARCHITECTURE (PHASE 95c)
# -------------------------------------------------------------------------
class DilatedFrontend(nn.Module):
    """
    Extracts features across a ~445ms temporal window.
    Crucial for aligning the EEG lag (150ms-250ms) with the Audio
    before instantaneous interaction occurs.
    """
    def __init__(self, in_channels, d_model=64):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, 32, kernel_size=9, padding=4, dilation=1)
        self.bn1 = nn.BatchNorm1d(32)
        
        self.conv2 = nn.Conv1d(32, 64, kernel_size=9, padding=8, dilation=2)
        self.bn2 = nn.BatchNorm1d(64)
        
        self.conv3 = nn.Conv1d(64, d_model, kernel_size=9, padding=16, dilation=4)
        self.bn3 = nn.BatchNorm1d(d_model)

    def forward(self, x):
        # x: [B, C, L]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return x

class ContinuousLSTMEncoder(nn.Module):
    def __init__(self, in_channels, d_model=64, num_layers=2):
        super().__init__()
        # Dilated frontend (Receptive Field ~ 445ms)
        self.frontend = DilatedFrontend(in_channels, d_model)
        
        self.lstm = nn.LSTM(d_model, d_model // 2, num_layers=num_layers, 
                            batch_first=True, bidirectional=True, dropout=0.3 if num_layers > 1 else 0)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        # x: [B, L, C] -> needs [B, C, L] for Conv1d
        x = x.transpose(1, 2)
        x = self.frontend(x)
        x = x.transpose(1, 2)
        
        # LSTM processes [B, L, d_model]
        out, _ = self.lstm(x)
        out = self.norm(out)
        return out


class ContinuousAADAUDModel(nn.Module):
    """
    An optimized Continuous Sequence Model for AAD.
    Phase 95c restores Siamese Symmetry and Lag Compensation.
    """
    def __init__(self, eeg_channels=8, audio_channels=16, d_model=64):
        super().__init__()
        
        self.eeg_encoder = ContinuousLSTMEncoder(in_channels=eeg_channels, d_model=d_model, num_layers=2)
        # Shared audio encoder to ensure symmetric latent space
        self.audio_encoder = ContinuousLSTMEncoder(in_channels=audio_channels, d_model=d_model, num_layers=2)
        
        # Integrator processes ONE stream at a time (Siamese)
        # It takes [z_eeg, z_aud, z_eeg - z_aud, z_eeg * z_aud] -> 4 * d_model = 256
        integrator_input_dim = d_model * 4
        self.integrator = nn.LSTM(integrator_input_dim, d_model, num_layers=1, 
                                  batch_first=True, bidirectional=True)
        
        # Final classifier with regularization
        self.classifier = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(d_model, 1)
        )

    def forward(self, eeg_seq, audio_a_seq, audio_b_seq):
        # Inputs: [B, L, C]
        
        # 1. Project to continuous latent manifolds (now with 445ms lag compensation)
        z_eeg = self.eeg_encoder(eeg_seq)      # [B, L, 64]
        z_a = self.audio_encoder(audio_a_seq)  # [B, L, 64]
        z_b = self.audio_encoder(audio_b_seq)  # [B, L, 64]
        
        # 2. Rich Interaction Features
        feat_a = torch.cat([z_eeg, z_a, z_eeg - z_a, z_eeg * z_a], dim=-1) # [B, L, 256]
        feat_b = torch.cat([z_eeg, z_b, z_eeg - z_b, z_eeg * z_b], dim=-1) # [B, L, 256]
        
        # 3. Symmetrical Siamese Integration
        integrated_a, _ = self.integrator(feat_a) # [B, L, 128]
        integrated_b, _ = self.integrator(feat_b) # [B, L, 128]
        
        # 4. Sequence Pooling
        pooled_a = integrated_a.mean(dim=1) # [B, 128]
        pooled_b = integrated_b.mean(dim=1) # [B, 128]
        
        # 5. Classification
        score_a = self.classifier(pooled_a).squeeze(-1) # [B]
        score_b = self.classifier(pooled_b).squeeze(-1) # [B]
        
        # Final Output is the difference (Strict Symmetry)
        return score_a - score_b
