import torch
import torch.nn as nn
import torch.nn.functional as F

# -------------------------------------------------------------------------
# CONTINUOUS SEQUENCE ARCHITECTURE (PHASE 95b)
# -------------------------------------------------------------------------
class ContinuousLSTMEncoder(nn.Module):
    def __init__(self, in_channels, d_model=64, num_layers=2):
        super().__init__()
        # Tiny local feature extraction (as recommended by GPT to denoise raw samples)
        self.frontend = nn.Conv1d(in_channels, d_model, kernel_size=5, padding=2)
        
        # Continuous sequence integration
        self.lstm = nn.LSTM(d_model, d_model // 2, num_layers=num_layers, 
                            batch_first=True, bidirectional=True, dropout=0.3 if num_layers > 1 else 0)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        # x: [B, L, C] -> needs [B, C, L] for Conv1d
        x = x.transpose(1, 2)
        x = self.frontend(x)
        x = x.transpose(1, 2)
        x = F.gelu(x)
        
        # LSTM processes [B, L, d_model]
        out, _ = self.lstm(x)
        out = self.norm(out)
        return out


class ContinuousAADAUDModel(nn.Module):
    """
    An optimized Continuous Sequence Model for AAD.
    Uses Bidirectional LSTMs to achieve the continuous infinite context window
    without the extreme Python overhead of custom pure-PyTorch SSMs.
    Also fixes the cosine information bottleneck by using rich interaction features.
    """
    def __init__(self, eeg_channels=8, audio_channels=16, d_model=64):
        super().__init__()
        
        self.eeg_encoder = ContinuousLSTMEncoder(in_channels=eeg_channels, d_model=d_model, num_layers=2)
        # Shared audio encoder to ensure symmetric latent space
        self.audio_encoder = ContinuousLSTMEncoder(in_channels=audio_channels, d_model=d_model, num_layers=2)
        
        # Integrator takes [feat_a, feat_b] -> 8 * d_model
        integrator_input_dim = d_model * 8
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
        
        # 1. Project to continuous latent manifolds
        z_eeg = self.eeg_encoder(eeg_seq)      # [B, L, 64]
        z_a = self.audio_encoder(audio_a_seq)  # [B, L, 64]
        z_b = self.audio_encoder(audio_b_seq)  # [B, L, 64]
        
        # 2. Rich Interaction Features (Solves GPT Bottleneck Warning)
        # Instead of 1 cosine scalar, we provide 256 dimensions of evidence per stream
        feat_a = torch.cat([z_eeg, z_a, z_eeg - z_a, z_eeg * z_a], dim=-1) # [B, L, 256]
        feat_b = torch.cat([z_eeg, z_b, z_eeg - z_b, z_eeg * z_b], dim=-1) # [B, L, 256]
        
        # [B, L, 512]
        sim_features = torch.cat([feat_a, feat_b], dim=-1)
        
        # 3. Continuous Temporal Integration
        # Evaluates the evolving evidence continuously over the 448 samples
        integrated, _ = self.integrator(sim_features) # [B, L, 128]
        
        # 4. Sequence Pooling
        # GPT warned that taking the last state (-1) could lose early decisive evidence.
        # Mean pooling aggregates evidence across the entire 3.5s sequence.
        pooled = integrated.mean(dim=1) # [B, 128]
        
        # 5. Classification
        logits = self.classifier(pooled) # [B, 1]
        
        return logits.squeeze(-1)
