import torch
import torch.nn as nn
import torch.nn.functional as F
from models.eegnet import EEGNet
from models.atcnet import ATCNet

class AudioEncoder(nn.Module):
    """
    Encodes 28-band Gammatone subbands into a latent representation.
    """
    def __init__(self, in_channels=28, latent_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=15, padding=7),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.Dropout(0.2),
            
            nn.Conv1d(32, 64, kernel_size=15, padding=7),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Dropout(0.2),
            
            nn.Conv1d(64, latent_dim, kernel_size=1)
        )
        
    def forward(self, x):
        return self.net(x)

class ContrastiveMatchNet(nn.Module):
    """
    A Siamese network that explicitly learns a matching function between EEG and Audio.
    """
    def __init__(self, eeg_model_type="eegnet", eeg_channels=8, audio_channels=28, latent_dim=64):
        super().__init__()
        
        # 1. EEG Encoder
        if eeg_model_type.lower() == "eegnet":
            # EEGNet outputs shape [B, 1, T] by default. We need [B, latent_dim, T]
            self.eeg_encoder = EEGNet(in_channels=eeg_channels)
            # Override the final projection
            self.eeg_encoder.output_proj = nn.Conv1d(16, latent_dim, kernel_size=1) # F2=16 by default
        elif eeg_model_type.lower() == "atcnet":
            self.eeg_encoder = ATCNet(in_channels=eeg_channels)
            # Override the final projection
            hidden_dim = 16 * 2 # F1 * D
            self.eeg_encoder.output_proj = nn.Conv1d(hidden_dim, latent_dim, kernel_size=1)
        else:
            raise ValueError(f"Unknown eeg_model_type: {eeg_model_type}")
            
        # 2. Audio Encoder
        self.audio_encoder = AudioEncoder(in_channels=audio_channels, latent_dim=latent_dim)
        
        self.latent_dim = latent_dim

    def encode_eeg(self, eeg):
        """ Returns [B, latent_dim, Time] """
        return self.eeg_encoder(eeg)
        
    def encode_audio(self, audio):
        """ Returns [B, latent_dim, Time] """
        return self.audio_encoder(audio)

    def forward(self, eeg, audio_a, audio_b):
        """
        Forward pass for training.
        eeg: [B, C, T]
        audio_a: [B, 28, T]
        audio_b: [B, 28, T]
        
        Returns latent representations.
        """
        z_eeg = self.encode_eeg(eeg)
        z_a = self.encode_audio(audio_a)
        z_b = self.encode_audio(audio_b)
        
        return z_eeg, z_a, z_b

def contrastive_loss(z_eeg, z_a, z_b, margin=0.1):
    """
    Computes a max-margin contrastive loss based on cosine similarity.
    We assume z_a is the attended audio, and z_b is the unattended audio.
    
    z_eeg, z_a, z_b shape: [B, latent_dim, T]
    We compute cosine similarity over the latent dimension, then average over Time.
    """
    # Cosine similarity: [B, T]
    sim_a = F.cosine_similarity(z_eeg, z_a, dim=1)
    sim_b = F.cosine_similarity(z_eeg, z_b, dim=1)
    
    # Average over time to get a single scalar per batch element
    sim_a_mean = sim_a.mean(dim=1)
    sim_b_mean = sim_b.mean(dim=1)
    
    # We want sim_a > sim_b + margin
    loss = F.relu(margin - (sim_a_mean - sim_b_mean)).mean()
    
    return loss, sim_a_mean, sim_b_mean

if __name__ == "__main__":
    model = ContrastiveMatchNet("eegnet")
    eeg = torch.randn(2, 8, 640)
    audio_a = torch.randn(2, 28, 640)
    audio_b = torch.randn(2, 28, 640)
    
    z_eeg, z_a, z_b = model(eeg, audio_a, audio_b)
    print("Z_eeg:", z_eeg.shape)
    loss, sa, sb = contrastive_loss(z_eeg, z_a, z_b)
    print("Loss:", loss.item())
