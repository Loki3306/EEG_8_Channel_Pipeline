import torch
import torch.nn as nn
import torch.nn.functional as F
from models.eegnet import EEGNet
from models.atcnet import ATCNet
from models.eegnet_tcn import EEGNetTCN

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
        elif eeg_model_type.lower() == "eegnet_tcn":
            self.eeg_encoder = EEGNetTCN(in_channels=eeg_channels)
            # Override the final projection. EEGNetTCN output_proj takes F2 channels (default 16)
            self.eeg_encoder.output_proj = nn.Conv1d(16, latent_dim, kernel_size=1)
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

def infonce_loss(z_eeg, z_a, z_b, temperature=0.1):
    """
    Computes an InfoNCE loss across the batch.
    For each EEG representation, the network must identify the correct audio (z_a[i])
    out of 2B candidates: all z_a and all z_b in the batch.
    
    z_eeg, z_a, z_b shape: [B, latent_dim, T]
    """
    B, D, T = z_eeg.shape
    
    # 1. Normalize over the latent dimension
    z_eeg_norm = F.normalize(z_eeg, dim=1)
    z_a_norm = F.normalize(z_a, dim=1)
    z_b_norm = F.normalize(z_b, dim=1)
    
    # 2. Compute time-averaged similarity matrix
    # torch.einsum('bdt,cdt->bc', X, Y) computes the dot product over D and T for all pairs of B and C
    # We divide by T to get the mean similarity over time.
    sim_a = torch.einsum('bdt,cdt->bc', z_eeg_norm, z_a_norm) / T  # [B, B]
    sim_b = torch.einsum('bdt,cdt->bc', z_eeg_norm, z_b_norm) / T  # [B, B]
    
    # 3. Concatenate all candidates
    # The first B columns are comparisons against z_a (positive is on the diagonal)
    # The next B columns are comparisons against z_b (all are negative, including diagonal)
    logits = torch.cat([sim_a, sim_b], dim=1) / temperature  # [B, 2B]
    
    # 4. The correct target for eeg i is z_a i, which is at index i
    labels = torch.arange(B, device=logits.device)
    
    loss = F.cross_entropy(logits, labels)
    
    # For tracking purposes, return the mean similarity of the true positive and the hard negative
    with torch.no_grad():
        sim_a_diag = torch.diag(sim_a).mean()
        sim_b_diag = torch.diag(sim_b).mean()
        
    return loss, sim_a_diag, sim_b_diag

if __name__ == "__main__":
    model = ContrastiveMatchNet("eegnet")
    eeg = torch.randn(16, 8, 320)
    audio_a = torch.randn(16, 28, 320)
    audio_b = torch.randn(16, 28, 320)
    
    z_eeg, z_a, z_b = model(eeg, audio_a, audio_b)
    print("Z_eeg:", z_eeg.shape)
    loss, sa, sb = infonce_loss(z_eeg, z_a, z_b)
    print(f"InfoNCE Loss: {loss.item():.4f} | Sim A: {sa:.4f} | Sim B: {sb:.4f}")
