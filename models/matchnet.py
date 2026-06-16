import torch
import torch.nn as nn
import torch.nn.functional as F
from models.eegnet import EEGNet, MultiScaleEEGNet
from models.atcnet import ATCNet
from models.eegnet_tcn import EEGNetTCN

def create_lagged_audio(audio, lags=[3, 6, 10, 13, 16]):
    """
    Shifts the audio tensor by discrete sample delays to explicitly model neural lag.
    audio: [B, C, T]
    lags: list of integer sample delays (e.g., at 64Hz, 3=~50ms, 16=250ms)
    Returns: [B, C * len(lags), T]
    """
    B, C, T = audio.shape
    lagged = []
    for lag in lags:
        if lag == 0:
            lagged.append(audio)
        else:
            # Shift right by 'lag', pad left with zeros
            shifted = torch.cat([torch.zeros(B, C, lag, device=audio.device), audio[:, :, :-lag]], dim=2)
            lagged.append(shifted)
    return torch.cat(lagged, dim=1)

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

class InceptionAudioEncoder(nn.Module):
    """
    Multi-scale audio encoder using parallel convolution branches.
    """
    def __init__(self, in_channels=28, latent_dim=64):
        super().__init__()
        
        # Parallel branches
        self.branch1 = nn.Conv1d(in_channels, 16, kernel_size=3, padding=1)
        self.branch2 = nn.Conv1d(in_channels, 16, kernel_size=7, padding=3)
        self.branch3 = nn.Conv1d(in_channels, 16, kernel_size=15, padding=7)
        self.branch4 = nn.Conv1d(in_channels, 16, kernel_size=31, padding=15)
        
        self.bn = nn.BatchNorm1d(64)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(0.2)
        
        self.proj = nn.Conv1d(64, latent_dim, kernel_size=1)
        
    def forward(self, x):
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x4 = self.branch4(x)
        
        out = torch.cat([x1, x2, x3, x4], dim=1) # [B, 64, T]
        out = self.bn(out)
        out = self.gelu(out)
        out = self.dropout(out)
        
        return self.proj(out)

class TemporalAttentionPooling(nn.Module):
    """
    Learns to dynamically weight time steps based on their informativeness.
    Replaces uniform averaging with attention.
    """
    def __init__(self, in_features):
        super().__init__()
        # 1x1 conv acting as a linear layer across the time dimension
        self.attn = nn.Conv1d(in_features, 1, kernel_size=1)
        
    def forward(self, x):
        # x: [B, D, T]
        scores = self.attn(x) # [B, 1, T]
        weights = F.softmax(scores, dim=2) # [B, 1, T]
        # Keep time dimension = 1 for backward compatibility with cosine/pearson functions
        pooled = torch.sum(x * weights, dim=2, keepdim=True) # [B, D, 1]
        return pooled

class ContrastiveMatchNet(nn.Module):
    """
    A Siamese network that explicitly learns a matching function between EEG and Audio.
    Includes explicit auditory delay modeling.
    """
    def __init__(self, eeg_model_type="eegnet", eeg_channels=8, audio_channels=28, latent_dim=64, lags=[3, 6, 10, 13, 16], audio_model_type="standard", temporal_pooling=False):
        super().__init__()
        self.lags = lags
        self.temporal_pooling = temporal_pooling
        
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
        elif eeg_model_type.lower() == "eegnet_multiscale":
            self.eeg_encoder = MultiScaleEEGNet(in_channels=eeg_channels)
            self.eeg_encoder.output_proj = nn.Conv1d(16, latent_dim, kernel_size=1)
        else:
            raise ValueError(f"Unknown eeg_model_type: {eeg_model_type}")
            
        # 2. Audio Encoder
        num_lags = len(self.lags) if self.lags else 1
        if audio_model_type == "inception":
            self.audio_encoder = InceptionAudioEncoder(in_channels=audio_channels * num_lags, latent_dim=latent_dim)
        else:
            self.audio_encoder = AudioEncoder(in_channels=audio_channels * num_lags, latent_dim=latent_dim)
        
        self.latent_dim = latent_dim
        
        if self.temporal_pooling:
            self.eeg_pool = TemporalAttentionPooling(latent_dim)
            self.audio_pool = TemporalAttentionPooling(latent_dim)
            
        self.use_late_attention = False # Disabled by default for baseline compatibility
        self.late_attention = None

    def enable_late_attention(self):
        self.use_late_attention = True
        self.late_attention = nn.Conv1d(self.latent_dim, 1, kernel_size=1)
        self.late_attention.to(next(self.parameters()).device)
        
    def compute_similarity(self, z_eeg, z_audio):
        """
        Computes the cosine similarity scalar for each sequence in the batch.
        If late_attention is enabled, it weights the temporal similarities using audio-guided attention.
        Otherwise, it falls back to standard mean pooling over time.
        """
        sim_t = F.cosine_similarity(z_eeg, z_audio, dim=1) # [B, T]
        
        if self.use_late_attention and self.late_attention is not None:
            # Audio-guided attention weights
            attn_logits = self.late_attention(z_audio) # [B, 1, T]
            alpha = F.softmax(attn_logits, dim=2).squeeze(1) # [B, T]
            return torch.sum(sim_t * alpha, dim=1) # [B]
        else:
            return sim_t.mean(dim=1) # [B]

    def encode_eeg(self, eeg):
        """ Returns [B, latent_dim, Time] """
        x = self.eeg_encoder(eeg)
        if self.temporal_pooling:
            x = self.eeg_pool(x)
        return x
        
    def encode_audio(self, audio):
        """ Returns [B, latent_dim, Time] """
        if self.lags:
            audio = create_lagged_audio(audio, self.lags)
        x = self.audio_encoder(audio)
        if self.temporal_pooling:
            x = self.audio_pool(x)
        return x

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

def contrastive_loss(z_eeg, z_a, z_b, margin=0.1, model=None):
    """
    Computes a max-margin contrastive loss based on cosine similarity.
    We assume z_a is the attended audio, and z_b is the unattended audio.
    """
    if model is not None and hasattr(model, 'compute_similarity'):
        sim_a = model.compute_similarity(z_eeg, z_a)
        sim_b = model.compute_similarity(z_eeg, z_b)
    else:
        sim_a = F.cosine_similarity(z_eeg, z_a, dim=1).mean(dim=1)
        sim_b = F.cosine_similarity(z_eeg, z_b, dim=1).mean(dim=1)
    
    loss = torch.clamp(margin - sim_a + sim_b, min=0)
    return loss.mean(), sim_a.mean(), sim_b.mean()

def infonce_loss(z_eeg, z_a, z_b, temperature=0.1, model=None):
    """
    Computes an InfoNCE loss across the batch.
    For each EEG representation, the network must identify the correct audio (z_a[i])
    out of 2B candidates: all z_a and all z_b in the batch.
    
    z_eeg, z_a, z_b shape: [B, latent_dim, T]
    """
    if model is not None and hasattr(model, 'compute_similarity'):
        # For InfoNCE, we need the matrix of similarities between all B eegs and B audios
        # If compute_similarity computes per-sample (B), we need a custom matrix approach
        # For backward compatibility with the original logic, fallback to mean-over-time 
        # or require the model to compute a [B, B] matrix. 
        # Given the instruction, we maintain the original logic for BxB but use model if applicable.
        pass

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
