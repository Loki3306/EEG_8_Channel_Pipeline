import torch
import torch.nn as nn
import torch.nn.functional as F
from models.eegnet import EEGNet
from models.atcnet import ATCNet
from models.eegnet_tcn import EEGNetTCN
from models.eegnet_multiscale import EEGNetMultiScaleM2

class ChannelLayerNorm(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, x):
        # [B, C, T] -> [B, T, C]
        x = x.transpose(1, 2)
        x = self.norm(x)
        return x.transpose(1, 2)
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

def _compute_mean_cosine_similarity(z_1, z_2):
    """Normalized cosine similarity averaged over the temporal dimension (dim=2)."""
    # F.cosine_similarity(dim=1) yields [B, T], which we average over time
    sim = F.cosine_similarity(z_1, z_2, dim=1)
    return sim.mean(dim=1)

class TemporalDeformationField(nn.Module):
    """
    1D Spatial Transformer Network to biologically align EEG and Audio.
    Learns a local temporal deformation field delta_t in [-1, 1].
    """
    def __init__(self, channels=64, max_shift_ms=500, fs=64.0):
        super().__init__()
        self.max_shift_seconds = max_shift_ms / 1000.0
        self.fs = fs
        
        # Lightweight CNN to predict delay field
        # Removed groups=channels to allow full cross-channel mixing
        self.align_net = nn.Sequential(
            nn.Conv1d(channels * 2, channels // 2, kernel_size=15, padding=7),
            nn.BatchNorm1d(channels // 2),
            nn.GELU(),
            nn.Conv1d(channels // 2, 1, kernel_size=1)
        )
        
        # Initialize the final layer to predict 0 shift at the start of training
        nn.init.zeros_(self.align_net[-1].weight)
        nn.init.zeros_(self.align_net[-1].bias)
        
    def forward(self, z_eeg, z_audio):
        B, C, T = z_eeg.shape
        
        # 1. Predict delay field in physical seconds
        concat_feats = torch.cat([z_eeg, z_audio], dim=1) # [B, 2C, T]
        raw_delta = self.align_net(concat_feats) # [B, 1, T]
        
        # Explicit unit conversion
        delta_seconds = self.max_shift_seconds * torch.tanh(raw_delta) # Bounded to [-max, +max]
        delta_samples = delta_seconds * self.fs
        
        # Convert delta to normalized grid coordinate shift
        # 2 / (T - 1) is the width of 1 sample in normalized coordinates [-1, 1]
        sample_width = 2.0 / (T - 1)
        normalized_shift = delta_samples * sample_width
        
        # 2. Create base grid
        base_grid = torch.linspace(-1, 1, T, device=z_eeg.device)
        base_grid = base_grid.view(1, 1, T).expand(B, 1, T)
        
        # Add shift to base grid
        shifted_grid_x = base_grid + normalized_shift # [B, 1, T]
        shifted_grid_y = torch.zeros_like(shifted_grid_x)
        
        # Stack into grid: [B, 1, T, 2]
        grid = torch.stack([shifted_grid_x, shifted_grid_y], dim=-1)
        
        # 3. Warp Audio
        z_audio_4d = z_audio.unsqueeze(2) # [B, C, 1, T]
        z_audio_warped = F.grid_sample(
            z_audio_4d, 
            grid, 
            mode='bilinear', 
            padding_mode='border', 
            align_corners=True
        )
        z_audio_warped = z_audio_warped.squeeze(2) # [B, C, T]
        
        return z_audio_warped, delta_seconds

class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.lambda_
        return output, None

class GradientReversalLayer(nn.Module):
    def __init__(self, lambda_=1.0):
        super().__init__()
        self.lambda_ = lambda_

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_)

class ContrastiveMatchNet(nn.Module):
    """
    A Siamese network that explicitly learns a matching function between EEG and Audio.
    """
    def __init__(self, eeg_model_type="eegnet", eeg_channels=8, audio_channels=28, latent_dim=64, num_subjects=17, use_temporal_transport=False, audio_model_type="gammatone"):
        super().__init__()
        
        self.use_temporal_transport = use_temporal_transport
        if self.use_temporal_transport:
            self.tdf = TemporalDeformationField(channels=latent_dim)
        
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
        elif eeg_model_type.lower() == "eegnet_s1":
            from models.eegnet_s1 import EEGNetS1
            self.eeg_encoder = EEGNetS1(in_channels=eeg_channels)
            # Override the final projection
            self.eeg_encoder.output_proj = nn.Conv1d(16, latent_dim, kernel_size=1)
        elif eeg_model_type.lower() == "eegnet_s2":
            from models.eegnet_s2 import EEGNetS2
            self.eeg_encoder = EEGNetS2(in_channels=eeg_channels)
            # Override the final projection
            self.eeg_encoder.output_proj = nn.Conv1d(16, latent_dim, kernel_size=1)
        elif eeg_model_type.lower() == "eegnet_multiscale_m2":
            from models.eegnet_multiscale import EEGNetMultiScaleM2
            self.eeg_encoder = EEGNetMultiScaleM2(in_channels=eeg_channels)
            self.eeg_encoder.output_proj = nn.Conv1d(16, latent_dim, kernel_size=1)
        elif eeg_model_type.lower() == "sincalignnet":
            from models.sincalignnet import SincAlignNet
            self.eeg_encoder = SincAlignNet(in_channels=eeg_channels)
            self.eeg_encoder.output_proj = nn.Conv1d(16, latent_dim, kernel_size=1)
        elif eeg_model_type.lower() == "msca":
            from models.msca_eeg_encoder import MSCA_EEGEncoder
            self.eeg_encoder = MSCA_EEGEncoder(in_channels=eeg_channels)
            self.eeg_encoder.output_proj = nn.Conv1d(64, latent_dim, kernel_size=1)
        else:
            raise ValueError(f"Unknown eeg_model_type: {eeg_model_type}")
            
        # 2. Audio Encoder
        if audio_model_type.lower() == "gammatone":
            self.audio_encoder = AudioEncoder(in_channels=audio_channels, latent_dim=latent_dim)
        elif audio_model_type.lower() == "wavlm":
            self.audio_encoder = nn.Sequential(
                nn.Conv1d(audio_channels, 256, kernel_size=1),
                ChannelLayerNorm(256),
                nn.GELU(),
                nn.Dropout(0.3),
                nn.Conv1d(256, latent_dim, kernel_size=1),
                ChannelLayerNorm(latent_dim)
            )
        else:
            raise ValueError(f"Unknown audio_model_type: {audio_model_type}")
        
        # 3. Domain Adversarial Components
        self.grl = GradientReversalLayer(lambda_=0.0)
            
        self.subject_classifier = nn.Sequential(
            nn.Linear(latent_dim, latent_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(latent_dim // 2, num_subjects)
        )
        
        self.latent_dim = latent_dim

    def encode_eeg(self, eeg):
        """ Returns [B, latent_dim, Time] """
        return self.eeg_encoder(eeg)
        
    def encode_audio(self, audio):
        """ Returns [B, latent_dim, Time] """
        return self.audio_encoder(audio)

    def forward(self, eeg, audio_a, audio_b=None, return_subject_logits=False, return_deltas=False):
        """
        Forward pass for training.
        eeg: [B, C, T]
        audio_a: [B, 28, T]
        audio_b: [B, 28, T]
        
        Returns latent representations (retaining simple 3-tensor signature for backward compatibility).
        """
        z_eeg = self.encode_eeg(eeg)
        z_a = self.encode_audio(audio_a)
        z_b = self.encode_audio(audio_b) if audio_b is not None else None
        
        delta_t_a, delta_t_b = None, None
        if hasattr(self, 'use_temporal_transport') and self.use_temporal_transport:
            z_a, delta_t_a = self.tdf(z_eeg, z_a)
            if z_b is not None:
                z_b, delta_t_b = self.tdf(z_eeg, z_b)
                
        # Return Tuple formatting
        out = (z_eeg, z_a, z_b)
        
        if return_subject_logits:
            z_pool = z_eeg.mean(dim=-1)
            subj_feat = self.grl(z_pool)
            subj_logits = self.subject_classifier(subj_feat)
            out = out + (subj_logits,)
        if return_deltas:
            out = out + ((delta_t_a, delta_t_b),)
            
        return out

    def compute_similarities(self, z_eeg, z_a, z_b):
        """Shared similarity computation path."""
        sim_a = _compute_mean_cosine_similarity(z_eeg, z_a)
        sim_b = _compute_mean_cosine_similarity(z_eeg, z_b)
        return sim_a, sim_b

    def get_confidence_from_latents(self, z_eeg, z_a, z_b, temperature=1.0):
        """
        Computes a confidence score based on the decision margin.
        This score is a probability proxy scaled via temperature, but is uncalibrated.
        """
        sim_a, sim_b = self.compute_similarities(z_eeg, z_a, z_b)
        margin = sim_a - sim_b
        confidence = torch.sigmoid(margin / temperature)
        return confidence

    def get_confidence(self, eeg, audio_a, audio_b, temperature=1.0):
        """
        Computes a confidence score based on the difference in cosine similarity 
        between z_eeg and the two audio candidates.
        """
        z_eeg, z_a, z_b = self(eeg, audio_a, audio_b)
        return self.get_confidence_from_latents(z_eeg, z_a, z_b, temperature=temperature)

def contrastive_loss(z_eeg, z_a, z_b, margin=0.1):
    """
    Computes a max-margin contrastive loss based on cosine similarity.
    We assume z_a is the attended audio, and z_b is the unattended audio.
    """
    sim_a_mean = _compute_mean_cosine_similarity(z_eeg, z_a)
    sim_b_mean = _compute_mean_cosine_similarity(z_eeg, z_b)
    
    loss = F.relu(margin - (sim_a_mean - sim_b_mean)).mean()
    
    return loss, sim_a_mean.mean(), sim_b_mean.mean()

def anchored_contrastive_loss(z_eeg, z_a, z_b, margin=0.1, lambda_align=0.5, align_target=0.1):
    """
    Computes a max-margin contrastive loss while explicitly anchoring r_A to be positive.
    Penalizes if sim_a_mean falls below align_target.
    """
    sim_a_mean = _compute_mean_cosine_similarity(z_eeg, z_a)
    sim_b_mean = _compute_mean_cosine_similarity(z_eeg, z_b)
    
    l_margin = F.relu(margin - (sim_a_mean - sim_b_mean)).mean()
    l_align = F.relu(align_target - sim_a_mean).mean()
    
    loss = l_margin + lambda_align * l_align
    
    return loss, sim_a_mean.mean(), sim_b_mean.mean()

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

def dcca_loss(H1, H2, lambda_decorr=0.1, eps=1e-9):
    """
    Computes a Coordinate-Aligned Deep Canonical Correlation Analysis (DCCA) loss.
    This is mathematically similar to Barlow Twins/VICReg, ensuring that the 
    representations are not just correlated under some arbitrary linear projection,
    but explicitly aligned coordinate-by-coordinate. This matches the evaluation
    metric which computes coordinate-wise Pearson correlation.
    
    H1, H2: tensors of shape [Batch * T, Features]
    """
    # 1. Mean center the batches
    H1_mean = H1 - H1.mean(dim=0, keepdim=True)
    H2_mean = H2 - H2.mean(dim=0, keepdim=True)
    
    # 2. Normalize by standard deviation (variance = 1)
    H1_std = torch.sqrt(H1_mean.var(dim=0) + eps)
    H2_std = torch.sqrt(H2_mean.var(dim=0) + eps)
    
    Z1 = H1_mean / H1_std
    Z2 = H2_mean / H2_std
    
    batch_size = Z1.size(0)
    
    # 3. Compute cross-correlation matrix (should be Identity)
    C = (Z1.t() @ Z2) / (batch_size - 1)
    
    # 4. Maximize correlation on the diagonal (Coordinate-wise alignment)
    diag_corr = torch.diagonal(C)
    invariance_loss = -diag_corr.mean()
    
    # 5. Decorrelate off-diagonal elements (Prevent representational collapse)
    # We penalize the off-diagonal elements of the auto-correlation matrices
    C1 = (Z1.t() @ Z1) / (batch_size - 1)
    C2 = (Z2.t() @ Z2) / (batch_size - 1)
    
    mask = ~torch.eye(C1.size(0), dtype=torch.bool, device=C1.device)
    decorr_loss = (C1[mask] ** 2).mean() + (C2[mask] ** 2).mean()
    
    loss = invariance_loss + lambda_decorr * decorr_loss
    
    return loss, diag_corr.mean(), diag_corr.mean()

if __name__ == "__main__":
    model = ContrastiveMatchNet("eegnet")
    eeg = torch.randn(16, 8, 320)
    audio_a = torch.randn(16, 28, 320)
    audio_b = torch.randn(16, 28, 320)
    
    z_eeg, z_a, z_b = model(eeg, audio_a, audio_b)
    print("Z_eeg:", z_eeg.shape)
    loss, sa, sb = infonce_loss(z_eeg, z_a, z_b)
    print(f"InfoNCE Loss: {loss.item():.4f} | Sim A: {sa:.4f} | Sim B: {sb:.4f}")
