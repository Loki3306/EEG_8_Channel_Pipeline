import torch
import torch.nn as nn
import torch.nn.functional as F
from .aad_tcn import LocalEncoder, FastAudioEncoder, SlowAudioEncoder

class PureCosineAADModel(nn.Module):
    """
    A pure cosine similarity AAD Model.
    No TCN, no temporal modeling beyond the window, no sequence fusion.
    Forces the network to explicitly map EEG and Audio into the exact same latent space.
    """
    def __init__(self, eeg_channels=8, latent_dim=64, audio_channels=1, encoder_type='baseline'):
        super().__init__()
        
        self.eeg_encoder = LocalEncoder(in_channels=eeg_channels, out_dim=latent_dim)
        
        if encoder_type == 'fast':
            self.audio_encoder = FastAudioEncoder(in_channels=audio_channels, out_dim=latent_dim)
        elif encoder_type == 'slow':
            self.audio_encoder = SlowAudioEncoder(in_channels=audio_channels, out_dim=latent_dim)
        else:
            self.audio_encoder = LocalEncoder(in_channels=audio_channels, out_dim=latent_dim)

        # Scale parameter to allow the network to scale the [-1, 1] cosine similarity to a wider range
        # for BCEWithLogitsLoss. Initialized to 10.0 (like in CLIP)
        self.logit_scale = nn.Parameter(torch.ones([]) * 10.0)
        self.bias = nn.Parameter(torch.zeros([]))

    def forward(self, eeg_seq, aud_seq):
        """
        eeg_seq: [B, SeqLen, EEG_Channels, Time]
        aud_seq: [B, SeqLen, Audio_Channels, Time]
        """
        B, SeqLen, C_e, T_e = eeg_seq.shape
        eeg_flat = eeg_seq.reshape(B * SeqLen, C_e, T_e)
        
        C_a, T_a = aud_seq.shape[2], aud_seq.shape[3]
        aud_flat = aud_seq.reshape(B * SeqLen, C_a, T_a)
        
        eeg_latent = self.eeg_encoder(eeg_flat)
        p_eeg = F.normalize(eeg_latent, dim=-1) # [B*SeqLen, latent_dim]
        
        aud_latent = self.audio_encoder(aud_flat)
        p_a = F.normalize(aud_latent, dim=-1) # [B*SeqLen, latent_dim]
        
        # Pure Cosine Similarity
        score = F.cosine_similarity(p_eeg, p_a, dim=-1) # [B*SeqLen]
        
        # Scale for BCE
        logits = score * self.logit_scale + self.bias
        
        # Average pooling over the sequence to produce one prediction per 3.5s trial
        logits = logits.reshape(B, SeqLen).mean(dim=1)
        
        return logits, None
