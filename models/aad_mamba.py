import torch
import torch.nn as nn
import torch.nn.functional as F

# -------------------------------------------------------------------------
# PURE PYTORCH MAMBA IMPLEMENTATION (Minimal)
# -------------------------------------------------------------------------
class MambaBlock(nn.Module):
    """
    A minimal, pure PyTorch implementation of the Mamba (State Space Model) block.
    This avoids CUDA compilation issues while preserving the exact mathematical formulation
    of the S4/Mamba selective state space mechanism.
    """
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)

        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)

        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=True,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
        )

        self.x_proj = nn.Linear(self.d_inner, self.d_state * 2 + 1, bias=False)
        self.dt_proj = nn.Linear(1, self.d_inner, bias=True)
        
        A = torch.arange(1, self.d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))

        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, u):
        # u: [B, L, D]
        batch, seq_len, _ = u.shape

        xz = self.in_proj(u) # [B, L, 2 * d_inner]
        x, z = xz.chunk(2, dim=-1)

        x = x.transpose(1, 2)
        x = self.conv1d(x)[:, :, :seq_len]
        x = x.transpose(1, 2)
        x = F.silu(x)

        x_dbl = self.x_proj(x)
        dt, B, C = torch.split(x_dbl, [1, self.d_state, self.d_state], dim=-1)

        dt = F.softplus(self.dt_proj(dt)) # [B, L, d_inner]
        
        A = -torch.exp(self.A_log.float()) # [d_inner, d_state]

        # Discretize
        dA = torch.exp(torch.einsum("bld,dn->bldn", dt, A))
        dB_x = torch.einsum("bld,bld,bln->bldn", dt, x, B)

        # Sequential Scan (Pure PyTorch)
        y = []
        h = torch.zeros((batch, self.d_inner, self.d_state), device=u.device, dtype=u.dtype)
        
        for i in range(seq_len):
            h = dA[:, i] * h + dB_x[:, i]
            y.append(torch.einsum("bdn,bn->bd", h, C[:, i]))
            
        y = torch.stack(y, dim=1) # [B, L, d_inner]
        
        y = y + x * self.D
        y = y * F.silu(z)

        out = self.out_proj(y)
        return out


# -------------------------------------------------------------------------
# MAMBA AAD ARCHITECTURE
# -------------------------------------------------------------------------
class MambaEncoder(nn.Module):
    def __init__(self, in_channels, d_model=64, num_layers=2):
        super().__init__()
        self.stem = nn.Linear(in_channels, d_model)
        self.mamba_layers = nn.ModuleList([
            MambaBlock(d_model=d_model, d_state=16, d_conv=4, expand=2)
            for _ in range(num_layers)
        ])
        
    def forward(self, x):
        # x: [B, L, C]
        h = F.gelu(self.stem(x))
        for layer in self.mamba_layers:
            # Mamba has a residual connection
            h = layer(h) + h
        return h

class ContinuousMambaAADModel(nn.Module):
    """
    A continuous State Space Model for AAD.
    Unlike previous models that chunk 2s overlapping windows, this model processes
    the entire 3.5s sequence (448 samples) continuously using Mamba's infinite receptive field.
    """
    def __init__(self, eeg_channels=8, audio_channels=16, d_model=64):
        super().__init__()
        
        self.eeg_encoder = MambaEncoder(in_channels=eeg_channels, d_model=d_model, num_layers=2)
        # Shared audio encoder to ensure symmetric latent space
        self.audio_encoder = MambaEncoder(in_channels=audio_channels, d_model=d_model, num_layers=2)
        
        # Integrator: Takes the similarity scores and processes them temporally
        self.integrator = MambaBlock(d_model=2, d_state=16, d_conv=4, expand=2)
        
        # Final classifier
        self.classifier = nn.Linear(2, 1)

    def forward(self, eeg_seq, audio_a_seq, audio_b_seq):
        # Inputs: [B, L, C]
        
        # 1. Project to continuous latent manifolds
        z_eeg = self.eeg_encoder(eeg_seq)      # [B, L, 64]
        z_a = self.audio_encoder(audio_a_seq)  # [B, L, 64]
        z_b = self.audio_encoder(audio_b_seq)  # [B, L, 64]
        
        # 2. Continuous Cosine Similarity (Sample-by-sample)
        z_eeg_norm = F.normalize(z_eeg, p=2, dim=-1)
        z_a_norm = F.normalize(z_a, p=2, dim=-1)
        z_b_norm = F.normalize(z_b, p=2, dim=-1)
        
        sim_a = (z_eeg_norm * z_a_norm).sum(dim=-1, keepdim=True) # [B, L, 1]
        sim_b = (z_eeg_norm * z_b_norm).sum(dim=-1, keepdim=True) # [B, L, 1]
        
        # [B, L, 2]
        sim_features = torch.cat([sim_a, sim_b], dim=-1)
        
        # 3. Temporal Integration via Mamba
        # Mamba accumulates the evidence over the 448 samples continuously
        integrated = self.integrator(sim_features) # [B, L, 2]
        
        # 4. Final Classification (take the last state, as it holds the accumulated evidence)
        last_state = integrated[:, -1, :] # [B, 2]
        logits = self.classifier(last_state) # [B, 1]
        
        return logits.squeeze(-1)
