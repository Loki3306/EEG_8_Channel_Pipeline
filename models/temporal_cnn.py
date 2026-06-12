from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def _make_norm(norm_type: str, channels: int) -> nn.Module:
    if norm_type == "batch":
        return nn.BatchNorm1d(channels)
    if norm_type == "layer":
        return nn.GroupNorm(1, channels)
    if norm_type == "none":
        return nn.Identity()
    raise ValueError(f"Unsupported norm_type: {norm_type}")


class ResidualTemporalBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        *,
        dilation: int,
        dropout: float = 0.1,
        norm_type: str = "layer",
    ) -> None:
        super().__init__()
        padding = dilation
        self.block = nn.Sequential(
            _make_norm(norm_type, channels),
            nn.GELU(),
            nn.Conv1d(channels, channels, kernel_size=3, dilation=dilation, padding=padding),
            _make_norm(norm_type, channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size=3, dilation=dilation, padding=padding),
        )
        self.skip = nn.Identity()
        self.out_norm = _make_norm(norm_type, channels)
        self.out_act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        x = self.block(x)
        x = x + residual
        return self.out_act(self.out_norm(x))


class TemporalCNNAAD(nn.Module):
    def __init__(
        self,
        in_channels: int = 2,
        stem_channels: int = 32,
        hidden_channels: int = 64,
        dropout: float = 0.1,
        norm_type: str = "layer",
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, stem_channels, kernel_size=5, padding=2),
            _make_norm(norm_type, stem_channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(stem_channels, hidden_channels, kernel_size=1),
        )
        # multi-kernel parallel convs to increase multi-resolution receptive field
        self.multires_kernels = [3, 7, 15]
        num_branches = len(self.multires_kernels)
        channels_per_branch = (hidden_channels + num_branches - 1) // num_branches  # ceiling division
        self.multires_convs = nn.ModuleList(
            [nn.Conv1d(hidden_channels, channels_per_branch, kernel_size=k, padding=k // 2) for k in self.multires_kernels]
        )
        self.multires_proj = nn.Conv1d(channels_per_branch * num_branches, hidden_channels, kernel_size=1)

        self.block1 = ResidualTemporalBlock(hidden_channels, dilation=2, dropout=dropout, norm_type=norm_type)
        self.block2 = ResidualTemporalBlock(hidden_channels, dilation=4, dropout=dropout, norm_type=norm_type)
        self.skip_projection = nn.Conv1d(hidden_channels, hidden_channels, kernel_size=1)
        self.head = nn.Sequential(
            _make_norm(norm_type, hidden_channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_channels, 1, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected 3D input, got shape {tuple(x.shape)}")

        if x.shape[-1] == self.in_channels:
            x = x.transpose(1, 2)
        elif x.shape[1] != self.in_channels:
            raise ValueError(
                f"Expected input with {self.in_channels} channels in either the last or second dimension, got shape {tuple(x.shape)}"
            )

        stem = self.stem(x)
        # multiresolution branch
        branches = [conv(stem) for conv in self.multires_convs]
        multi = torch.cat(branches, dim=1)
        multi = self.multires_proj(multi)

        block1 = self.block1(multi)
        block2 = self.block2(block1)
        combined = block2 + self.skip_projection(multi)
        output = self.head(combined)
        return output.squeeze(1)


class VLAAILiteAAD(nn.Module):
    def __init__(
        self,
        in_channels: int = 8,
        virtual_channels: int = 16,
        hidden_channels: int = 32,
        dropout: float = 0.2,
        norm_type: str = "layer",
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        
        self.spatial_projection = nn.Sequential(
            nn.Conv1d(in_channels, virtual_channels, kernel_size=1),
            _make_norm(norm_type, virtual_channels),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        self.temporal_block = nn.Sequential(
            nn.Conv1d(virtual_channels, hidden_channels, kernel_size=9, padding=4, groups=virtual_channels),
            _make_norm(norm_type, hidden_channels),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        self.context_block1 = nn.Sequential(
            nn.Conv1d(hidden_channels, hidden_channels, kernel_size=5, padding=4, dilation=2),
            _make_norm(norm_type, hidden_channels),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        self.context_block2 = nn.Sequential(
            nn.Conv1d(hidden_channels, hidden_channels, kernel_size=5, padding=8, dilation=4),
            _make_norm(norm_type, hidden_channels),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        self.output_context = nn.Sequential(
            nn.Conv1d(hidden_channels, hidden_channels, kernel_size=1),
            _make_norm(norm_type, hidden_channels),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        self.head = nn.Conv1d(hidden_channels, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected 3D input, got shape {tuple(x.shape)}")

        if x.shape[-1] == self.in_channels:
            x = x.transpose(1, 2)
        elif x.shape[1] != self.in_channels:
            raise ValueError(
                f"Expected input with {self.in_channels} channels in either the last or second dimension, got shape {tuple(x.shape)}"
            )

        x = self.spatial_projection(x)
        x = self.temporal_block(x)
        
        res1 = x
        x = self.context_block1(x)
        x = x + res1
        
        res2 = x
        x = self.context_block2(x)
        x = x + res2
        
        x = self.output_context(x)
        output = self.head(x)
        return output.squeeze(1)


class TemporalEmbeddingEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        *,
        stem_channels: int = 24,
        hidden_channels: int = 48,
        embedding_dim: int = 48,
        dropout: float = 0.1,
        norm_type: str = "layer",
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, stem_channels, kernel_size=5, padding=2),
            _make_norm(norm_type, stem_channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(stem_channels, hidden_channels, kernel_size=1),
        )
        self.block1 = ResidualTemporalBlock(hidden_channels, dilation=2, dropout=dropout, norm_type=norm_type)
        self.block2 = ResidualTemporalBlock(hidden_channels, dilation=4, dropout=dropout, norm_type=norm_type)
        self.pool_norm = nn.LayerNorm(hidden_channels)
        self.projection = nn.Sequential(
            nn.Linear(hidden_channels, embedding_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, embedding_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected 3D input, got shape {tuple(x.shape)}")

        if x.shape[-1] == self.in_channels:
            x = x.transpose(1, 2)
        elif x.shape[1] != self.in_channels:
            raise ValueError(
                f"Expected input with {self.in_channels} channels in either the last or second dimension, got shape {tuple(x.shape)}"
            )

        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = x.mean(dim=-1)
        x = self.pool_norm(x)
        x = self.projection(x)
        return F.normalize(x, dim=-1)


class TemporalContrastiveAAD(nn.Module):
    def __init__(
        self,
        eeg_channels: int = 2,
        audio_channels: int = 1,
        *,
        embedding_dim: int = 48,
        dropout: float = 0.1,
        norm_type: str = "layer",
    ) -> None:
        super().__init__()
        self.eeg_encoder = TemporalEmbeddingEncoder(
            eeg_channels,
            embedding_dim=embedding_dim,
            dropout=dropout,
            norm_type=norm_type,
        )
        self.audio_encoder = TemporalEmbeddingEncoder(
            audio_channels,
            embedding_dim=embedding_dim,
            dropout=dropout,
            norm_type=norm_type,
        )

    def encode_eeg(self, x: torch.Tensor) -> torch.Tensor:
        return self.eeg_encoder(x)

    def encode_audio(self, x: torch.Tensor) -> torch.Tensor:
        return self.audio_encoder(x)

    def forward(self, eeg: torch.Tensor, audio: torch.Tensor | None = None) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        eeg_embedding = self.encode_eeg(eeg)
        if audio is None:
            return eeg_embedding
        return eeg_embedding, self.encode_audio(audio)


def cosine_similarity_matrix(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    left = F.normalize(left, dim=-1)
    right = F.normalize(right, dim=-1)
    return left @ right.transpose(0, 1)


def info_nce_loss(
    eeg_embedding: torch.Tensor,
    positive_audio_embedding: torch.Tensor,
    negative_audio_embedding: torch.Tensor,
    *,
    temperature: float = 0.1,
) -> torch.Tensor:
    audio_pool = torch.cat([positive_audio_embedding, negative_audio_embedding], dim=0)
    logits = cosine_similarity_matrix(eeg_embedding, audio_pool) / max(float(temperature), 1e-6)
    labels = torch.arange(eeg_embedding.shape[0], device=eeg_embedding.device)
    return F.cross_entropy(logits, labels)


def pearson_corr(prediction: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    if prediction.ndim == 1:
        prediction = prediction.unsqueeze(0)
    if target.ndim == 1:
        target = target.unsqueeze(0)

    prediction = prediction - prediction.mean(dim=-1, keepdim=True)
    target = target - target.mean(dim=-1, keepdim=True)

    numerator = (prediction * target).sum(dim=-1)
    denominator = torch.sqrt((prediction.square().sum(dim=-1) * target.square().sum(dim=-1)).clamp_min(eps))
    return numerator / denominator


def correlation_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return -pearson_corr(prediction, target).mean()


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
