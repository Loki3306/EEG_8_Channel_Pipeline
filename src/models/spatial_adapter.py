import torch
import torch.nn as nn
import torch.nn.functional as F

class SpatialEEGAdapter(nn.Module):
    """
    Subject-Specific Linear Spatial Adapter for 8-Channel Near-Ear EEG.
    
    Architecture:
      Strictly C * C = 64 trainable scalar parameters.
      Initialized as Identity matrix I_8.
      Forward Pass: x_adapted = W * x
      
    Regularization:
      Penalizes Frobenius distance from Identity: ||W - I||_F^2
      Ensures that under ambiguous calibration data, the adapter smoothly
      reverts to the universal zero-shot baseline rather than distorting channels.
    """
    def __init__(self, channels: int = 8):
        super().__init__()
        self.channels = channels
        # 1x1 1D convolution across electrode channels
        self.proj = nn.Conv1d(channels, channels, kernel_size=1, bias=False)
        self.reset_to_identity()
        
    def reset_to_identity(self):
        """Initializes weights strictly to Identity matrix I_C."""
        with torch.no_grad():
            self.proj.weight.zero_()
            for i in range(self.channels):
                self.proj.weight[i, i, 0] = 1.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, C, T]
        Returns: [B, C, T] adapted EEG tensor
        """
        return self.proj(x)

    def identity_regularization_loss(self) -> torch.Tensor:
        """
        Computes Frobenius norm distance from Identity matrix:
        L_reg = ||W - I_C||_F^2
        """
        w = self.proj.weight.squeeze(-1)  # [C, C]
        eye = torch.eye(self.channels, device=w.device, dtype=w.dtype)
        return torch.sum((w - eye) ** 2)

    def get_weight_matrix(self) -> torch.Tensor:
        """Returns the [C, C] spatial projection matrix."""
        return self.proj.weight.squeeze(-1).detach().cpu()
