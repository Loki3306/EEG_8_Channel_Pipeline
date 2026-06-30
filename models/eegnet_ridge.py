import torch
import torch.nn as nn
from models.eegnet import EEGNet

class ResidualEEGNetRidge(nn.Module):
    """
    Hybrid Teacher-Student Model for AAD.
    Uses an analytical (frozen) Ridge regression mapping as a base prediction,
    and trains a deep neural network (EEGNet) to model the residual error.
    
    Target: true envelope (not Ridge imitation).
    """
    def __init__(self, ridge_feature_count, in_channels=64):
        super().__init__()
        
        # Base Linear Ridge Branch
        # Takes lagged EEG of shape [Batch, Time, ridge_feature_count] and outputs [Batch, Time, 1]
        # Bias is False because the analytical Ridge baseline does not use an intercept (data is standardized).
        self.base_ridge = nn.Linear(ridge_feature_count, 1, bias=False)
        for p in self.base_ridge.parameters():
            p.requires_grad = False
            
        # Non-linear Residual Branch
        self.eegnet = EEGNet(in_channels=in_channels, F1=8, D=2, F2=16, kernel_length=64)
        
        # Learnable scaling for the residual to ensure it starts small and doesn't destabilize Ridge
        self.raw_alpha = nn.Parameter(torch.tensor(-3.0)) # sigmoid(-3) is approx 0.05
        
    def forward(self, eeg_raw, eeg_lagged):
        """
        eeg_raw: [Batch, Channels, Time] (for EEGNet)
        eeg_lagged: [Batch, Time, Features] (for Ridge)
        
        Returns: [Batch, Time] predicted envelope, strictly aligned
        """
        # 1. Base Ridge prediction
        # Linear expects [..., in_features], outputs [Batch, Time, 1]
        base_pred = self.base_ridge(eeg_lagged).squeeze(-1)
        
        # 2. Non-linear Residual
        # EEGNet returns [Batch, 1, Time]
        residual_pred = self.eegnet(eeg_raw).squeeze(1)
        
        # Temporal Alignment: Both branches operate on the full Time length.
        # `lagged_eeg_matrix` uses zero-padding, so it does not discard samples.
        # Thus, `base_pred` and `residual_pred` are perfectly aligned at t=0.
        
        # Match minimum lengths in case of off-by-one discrepancies due to convolutions
        min_len = min(base_pred.size(1), residual_pred.size(1))
        
        base_pred = base_pred[:, :min_len]
        residual_pred = residual_pred[:, :min_len]
        
        # 3. Hybrid Output
        # Constrain alpha between 0 and 0.5 to prevent it from overwhelming the base predictor
        alpha = torch.sigmoid(self.raw_alpha) * 0.5
        return base_pred + alpha * residual_pred

    def load_ridge_weights(self, weights):
        """
        Injects analytical ridge weights into the frozen linear branch.
        weights: numpy array of shape (ridge_feature_count,)
        """
        with torch.no_grad():
            self.base_ridge.weight.copy_(torch.FloatTensor(weights).unsqueeze(0))
