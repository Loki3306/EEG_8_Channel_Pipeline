import torch
import torch.nn as nn
import torch.nn.functional as F

class Conv2dWithConstraint(nn.Conv2d):
    def __init__(self, *args, max_norm=1, **kwargs):
        self.max_norm = max_norm
        super().__init__(*args, **kwargs)

    def forward(self, x):
        self.weight.data = torch.renorm(self.weight.data, p=2, dim=0, maxnorm=self.max_norm)
        return super().forward(x)

class InceptionBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_sizes, in_eeg_channels):
        super().__init__()
        self.branches = nn.ModuleList()
        for k in kernel_sizes:
            branch = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, (1, k), padding=(0, k//2), bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ELU(),
                nn.Dropout(0.25),
                Conv2dWithConstraint(out_channels, out_channels * 2, (in_eeg_channels, 1), 
                                     groups=out_channels, bias=False, max_norm=1.0),
                nn.BatchNorm2d(out_channels * 2),
                nn.ELU(),
                nn.Dropout(0.25)
            )
            self.branches.append(branch)
            
    def forward(self, x):
        outputs = [branch(x) for branch in self.branches]
        return torch.cat(outputs, dim=1) # Concat along channel dim

class EEGInception(nn.Module):
    """
    EEG-Inception adapted for continuous auditory envelope reconstruction (regression).
    Utilizes multi-scale temporal filters to capture varying temporal dynamics.
    """
    def __init__(self, in_channels=8, scales=3, filters_per_scale=8, dropout_rate=0.25):
        super().__init__()
        
        # Temporal kernels: e.g., 31, 15, 7 (to preserve sequence length we need odd kernels)
        kernel_sizes = [31, 15, 7][:scales]
        
        self.inception1 = InceptionBlock(1, filters_per_scale, kernel_sizes, in_channels)
        
        out_c1 = filters_per_scale * 2 * len(kernel_sizes)
        
        self.inception2 = nn.Sequential(
            nn.Conv2d(out_c1, out_c1 // 2, (1, 7), padding=(0, 3), bias=False),
            nn.BatchNorm2d(out_c1 // 2),
            nn.ELU(),
            nn.Dropout(dropout_rate)
        )
        
        self.inception3 = nn.Sequential(
            nn.Conv2d(out_c1 // 2, out_c1 // 4, (1, 3), padding=(0, 1), bias=False),
            nn.BatchNorm2d(out_c1 // 4),
            nn.ELU(),
            nn.Dropout(dropout_rate)
        )
        
        out_final = out_c1 // 4
        
        self.output_proj = nn.Conv1d(out_final, 1, kernel_size=1)
        
    def forward(self, x):
        # x: [B, Channels, Time]
        # Reshape to [B, 1, Channels, Time] for Conv2d
        x = x.unsqueeze(1)
        
        x = self.inception1(x)
        x = self.inception2(x)
        x = self.inception3(x)
        
        # x: [B, out_final, 1, Time]
        x = x.squeeze(2) # [B, out_final, Time]
        
        x = self.output_proj(x) # [B, 1, Time]
        
        return x
