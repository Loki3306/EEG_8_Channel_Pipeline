import torch
import torch.nn as nn
import torch.nn.functional as F

class EEGEncoder(nn.Module):
    """
    Encodes 60-channel EEG into a D-dimensional latent space using EEGNet-style blocks.
    """
    def __init__(self, in_channels=60, samples=256, F1=8, D=2, F2=16, temporal_kernel=64, out_dim=16):
        super(EEGEncoder, self).__init__()
        
        # Block 1
        self.temporal_conv = nn.Conv2d(1, F1, kernel_size=(1, temporal_kernel), padding=(0, temporal_kernel//2), bias=False)
        self.bn1 = nn.BatchNorm2d(F1)
        self.spatial_conv = nn.Conv2d(F1, F1 * D, kernel_size=(in_channels, 1), groups=F1, bias=False)
        self.bn2 = nn.BatchNorm2d(F1 * D)
        self.elu = nn.ELU()
        self.avg_pool1 = nn.AvgPool2d(kernel_size=(1, 4))
        self.dropout1 = nn.Dropout(0.50)
        
        # Block 2
        self.separable_depth = nn.Conv2d(F1 * D, F1 * D, kernel_size=(1, 16), groups=F1 * D, padding=(0, 8), bias=False)
        self.separable_point = nn.Conv2d(F1 * D, F2, kernel_size=(1, 1), bias=False)
        self.bn3 = nn.BatchNorm2d(F2)
        self.avg_pool2 = nn.AvgPool2d(kernel_size=(1, 8))
        self.dropout2 = nn.Dropout(0.50)
        
        out_time = samples // 32
        flatten_size = F2 * out_time
        
        self.fc = nn.Linear(flatten_size, out_dim)

    def forward(self, x):
        x = x.unsqueeze(1) # [B, 1, C, T]
        x = self.dropout1(self.avg_pool1(self.elu(self.bn2(self.spatial_conv(self.bn1(self.temporal_conv(x)))))))
        x = self.dropout2(self.avg_pool2(self.elu(self.bn3(self.separable_point(self.separable_depth(x))))))
        x = x.view(x.size(0), -1)
        return self.fc(x)

class AudioEncoder(nn.Module):
    """
    Encodes 1-channel Audio Envelope into the same D-dimensional latent space.
    """
    def __init__(self, samples=256, F1=8, F2=16, temporal_kernel=64, out_dim=16):
        super(AudioEncoder, self).__init__()
        
        self.conv1 = nn.Conv2d(1, F1, kernel_size=(1, temporal_kernel), padding=(0, temporal_kernel//2), bias=False)
        self.bn1 = nn.BatchNorm2d(F1)
        self.elu1 = nn.ELU()
        self.pool1 = nn.AvgPool2d(kernel_size=(1, 4))
        self.drop1 = nn.Dropout(0.50)
        
        self.conv2 = nn.Conv2d(F1, F2, kernel_size=(1, 16), padding=(0, 8), bias=False)
        self.bn2 = nn.BatchNorm2d(F2)
        self.elu2 = nn.ELU()
        self.pool2 = nn.AvgPool2d(kernel_size=(1, 8))
        self.drop2 = nn.Dropout(0.50)
        
        out_time = samples // 32
        flatten_size = F2 * out_time
        
        self.fc = nn.Linear(flatten_size, out_dim)

    def forward(self, x):
        x = x.unsqueeze(1).unsqueeze(2) # [B, 1, 1, T]
        x = self.drop1(self.pool1(self.elu1(self.bn1(self.conv1(x)))))
        x = self.drop2(self.pool2(self.elu2(self.bn2(self.conv2(x)))))
        x = x.view(x.size(0), -1)
        return self.fc(x)

class MatchMismatchNet(nn.Module):
    """
    Dual-Encoder Network that uses Cosine Similarity Fusion to predict if EEG matches Audio.
    """
    def __init__(self, in_channels=60, samples=256, latent_dim=16):
        super(MatchMismatchNet, self).__init__()
        self.eeg_encoder = EEGEncoder(in_channels=in_channels, samples=samples, out_dim=latent_dim)
        self.audio_encoder = AudioEncoder(samples=samples, out_dim=latent_dim)
        
        # Learnable scale and shift for Cosine Similarity
        # BCEWithLogits requires inputs in roughly [-5, 5] for strong gradients,
        # but Cosine Sim is strictly [-1, 1]. This temperature scaling fixes that.
        self.scale = nn.Parameter(torch.tensor(5.0))
        self.bias = nn.Parameter(torch.tensor(0.0))

    def forward(self, eeg, audio):
        # Embed
        z_e = self.eeg_encoder(eeg)
        z_a = self.audio_encoder(audio)
        
        # Cosine Similarity
        cos_sim = F.cosine_similarity(z_e, z_a, dim=1) # [B]
        
        # Shift and Scale to Logits
        logits = cos_sim * self.scale + self.bias
        
        return logits
