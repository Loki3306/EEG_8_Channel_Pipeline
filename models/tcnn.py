import torch
import torch.nn as nn
import torch.nn.functional as F

class TCNN(nn.Module):
    """
    Temporal Convolutional Neural Network (TCNN) for EEG sequence classification.
    Expects input shape: [Batch, Channels, Time]
    Outputs: [Batch, 2] logits for binary classification (Left vs Right ear / Track 1 vs Track 2)
    """
    def __init__(self, in_channels=8, num_classes=2):
        super(TCNN, self).__init__()
        
        # Temporal Convolution Block 1
        self.conv1 = nn.Conv1d(in_channels, 32, kernel_size=11, padding=5)
        self.bn1 = nn.BatchNorm1d(32)
        self.pool1 = nn.MaxPool1d(2)
        
        # Temporal Convolution Block 2
        self.conv2 = nn.Conv1d(32, 64, kernel_size=9, padding=4)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool2 = nn.MaxPool1d(2)
        
        # Temporal Convolution Block 3
        self.conv3 = nn.Conv1d(64, 128, kernel_size=7, padding=3)
        self.bn3 = nn.BatchNorm1d(128)
        self.pool3 = nn.MaxPool1d(2)
        
        # Global Average Pooling over time will reduce [B, 128, T'] to [B, 128]
        
        self.dropout = nn.Dropout(0.5)
        
        # Classification Head
        self.fc = nn.Linear(128, num_classes)
        
    def forward(self, x):
        # Block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool1(x)
        
        # Block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool2(x)
        
        # Block 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.pool3(x)
        
        # Global Average Pooling
        x = x.mean(dim=-1)
        
        # Classification
        x = self.dropout(x)
        logits = self.fc(x)
        
        return logits

def print_summary():
    model = TCNN(in_channels=8)
    x = torch.randn(2, 8, 320) # 5 seconds @ 64Hz = 320 samples
    out = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"TCNN Parameter Count: {params:,}")
    
if __name__ == "__main__":
    print_summary()
