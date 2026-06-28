import torch
import torch.nn as nn

class KuruvilaCNNLSTM(nn.Module):
    """
    Joint CNN-BiLSTM AAD Architecture (Kuruvila et al., 2021)
    
    This model predicts the attended speaker (Speaker 1 vs Speaker 2) from:
    1. EEG signals
    2. Speaker 1 Audio Spectrogram / Gammatone
    3. Speaker 2 Audio Spectrogram / Gammatone
    """
    def __init__(self, eeg_channels=8, audio_channels=28, num_classes=2):
        super().__init__()
        
        # ---------------------------------------------------------------------
        # EEG CNN Branch
        # ---------------------------------------------------------------------
        # Paper specifies first kernel is ~375ms (24 samples at 64Hz)
        self.eeg_cnn = nn.Sequential(
            # Block 1
            nn.Conv1d(in_channels=eeg_channels, out_channels=32, kernel_size=25, padding='same'),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(0.2),
            
            # Block 2
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=11, padding='same'),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(0.2),
            
            # Block 3
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5, padding='same'),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(0.2),
            
            # Block 4
            nn.Conv1d(in_channels=128, out_channels=64, kernel_size=3, padding='same'),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(0.2)
        )
        
        # ---------------------------------------------------------------------
        # Audio CNN Branch (Shared Weights for Speaker 1 and Speaker 2)
        # ---------------------------------------------------------------------
        # We match the 4-block structure of EEG so the time dimensions align
        self.audio_cnn = nn.Sequential(
            # Block 1
            nn.Conv1d(in_channels=audio_channels, out_channels=32, kernel_size=11, padding='same'),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(0.2),
            
            # Block 2
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, padding='same'),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(0.2),
            
            # Block 3
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5, padding='same'),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(0.2),
            
            # Block 4
            nn.Conv1d(in_channels=128, out_channels=64, kernel_size=3, padding='same'),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(0.2)
        )
        
        # ---------------------------------------------------------------------
        # Temporal Module (BiLSTM)
        # ---------------------------------------------------------------------
        # Concatenated features: EEG (64) + Audio1 (64) + Audio2 (64) = 192
        fused_channels = 64 + 64 + 64
        self.lstm = nn.LSTM(input_size=fused_channels, hidden_size=64, num_layers=1, 
                            batch_first=True, bidirectional=True)
        
        # ---------------------------------------------------------------------
        # Classification Head
        # ---------------------------------------------------------------------
        self.fc = nn.Sequential(
            nn.Linear(64 * 2, 64), # 64 * 2 because BiLSTM
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes)
        )

    def forward(self, eeg, audio1, audio2):
        """
        Shapes:
        eeg: (Batch, Channels, Time)
        audio1: (Batch, Channels, Time)
        audio2: (Batch, Channels, Time)
        """
        
        # Extract Embeddings
        # Shape: (Batch, 64, Time / 16)
        eeg_emb = self.eeg_cnn(eeg)
        aud1_emb = self.audio_cnn(audio1)
        aud2_emb = self.audio_cnn(audio2)
        
        # Ensure time dimensions perfectly match (they should if input time is identical)
        # but just in case of rounding errors with padding, truncate to minimum length
        min_len = min(eeg_emb.shape[-1], aud1_emb.shape[-1], aud2_emb.shape[-1])
        eeg_emb = eeg_emb[:, :, :min_len]
        aud1_emb = aud1_emb[:, :, :min_len]
        aud2_emb = aud2_emb[:, :, :min_len]
        
        # Fusion (Concatenate along Feature Dimension)
        # Shape: (Batch, 192, Time / 16)
        fused = torch.cat([eeg_emb, aud1_emb, aud2_emb], dim=1)
        
        # Permute for LSTM: (Batch, Time, Features)
        fused = fused.permute(0, 2, 1)
        
        # BiLSTM
        # out shape: (Batch, Time, Hidden*2)
        lstm_out, _ = self.lstm(fused)
        
        # Global Average Pooling over Time
        # Shape: (Batch, Hidden*2)
        time_pool = torch.mean(lstm_out, dim=1)
        
        # Classification
        # Shape: (Batch, num_classes)
        logits = self.fc(time_pool)
        
        return logits
