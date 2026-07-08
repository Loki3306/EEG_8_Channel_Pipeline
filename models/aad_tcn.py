import torch
import torch.nn as nn
import torch.nn.functional as F

class LocalEncoder(nn.Module):
    def __init__(self, in_channels, out_dim=64):
        super().__init__()
        # Explicit depthwise-separable style for EEG to reduce parameters
        self.conv1 = nn.Conv1d(in_channels, 16, kernel_size=33, padding=16)
        self.bn1 = nn.BatchNorm1d(16)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=17, padding=8)
        self.bn2 = nn.BatchNorm1d(32)
        self.pool = nn.AdaptiveAvgPool1d(4)
        self.fc = nn.Linear(32 * 4, out_dim)
        
    def forward(self, x):
        if x.dim() == 2: x = x.unsqueeze(1)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool1d(x, 2)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = torch.flatten(x, 1) 
        return self.fc(x)

class WavLMEncoder(nn.Module):
    """Projects 768-dimensional WavLM embeddings down to latent_dim without blowing up parameters."""
    def __init__(self, in_channels=768, out_dim=64):
        super().__init__()
        # 1x1 Convolution to reduce dimensionality drastically
        self.proj = nn.Conv1d(in_channels, 64, kernel_size=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 32, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(32)
        self.pool = nn.AdaptiveAvgPool1d(4)
        self.fc = nn.Linear(32 * 4, out_dim)
        
    def forward(self, x):
        if x.dim() == 2: x = x.unsqueeze(1)
        x = F.relu(self.bn1(self.proj(x)))
        x = F.max_pool1d(x, 2)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = torch.flatten(x, 1) 
        return self.fc(x)

class Chomp1d(nn.Module):
    """Slices off the padding added for causal convolutions."""
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        if self.chomp_size == 0:
            return x
        return x[:, :, :-self.chomp_size].contiguous()

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        # Causal Dilated Conv 1
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        # Causal Dilated Conv 2
        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
                                 
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            # Padding is strictly defined to ensure causality
            padding = (kernel_size - 1) * dilation_size
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=padding, dropout=dropout)]
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

class TCNAADModel(nn.Module):
    def __init__(self, eeg_channels=8, latent_dim=64, tcn_channels=[64, 64, 64], kernel_size=2, dropout=0.3, use_wavlm=False, audio_channels=1):
        super().__init__()
        self.eeg_encoder = LocalEncoder(in_channels=eeg_channels, out_dim=latent_dim)
        
        if use_wavlm:
            self.aud_encoder = WavLMEncoder(in_channels=768, out_dim=latent_dim)
        else:
            self.aud_encoder = LocalEncoder(in_channels=audio_channels, out_dim=latent_dim)
        
        # Total concatenated features per timestep
        tcn_input_dim = latent_dim * 3 + 3 
        
        self.tcn = TemporalConvNet(
            num_inputs=tcn_input_dim, 
            num_channels=tcn_channels, 
            kernel_size=kernel_size, 
            dropout=dropout
        )
        
        # Classifier runs on the TCN outputs
        self.classifier = nn.Linear(tcn_channels[-1], 1)
        
    def forward(self, eeg_seq, aud_a_seq, aud_b_seq, hidden=None):
        # Determine if input is Envelope (1D) or WavLM (768D)
        is_wavlm = aud_a_seq.shape[-1] == 768
        
        B, SeqLen = eeg_seq.shape[0], eeg_seq.shape[1]
        
        # eeg_seq is [B, SeqLen, C, T]
        C, T = eeg_seq.shape[2], eeg_seq.shape[3]
        eeg_flat = eeg_seq.reshape(B * SeqLen, C, T)
        
        if is_wavlm:
            # aud_seq is [B, SeqLen, T, 768] -> reshape to [B*SeqLen, 768, T]
            T_aud = aud_a_seq.shape[2]
            aud_a_flat = aud_a_seq.reshape(B * SeqLen, T_aud, 768).transpose(1, 2)
            aud_b_flat = aud_b_seq.reshape(B * SeqLen, T_aud, 768).transpose(1, 2)
        else:
            if aud_a_seq.dim() == 3:
                # 1D Envelope: aud_seq is [B, SeqLen, T] -> reshape to [B*SeqLen, 1, T]
                T_aud = aud_a_seq.shape[2]
                aud_a_flat = aud_a_seq.reshape(B * SeqLen, 1, T_aud)
                aud_b_flat = aud_b_seq.reshape(B * SeqLen, 1, T_aud)
            else:
                # Multiband Envelope: aud_seq is [B, SeqLen, Channels, T]
                C_aud = aud_a_seq.shape[2]
                T_aud = aud_a_seq.shape[3]
                aud_a_flat = aud_a_seq.reshape(B * SeqLen, C_aud, T_aud)
                aud_b_flat = aud_b_seq.reshape(B * SeqLen, C_aud, T_aud)
        
        p_eeg = F.normalize(self.eeg_encoder(eeg_flat), dim=-1)
        p_a = F.normalize(self.aud_encoder(aud_a_flat), dim=-1)
        p_b = F.normalize(self.aud_encoder(aud_b_flat), dim=-1)
        
        score_a = F.cosine_similarity(p_eeg, p_a, dim=-1)
        score_b = F.cosine_similarity(p_eeg, p_b, dim=-1)
        score_diff = score_a - score_b
        
        # [B, SeqLen, 195]
        seq_feat = torch.cat([p_eeg, p_a, p_b, score_a.unsqueeze(-1), score_b.unsqueeze(-1), score_diff.unsqueeze(-1)], dim=-1)
        seq_feat = seq_feat.reshape(B, SeqLen, -1)
        
        # TCN expects [B, Channels, Time]
        seq_feat_tcn = seq_feat.transpose(1, 2)
        
        # Pass through TCN
        tcn_out = self.tcn(seq_feat_tcn)
        
        # Transpose back to [B, Time, Channels]
        tcn_out = tcn_out.transpose(1, 2)
        
        # Global Average Pooling over time dimension (as recommended by GPT IPC)
        tcn_pool = tcn_out.mean(dim=1)
        logits = self.classifier(tcn_pool).squeeze(-1)
        
        return logits, None

class HybridMoEAADModel(nn.Module):
    def __init__(self, eeg_channels=8, latent_dim=64, tcn_channels=[64, 64, 64], kernel_size=2, dropout=0.3):
        super().__init__()
        # 1. Shared EEG Encoder (Saves massive GPU memory and compute)
        self.eeg_encoder = LocalEncoder(in_channels=eeg_channels, out_dim=latent_dim)
        
        # 2. Independent Audio Encoders
        self.wavlm_encoder = WavLMEncoder(in_channels=768, out_dim=latent_dim)
        self.multi_encoder = LocalEncoder(in_channels=16, out_dim=latent_dim)
        
        tcn_input_dim = latent_dim * 3 + 3 
        
        # 3. Independent TCN Backbones (One for each expert)
        self.wavlm_tcn = TemporalConvNet(tcn_input_dim, tcn_channels, kernel_size, dropout)
        self.wavlm_classifier = nn.Linear(tcn_channels[-1], 1)
        
        self.multi_tcn = TemporalConvNet(tcn_input_dim, tcn_channels, kernel_size, dropout)
        self.multi_classifier = nn.Linear(tcn_channels[-1], 1)
        
        # 4. Temporal Gating Network
        # Takes the encoded EEG and ALL encoded audio representations (much richer input)
        self.gate = nn.Sequential(
            nn.Linear(latent_dim * 5, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid() # Outputs alpha (0 to 1)
        )
        
    def forward(self, eeg_seq, wavlm_a, wavlm_b, multi_a, multi_b):
        B, SeqLen = eeg_seq.shape[0], eeg_seq.shape[1]
        
        # 1. Shared EEG Encoding
        C, T = eeg_seq.shape[2], eeg_seq.shape[3]
        eeg_flat = eeg_seq.reshape(B * SeqLen, C, T)
        p_eeg = F.normalize(self.eeg_encoder(eeg_flat), dim=-1) # [B*SeqLen, latent_dim]
        
        # 2. WavLM Encoding
        T_aud = wavlm_a.shape[2]
        wa_flat = wavlm_a.reshape(B * SeqLen, T_aud, 768).transpose(1, 2)
        wb_flat = wavlm_b.reshape(B * SeqLen, T_aud, 768).transpose(1, 2)
        p_wa = F.normalize(self.wavlm_encoder(wa_flat), dim=-1)
        p_wb = F.normalize(self.wavlm_encoder(wb_flat), dim=-1)
        
        # 3. Multiband Encoding
        C_m, T_m = multi_a.shape[2], multi_a.shape[3]
        ma_flat = multi_a.reshape(B * SeqLen, C_m, T_m)
        mb_flat = multi_b.reshape(B * SeqLen, C_m, T_m)
        p_ma = F.normalize(self.multi_encoder(ma_flat), dim=-1)
        p_mb = F.normalize(self.multi_encoder(mb_flat), dim=-1)
        
        # 4. Temporal Gating
        # The gate dynamically adapts PER TIMESTEP, and looks at both Audio A and Audio B!
        gate_input = torch.cat([p_eeg, p_wa, p_wb, p_ma, p_mb], dim=-1)
        alpha = self.gate(gate_input).reshape(B, SeqLen) # [B, SeqLen]
        
        # 5. WavLM Expert TCN
        score_wa = F.cosine_similarity(p_eeg, p_wa, dim=-1)
        score_wb = F.cosine_similarity(p_eeg, p_wb, dim=-1)
        diff_w = score_wa - score_wb
        feat_w = torch.cat([p_eeg, p_wa, p_wb, score_wa.unsqueeze(-1), score_wb.unsqueeze(-1), diff_w.unsqueeze(-1)], dim=-1)
        feat_w = feat_w.reshape(B, SeqLen, -1).transpose(1, 2)
        tcn_out_w = self.wavlm_tcn(feat_w).transpose(1, 2) # [B, SeqLen, Channels]
        
        # 6. Multiband Expert TCN
        score_ma = F.cosine_similarity(p_eeg, p_ma, dim=-1)
        score_mb = F.cosine_similarity(p_eeg, p_mb, dim=-1)
        diff_m = score_ma - score_mb
        feat_m = torch.cat([p_eeg, p_ma, p_mb, score_ma.unsqueeze(-1), score_mb.unsqueeze(-1), diff_m.unsqueeze(-1)], dim=-1)
        feat_m = feat_m.reshape(B, SeqLen, -1).transpose(1, 2)
        tcn_out_m = self.multi_tcn(feat_m).transpose(1, 2) # [B, SeqLen, Channels]
        
        # 7. Classification and Fusion
        logits_w = self.wavlm_classifier(tcn_out_w).squeeze(-1) # [B, SeqLen]
        logits_m = self.multi_classifier(tcn_out_m).squeeze(-1) # [B, SeqLen]
        
        # LOGIT Fusion (Standard MoE practice)
        hybrid_logits = (alpha * logits_w) + ((1 - alpha) * logits_m)
        
        # Global Average Pooling for final prediction
        hybrid_logits_pool = hybrid_logits.mean(dim=1)
        
        # Return alpha as the second argument so the training loop can apply Entropy Regularization!
        return hybrid_logits_pool, alpha
