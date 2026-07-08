import torch
import torch.nn as nn
import torch.nn.functional as F

class LocalEncoder(nn.Module):
    def __init__(self, in_channels, out_dim=64):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, 32, kernel_size=15, padding=7)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=7, padding=3)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool = nn.AdaptiveAvgPool1d(4)
        self.fc = nn.Linear(64 * 4, out_dim)

    def forward(self, x):
        if x.dim() == 2: x = x.unsqueeze(1)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool1d(x, 2)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = torch.flatten(x, 1) 
        return self.fc(x)

class InceptionAudioEncoder(nn.Module):
    def __init__(self, in_channels, out_dim=64):
        super().__init__()
        
        # Branch 1: Fast (RF ~23ms)
        self.branch1 = nn.Sequential(
            nn.Conv1d(in_channels, 24, kernel_size=3, padding=1, dilation=1),
            nn.BatchNorm1d(24),
            nn.ReLU()
        )
        
        # Branch 2: Medium (RF ~117ms)
        self.branch2 = nn.Sequential(
            nn.Conv1d(in_channels, 24, kernel_size=15, padding=7, dilation=1),
            nn.BatchNorm1d(24),
            nn.ReLU()
        )
        
        # Branch 3: Slow (RF ~445ms)
        self.branch3 = nn.Sequential(
            nn.Conv1d(in_channels, 24, kernel_size=15, padding=28, dilation=4),
            nn.BatchNorm1d(24),
            nn.ReLU()
        )
        
        # Fusion 1x1 Conv
        self.fusion = nn.Sequential(
            nn.Conv1d(24 * 3, 64, kernel_size=1),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )
        
        self.pool = nn.AdaptiveAvgPool1d(4)
        self.fc = nn.Linear(64 * 4, out_dim)
        
    def forward(self, x):
        if x.dim() == 2: x = x.unsqueeze(1)
        
        out1 = self.branch1(x)
        out2 = self.branch2(x)
        out3 = self.branch3(x)
        
        fused = self.fusion(torch.cat([out1, out2, out3], dim=1))
        
        p = self.pool(fused)
        p = torch.flatten(p, 1)
        return self.fc(p)

class TemporalPyramidEncoder(nn.Module):
    """
    Multi-Resolution Temporal Pyramid
    Physically downsamples the audio to force clean low-frequency tracking,
    then upsamples and adds the features together (ResNet style).
    """
    def __init__(self, in_channels, out_dim=64):
        super().__init__()
        
        # Branch 1: 128 Hz (Fast transients, kernel=3)
        self.branch_128 = nn.Sequential(
            nn.Conv1d(in_channels, 24, kernel_size=3, padding=1),
            nn.BatchNorm1d(24),
            nn.ReLU()
        )
        
        # Branch 2: 64 Hz (Medium scales, stride=2)
        self.branch_64 = nn.Sequential(
            nn.Conv1d(in_channels, 24, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(24),
            nn.ReLU()
        )
        
        # Branch 3: 32 Hz (Slow semantics, stride=4)
        self.branch_32 = nn.Sequential(
            nn.Conv1d(in_channels, 24, kernel_size=9, stride=4, padding=4),
            nn.BatchNorm1d(24),
            nn.ReLU()
        )
        
        self.fusion = nn.Sequential(
            nn.Conv1d(24, 64, kernel_size=1),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )
        
        self.pool = nn.AdaptiveAvgPool1d(4)
        self.fc = nn.Linear(64 * 4, out_dim)
        
    def forward(self, x):
        if x.dim() == 2: x = x.unsqueeze(1)
        
        out_128 = self.branch_128(x)
        out_64 = self.branch_64(x)
        out_32 = self.branch_32(x)
        
        T = out_128.shape[2]
        
        # Upsample back to original T (128Hz)
        out_64_up = F.interpolate(out_64, size=T, mode='linear', align_corners=False)
        out_32_up = F.interpolate(out_32, size=T, mode='linear', align_corners=False)
        
        # Additive fusion (No unstable gating, pure gradient stability)
        fused = out_128 + out_64_up + out_32_up
        
        p = self.pool(self.fusion(fused))
        p = torch.flatten(p, 1)
        return self.fc(p)

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

class SpatialAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(in_channels, max(1, in_channels // 2), kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(max(1, in_channels // 2), in_channels, kernel_size=1),
            nn.Sigmoid() 
        )
    def forward(self, x):
        weights = self.se(x) 
        return x * weights 

class SpectralAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(in_channels, max(1, in_channels // 2), kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(max(1, in_channels // 2), in_channels, kernel_size=1),
            nn.Sigmoid()
        )
    def forward(self, x):
        weights = self.se(x)
        return x * weights

class CrossModalGate(nn.Module):
    def __init__(self, latent_dim, audio_channels):
        super().__init__()
        # Takes the Brain State and predicts bounded weights for the audio channels
        self.net = nn.Sequential(
            nn.Linear(latent_dim, latent_dim // 2),
            nn.ReLU(),
            nn.Linear(latent_dim // 2, audio_channels),
            nn.Tanh() # Outputs between -1 and 1
        )
    def forward(self, eeg_latent):
        # We use (1 + 0.5 * tanh) to bound the weights between 0.5 (suppress) and 1.5 (amplify)
        weights = 1.0 + 0.5 * self.net(eeg_latent) 
        return weights.unsqueeze(-1) # [B*SeqLen, audio_channels, 1]

class DeepTemporalCrossModalGate(nn.Module):
    def __init__(self, eeg_channels, audio_channels):
        super().__init__()
        # A deep temporal convolutional network to extract clean EEG features
        # while strictly preserving the temporal sequence dimension 'T'.
        self.net = nn.Sequential(
            nn.Conv1d(eeg_channels, 32, kernel_size=15, padding=7),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, audio_channels, kernel_size=3, padding=1),
            nn.Tanh() # Outputs between -1 and 1
        )
        
    def forward(self, eeg_seq):
        # eeg_seq: [B*SeqLen, eeg_channels, T]
        # output bounds: [0.5, 1.5]
        # Mathematics: A * (1 + 0.5w) = A + 0.5(A*w), which acts as a residual connection
        weights = 1.0 + 0.5 * self.net(eeg_seq) 
        return weights # [B*SeqLen, audio_channels, T]

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
    def __init__(self, eeg_channels=8, latent_dim=64, tcn_channels=[64, 64, 64], kernel_size=2, dropout=0.2, use_wavlm=False, audio_channels=1, attention_type='none', use_inception=False, use_pyramid=False):
        super().__init__()
        self.use_wavlm = use_wavlm
        
        self.eeg_encoder = LocalEncoder(in_channels=eeg_channels, out_dim=latent_dim)
        
        if attention_type in ['spatial_spectral', 'cross_modal']:
            self.spatial_attention = SpatialAttention(eeg_channels)
        else:
            self.spatial_attention = None
            
        if use_wavlm:
            self.audio_encoder = WavLMEncoder(in_channels=768, out_dim=latent_dim)
            self.cross_modal_gate = None
            self.spectral_attention = None
            self.temporal_cross_modal_gate = None
        else:
            if use_inception:
                self.audio_encoder = InceptionAudioEncoder(in_channels=audio_channels, out_dim=latent_dim)
            elif use_pyramid:
                self.audio_encoder = TemporalPyramidEncoder(in_channels=audio_channels, out_dim=latent_dim)
            else:
                self.audio_encoder = LocalEncoder(in_channels=audio_channels, out_dim=latent_dim)
                
            if audio_channels > 1 and attention_type == 'spatial_spectral':
                self.spectral_attention = SpectralAttention(audio_channels)
                self.cross_modal_gate = None
                self.temporal_cross_modal_gate = None
            elif audio_channels > 1 and attention_type == 'cross_modal':
                self.cross_modal_gate = CrossModalGate(latent_dim, audio_channels)
                self.spectral_attention = None
                self.temporal_cross_modal_gate = None
            elif audio_channels > 1 and attention_type == 'temporal_cross_modal':
                self.temporal_cross_modal_gate = DeepTemporalCrossModalGate(eeg_channels, audio_channels)
                self.cross_modal_gate = None
                self.spectral_attention = None
            else:
                self.cross_modal_gate = None
                self.spectral_attention = None
                self.temporal_cross_modal_gate = None
                
        # Input to TCN: eeg_latent (64) + aud_a_latent (64) + aud_b_latent (64) + 
        #               cos_sim_a (1) + cos_sim_b (1) + diff (1) = 195
        tcn_input_dim = latent_dim * 3 + 3 
        
        self.tcn = TemporalConvNet(tcn_input_dim, tcn_channels, kernel_size=kernel_size, dropout=dropout)
        self.classifier = nn.Linear(tcn_channels[-1], 1)

    def forward(self, eeg_seq, aud_seq_a, aud_seq_b):
        B, SeqLen, C, T = eeg_seq.shape
        eeg_flat = eeg_seq.reshape(B * SeqLen, C, T)
        
        if self.spatial_attention is not None:
            eeg_flat = self.spatial_attention(eeg_flat)
            
        if self.use_wavlm:
            pass # Temporal cross-modal gate not supported for WavLM
        else:
            if getattr(self, 'temporal_cross_modal_gate', None) is not None:
                # Apply residual temporal gating to audio BEFORE audio flattening/encoding
                temporal_weights = self.temporal_cross_modal_gate(eeg_flat) # [B*SeqLen, C_a, T]
                C_a, T_a = aud_seq_a.shape[2], aud_seq_a.shape[3]
                aud_a_flat = aud_seq_a.reshape(B * SeqLen, C_a, T_a)
                aud_b_flat = aud_seq_b.reshape(B * SeqLen, C_a, T_a)
                
                aud_a_flat = aud_a_flat * temporal_weights
                aud_b_flat = aud_b_flat * temporal_weights
                
                aud_seq_a = aud_a_flat.reshape(B, SeqLen, C_a, T_a)
                aud_seq_b = aud_b_flat.reshape(B, SeqLen, C_a, T_a)
                
        eeg_latent_raw = self.eeg_encoder(eeg_flat)
        p_eeg = F.normalize(eeg_latent_raw, dim=-1) # [B*SeqLen, latent_dim]
        
        if self.use_wavlm:
            T_aud = aud_seq_a.shape[2]
            aud_a_flat = aud_seq_a.reshape(B * SeqLen, T_aud, 768).transpose(1, 2)
            aud_b_flat = aud_seq_b.reshape(B * SeqLen, T_aud, 768).transpose(1, 2)
            p_a = F.normalize(self.audio_encoder(aud_a_flat), dim=-1)
            p_b = F.normalize(self.audio_encoder(aud_b_flat), dim=-1)
        else:
            C_a, T_a = aud_seq_a.shape[2], aud_seq_a.shape[3]
            aud_a_flat = aud_seq_a.reshape(B * SeqLen, C_a, T_a)
            aud_b_flat = aud_seq_b.reshape(B * SeqLen, C_a, T_a)
            
            if self.spectral_attention is not None:
                aud_a_flat = self.spectral_attention(aud_a_flat)
                aud_b_flat = self.spectral_attention(aud_b_flat)
            elif self.cross_modal_gate is not None:
                spectral_weights = self.cross_modal_gate(eeg_latent_raw) # [B*SeqLen, C_a, 1]
                aud_a_flat = aud_a_flat * spectral_weights
                aud_b_flat = aud_b_flat * spectral_weights
                
            p_a = F.normalize(self.audio_encoder(aud_a_flat), dim=-1)
            p_b = F.normalize(self.audio_encoder(aud_b_flat), dim=-1)
        
        score_a = F.cosine_similarity(p_eeg, p_a, dim=-1)
        score_b = F.cosine_similarity(p_eeg, p_b, dim=-1)
        score_diff = score_a - score_b
        
        seq_feat = torch.cat([p_eeg, p_a, p_b, score_a.unsqueeze(-1), score_b.unsqueeze(-1), score_diff.unsqueeze(-1)], dim=-1)
        seq_feat = seq_feat.reshape(B, SeqLen, -1).transpose(1, 2)
        
        tcn_out = self.tcn(seq_feat).transpose(1, 2)
        logits = self.classifier(tcn_out.mean(dim=1)).squeeze(-1)
        
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

class LateFusionAADModel(nn.Module):
    def __init__(self, eeg_channels=8, latent_dim=64, tcn_channels=[64, 64, 64], kernel_size=2, dropout=0.3):
        super().__init__()
        
        # 1. Shared EEG Encoder
        self.eeg_encoder = LocalEncoder(in_channels=eeg_channels, out_dim=latent_dim)
        
        tcn_input_dim = latent_dim * 3 + 3
        
        # 2. Fast Expert (128 Hz)
        self.fast_audio = nn.Sequential(
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(4),
            nn.Flatten(1),
            nn.Linear(32 * 4, latent_dim)
        )
        self.fast_tcn = TemporalConvNet(tcn_input_dim, tcn_channels, kernel_size, dropout)
        self.fast_classifier = nn.Linear(tcn_channels[-1], 1)
        
        # 3. Slow Expert (32 Hz)
        self.slow_audio = nn.Sequential(
            nn.Conv1d(16, 32, kernel_size=9, stride=4, padding=4),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(4),
            nn.Flatten(1),
            nn.Linear(32 * 4, latent_dim)
        )
        self.slow_tcn = TemporalConvNet(tcn_input_dim, tcn_channels, kernel_size, dropout)
        self.slow_classifier = nn.Linear(tcn_channels[-1], 1)
        
        # 4. Combiner (Static Subject-Specific Router)
        self.combiner = nn.Linear(2, 1)
        
    def forward(self, eeg_seq, aud_a, aud_b):
        B, SeqLen = eeg_seq.shape[0], eeg_seq.shape[1]
        
        # -- Shared EEG --
        C_e, T_e = eeg_seq.shape[2], eeg_seq.shape[3]
        eeg_flat = eeg_seq.reshape(B * SeqLen, C_e, T_e)
        p_eeg = F.normalize(self.eeg_encoder(eeg_flat), dim=-1)
        
        # -- Audio Flattening --
        C_a, T_a = aud_a.shape[2], aud_a.shape[3]
        a_flat = aud_a.reshape(B * SeqLen, C_a, T_a)
        b_flat = aud_b.reshape(B * SeqLen, C_a, T_a)
        
        # ==========================================
        # FAST EXPERT
        # ==========================================
        p_f_a = F.normalize(self.fast_audio(a_flat), dim=-1)
        p_f_b = F.normalize(self.fast_audio(b_flat), dim=-1)
        
        score_f_a = F.cosine_similarity(p_eeg, p_f_a, dim=-1)
        score_f_b = F.cosine_similarity(p_eeg, p_f_b, dim=-1)
        diff_f = score_f_a - score_f_b
        
        feat_f = torch.cat([p_eeg, p_f_a, p_f_b, score_f_a.unsqueeze(-1), score_f_b.unsqueeze(-1), diff_f.unsqueeze(-1)], dim=-1)
        feat_f = feat_f.reshape(B, SeqLen, -1).transpose(1, 2)
        
        tcn_out_f = self.fast_tcn(feat_f).transpose(1, 2)
        logits_f = self.fast_classifier(tcn_out_f).squeeze(-1) # [B, SeqLen]
        
        # ==========================================
        # SLOW EXPERT
        # ==========================================
        p_s_a = F.normalize(self.slow_audio(a_flat), dim=-1)
        p_s_b = F.normalize(self.slow_audio(b_flat), dim=-1)
        
        score_s_a = F.cosine_similarity(p_eeg, p_s_a, dim=-1)
        score_s_b = F.cosine_similarity(p_eeg, p_s_b, dim=-1)
        diff_s = score_s_a - score_s_b
        
        feat_s = torch.cat([p_eeg, p_s_a, p_s_b, score_s_a.unsqueeze(-1), score_s_b.unsqueeze(-1), diff_s.unsqueeze(-1)], dim=-1)
        feat_s = feat_s.reshape(B, SeqLen, -1).transpose(1, 2)
        
        tcn_out_s = self.slow_tcn(feat_s).transpose(1, 2)
        logits_s = self.slow_classifier(tcn_out_s).squeeze(-1) # [B, SeqLen]
        
        # ==========================================
        # COMBINER (STATIC ROUTING)
        # ==========================================
        # Average pooling over the sequence BEFORE the combiner, to give a single robust logit per expert
        pool_f = logits_f.mean(dim=1).unsqueeze(-1) # [B, 1]
        pool_s = logits_s.mean(dim=1).unsqueeze(-1) # [B, 1]
        
        combined_features = torch.cat([pool_f, pool_s], dim=-1) # [B, 2]
        final_logits = self.combiner(combined_features).squeeze(-1) # [B]
        
        return final_logits, None
