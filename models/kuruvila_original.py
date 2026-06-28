import torch
import torch.nn as nn
import torch.nn.functional as F

class KuruvilaOriginalCNNLSTM(nn.Module):
    def __init__(self, eeg_channels=8, audio_channels=28, num_spkr=2):
        super().__init__()

        self.num_spkr = num_spkr
        self.E_dp_prob = 0.25
        self.aud_dp_prob = 0.4
        self.fc_dp_prob = 0.25
        
        self.num_conv_kernels_1 = 32
        self.num_conv_kernels_2 = 16   
        self.num_conv_kernels_3 = 8   
        self.num_conv_kernels_4 = 1  
        
        self.eeg_channels = eeg_channels

        self.EEG_model_init()
        self.Audio_model_init(audio_channels)

        # Get shapes via dummy forward passes
        x_eeg = torch.randn(1, 1, 192, self.eeg_channels)
        self.eeg_conv_shape = None
        self.EEG_model_convs(x_eeg) 
        print(f"[Init] EEG CNN Output Shape: {self.eeg_conv_shape}")
        
        x_aud = torch.randn(1, 1, 192, audio_channels)
        self.audio_conv_shape = None
        self.Audio_model_convs(x_aud)
        print(f"[Init] Audio CNN Output Shape: {self.audio_conv_shape}")

        self.lstm_hidden_size = 48
        use_bidirectional = True

        if use_bidirectional:
            self.num_direction = 2
            self.direction_scale = 0.5
        else:
            self.num_direction = 1
            self.direction_scale = 1 

        lstm_input_size = self.eeg_conv_shape[1] + 2 * self.audio_conv_shape[1] * self.audio_conv_shape[3]
        
        self.lstm1 = nn.LSTM(
            lstm_input_size, 
            int(self.lstm_hidden_size * self.direction_scale), 
            bidirectional=use_bidirectional, 
            batch_first=True
        )

        tmp = self.lstm_hidden_size * self.eeg_conv_shape[2] * self.eeg_conv_shape[3]        
        self.fc1 = nn.Linear(tmp, 128) 
        self.fc1_dp = nn.Dropout(p=self.fc_dp_prob)

        self.fc2 = nn.Linear(128, 128) 
        self.fc2_dp = nn.Dropout(p=self.fc_dp_prob)

        self.fc3 = nn.Linear(128, 32) 
        self.fc3_dp = nn.Dropout(p=self.fc_dp_prob)

        self.fc4 = nn.Linear(32, self.num_spkr)        

    def EEG_model_init(self):
        self.E_conv1 = nn.Conv2d(1, self.num_conv_kernels_1, kernel_size=(24,1), padding=(12,0))
        self.E_mPool1 = nn.MaxPool2d((2,1))
        self.E_conv1_bn = nn.BatchNorm2d(self.num_conv_kernels_1)
        self.E_conv1_dp = nn.Dropout(p=self.E_dp_prob) 
 
        self.E_conv2 = nn.Conv2d(self.num_conv_kernels_1, self.num_conv_kernels_1, kernel_size=(7,1), padding=(6,0), dilation=(2, 1))
        self.E_mPool2 = nn.MaxPool2d((1,2))
        self.E_conv2_bn = nn.BatchNorm2d(self.num_conv_kernels_1)
        self.E_conv2_dp = nn.Dropout(p=self.E_dp_prob) 

        self.E_conv3 = nn.Conv2d(self.num_conv_kernels_1, self.num_conv_kernels_1, kernel_size=(7,5), padding=(3,2))
        
        # Adaptation for 8 channels instead of 10 channels:
        # After Pool2, we have 4 channels left.
        # Pool3 needs to pool these 4 channels down to 1.
        if self.eeg_channels == 8:
            self.E_mPool3 = nn.MaxPool2d((2,4))
        else:
            self.E_mPool3 = nn.MaxPool2d((2,5))
            
        self.E_conv3_bn = nn.BatchNorm2d(self.num_conv_kernels_1)   
        self.E_conv3_dp = nn.Dropout(p=self.E_dp_prob) 

        self.E_conv4 = nn.Conv2d(self.num_conv_kernels_1, self.num_conv_kernels_1, kernel_size=(7,1), padding=(3,0))
        self.E_conv4_bn = nn.BatchNorm2d(self.num_conv_kernels_1)
        self.E_conv4_dp = nn.Dropout(p=self.E_dp_prob) 

    def Audio_model_init(self, audio_channels):
        # Adaptation to accept Gammatone envelopes (1, 192, 28) 
        # and output exactly (1, 48, 16) to match the paper's fusion shape.
        
        self.A_conv1 = nn.Conv2d(1, self.num_conv_kernels_1, kernel_size=(1,7), padding='same') 
        self.A_conv1_bn = nn.BatchNorm2d(self.num_conv_kernels_1)
        self.A_conv1_dp = nn.Dropout(p=self.aud_dp_prob)

        self.A_conv2 = nn.Conv2d(self.num_conv_kernels_1, self.num_conv_kernels_1, kernel_size=(7,1), padding='same') 
        self.A_mPool2 = nn.MaxPool2d((2,1)) # 192 -> 96
        self.A_conv2_bn = nn.BatchNorm2d(self.num_conv_kernels_1)  
        self.A_conv2_dp = nn.Dropout(p=self.aud_dp_prob)
    
        self.A_conv3 = nn.Conv2d(self.num_conv_kernels_1, self.num_conv_kernels_1, kernel_size=(3,5), padding='same')
        self.A_mPool3 = nn.MaxPool2d((2,1)) # 96 -> 48
        self.A_conv3_bn = nn.BatchNorm2d(self.num_conv_kernels_1)  
        self.A_conv3_dp = nn.Dropout(p=self.aud_dp_prob)

        self.A_conv4 = nn.Conv2d(self.num_conv_kernels_1, self.num_conv_kernels_1, kernel_size=(3,3), padding='same')
        self.A_conv4_bn = nn.BatchNorm2d(self.num_conv_kernels_1)  
        self.A_conv4_dp = nn.Dropout(p=self.aud_dp_prob)

        self.A_conv5 = nn.Conv2d(self.num_conv_kernels_1, self.num_conv_kernels_4, kernel_size=(1,1))
        
        # Adaptive pooling to force exactly (48 time, 16 freq) out
        self.A_mPool5 = nn.AdaptiveMaxPool2d((48, 16))
        
        self.A_conv5_bn = nn.BatchNorm2d(self.num_conv_kernels_4)  
        self.A_conv5_dp = nn.Dropout(p=self.aud_dp_prob) 

    def EEG_model_convs(self, x, verbose=False):
        if verbose: print(f"EEG Input: {x.shape}")
        x = self.E_conv1_bn(self.E_mPool1(self.E_conv1(x)))
        x = self.E_conv1_dp(F.relu(x))
        if verbose: print(f"EEG Pool1: {x.shape}")

        x = self.E_conv2_bn(self.E_mPool2(self.E_conv2(x)))
        x = self.E_conv2_dp(F.relu(x))
        if verbose: print(f"EEG Pool2: {x.shape}")

        x = self.E_conv3_bn(self.E_mPool3(self.E_conv3(x)))
        x = self.E_conv3_dp(F.relu(x))
        if verbose: print(f"EEG Pool3: {x.shape}")

        x = self.E_conv4_bn(self.E_conv4(x))
        x = self.E_conv4_dp(F.relu(x))        
        if verbose: print(f"EEG Conv4: {x.shape}")

        if self.eeg_conv_shape is None:
            self.eeg_conv_shape = x.shape
        return x

    def Audio_model_convs(self, x, verbose=False):
        if verbose: print(f"Audio Input: {x.shape}")
        x = self.A_conv1_bn(self.A_conv1(x))
        x = self.A_conv1_dp(F.relu(x))
        if verbose: print(f"Audio Conv1: {x.shape}")

        x = self.A_conv2_bn(self.A_mPool2(self.A_conv2(x)))
        x = self.A_conv2_dp(F.relu(x))
        if verbose: print(f"Audio Pool2: {x.shape}")

        x = self.A_conv3_bn(self.A_mPool3(self.A_conv3(x)))
        x = self.A_conv3_dp(F.relu(x))
        if verbose: print(f"Audio Pool3: {x.shape}")

        x = self.A_conv4_bn(self.A_conv4(x))
        x = self.A_conv4_dp(F.relu(x))
        if verbose: print(f"Audio Conv4: {x.shape}")

        x = self.A_conv5_bn(self.A_mPool5(self.A_conv5(x)))
        x = self.A_conv5_dp(F.relu(x)) 
        if verbose: print(f"Audio Pool5: {x.shape}")
        
        if self.audio_conv_shape is None:
            self.audio_conv_shape = x.shape
        return x

    def forward(self, eeg_x, aud1, aud2, verbose=False):
        # Input expected: (batch, channels, time) for both eeg and audio
        # But CNN requires (batch, 1, time, channels)
        eeg_x = eeg_x.permute(0, 2, 1).unsqueeze(1) # (batch, 1, 192, 8)
        aud1 = aud1.permute(0, 2, 1).unsqueeze(1)   # (batch, 1, 192, 28)
        aud2 = aud2.permute(0, 2, 1).unsqueeze(1)
        
        eeg_x = self.EEG_model_convs(eeg_x, verbose=verbose)
        # for the lstm input, first dim after batch size should be seq_len 
        eeg_x = eeg_x.view(-1, self.eeg_conv_shape[2], self.eeg_conv_shape[3]*self.eeg_conv_shape[1])
        if verbose: print(f"EEG Reshaped for LSTM: {eeg_x.shape}")

        a1 = self.Audio_model_convs(aud1, verbose=verbose)
        a1 = a1.reshape(-1, self.audio_conv_shape[2], self.audio_conv_shape[3]*self.audio_conv_shape[1])
        
        a2 = self.Audio_model_convs(aud2, verbose=verbose)
        a2 = a2.reshape(-1, self.audio_conv_shape[2], self.audio_conv_shape[3]*self.audio_conv_shape[1])
        
        if verbose: 
            print(f"Audio1 Reshaped for LSTM: {a1.shape}")
            print(f"Audio2 Reshaped for LSTM: {a2.shape}")

        # Exact Fusion Ordering from original code
        x = torch.cat([a1, eeg_x, a2], dim=2)
        if verbose: print(f"Fused LSTM Input: {x.shape}")
        
        x, (hidden_state, cell_state) = self.lstm1(x)
        if verbose: print(f"LSTM Output: {x.shape}")
        
        # Exact Flattening (Time * Features)
        x = x.reshape(-1, self.lstm_hidden_size*self.eeg_conv_shape[2])
        if verbose: print(f"Flattened for FC: {x.shape}")

        x = self.fc1_dp(F.relu(self.fc1(x)))
        x = self.fc2_dp(F.relu(self.fc2(x)))
        x = self.fc3_dp(F.relu(self.fc3(x)))
        
        # Note: Original code uses Softmax inside the model
        x = F.softmax(self.fc4(x), dim=1) 
        if verbose: print(f"Final Output: {x.shape}")
        
        return x
