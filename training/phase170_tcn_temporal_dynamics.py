import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from scipy import signal
from sklearn.model_selection import KFold
import concurrent.futures
import multiprocessing as mp

# -------------------------------------------------------------------------
# CONSTANTS & CONFIGURATION
# -------------------------------------------------------------------------
SR = 128
EAR_CHANNEL_INDICES = [23, 31, 32, 40, 14, 22, 41, 49]
SEQ_SAMPLES = int(10.0 * SR)  # 10 seconds context (to preserve dataset size)
SEQ_HOP = int(2.0 * SR)       # 2 second hop
BROADBAND = (0.5, 8.0)

# -------------------------------------------------------------------------
# TCN ARCHITECTURE
# -------------------------------------------------------------------------
class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size
    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        # Chomp1d removes the future padding to make it causal
        self.chomp1 = Chomp1d(padding)
        # We must use BatchNorm1d instead of LayerNorm for Ear-EEG per workspace rules
        self.bn1 = nn.BatchNorm1d(n_outputs)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.bn2 = nn.BatchNorm1d(n_outputs)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.bn1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.bn2, self.relu2, self.dropout2)
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
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

class AttentionTCN(nn.Module):
    # To cover 10s = 1280 samples with kernel=7, we need dilations up to 2^6=64.
    # Receptive field = 1 + 2*(7-1)*(1+2+4+8+16+32+64) = 1 + 12*127 = 1525 samples = 11.9 seconds
    def __init__(self, num_inputs=8, num_channels=[16, 32, 64, 64, 64, 64, 64], kernel_size=7, dropout=0.3):
        super(AttentionTCN, self).__init__()
        self.tcn = TemporalConvNet(num_inputs, num_channels, kernel_size=kernel_size, dropout=dropout)
        self.linear = nn.Linear(num_channels[-1], 1)
        
    def forward(self, x):
        # x: (B, C, T)
        y1 = self.tcn(x)
        # We predict the attention state of the final 2 seconds
        pool = y1[:, :, -int(2 * SR):].mean(dim=2)  # (B, C)
        out = self.linear(pool) # (B, 1)
        return out

class EEGSequenceDataset(Dataset):
    def __init__(self, eeg_sequences, labels):
        self.X = torch.tensor(np.array(eeg_sequences), dtype=torch.float32)
        self.Y = torch.tensor(np.array(labels), dtype=torch.float32).unsqueeze(1)
        
    def __len__(self):
        return len(self.X)
        
    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

def apply_modulation_filter(eeg_raw, lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    b, a = signal.butter(order, [lowcut / nyq, highcut / nyq], btype='band')
    return signal.filtfilt(b, a, eeg_raw, axis=1)

def extract_sequences(cache_file):
    cached = torch.load(cache_file, map_location='cpu', weights_only=False)['raw']
    sequences = []
    labels = []
    
    for tr in cached:
        eeg_raw = tr['eeg'].numpy()[EAR_CHANNEL_INDICES, :]
        eeg_f = apply_modulation_filter(eeg_raw, BROADBAND[0], BROADBAND[1], SR)
        
        # Standardize per trial (mean 0, var 1)
        eeg_f = (eeg_f - np.mean(eeg_f, axis=1, keepdims=True)) / (np.std(eeg_f, axis=1, keepdims=True) + 1e-8)
        
        T = eeg_f.shape[1]
        
        sp = tr['meta']['switch_points']
        boundaries = [0] + [idx for spk, idx in sp]
        boundaries = sorted(set(boundaries))
        if boundaries[-1] != T: boundaries.append(T)
            
        for i in range(len(boundaries) - 1):
            start_idx = boundaries[i]
            end_idx = boundaries[i+1]
            
            current_spk = 'L'
            for spk, idx in sp:
                if idx <= start_idx: current_spk = spk
                else: break
                
            safe_start = start_idx + int(1.5 * SR)
            safe_end = end_idx
            
            if safe_end - safe_start >= SEQ_SAMPLES:
                for seq_start in range(safe_start, safe_end - SEQ_SAMPLES + 1, SEQ_HOP):
                    eeg_seq = eeg_f[:, seq_start:seq_start + SEQ_SAMPLES]
                    label = 1 if current_spk == 'L' else 0
                    
                    sequences.append(eeg_seq)
                    labels.append(label)
                    
    return sequences, labels

def process_subject(cache_file):
    torch.set_num_threads(1)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    subj_name = cache_file.stem.split('_')[0]
    
    sequences, labels = extract_sequences(cache_file)
    if len(sequences) < 50:
        return subj_name, None
        
    X_all = np.array(sequences)
    Y_all = np.array(labels)
    
    # 5-Fold Chronological Split (Strictly TimeSeriesSplit)
    from sklearn.model_selection import TimeSeriesSplit
    tscv = TimeSeriesSplit(n_splits=5)
    
    accuracies = []
    
    for fold, (train_idx, test_idx) in enumerate(tscv.split(X_all)):
        X_train, X_test = X_all[train_idx], X_all[test_idx]
        Y_train, Y_test = Y_all[train_idx], Y_all[test_idx]
        
        train_dataset = EEGSequenceDataset(X_train, Y_train)
        test_dataset = EEGSequenceDataset(X_test, Y_test)
        
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        
        model = AttentionTCN().to(device)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        
        # Train for 15 epochs
        for epoch in range(15):
            model.train()
            for batch_X, batch_Y in train_loader:
                batch_X, batch_Y = batch_X.to(device), batch_Y.to(device)
                optimizer.zero_grad()
                out = model(batch_X)
                loss = criterion(out, batch_Y)
                loss.backward()
                optimizer.step()
                
        # Evaluate
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_X, batch_Y in test_loader:
                batch_X, batch_Y = batch_X.to(device), batch_Y.to(device)
                out = model(batch_X)
                preds = (torch.sigmoid(out) > 0.5).float()
                correct += (preds == batch_Y).sum().item()
                total += batch_Y.size(0)
                
        fold_acc = correct / total
        accuracies.append(fold_acc)
        
    mean_acc = np.mean(accuracies)
    print(f"[{subj_name}] TCN End-to-End Accuracy: {mean_acc*100:.1f}%")
    
    return subj_name, mean_acc

def main():
    print("=======================================================")
    print(" PHASE 170A: LONG-RANGE TEMPORAL SUFFICIENCY TEST")
    print(" Causal TCN (30s) directly predicting Spatial Attention")
    print("=======================================================\n")
    
    cache_dir = Path('/kaggle/input/datasets/lowkieee/multiband-cache/kaggle/working/multiband_cache')
    possible_paths = [
        Path('/kaggle/input/datasets/lokeshgile/aasd-universal-cache-v1'),
        cache_dir,
        Path('/kaggle/working/multiband_cache')
    ]
    for p in possible_paths:
        if p.exists() and len(list(p.glob('*_multiband.pt'))) > 0:
            cache_dir = p
            break
            
    cache_files = sorted(list(cache_dir.glob('*_multiband.pt')))
    
    start_time = time.time()
    results = {}
    
    # We will process subjects sequentially or with minimal workers if GPU is available
    if torch.cuda.is_available():
        print("Using CUDA for TCN training...")
        for cf in cache_files:
            subj, acc = process_subject(cf)
            if acc is not None:
                results[subj] = acc
    else:
        print("Using CPU. Distributing across processes...")
        with concurrent.futures.ProcessPoolExecutor(max_workers=max(1, mp.cpu_count()//2)) as executor:
            futures = {executor.submit(process_subject, cf): cf for cf in cache_files}
            for future in concurrent.futures.as_completed(futures):
                subj, acc = future.result()
                if acc is not None:
                    results[subj] = acc
                
    print(f"\nExtraction & Training Time: {time.time() - start_time:.2f}s\n")
    
    global_acc = []
    
    subjects_sorted = sorted(results.keys())
    for subj in subjects_sorted:
        acc = results[subj]
        global_acc.append(acc * 100)
        print(f"--- Subject: {subj} ---")
        print(f"  TCN Accuracy : {acc*100:.1f}%\n")
        
    print("=======================================================")
    print(" GLOBAL OBSERVABILITY AVERAGES")
    print("=======================================================")
    print(f"Global TCN Spatial Acc : {np.mean(global_acc):.2f}%")
    print("=======================================================")
    
if __name__ == '__main__':
    main()
