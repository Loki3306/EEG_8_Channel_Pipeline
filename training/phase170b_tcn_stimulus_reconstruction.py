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
from sklearn.model_selection import TimeSeriesSplit
import concurrent.futures
import multiprocessing as mp

# -------------------------------------------------------------------------
# CONSTANTS & CONFIGURATION
# -------------------------------------------------------------------------
SR = 128
EAR_CHANNEL_INDICES = [23, 31, 32, 40, 14, 22, 41, 49]
SEQ_SAMPLES = int(15.0 * SR)  # 15 seconds context
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
        self.chomp1 = Chomp1d(padding)
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

class ReconstructionTCN(nn.Module):
    # Dilations: 1, 2, 4, 8, 16, 32. Receptive field = 1 + 2*(7-1)*(63) = 757 samples = ~5.9 seconds
    def __init__(self, num_inputs=8, num_channels=[16, 32, 64, 64, 64, 64], kernel_size=7, dropout=0.3):
        super(ReconstructionTCN, self).__init__()
        self.tcn = TemporalConvNet(num_inputs, num_channels, kernel_size=kernel_size, dropout=dropout)
        self.head = nn.Sequential(
            nn.Conv1d(num_channels[-1], 32, 1),
            nn.GELU(),
            nn.Conv1d(32, 1, 1)
        )
        
    def forward(self, x):
        y = self.tcn(x)
        out = self.head(y)
        return out

class DiscriminativePearsonLoss(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, pred, target_att, target_unatt):
        # Skip the receptive field (6 seconds)
        skip = int(6.0 * SR)
        p = pred[:, 0, skip:]
        t_a = target_att[:, 0, skip:]
        t_u = target_unatt[:, 0, skip:]
        
        def calc_corr(pred_c, target_c):
            p_mean = pred_c.mean(dim=1, keepdim=True)
            t_mean = target_c.mean(dim=1, keepdim=True)
            p_cen = pred_c - p_mean
            t_cen = target_c - t_mean
            num = (p_cen * t_cen).sum(dim=1)
            den = torch.sqrt((p_cen**2).sum(dim=1) * (t_cen**2).sum(dim=1) + 1e-8)
            return num / den
            
        corr_att = calc_corr(p, t_a)
        corr_unatt = calc_corr(p, t_u)
        
        # Discriminative ranking loss
        return -(corr_att - corr_unatt).mean()

class EEGEnvelopeDataset(Dataset):
    def __init__(self, X, Y_att, Y_unatt):
        self.X = torch.tensor(np.array(X), dtype=torch.float32)
        self.Y_att = torch.tensor(np.array(Y_att), dtype=torch.float32).unsqueeze(1)
        self.Y_unatt = torch.tensor(np.array(Y_unatt), dtype=torch.float32).unsqueeze(1)
        
    def __len__(self):
        return len(self.X)
        
    def __getitem__(self, idx):
        return self.X[idx], self.Y_att[idx], self.Y_unatt[idx]

def apply_modulation_filter(eeg_raw, lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    b, a = signal.butter(order, [lowcut / nyq, highcut / nyq], btype='band')
    return signal.filtfilt(b, a, eeg_raw, axis=1)

def extract_sequences(cache_file):
    cached = torch.load(cache_file, map_location='cpu', weights_only=False)['raw']
    sequences_X = []
    sequences_Y_att = []
    sequences_Y_unatt = []
    
    for tr in cached:
        eeg_raw = tr['eeg'].numpy()[EAR_CHANNEL_INDICES, :]
        env_L = tr['env_l'].numpy().flatten()
        env_R = tr['env_r'].numpy().flatten()
        
        eeg_f = apply_modulation_filter(eeg_raw, BROADBAND[0], BROADBAND[1], SR)
        env_L = apply_modulation_filter(np.expand_dims(env_L, 0), BROADBAND[0], BROADBAND[1], SR).flatten()
        env_R = apply_modulation_filter(np.expand_dims(env_R, 0), BROADBAND[0], BROADBAND[1], SR).flatten()
        
        # Standardize
        eeg_f = (eeg_f - np.mean(eeg_f, axis=1, keepdims=True)) / (np.std(eeg_f, axis=1, keepdims=True) + 1e-8)
        env_L = (env_L - np.mean(env_L)) / (np.std(env_L) + 1e-8)
        env_R = (env_R - np.mean(env_R)) / (np.std(env_R) + 1e-8)
        
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
                    env_L_seq = env_L[seq_start:seq_start + SEQ_SAMPLES]
                    env_R_seq = env_R[seq_start:seq_start + SEQ_SAMPLES]
                    
                    if current_spk == 'L':
                        y_att, y_unatt = env_L_seq, env_R_seq
                    else:
                        y_att, y_unatt = env_R_seq, env_L_seq
                        
                    sequences_X.append(eeg_seq)
                    sequences_Y_att.append(y_att)
                    sequences_Y_unatt.append(y_unatt)
                    
    return sequences_X, sequences_Y_att, sequences_Y_unatt

def calculate_pearsonr(pred, target):
    # Skip receptive field of 6s
    skip = int(6.0 * SR)
    p = pred[skip:]
    t = target[skip:]
    p = p - np.mean(p)
    t = t - np.mean(t)
    num = np.sum(p * t)
    den = np.sqrt(np.sum(p**2) * np.sum(t**2)) + 1e-8
    return num / den

def process_subject(cache_file):
    torch.set_num_threads(1)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    subj_name = cache_file.stem.split('_')[0]
    
    X_all, Y_att_all, Y_unatt_all = extract_sequences(cache_file)
    if len(X_all) < 50:
        return subj_name, None
        
    X_all = np.array(X_all)
    Y_att_all = np.array(Y_att_all)
    Y_unatt_all = np.array(Y_unatt_all)
    
    tscv = TimeSeriesSplit(n_splits=5)
    accuracies = []
    
    for fold, (train_idx, test_idx) in enumerate(tscv.split(X_all)):
        # Prevent sequence overlap leakage between train and test
        # SEQ_SAMPLES is 15s, SEQ_HOP is 2s, so overlap is up to 13s / 2s = 7 samples
        overlap_margin = int(SEQ_SAMPLES / SEQ_HOP)
        if len(test_idx) > overlap_margin:
            test_idx = test_idx[overlap_margin:]
            
        X_train, X_test = X_all[train_idx], X_all[test_idx]
        Y_att_train, Y_att_test = Y_att_all[train_idx], Y_att_all[test_idx]
        Y_unatt_train, Y_unatt_test = Y_unatt_all[train_idx], Y_unatt_all[test_idx]
        
        train_dataset = EEGEnvelopeDataset(X_train, Y_att_train, Y_unatt_train)
        test_dataset = EEGEnvelopeDataset(X_test, Y_att_test, Y_unatt_test)
        
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        
        model = ReconstructionTCN().to(device)
        criterion = DiscriminativePearsonLoss()
        optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        
        for epoch in range(15):
            model.train()
            for batch_X, batch_Y_att, batch_Y_unatt in train_loader:
                batch_X, batch_Y_att, batch_Y_unatt = batch_X.to(device), batch_Y_att.to(device), batch_Y_unatt.to(device)
                optimizer.zero_grad()
                out = model(batch_X)
                loss = criterion(out, batch_Y_att, batch_Y_unatt)
                loss.backward()
                optimizer.step()
                
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_X, batch_Y_att, batch_Y_unatt in test_loader:
                batch_X = batch_X.to(device)
                out = model(batch_X).cpu().numpy() # (B, 1, T)
                
                y_att = batch_Y_att.numpy()
                y_unatt = batch_Y_unatt.numpy()
                
                for b in range(out.shape[0]):
                    pred = out[b, 0, :]
                    att = y_att[b, 0, :]
                    unatt = y_unatt[b, 0, :]
                    
                    corr_att = calculate_pearsonr(pred, att)
                    corr_unatt = calculate_pearsonr(pred, unatt)
                    
                    if corr_att > corr_unatt:
                        correct += 1
                    total += 1
                    
        fold_acc = correct / total
        accuracies.append(fold_acc)
        
    mean_acc = np.mean(accuracies)
    print(f"[{subj_name}] TCN SR Accuracy: {mean_acc*100:.1f}%")
    
    return subj_name, mean_acc

def main():
    print("=======================================================")
    print(" PHASE 170B: NON-LINEAR STIMULUS RECONSTRUCTION (TCN)")
    print(" Predicting Auditory Envelopes via Causal ConvNet")
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
        print(f"  TCN SR Accuracy : {acc*100:.1f}%\n")
        
    print("=======================================================")
    print(" GLOBAL OBSERVABILITY AVERAGES")
    print("=======================================================")
    print(f"Global TCN SR Acc : {np.mean(global_acc):.2f}%")
    print("=======================================================")
    
if __name__ == '__main__':
    main()
