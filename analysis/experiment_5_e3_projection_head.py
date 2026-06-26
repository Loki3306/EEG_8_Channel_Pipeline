import os
import sys
import argparse
import scipy.io as sio
import scipy.io.wavfile as wavfile
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from scipy.signal import resample, butter, filtfilt
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path

# Add repository root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from models.matchnet import ContrastiveMatchNet, contrastive_loss
except ImportError as e:
    print(f"Could not import MatchNet: {e}")
    ContrastiveMatchNet = None
    
from preprocessing.euclidean_alignment import prepare_alignment_matrices, apply_alignment

# --- ERB Filterbank Functions ---
def erb_space(low_freq, high_freq, num_bands):
    erb_low = 21.4 * np.log10(4.37 * low_freq / 1000 + 1)
    erb_high = 21.4 * np.log10(4.37 * high_freq / 1000 + 1)
    erb_points = np.linspace(erb_low, erb_high, num_bands)
    cf = (10 ** (erb_points / 21.4) - 1) * 1000 / 4.37
    return cf

def get_erb_bands(cfs):
    bws = 24.7 * (4.37 * cfs / 1000 + 1)
    lows = cfs - bws / 2
    highs = cfs + bws / 2
    return lows, highs

def apply_bandpass(data, low, high, fs, order=2):
    nyq = 0.5 * fs
    low = max(low / nyq, 0.001)
    high = min(high / nyq, 0.999)
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)

def _process_band(audio_data, low, high, fs_in):
    band_audio = apply_bandpass(audio_data, low, high, fs_in)
    return np.abs(band_audio) ** 0.3

def extract_28_band_envelope(audio_data, fs_in, fs_out=64, num_bands=28):
    cfs = erb_space(50, 8000, num_bands)
    lows, highs = get_erb_bands(cfs)
    
    from joblib import Parallel, delayed
    envelopes = Parallel(n_jobs=-1)(
        delayed(_process_band)(audio_data, lows[i], highs[i], fs_in) 
        for i in range(num_bands)
    )
        
    envelopes = np.array(envelopes)
    
    import math
    from scipy.signal import resample_poly
    g = math.gcd(fs_out, fs_in)
    up = fs_out // g
    down = fs_in // g
    
    return resample_poly(envelopes, up, down, axis=1)

# --- Normalization ---
def normalize_array(arr):
    arr = arr - arr.mean(axis=0, keepdims=True)
    scale = arr.std(axis=0, keepdims=True) + 1e-12
    return arr / scale

def find_checkpoint(root_dir):
    candidates = []
    for root, dirs, files in os.walk(root_dir):
        for f in files:
            if f.endswith('.pth') or f.endswith('.pt'):
                candidates.append(os.path.join(root, f))
    for c in candidates:
        if 'best' in c.lower() or 'matchnet' in c.lower(): return c
    return candidates[0] if candidates else None

class WindowDataset(Dataset):
    def __init__(self, eeg_list, aa_list, ab_list):
        self.eeg = []
        self.aa = []
        self.ab = []
        for e, a, b in zip(eeg_list, aa_list, ab_list):
            self.eeg.append(torch.FloatTensor(e).unsqueeze(0)) 
            self.aa.append(torch.FloatTensor(a).unsqueeze(0))
            self.ab.append(torch.FloatTensor(b).unsqueeze(0))
            
    def __len__(self):
        return len(self.eeg)
        
    def __getitem__(self, idx):
        return self.eeg[idx].squeeze(0), self.aa[idx].squeeze(0), self.ab[idx].squeeze(0)

def main():
    print("\n" + "="*60)
    print("PHASE KUL-5: E3 - FINE-TUNE EEG ENCODER + PROJECTION HEAD")
    print("="*60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # 1. Load Model
    checkpoint_dir = "/kaggle/working/EEG_8_Channel_Pipeline/checkpoints"
    if not os.path.exists(checkpoint_dir):
        checkpoint_dir = "checkpoints" # fallback
        
    chk_path = find_checkpoint(checkpoint_dir)
    if not chk_path:
        chk_path = find_checkpoint(".")
        
    if not chk_path:
        print("ERROR: No DTU MatchNet checkpoint found!")
        return
        
    print(f"Loading DTU Checkpoint: {chk_path}")
    model = ContrastiveMatchNet(eeg_model_type="eegnet", eeg_channels=8, audio_channels=28, latent_dim=64).to(device)
    model.load_state_dict(torch.load(chk_path, map_location=device))
    
    # 2. Freeze/Unfreeze Logic
    print("Freezing layers...")
    for param in model.parameters():
        param.requires_grad = False
        
    print("Unfreezing entire EEG Encoder AND Projection Head...")
    for param in model.eeg_encoder.block1.parameters():
        param.requires_grad = True
    for param in model.eeg_encoder.block2.parameters():
        param.requires_grad = True
    for param in model.eeg_encoder.output_proj.parameters():
        param.requires_grad = True
        
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable Parameters: {trainable_params:,} / {total_params:,}")
    
    # 3. Config
    mat_path = "/kaggle/input/datasets/lowk1ee/s1-klu/S1_KLU.mat"
    wav_dir = "/kaggle/input/datasets/lowk1ee/audio-klu"
    target_channels = ['Fp1', 'Fp2', 'F7', 'F8', 'T7', 'T8', 'P7', 'P8']
    fs_dtu = 64
    window_sec, stride_sec = 3.0, 1.5
    win_samples = int(window_sec * fs_dtu)
    stride_samples = int(stride_sec * fs_dtu)
    
    print("\nLoading KUL MAT file...")
    mat = sio.loadmat(mat_path, squeeze_me=True, struct_as_record=False)
    trials = mat['trials'] if 'trials' in mat else mat['trial']
    
    # 4. Extract data (Train/Test split)
    print("Extracting trials and computing envelopes...")
    audio_cache = {}
    def find_wav(name):
        for r, d, f in os.walk(wav_dir):
            if name in f: return os.path.join(r, name)
            if name+".wav" in f: return os.path.join(r, name+".wav")
        return None

    train_eeg, train_aa, train_ab = [], [], []
    test_eeg, test_aa, test_ab = [], [], []
    test_trial_windows = [] # Store number of windows per test trial
    
    for t_idx, trial in enumerate(trials):
        print(f"  [Trial {t_idx+1:02d}/{len(trials):02d}] Extracting...")
        eeg_data = trial.RawData.EegData
        fs_eeg = trial.FileHeader.SampleRate
        channel_names = [ch.Label for ch in trial.FileHeader.Channels]
        
        selected_indices = [channel_names.index(tc) if tc in channel_names else [c.upper() for c in channel_names].index(tc.upper()) for tc in target_channels]
        eeg_8 = eeg_data[:, selected_indices]
        
        nyq = 0.5 * fs_eeg
        b, a = butter(2, [1.0/nyq, 6.0/nyq], btype='band')
        eeg_8 = filtfilt(b, a, eeg_8, axis=0)
        eeg_64 = resample(eeg_8, int(len(eeg_8) * fs_dtu / fs_eeg), axis=0)
        
        att_ear = trial.attended_ear
        stimuli = trial.stimuli
        att_wav_name = stimuli[0] if att_ear == 'L' else stimuli[1]
        unatt_wav_name = stimuli[1] if att_ear == 'L' else stimuli[0]
        
        att_wav_path = find_wav(str(att_wav_name))
        unatt_wav_path = find_wav(str(unatt_wav_name))
        
        if not att_wav_path or not unatt_wav_path:
            print(f"    WARNING: Missing audio for Trial {t_idx}. Skipping.")
            continue
            
        def get_cached_env(wav_path):
            if wav_path in audio_cache:
                return audio_cache[wav_path]
            fs, audio = wavfile.read(wav_path)
            if len(audio.shape) > 1: audio = audio.mean(axis=1)
            env = extract_28_band_envelope(audio, fs, fs_out=64, num_bands=28)
            audio_cache[wav_path] = env
            return env
            
        env_att = get_cached_env(att_wav_path)
        env_unatt = get_cached_env(unatt_wav_path)
        
        min_len = min(len(eeg_64), env_att.shape[1], env_unatt.shape[1])
        eeg_64 = eeg_64[:min_len]
        env_att = env_att[:, :min_len]
        env_unatt = env_unatt[:, :min_len]
        
        eeg_norm = normalize_array(eeg_64) 
        env_att = normalize_array(env_att.T).T 
        env_unatt = normalize_array(env_unatt.T).T
        
        num_windows_for_this_trial = 0
        for start in range(0, min_len - win_samples + 1, stride_samples):
            x = eeg_norm[start:start+win_samples].T 
            ya = env_att[:, start:start+win_samples] 
            yb = env_unatt[:, start:start+win_samples]
            
            if t_idx < 15: # Train Split
                train_eeg.append(x)
                train_aa.append(ya)
                train_ab.append(yb)
            else: # Test Split
                test_eeg.append(x)
                test_aa.append(ya)
                test_ab.append(yb)
                num_windows_for_this_trial += 1
                
        if t_idx >= 15:
            test_trial_windows.append(num_windows_for_this_trial)
            
    # 5. Training Setup
    print("\nSetting up DataLoaders...")
    train_dataset = WindowDataset(train_eeg, train_aa, train_ab)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    
    # Combine parameters for block 1, block 2, and output_proj
    params_to_update = list(model.eeg_encoder.block1.parameters()) + \
                       list(model.eeg_encoder.block2.parameters()) + \
                       list(model.eeg_encoder.output_proj.parameters())
    optimizer = torch.optim.Adam(params_to_update, lr=5e-4)
    epochs = 15
    
    print("\n" + "="*40)
    print("STARTING PARTIAL FINE-TUNING")
    print("="*40)
    def evaluate_split(loader, trial_windows=None):
        model.eval()
        all_sim_a = []
        all_sim_b = []
        all_norms = []
        total_loss = 0
        
        with torch.no_grad():
            for eeg, aa, ab in loader:
                eeg, aa, ab = eeg.to(device), aa.to(device), ab.to(device)
                z_eeg, z_a, z_b = model(eeg, aa, ab)
                
                loss, _, _ = contrastive_loss(z_eeg, z_a, z_b, margin=0.1)
                total_loss += loss.item()
                
                sim_a = F.cosine_similarity(z_eeg, z_a, dim=1).cpu().numpy()
                sim_b = F.cosine_similarity(z_eeg, z_b, dim=1).cpu().numpy()
                norms = torch.norm(z_eeg, dim=1).cpu().numpy()
                
                all_sim_a.extend(sim_a)
                all_sim_b.extend(sim_b)
                all_norms.extend(norms)
                
        all_sim_a = np.array(all_sim_a)
        all_sim_b = np.array(all_sim_b)
        all_norms = np.array(all_norms)
        margins = all_sim_a - all_sim_b
        
        avg_loss = total_loss / len(loader)
        window_acc = (margins > 0).mean()
        mean_margin = margins.mean()
        mean_sim_a = all_sim_a.mean()
        mean_sim_b = all_sim_b.mean()
        mean_norm = all_norms.mean()
        
        trial_acc = 0.0
        if trial_windows:
            offset = 0
            correct_trials = 0
            for num_wins in trial_windows:
                trial_margins = margins[offset:offset+num_wins]
                if trial_margins.mean() > 0:
                    correct_trials += 1
                offset += num_wins
            trial_acc = correct_trials / len(trial_windows)
            
        delta = mean_sim_a - mean_sim_b
            
        return avg_loss, trial_acc, window_acc, mean_margin, mean_sim_a, mean_sim_b, mean_norm, delta, margins

    print("\n" + "="*40)
    print("STARTING PARTIAL FINE-TUNING")
    print("="*40)
    
    test_dataset = WindowDataset(test_eeg, test_aa, test_ab)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    # E0: Zero-shot baseline
    print("\n--- ZERO-SHOT BASELINE (E0) ---")
    tr_loss, _, tr_wacc, tr_marg, tr_sa, tr_sb, tr_norm, tr_delta, _ = evaluate_split(train_loader)
    te_loss, te_tacc, te_wacc, te_marg, te_sa, te_sb, te_norm, te_delta, _ = evaluate_split(test_loader, test_trial_windows)
    
    print(f"[Train] Loss: {tr_loss:.4f} | WinAcc: {tr_wacc*100:.1f}% | Margin: {tr_marg:.4f} | SimA: {tr_sa:.4f} | SimB: {tr_sb:.4f} | Delta: {tr_delta:.4f} | Norm: {tr_norm:.4f}")
    print(f"[Test]  Loss: {te_loss:.4f} | TrialAcc: {te_tacc*100:.1f}% | WinAcc: {te_wacc*100:.1f}% | Margin: {te_marg:.4f} | SimA: {te_sa:.4f} | SimB: {te_sb:.4f} | Delta: {te_delta:.4f} | Norm: {te_norm:.4f}")
    
    for epoch in range(epochs):
        model.train()
        for eeg, aa, ab in train_loader:
            eeg, aa, ab = eeg.to(device), aa.to(device), ab.to(device)
            optimizer.zero_grad()
            z_eeg, z_a, z_b = model(eeg, aa, ab)
            loss, _, _ = contrastive_loss(z_eeg, z_a, z_b, margin=0.1)
            loss.backward()
            optimizer.step()
            
        print(f"\n--- EPOCH {epoch+1} ---")
        tr_loss, _, tr_wacc, tr_marg, tr_sa, tr_sb, tr_norm, tr_delta, final_tr_margins = evaluate_split(train_loader)
        te_loss, te_tacc, te_wacc, te_marg, te_sa, te_sb, te_norm, te_delta, final_te_margins = evaluate_split(test_loader, test_trial_windows)
        
        print(f"[Train] Loss: {tr_loss:.4f} | WinAcc: {tr_wacc*100:.1f}% | Margin: {tr_marg:.4f} | SimA: {tr_sa:.4f} | SimB: {tr_sb:.4f} | Delta: {tr_delta:.4f} | Norm: {tr_norm:.4f}")
        print(f"[Test]  Loss: {te_loss:.4f} | TrialAcc: {te_tacc*100:.1f}% | WinAcc: {te_wacc*100:.1f}% | Margin: {te_marg:.4f} | SimA: {te_sa:.4f} | SimB: {te_sb:.4f} | Delta: {te_delta:.4f} | Norm: {te_norm:.4f}")
    
    # Save finetuned model
    save_path = os.path.join(checkpoint_dir, "matchnet_kul_finetuned_e3.pth")
    torch.save(model.state_dict(), save_path)
    print(f"\nSaved finetuned model to {save_path}")
    
    # Generate final distribution statistics and plot
    print("\n" + "="*40)
    print("FINAL DELTA DISTRIBUTION ANALYSIS")
    print("="*40)
    
    print(f"Train Delta Mean: {final_tr_margins.mean():.4f} | Std: {final_tr_margins.std():.4f}")
    print(f"Test  Delta Mean: {final_te_margins.mean():.4f} | Std: {final_te_margins.std():.4f}")
    
    plt.figure(figsize=(10, 6))
    plt.hist(final_tr_margins, bins=50, alpha=0.5, label='Train Delta (SimA - SimB)', density=True)
    plt.hist(final_te_margins, bins=50, alpha=0.5, label='Test Delta (SimA - SimB)', density=True)
    plt.axvline(0, color='red', linestyle='dashed', linewidth=2)
    plt.title("E3: Delta (SimA - SimB) Distribution after Projection Head Fine-Tuning")
    plt.xlabel("Delta (SimA - SimB)")
    plt.ylabel("Density")
    plt.legend()
    
    os.makedirs("analysis/figures/layer_inspector", exist_ok=True)
    plot_path = "analysis/figures/layer_inspector/e3_delta_distribution.png"
    plt.savefig(plot_path)
    print(f"Saved distribution plot to {plot_path}")

if __name__ == "__main__":
    main()
