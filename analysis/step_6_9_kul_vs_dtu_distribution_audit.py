import os
import sys
import numpy as np
import pandas as pd
import scipy.io as sio
import scipy.io.wavfile as wavfile
import torch
import matplotlib.pyplot as plt
from scipy.signal import resample, butter, filtfilt, welch

# Add repository root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from models.matchnet import ContrastiveMatchNet
except ImportError:
    ContrastiveMatchNet = None

from baselines.ridge_aad import load_subject_examples, subject_files
import json
import pickle

# --- ERB Filterbank Functions (KUL) ---
def erb_space(low_freq, high_freq, num_bands):
    erb_low = 21.4 * np.log10(4.37 * low_freq / 1000 + 1)
    erb_high = 21.4 * np.log10(4.37 * high_freq / 1000 + 1)
    erb_points = np.linspace(erb_low, erb_high, num_bands)
    cf = (10 ** (erb_points / 21.4) - 1) * 1000 / 4.37
    return cf

def get_erb_bands(cfs):
    bws = 24.7 * (4.37 * cfs / 1000 + 1)
    return cfs - bws / 2, cfs + bws / 2

def apply_bandpass(data, low, high, fs, order=2):
    nyq = 0.5 * fs
    low = max(low / nyq, 0.001)
    high = min(high / nyq, 0.999)
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)

def extract_28_band_envelope(audio_data, fs_in, fs_out=64, num_bands=28):
    cfs = erb_space(50, 8000, num_bands)
    lows, highs = get_erb_bands(cfs)
    envelopes = []
    for i in range(num_bands):
        band_audio = apply_bandpass(audio_data, lows[i], highs[i], fs_in)
        envelopes.append((np.abs(band_audio) ** 0.3))
    envelopes = np.array(envelopes)
    num_samples_out = int(envelopes.shape[1] * fs_out / fs_in)
    return resample(envelopes, num_samples_out, axis=1)

# --- Normalization & Misc ---
def normalize_array(arr):
    return (arr - arr.mean(axis=0, keepdims=True)) / (arr.std(axis=0, keepdims=True) + 1e-12)

def pearson_corr(x, y, dim=1):
    x_c = x - x.mean(dim=dim, keepdim=True)
    y_c = y - y.mean(dim=dim, keepdim=True)
    cov = (x_c * y_c).sum(dim=dim)
    return cov / torch.sqrt((x_c**2).sum(dim=dim) * (y_c**2).sum(dim=dim) + 1e-8)

def compute_psd(data, fs):
    f, Pxx = welch(data, fs, nperseg=fs*2)
    return f, Pxx

def main():
    print("="*60)
    print("PHASE KUL-4.5: DTU vs KUL DISTRIBUTION COMPARISON")
    print("="*60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs("analysis/figures/distribution_audit", exist_ok=True)
    
    # ---------------------------------------------------------
    # LOAD DTU S1 DATA
    # ---------------------------------------------------------
    print("\nLoading DTU S1 Data...")
    from pathlib import Path
    REPO_ROOT = Path(__file__).resolve().parents[1]
    
    # Load DTU S1 raw eeg and get the first example
    dtu_s1_examples = load_subject_examples('S1')
    if not dtu_s1_examples:
        print("ERROR: DTU S1 data not found!")
        return
        
    dtu_eeg = dtu_s1_examples[0].eeg  # shape: (T, 8)
    dtu_eeg_norm = normalize_array(dtu_eeg)
    dtu_label = dtu_s1_examples[0].label # 1 or 2
    dtu_trial_key = dtu_s1_examples[0].id
    
    # Get Audio Envelope for DTU
    map_file = REPO_ROOT / "data" / "audio_mapping.json"
    kaggle_dir = Path("/kaggle/input/datasets/lokeshgile/gammatone-envelope")
    env_file = list(kaggle_dir.glob("*.pkl"))[0] if kaggle_dir.exists() and list(kaggle_dir.glob("*.pkl")) else REPO_ROOT / "data" / "gammatone_envelopes.pkl"
    
    with open(map_file, 'r') as f: mapping = json.load(f)
    with open(env_file, 'rb') as f: envelopes = pickle.load(f)
    
    fname_a = mapping['S1'][dtu_trial_key]["wavA"]["filename"]
    fname_b = mapping['S1'][dtu_trial_key]["wavB"]["filename"]
    dtu_env_a = envelopes[fname_a].T  # (T, 28)
    dtu_env_b = envelopes[fname_b].T
    
    dtu_env_att = dtu_env_a if dtu_label == 1 else dtu_env_b
    dtu_env_unatt = dtu_env_b if dtu_label == 1 else dtu_env_a
    dtu_env_att = normalize_array(dtu_env_att).T # (28, T)
    dtu_env_unatt = normalize_array(dtu_env_unatt).T
    
    # ---------------------------------------------------------
    # LOAD KUL S1 TRIAL 0 DATA
    # ---------------------------------------------------------
    print("Loading KUL S1 Trial 0 Data...")
    mat_path = "/kaggle/input/datasets/lowk1ee/s1-klu/S1_KLU.mat"
    wav_dir = "/kaggle/input/datasets/lowk1ee/audio-klu"
    
    mat = sio.loadmat(mat_path, squeeze_me=True, struct_as_record=False)
    trials = mat['trials'] if 'trials' in mat else mat['trial']
    kul_trial = trials[0]
    
    fs_eeg = kul_trial.FileHeader.SampleRate
    channel_names = [ch.Label for ch in kul_trial.FileHeader.Channels]
    target_channels = ['Fp1', 'Fp2', 'F7', 'F8', 'T7', 'T8', 'P7', 'P8']
    selected_indices = [channel_names.index(tc) if tc in channel_names else [c.upper() for c in channel_names].index(tc.upper()) for tc in target_channels]
    
    kul_eeg = kul_trial.RawData.EegData[:, selected_indices]
    nyq = 0.5 * fs_eeg
    b, a = butter(2, [1.0/nyq, 6.0/nyq], btype='band')
    kul_eeg = filtfilt(b, a, kul_eeg, axis=0)
    kul_eeg = resample(kul_eeg, int(len(kul_eeg) * 64 / fs_eeg), axis=0)
    kul_eeg_norm = normalize_array(kul_eeg) # (T, 8)
    
    def find_wav(name):
        for r, d, f in os.walk(wav_dir):
            if name in f: return os.path.join(r, name)
            if name+".wav" in f: return os.path.join(r, name+".wav")
        return None
        
    att_ear = kul_trial.attended_ear
    stimuli = kul_trial.stimuli
    att_wav_name = stimuli[0] if att_ear == 'L' else stimuli[1]
    unatt_wav_name = stimuli[1] if att_ear == 'L' else stimuli[0]
    
    fs_att, audio_att = wavfile.read(find_wav(str(att_wav_name)))
    fs_unatt, audio_unatt = wavfile.read(find_wav(str(unatt_wav_name)))
    if len(audio_att.shape) > 1: audio_att = audio_att.mean(axis=1)
    if len(audio_unatt.shape) > 1: audio_unatt = audio_unatt.mean(axis=1)
    
    kul_env_att = extract_28_band_envelope(audio_att, fs_att)
    kul_env_unatt = extract_28_band_envelope(audio_unatt, fs_unatt)
    
    min_len = min(len(kul_eeg_norm), kul_env_att.shape[1], kul_env_unatt.shape[1])
    kul_eeg_norm = kul_eeg_norm[:min_len]
    kul_env_att = normalize_array(kul_env_att[:, :min_len].T).T # (28, T)
    kul_env_unatt = normalize_array(kul_env_unatt[:, :min_len].T).T
    
    # ---------------------------------------------------------
    # STUDY 1: EEG DISTRIBUTION AUDIT
    # ---------------------------------------------------------
    print("\n--- STUDY 1: EEG Distribution Audit ---")
    print(f"{'Metric':<15} | {'DTU S1 (T=~30s)':<20} | {'KUL S1 (T=~389s)'}")
    print("-" * 60)
    print(f"{'Mean':<15} | {dtu_eeg.mean():.6f}             | {kul_eeg.mean():.6f}")
    print(f"{'Std':<15} | {dtu_eeg.std():.6f}             | {kul_eeg.std():.6f}")
    print(f"{'RMS':<15} | {np.sqrt(np.mean(dtu_eeg**2)):.6f}             | {np.sqrt(np.mean(kul_eeg**2)):.6f}")
    print(f"{'Min':<15} | {dtu_eeg.min():.6f}             | {kul_eeg.min():.6f}")
    print(f"{'Max':<15} | {dtu_eeg.max():.6f}             | {kul_eeg.max():.6f}")
    
    # PSD Comparison (normalized)
    f_dtu, Pxx_dtu = compute_psd(dtu_eeg_norm[:,0], 64)
    f_kul, Pxx_kul = compute_psd(kul_eeg_norm[:,0], 64)
    
    plt.figure()
    plt.semilogy(f_dtu, Pxx_dtu, label='DTU (Ch Fp1)')
    plt.semilogy(f_kul, Pxx_kul, label='KUL (Ch Fp1)')
    plt.xlim([1, 20])
    plt.title("EEG PSD Comparison (1-20 Hz)")
    plt.legend()
    plt.savefig("analysis/figures/distribution_audit/eeg_psd.png")
    plt.close()
    
    # ---------------------------------------------------------
    # STUDY 2: AUDIO DISTRIBUTION AUDIT
    # ---------------------------------------------------------
    print("\n--- STUDY 2: Audio Envelope Distribution Audit ---")
    print("Comparing 28-Band Envelope (Attended Stream)")
    dtu_band_means = dtu_env_att.mean(axis=1)
    dtu_band_stds = dtu_env_att.std(axis=1)
    kul_band_means = kul_env_att.mean(axis=1)
    kul_band_stds = kul_env_att.std(axis=1)
    
    print(f"{'Metric':<15} | {'DTU Envelope':<20} | {'KUL Envelope'}")
    print("-" * 60)
    print(f"{'Mean of Means':<15} | {dtu_band_means.mean():.6f}             | {kul_band_means.mean():.6f}")
    print(f"{'Mean of Stds':<15} | {dtu_band_stds.mean():.6f}             | {kul_band_stds.mean():.6f}")
    
    # ---------------------------------------------------------
    # STUDY 3: EMBEDDING AUDIT
    # ---------------------------------------------------------
    print("\n--- STUDY 3: MatchNet Embedding Audit ---")
    chk_path = "checkpoints/matchnet_fold_S1_best.pth"
    if not os.path.exists(chk_path):
        candidates = []
        for r, d, f in os.walk("/kaggle/working/EEG_8_Channel_Pipeline/checkpoints"):
            for file in f:
                if 'best' in file.lower() or 'matchnet' in file.lower():
                    candidates.append(os.path.join(r, file))
        if not candidates:
            candidates = [os.path.join(r, file) for r, d, f in os.walk(".") for file in f if 'best' in file.lower()]
        chk_path = candidates[0] if candidates else None

    if not chk_path:
        print("ERROR: Could not find checkpoint!")
        return

    model = ContrastiveMatchNet(eeg_model_type="eegnet", eeg_channels=8, audio_channels=28, latent_dim=64).to(device)
    model.load_state_dict(torch.load(chk_path, map_location=device))
    model.eval()

    def get_embeddings(eeg_n, env_a, env_b, w_sec=3.0, s_sec=1.5):
        win_s = int(w_sec * 64)
        str_s = int(s_sec * 64)
        m_len = min(eeg_n.shape[0], env_a.shape[1], env_b.shape[1])
        xl, yal, ybl = [], [], []
        for start in range(0, m_len - win_s + 1, str_s):
            xl.append(eeg_n[start:start+win_s].T)
            yal.append(env_a[:, start:start+win_s])
            ybl.append(env_b[:, start:start+win_s])
            
        with torch.no_grad():
            x_t = torch.FloatTensor(np.array(xl)).to(device)
            ya_t = torch.FloatTensor(np.array(yal)).to(device)
            yb_t = torch.FloatTensor(np.array(ybl)).to(device)
            ze, za, zb = model(x_t, ya_t, yb_t)
            sim_a = pearson_corr(ze, za, dim=1).mean(dim=1)
            sim_b = pearson_corr(ze, zb, dim=1).mean(dim=1)
            return ze.cpu(), za.cpu(), zb.cpu(), sim_a.cpu(), sim_b.cpu()

    dtu_ze, dtu_za, dtu_zb, dtu_sa, dtu_sb = get_embeddings(dtu_eeg_norm, dtu_env_att, dtu_env_unatt)
    kul_ze, kul_za, kul_zb, kul_sa, kul_sb = get_embeddings(kul_eeg_norm, kul_env_att, kul_env_unatt)

    def print_emb_stats(name, z_e, z_a):
        n_e = torch.norm(z_e, dim=1).mean().item()
        n_a = torch.norm(z_a, dim=1).mean().item()
        v_e = z_e.var().item()
        v_a = z_a.var().item()
        print(f"[{name}] ||Ze||: {n_e:.4f}  ||Za||: {n_a:.4f}  Var(Ze): {v_e:.4f}  Var(Za): {v_a:.4f}")

    print_emb_stats("DTU", dtu_ze, dtu_za)
    print_emb_stats("KUL", kul_ze, kul_za)

    # ---------------------------------------------------------
    # STUDY 4 & 5: MARGIN & CONFIDENCE FEATURE AUDIT
    # ---------------------------------------------------------
    print("\n--- STUDY 4: Margin Distribution Audit ---")
    dtu_margin = torch.abs(dtu_sa - dtu_sb).numpy()
    kul_margin = torch.abs(kul_sa - kul_sb).numpy()

    dtu_pos_rate = (dtu_sa > dtu_sb).float().mean().item()
    kul_pos_rate = (kul_sa > kul_sb).float().mean().item()

    print(f"{'Metric':<15} | {'DTU Margins':<20} | {'KUL Margins'}")
    print("-" * 60)
    print(f"{'Mean':<15} | {dtu_margin.mean():.4f}               | {kul_margin.mean():.4f}")
    print(f"{'Std':<15} | {dtu_margin.std():.4f}               | {kul_margin.std():.4f}")
    print(f"{'Positive Rate':<15} | {dtu_pos_rate*100:.1f}%                | {kul_pos_rate*100:.1f}%")

    plt.figure()
    plt.hist(dtu_margin, bins=20, alpha=0.5, label='DTU', density=True)
    plt.hist(kul_margin, bins=20, alpha=0.5, label='KUL', density=True)
    plt.title("Margin Distribution Comparison")
    plt.legend()
    plt.savefig("analysis/figures/distribution_audit/margin_histogram.png")
    plt.close()

    print("\n--- FINAL VERDICT ---")
    margin_diff = abs(dtu_margin.mean() - kul_margin.mean())
    if margin_diff < 0.05 and abs(dtu_pos_rate - kul_pos_rate) < 0.2:
        print("Verdict: A. DTU and KUL occupy similar distributions")
    elif margin_diff < 0.15:
        print("Verdict: B. Moderate distribution shift")
    else:
        print("Verdict: C. Severe distribution mismatch")
        
    print("\nAll figures saved to analysis/figures/distribution_audit/")

if __name__ == "__main__":
    main()
