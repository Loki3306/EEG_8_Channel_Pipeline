import os
import sys
import numpy as np
import scipy.io as sio
import scipy.io.wavfile as wavfile
import torch
import torch.nn.functional as F
import pandas as pd
from pathlib import Path
from scipy.signal import resample, butter, filtfilt

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT))

from models.matchnet import ContrastiveMatchNet
from preprocessing.euclidean_alignment import prepare_alignment_matrices, apply_alignment

def normalize_array(arr):
    return (arr - arr.mean(axis=0, keepdims=True)) / (arr.std(axis=0, keepdims=True) + 1e-12)

def extract_28_band_envelope(audio_data, fs_in, fs_out=64, num_bands=28):
    def erb_space(low, high, n):
        erb_low = 21.4 * np.log10(4.37 * low / 1000 + 1)
        erb_high = 21.4 * np.log10(4.37 * high / 1000 + 1)
        erb_points = np.linspace(erb_low, erb_high, n)
        return (10 ** (erb_points / 21.4) - 1) * 1000 / 4.37
    def get_erb_bands(cfs):
        bws = 24.7 * (4.37 * cfs / 1000 + 1)
        return cfs - bws / 2, cfs + bws / 2
    cfs = erb_space(50, 8000, num_bands)
    lows, highs = get_erb_bands(cfs)
    envelopes = []
    for i in range(num_bands):
        nyq = 0.5 * fs_in
        l, h = max(lows[i]/nyq, 0.001), min(highs[i]/nyq, 0.999)
        b, a = butter(2, [l, h], btype='band')
        band_audio = filtfilt(b, a, audio_data)
        envelopes.append((np.abs(band_audio) ** 0.3))
    envelopes = np.array(envelopes)
    num_samples_out = int(envelopes.shape[1] * fs_out / fs_in)
    return resample(envelopes, num_samples_out, axis=1)

def pearson_corr(x, y, dim=1):
    x_c = x - x.mean(dim=dim, keepdim=True)
    y_c = y - y.mean(dim=dim, keepdim=True)
    return (x_c * y_c).sum(dim=dim) / torch.sqrt((x_c**2).sum(dim=dim) * (y_c**2).sum(dim=dim) + 1e-8)

def load_all_kul_data():
    mat_path = "/kaggle/input/datasets/lowk1ee/s1-klu/S1_KLU.mat"
    if not os.path.exists(mat_path):
        mat_path = str(REPO_ROOT / "data" / "S1_KLU.mat")
        if not os.path.exists(mat_path):
            raise FileNotFoundError("KUL dataset not found.")
            
    mat_data = sio.loadmat(mat_path, squeeze_me=True, struct_as_record=False)
    trials = mat_data['data']
    
    fs_eeg = 128
    fs_dtu = 64
    selected_indices = [13, 46, 43, 23, 50, 0, 52, 14]
    
    all_trials = []
    for t_idx, trial in enumerate(trials):
        eeg_data = trial.EEG
        if len(eeg_data.shape) > 2: eeg_data = eeg_data[0]
        eeg_8 = eeg_data[:, selected_indices]
        
        nyq = 0.5 * fs_eeg
        b, a = butter(2, [1.0/nyq, 6.0/nyq], btype='band')
        eeg_8 = filtfilt(b, a, eeg_8, axis=0)
        eeg_64 = resample(eeg_8, int(len(eeg_8) * fs_dtu / fs_eeg), axis=0)
        
        att_ear = trial.attended_ear
        stimuli = trial.stimuli
        att_wav_name = stimuli[0] if att_ear == 'L' else stimuli[1]
        unatt_wav_name = stimuli[1] if att_ear == 'L' else stimuli[0]
        
        # We need to simulate audio extraction or load if available.
        # But for this experiment, we can just load the audio.
        # In kaggle it's under /kaggle/input/datasets/lowk1ee/s1-klu/
        audio_dir = Path(mat_path).parent
        att_wav_path = list(audio_dir.glob(f"*{att_wav_name}*.wav"))
        unatt_wav_path = list(audio_dir.glob(f"*{unatt_wav_name}*.wav"))
        
        if not att_wav_path or not unatt_wav_path:
            continue
            
        fs_a, audio_a = wavfile.read(att_wav_path[0])
        fs_u, audio_u = wavfile.read(unatt_wav_path[0])
        if len(audio_a.shape) > 1: audio_a = audio_a.mean(axis=1)
        if len(audio_u.shape) > 1: audio_u = audio_u.mean(axis=1)
        
        env_a = extract_28_band_envelope(audio_a, fs_a)
        env_u = extract_28_band_envelope(audio_u, fs_u)
        
        min_len = min(len(eeg_64), env_a.shape[1], env_u.shape[1])
        all_trials.append({
            'eeg': eeg_64[:min_len].T, # (8, T)
            'env_att': env_a[:, :min_len], # (28, T)
            'env_unatt': env_u[:, :min_len] # (28, T)
        })
    return all_trials

def load_dtu_eeg_data():
    dtu_path = "/kaggle/input/datasets/lokeshgile/dataset-eeg/S1_data_preproc.mat"
    if not os.path.exists(dtu_path):
        dtu_path = str(REPO_ROOT / "data" / "S1_data_preproc.mat")
        if not os.path.exists(dtu_path):
            return []
            
    mat = sio.loadmat(dtu_path, squeeze_me=True, struct_as_record=False)
    trials = mat['data']
    channels = [13, 46, 43, 23, 50, 0, 52, 14]
    
    eegs = []
    for ex in trials:
        eeg = ex.eeg[:, channels].T
        nyq = 0.5 * 64
        b, a = butter(2, [1.0/nyq, 6.0/nyq], btype='band')
        eeg = filtfilt(b, a, eeg, axis=1)
        eegs.append(eeg)
    return eegs

def run_evaluation(model, trials, R, window_sec, device):
    win_samples = int(window_sec * 64)
    stride_samples = win_samples
    
    correct = 0
    total = 0
    
    for t in trials:
        # Apply alignment
        eeg = t['eeg']
        if R is not None:
            eeg = apply_alignment(eeg, R)
            
        eeg_norm = normalize_array(eeg.T).T
        env_a = normalize_array(t['env_att'].T).T
        env_b = normalize_array(t['env_unatt'].T).T
        
        for start in range(0, eeg_norm.shape[1] - win_samples + 1, stride_samples):
            x = eeg_norm[:, start:start+win_samples]
            ya = env_a[:, start:start+win_samples]
            yb = env_b[:, start:start+win_samples]
            
            x_t = torch.FloatTensor(x).unsqueeze(0).to(device)
            ya_t = torch.FloatTensor(ya).unsqueeze(0).to(device)
            yb_t = torch.FloatTensor(yb).unsqueeze(0).to(device)
            
            z_eeg, z_a, z_b = model(x_t, ya_t, yb_t)
            
            sim_a = pearson_corr(z_eeg, z_a).item()
            sim_b = pearson_corr(z_eeg, z_b).item()
            
            if sim_a > sim_b: correct += 1
            elif sim_a == sim_b: correct += 0.5
            total += 1
            
    return correct / max(total, 1)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    candidates = []
    if Path("/kaggle/input").exists():
        for r, d, f in os.walk("/kaggle/input"):
            for file in f:
                if file.endswith('.pth'): candidates.append(os.path.join(r, file))
    for r, d, f in os.walk(REPO_ROOT / "checkpoints"):
        for file in f:
            if file.endswith('.pth'): candidates.append(os.path.join(r, file))
            
    if not candidates:
        print("Model not found!")
        return
        
    chk_path = candidates[0]
    model = ContrastiveMatchNet(eeg_model_type="eegnet", eeg_channels=8, audio_channels=28, latent_dim=64).to(device)
    model.load_state_dict(torch.load(chk_path, map_location=device))
    model.eval()
    
    print("Loading KUL...")
    kul_trials = load_all_kul_data()
    print("Loading DTU...")
    dtu_eegs = load_dtu_eeg_data()
    
    kul_eegs = [t['eeg'] for t in kul_trials]
    
    R_whiten, R_recolor = prepare_alignment_matrices(kul_eegs, dtu_eegs if len(dtu_eegs)>0 else None)
    
    windows = [2, 5, 10, 20, 30]
    results = []
    
    with torch.no_grad():
        for w in windows:
            print(f"Evaluating {w}s...")
            acc_none = run_evaluation(model, kul_trials, None, w, device)
            acc_euclidean = run_evaluation(model, kul_trials, R_whiten, w, device)
            acc_align = run_evaluation(model, kul_trials, R_recolor @ R_whiten, w, device)
            
            results.append({
                "Window": f"{w}s",
                "Baseline": acc_none,
                "EA_Whiten": acc_euclidean,
                "EA_AlignDTU": acc_align
            })
            
    df = pd.DataFrame(results)
    os.makedirs(REPO_ROOT / "analysis", exist_ok=True)
    df.to_csv(REPO_ROOT / "analysis" / "ea_results.csv", index=False)
    print("\nResults:")
    print(df.to_string(index=False))

if __name__ == "__main__":
    main()
