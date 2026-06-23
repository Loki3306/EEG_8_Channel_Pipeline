import os
import sys
import scipy.io as sio
import scipy.io.wavfile as wavfile
import numpy as np
import pandas as pd
import torch
from scipy.signal import resample, butter, filtfilt
from sklearn.metrics import roc_auc_score
import xgboost as xgb
from tqdm import tqdm

# Add repository root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from models.matchnet import ContrastiveMatchNet
except ImportError as e:
    print(f"Could not import MatchNet: {e}")
    ContrastiveMatchNet = None

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

def extract_28_band_envelope(audio_data, fs_in, fs_out=64, num_bands=28):
    cfs = erb_space(50, 8000, num_bands)
    lows, highs = get_erb_bands(cfs)
    
    envelopes = []
    for i in range(num_bands):
        band_audio = apply_bandpass(audio_data, lows[i], highs[i], fs_in)
        env = np.abs(band_audio)
        env = env ** 0.3
        envelopes.append(env)
        
    envelopes = np.array(envelopes)
    num_samples_out = int(envelopes.shape[1] * fs_out / fs_in)
    return resample(envelopes, num_samples_out, axis=1)

# --- Normalization ---
def normalize_array(arr):
    arr = arr - arr.mean(axis=0, keepdims=True)
    scale = arr.std(axis=0, keepdims=True) + 1e-12
    return arr / scale

def pearson_corr(x, y, dim=1):
    x_centered = x - x.mean(dim=dim, keepdim=True)
    y_centered = y - y.mean(dim=dim, keepdim=True)
    cov = (x_centered * y_centered).sum(dim=dim)
    var_x = (x_centered ** 2).sum(dim=dim)
    var_y = (y_centered ** 2).sum(dim=dim)
    return cov / torch.sqrt(var_x * var_y + 1e-8)

def find_checkpoint(root_dir):
    candidates = []
    for root, dirs, files in os.walk(root_dir):
        for f in files:
            if f.endswith('.pth') or f.endswith('.pt'):
                candidates.append(os.path.join(root, f))
    for c in candidates:
        if 'best' in c.lower() or 'matchnet' in c.lower(): return c
    return candidates[0] if candidates else None

def add_temporal_features(df):
    df = df.sort_values(['subject_id', 'trial_id', 'window_id']).reset_index(drop=True)
    df['sim_chosen'] = df[['sim_A', 'sim_B']].max(axis=1)
    df['sim_unchosen'] = df[['sim_A', 'sim_B']].min(axis=1)
    df['rolling_std_margin'] = df.groupby(['subject_id', 'trial_id'])['margin'].rolling(window=5, min_periods=1).std().reset_index(level=[0,1], drop=True)
    df['rolling_std_margin'] = df['rolling_std_margin'].fillna(0.0)
    
    def compute_consistency(group):
        preds = group['prediction'].values
        consistencies = []
        for i in range(len(preds)):
            if i == 0:
                consistencies.append(1.0)
            else:
                consistencies.append(np.mean(preds[:i] == preds[i]))
        group['trial_consistency'] = consistencies
        return group

    df = df.groupby(['subject_id', 'trial_id'], group_keys=False).apply(compute_consistency)
    return df

def main():
    print("\n" + "="*60)
    print("PHASE KUL-4B: MULTI-WINDOW ABLATION & CONFIDENCE SYSTEM EVALUATION")
    print("="*60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # 1. Load MatchNet Model
    checkpoint_dir = "/kaggle/working/EEG_8_Channel_Pipeline/checkpoints"
    if not os.path.exists(checkpoint_dir): checkpoint_dir = "checkpoints"
    chk_path = find_checkpoint(checkpoint_dir)
    if not chk_path: chk_path = find_checkpoint(".")
    if not chk_path:
        print("ERROR: No DTU MatchNet checkpoint found!")
        return
        
    print(f"Loading DTU Checkpoint: {chk_path}")
    model = ContrastiveMatchNet(eeg_model_type="eegnet", eeg_channels=8, audio_channels=28, latent_dim=64).to(device)
    model.load_state_dict(torch.load(chk_path, map_location=device))
    model.eval()
    
    # 2. Load Confidence Model
    conf_model_path = "/kaggle/working/EEG_8_Channel_Pipeline/models/confidence_model.json"
    if not os.path.exists(conf_model_path): conf_model_path = "models/confidence_model.json"
    if not os.path.exists(conf_model_path):
        print(f"ERROR: Confidence model not found at {conf_model_path}!")
        return
        
    print(f"Loading DTU Confidence Model: {conf_model_path}")
    conf_model = xgb.XGBClassifier()
    conf_model.load_model(conf_model_path)
    
    # 3. Process KUL Data (Memory Cached)
    mat_path = "/kaggle/input/datasets/lowk1ee/s1-klu/S1_KLU.mat"
    wav_dir = "/kaggle/input/datasets/lowk1ee/audio-klu"
    target_channels = ['Fp1', 'Fp2', 'F7', 'F8', 'T7', 'T8', 'P7', 'P8']
    fs_dtu = 64
    
    print("\nLoading and precaching all KUL trials...")
    mat = sio.loadmat(mat_path, squeeze_me=True, struct_as_record=False)
    trials = mat['trials'] if 'trials' in mat else mat['trial']
    
    def find_wav(name):
        for r, d, f in os.walk(wav_dir):
            if name in f: return os.path.join(r, name)
            if name+".wav" in f: return os.path.join(r, name+".wav")
        return None

    preprocessed_trials = []
    
    print("\n" + "="*60)
    print("VERIFICATION: KUL LABEL MAPPING LOGIC")
    print("="*60)
    print("Logic for determining the attended stream:")
    print("  1. 'trial.stimuli' contains two filenames: [left_stream, right_stream]")
    print("  2. 'trial.attended_ear' indicates 'L' or 'R'")
    print("  3. IF attended_ear == 'L' -> Attended Audio = stimuli[0]")
    print("  4. IF attended_ear == 'R' -> Attended Audio = stimuli[1]")
    print("If this mapping is inverted, accuracy will artificially drop to 0%!")
    print("-" * 60)
    
    with torch.no_grad():
        for t_idx, trial in enumerate(trials):
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
            
            if t_idx == 0:
                print(f"Trial 0 Example:")
                print(f"  Stimuli: L={stimuli[0]}, R={stimuli[1]}")
                print(f"  Attended Ear: {att_ear}")
                print(f"  => Attended Stream: {att_wav_name}")
                print("-" * 60)
                
            att_wav_path = find_wav(str(att_wav_name))
            unatt_wav_path = find_wav(str(unatt_wav_name))
            
            if not att_wav_path or not unatt_wav_path: continue
                
            fs_att, audio_att = wavfile.read(att_wav_path)
            fs_unatt, audio_unatt = wavfile.read(unatt_wav_path)
            if len(audio_att.shape) > 1: audio_att = audio_att.mean(axis=1)
            if len(audio_unatt.shape) > 1: audio_unatt = audio_unatt.mean(axis=1)
            
            env_att = extract_28_band_envelope(audio_att, fs_att, fs_out=64, num_bands=28)
            env_unatt = extract_28_band_envelope(audio_unatt, fs_unatt, fs_out=64, num_bands=28)
            
            min_len = min(len(eeg_64), env_att.shape[1], env_unatt.shape[1])
            eeg_64 = eeg_64[:min_len]
            env_att = env_att[:, :min_len]
            env_unatt = env_unatt[:, :min_len]
            
            eeg_norm = normalize_array(eeg_64) 
            env_att = normalize_array(env_att.T).T 
            env_unatt = normalize_array(env_unatt.T).T
            
            preprocessed_trials.append((t_idx, eeg_norm, env_att, env_unatt))

    # 4. Ablation Loop
    window_sizes = [2, 5, 10, 15, 20, 30]
    
    print("\n" + "="*60)
    print(f"{'Window':<8} | {'Win Acc':<8} | {'Trial Acc':<10} | {'Conf AUROC':<10} | {'Conf Mean':<10}")
    print("-" * 60)
    
    for window_sec in window_sizes:
        win_samples = int(window_sec * fs_dtu)
        stride_samples = win_samples # No overlap for clean metric
        
        csv_rows = []
        total_trials_correct = 0
        total_trials = 0
        
        with torch.no_grad():
            for t_idx, eeg_norm, env_att, env_unatt in preprocessed_trials:
                min_len = eeg_norm.shape[0]
                trial_corrects = []
                window_id = 0
                
                for start in range(0, min_len - win_samples + 1, stride_samples):
                    x = eeg_norm[start:start+win_samples].T 
                    ya = env_att[:, start:start+win_samples] 
                    yb = env_unatt[:, start:start+win_samples]
                    
                    x_t = torch.FloatTensor(x).unsqueeze(0).to(device)
                    ya_t = torch.FloatTensor(ya).unsqueeze(0).to(device)
                    yb_t = torch.FloatTensor(yb).unsqueeze(0).to(device)
                    
                    z_eeg, z_a, z_b = model(x_t, ya_t, yb_t)
                    
                    sim_a = pearson_corr(z_eeg, z_a, dim=1).mean().item()
                    sim_b = pearson_corr(z_eeg, z_b, dim=1).mean().item()
                    
                    prediction = 'A' if sim_a > sim_b else 'B'
                    correct = 1 if prediction == 'A' else 0
                    margin = abs(sim_a - sim_b)
                    
                    csv_rows.append({
                        'subject_id': 'S1_KUL',
                        'trial_id': t_idx,
                        'window_id': window_id,
                        'sim_A': sim_a,
                        'sim_B': sim_b,
                        'margin': margin,
                        'prediction': prediction,
                        'correct': correct
                    })
                    
                    trial_corrects.append(correct)
                    window_id += 1
                    
                if np.mean(trial_corrects) > 0.5:
                    total_trials_correct += 1
                total_trials += 1
                
        df = pd.DataFrame(csv_rows)
        win_acc = df['correct'].mean()
        trial_acc = total_trials_correct / max(total_trials, 1)
        
        # Add Confidence Features
        df = add_temporal_features(df)
        features = ['margin', 'sim_chosen', 'sim_unchosen', 'rolling_std_margin', 'trial_consistency']
        X = df[features]
        y_true = df['correct'].values
        
        # Predict Confidence Probability
        conf_probs = conf_model.predict_proba(X)[:, 1]
        
        try:
            auroc = roc_auc_score(y_true, conf_probs)
        except ValueError:
            auroc = float('nan') # Fails if all are correct or all incorrect
            
        conf_mean = np.mean(conf_probs)
        
        print(f"{window_sec:>5}s   | {win_acc*100:>5.1f}%   | {trial_acc*100:>6.1f}%    | {auroc:>8.4f}   | {conf_mean:>8.4f}")

if __name__ == "__main__":
    main()
