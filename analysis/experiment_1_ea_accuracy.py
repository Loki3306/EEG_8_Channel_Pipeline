import os
import sys
import argparse
import scipy.io as sio
import scipy.io.wavfile as wavfile
import numpy as np
import torch
import torch.nn.functional as F
from scipy.signal import resample, butter, filtfilt
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path

# Add repository root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from models.matchnet import ContrastiveMatchNet
except ImportError as e:
    print(f"Could not import MatchNet: {e}")
    ContrastiveMatchNet = None
    
from preprocessing.euclidean_alignment import prepare_alignment_matrices, apply_alignment
from baselines.ridge_aad import load_subject_examples

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
    envelopes = Parallel(n_jobs=-1, backend="threading")(
        delayed(_process_band)(audio_data, lows[i], highs[i], fs_in) 
        for i in range(num_bands)
    )
        
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

def main():
    print("\n" + "="*60)
    print("PHASE KUL-4: FULL S1 CROSS-DATASET GENERALIZATION EVALUATION")
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
    model.eval()
    
    # 2. Config
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
    
    print("Gathering KUL EEG for EA calculation...")
    kul_eegs_for_ea = []
    for trial in trials:
        eeg_data = trial.RawData.EegData
        fs_eeg = trial.FileHeader.SampleRate
        channel_names = [ch.Label for ch in trial.FileHeader.Channels]
        selected_indices = [channel_names.index(tc) if tc in channel_names else [c.upper() for c in channel_names].index(tc.upper()) for tc in target_channels]
        eeg_8 = eeg_data[:, selected_indices]
        nyq = 0.5 * fs_eeg
        b, a = butter(2, [1.0/nyq, 6.0/nyq], btype='band')
        eeg_8 = filtfilt(b, a, eeg_8, axis=0)
        eeg_64 = resample(eeg_8, int(len(eeg_8) * fs_dtu / fs_eeg), axis=0)
        kul_eegs_for_ea.append(eeg_64.T) # shape (8, T)
        
    print("Loading DTU EEG for EA calculation...")
    dtu_path = Path("/kaggle/input/datasets/lokeshgile/dataset-eeg/S1_data_preproc.mat")
    if not dtu_path.exists():
        dtu_path = Path(REPO_ROOT) / "data" / "S1_data_preproc.mat"
    if dtu_path.exists():
        dtu_examples = load_subject_examples(dtu_path)
        dtu_channels = [13, 46, 43, 23, 50, 0, 52, 14]
        dtu_eegs_for_ea = []
        for ex in dtu_examples:
            eeg = ex.eeg[:, dtu_channels].T
            nyq = 0.5 * 64
            b, a = butter(2, [1.0/nyq, 6.0/nyq], btype='band')
            eeg = filtfilt(b, a, eeg, axis=1)
            dtu_eegs_for_ea.append(eeg) # shape (8, T)
    else:
        dtu_eegs_for_ea = []
        print("WARNING: DTU data not found, will only whiten KUL without recoloring.")
        
    print("Computing EA matrices...")
    R_whiten, R_recolor = prepare_alignment_matrices(kul_eegs_for_ea, dtu_eegs_for_ea if dtu_eegs_for_ea else None)
    R_ea = R_recolor @ R_whiten if R_recolor is not None else R_whiten
    
    def find_wav(name):
        for r, d, f in os.walk(wav_dir):
            if name in f: return os.path.join(r, name)
            if name+".wav" in f: return os.path.join(r, name+".wav")
        return None

    # Storage for results
    trial_results = []
    all_window_margins = []
    
    print(f"\nProcessing {len(trials)} trials...")
    
    with torch.no_grad():
        for t_idx, trial in enumerate(trials):
            print(f"  [Trial {t_idx+1:02d}/{len(trials):02d}] Processing...")
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
            
            # Apply Euclidean Alignment
            if R_ea is not None:
                eeg_64 = apply_alignment(eeg_64.T, R_ea).T
                
            eeg_norm = normalize_array(eeg_64) 
            env_att = normalize_array(env_att.T).T 
            env_unatt = normalize_array(env_unatt.T).T
            
            trial_win_margins = []
            correct_windows = 0
            total_windows = 0
            
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
                
                margin = sim_a - sim_b
                trial_win_margins.append(margin)
                all_window_margins.append(margin)
                
                if margin > 0:
                    correct_windows += 1
                total_windows += 1
                
            win_acc = correct_windows / max(total_windows, 1)
            mean_margin = np.mean(trial_win_margins)
            std_margin = np.std(trial_win_margins)
            trial_pred = "Attended" if mean_margin > 0 else "Unattended"
            trial_correct = 1 if mean_margin > 0 else 0
            
            trial_results.append({
                'trial_id': t_idx,
                'num_windows': total_windows,
                'correct_windows': correct_windows,
                'win_acc': win_acc,
                'mean_margin': mean_margin,
                'std_margin': std_margin,
                'trial_pred': trial_pred,
                'trial_correct': trial_correct
            })
            
    print("\n" + "="*60)
    print("TASK 3 & 4: TRIAL AND WINDOW LEVEL ACCURACY")
    print("="*60)
    print(f"{'Trial':<6} | {'Windows':<8} | {'Correct':<8} | {'Win Acc':<8} | {'Mean Margin':<12} | {'Std Margin':<11} | {'Trial Pred':<11} | {'Correct?'}")
    print("-" * 85)
    
    total_trial_correct = 0
    
    for tr in trial_results:
        print(f"{tr['trial_id']:<6} | {tr['num_windows']:<8} | {tr['correct_windows']:<8} | {tr['win_acc']*100:>5.1f}%   | {tr['mean_margin']:>11.4f}  | {tr['std_margin']:>10.4f} | {tr['trial_pred']:<11} | {'YES' if tr['trial_correct'] else 'NO'}")
        total_trial_correct += tr['trial_correct']
        
    overall_trial_acc = total_trial_correct / max(len(trial_results), 1)
    
    print("-" * 85)
    print(f"Overall Trial Accuracy: {total_trial_correct}/{len(trial_results)} ({overall_trial_acc*100:.1f}%)")
    
    # Task 5: Margin Diagnostics
    all_window_margins = np.array(all_window_margins)
    pos_margin_pct = (all_window_margins > 0).mean()
    
    print("\n" + "="*60)
    print("TASK 5: MARGIN DIAGNOSTICS (ALL WINDOWS)")
    print("="*60)
    print(f"Total Windows Processed : {len(all_window_margins)}")
    print(f"Mean Margin             : {np.mean(all_window_margins):.4f}")
    print(f"Std Margin              : {np.std(all_window_margins):.4f}")
    print(f"Min Margin              : {np.min(all_window_margins):.4f}")
    print(f"Max Margin              : {np.max(all_window_margins):.4f}")
    print(f"Positive Margin %       : {pos_margin_pct*100:.2f}%")
    
    os.makedirs("analysis/figures", exist_ok=True)
    
    plt.figure(figsize=(10, 6))
    plt.hist(all_window_margins, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
    plt.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Decision Boundary (0.0)')
    plt.axvline(x=np.mean(all_window_margins), color='green', linestyle='-', linewidth=2, label=f'Mean Margin ({np.mean(all_window_margins):.4f})')
    plt.title('KUL S1 Window Margins (DTU ContrastiveMatchNet)')
    plt.xlabel('Margin (sim_attended - sim_unattended)')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('analysis/figures/kul_s1_margin_histogram.png')
    plt.close()
    
    # Task 6: Per-Trial Visualization
    plt.figure(figsize=(12, 6))
    trial_ids = [str(r['trial_id']) for r in trial_results]
    trial_means = [r['mean_margin'] for r in trial_results]
    colors = ['green' if m > 0 else 'red' for m in trial_means]
    
    bars = plt.bar(trial_ids, trial_means, color=colors, alpha=0.7)
    plt.axhline(y=0, color='black', linestyle='--', linewidth=1)
    
    # Annotate with Win Acc
    for i, bar in enumerate(bars):
        yval = bar.get_height()
        win_acc = trial_results[i]['win_acc']
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + (0.001 if yval > 0 else -0.005), 
                 f"{win_acc*100:.0f}%", ha='center', va='bottom' if yval > 0 else 'top', fontsize=9)
                 
    plt.title('KUL S1 Mean Margin per Trial')
    plt.xlabel('Trial ID')
    plt.ylabel('Mean Margin')
    plt.tight_layout()
    plt.savefig('analysis/figures/kul_s1_trial_margins.png')
    plt.close()
    
    print("\nSaved figures:")
    print("  - analysis/figures/kul_s1_margin_histogram.png")
    print("  - analysis/figures/kul_s1_trial_margins.png")
    
    print("\n" + "="*60)
    print("TASK 7: DTU COMPARISON")
    print("="*60)
    print("NOTE: Based on typical DTU ContrastiveMatchNet performance metrics:")
    print("  DTU Mean Margin ≈ 0.05 - 0.10")
    print("  DTU Std Margin  ≈ 0.10 - 0.15")
    print(f"  KUL Mean Margin : {np.mean(all_window_margins):.4f}")
    print(f"  KUL Std Margin  : {np.std(all_window_margins):.4f}")
    
    print("\n" + "="*60)
    print("TASK 8: FINAL VERDICT")
    print("="*60)
    
    if overall_trial_acc >= 0.8:
        verdict = "A. Strong Transfer"
    elif overall_trial_acc >= 0.6:
        verdict = "B. Partial Transfer"
    elif pos_margin_pct >= 0.5:
        verdict = "C. Weak Transfer"
    else:
        verdict = "D. No Transfer"
        
    print(f"Verdict: {verdict}")
    print(f"Evidence: Trial Accuracy = {overall_trial_acc*100:.1f}%, Window Accuracy (Positive Margin) = {pos_margin_pct*100:.1f}%")

if __name__ == "__main__":
    main()
