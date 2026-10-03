import numpy as np
from scipy.signal import butter, filtfilt

FS = 64.0  # Hz

def butter_lowpass(data, cutoff, fs, order=4):
    nyq = 0.5 * fs
    b, a = butter(order, cutoff / nyq, btype='low')
    return filtfilt(b, a, data)

def test_lag_sign_convention():
    """
    Synthetic Sanity Test for Neural Lag Alignment:
    Physical Ground Truth:
      Cortical EEG response lags acoustic stimulus by +200 ms (13 samples at 64 Hz).
      EEG(t) = Audio(t - 200ms) + noise
      
    This test verifies that the lag sweep peaks at exactly +200 ms (+0.20s).
    """
    np.random.seed(42)
    duration_sec = 60.0
    n_samples = int(duration_sec * FS)
    
    # 1. Ground truth acoustic envelope (filtered 1-8 Hz)
    raw_audio = np.random.randn(n_samples)
    audio_env = butter_lowpass(raw_audio, 8.0, FS)
    audio_env = (audio_env - audio_env.mean()) / audio_env.std()
    
    # 2. Known biological lag: tau = +200 ms (13 samples at 64 Hz)
    tau_true_ms = 200
    true_shift_samples = int(round((tau_true_ms / 1000.0) * FS)) # 13 samples
    
    # EEG is the delayed version of audio: EEG at sample t reflects Audio at t - 13
    # That is: EEG[13:] corresponds to Audio[:-13]
    eeg = np.zeros(n_samples)
    eeg[true_shift_samples:] = audio_env[:-true_shift_samples]
    # Add noise
    eeg += 0.5 * np.random.randn(n_samples)
    eeg = (eeg - eeg.mean()) / eeg.std()
    
    print(f"--- SYNTHETIC NEURAL LAG SANITY TEST ---")
    print(f"Ground Truth Audio -> EEG latency: +{tau_true_ms} ms (+{true_shift_samples} samples at {FS} Hz)")
    print(f"{'Lag (ms)':>10} | {'Shift (samples)':>15} | {'Correlation':>12}")
    print("-" * 45)
    
    sweep_lags = [-1000, -750, -500, -250, 0, 100, 150, 200, 250, 500, 750, 1000]
    corrs = {}
    
    for lag_ms in sweep_lags:
        lag_sec = lag_ms / 1000.0
        shift_samples = int(round(lag_sec * FS))
        
        x_norm = eeg.copy()
        env_norm = audio_env.copy()
        
        # Exact alignment code from prepare_dataset:
        if shift_samples > 0:
            x_norm = x_norm[shift_samples:]
            env_norm = env_norm[:-shift_samples]
        elif shift_samples < 0:
            abs_shift = abs(shift_samples)
            x_norm = x_norm[:-abs_shift]
            env_norm = env_norm[abs_shift:]
            
        min_len = min(len(x_norm), len(env_norm))
        x_norm = x_norm[:min_len]
        env_norm = env_norm[:min_len]
        
        # Compute correlation
        r = np.corrcoef(x_norm, env_norm)[0, 1]
        corrs[lag_ms] = r
        marker = " <-- (PEAK!)" if lag_ms == tau_true_ms else ""
        print(f"{lag_ms:+9d} ms | {shift_samples:+14d} | {r:12.4f}{marker}")
        
    best_lag = max(corrs.keys(), key=lambda k: corrs[k])
    print("-" * 45)
    print(f"Result: Peak correlation observed at: {best_lag:+d} ms")
    assert best_lag == tau_true_ms, f"Sign error! Expected peak at {tau_true_ms} ms, got {best_lag} ms"
    print("VERDICT: PASSED! Sign convention perfectly aligns with physical neural latency.")

if __name__ == "__main__":
    test_lag_sign_convention()
