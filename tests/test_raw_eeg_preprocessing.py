import pytest
import numpy as np
from src.streaming.causal_raw_preprocessor import (
    Causal50HzNotchFilter,
    CausalDecimator,
    CausalRollingZScoreNormalizer,
    StreamingCausalRawEEGPreprocessor
)


def test_notch_filter_attenuation():
    fs = 512.0
    t = np.linspace(0, 2.0, int(2.0 * fs), endpoint=False)
    
    # 50 Hz powerline signal + 4 Hz neural signal
    sig_50hz = np.sin(2 * np.pi * 50.0 * t)[:, np.newaxis] # [N, 1]
    sig_4hz = np.sin(2 * np.pi * 4.0 * t)[:, np.newaxis]
    composite = sig_50hz + sig_4hz
    
    notch = Causal50HzNotchFilter(fs=fs, f0=50.0, q=30.0, n_channels=1)
    filtered = notch.process_chunk(composite)
    
    # 50 Hz energy should be dramatically reduced (>20 dB)
    rms_in_50 = np.sqrt(np.mean(sig_50hz[int(fs):] ** 2))
    rms_out_err = np.sqrt(np.mean((filtered[int(fs):] - sig_4hz[int(fs):]) ** 2))
    assert rms_out_err < 0.2 * rms_in_50


def test_causal_decimator_chunk_invariance():
    fs_in = 512.0
    fs_out = 64.0
    n_samples = 1024
    t = np.linspace(0, n_samples / fs_in, n_samples, endpoint=False)
    # 4 Hz test signal across 8 channels
    sig = np.sin(2 * np.pi * 4.0 * t)[:, np.newaxis] * np.ones((1, 8))
    
    # Mode A: Single batch
    decim_batch = CausalDecimator(in_fs=fs_in, out_fs=fs_out, n_channels=8)
    out_batch = decim_batch.process_chunk(sig)
    
    # Mode B: Streamed in chunks of 16 samples
    decim_stream = CausalDecimator(in_fs=fs_in, out_fs=fs_out, n_channels=8)
    streamed_chunks = []
    chunk_size = 16
    for i in range(0, n_samples, chunk_size):
        c = sig[i:i + chunk_size]
        res = decim_stream.process_chunk(c)
        if len(res) > 0:
            streamed_chunks.append(res)
    out_stream = np.concatenate(streamed_chunks, axis=0)
    
    assert out_batch.shape == out_stream.shape
    # Maximum absolute difference across all samples and channels should be virtually zero
    max_diff = np.max(np.abs(out_batch - out_stream))
    assert max_diff < 1e-4, f"Decimator is not chunk-invariant: max diff = {max_diff}"


def test_rolling_zscore_normalizer():
    fs = 64.0
    normalizer = CausalRollingZScoreNormalizer(n_channels=4, half_life_sec=2.0, fs=fs)
    
    # Input with large DC offset (100.0) and high amplitude (50.0)
    t = np.linspace(0, 10.0, int(10.0 * fs), endpoint=False)
    raw = 100.0 + 50.0 * np.sin(2 * np.pi * 2.0 * t)[:, np.newaxis] * np.ones((1, 4))
    
    # Process in 32-sample streaming blocks
    out_blocks = []
    for i in range(0, len(raw), 32):
        out_blocks.append(normalizer.process_chunk(raw[i:i+32]))
    out = np.concatenate(out_blocks, axis=0)
    
    # After warm-up (last 3 seconds), mean should be near 0 and std near 1
    warm_out = out[int(7.0 * fs):]
    mean_val = np.mean(warm_out, axis=0)
    std_val = np.std(warm_out, axis=0)
    
    assert np.all(np.abs(mean_val) < 0.3)
    assert np.all(np.abs(std_val - 1.0) < 0.3)


def test_streaming_causal_raw_preprocessor_end_to_end():
    raw_fs = 512.0
    preprocessor = StreamingCausalRawEEGPreprocessor(
        raw_fs=raw_fs,
        target_fs=64.0,
        n_scalp_channels=64,
        montage_name="near_ear_expanded"
    )
    
    # Simulate 3 seconds of 64-channel raw EEG at 512 Hz
    n_samples = int(3.0 * raw_fs)
    rng = np.random.RandomState(42)
    raw_eeg = rng.randn(n_samples, 64) * 20.0 + 50.0 # noise + DC
    veog = rng.randn(n_samples) * 100.0 # blinks
    heog = rng.randn(n_samples) * 30.0 # saccades
    
    # Calibrate EOG regression
    preprocessor.calibrate_eog_weights(raw_eeg, veog, heog)
    assert preprocessor.eog_weights.shape == (64, 2)
    
    # Stream in realistic hardware packet sizes: 16 samples per chunk (~31.25 ms per packet)
    chunk_size = 16
    processed_blocks = []
    for i in range(0, n_samples, chunk_size):
        eeg_chunk = raw_eeg[i:i+chunk_size]
        v_chunk = veog[i:i+chunk_size]
        h_chunk = heog[i:i+chunk_size]
        out_chunk = preprocessor.process_raw_chunk(eeg_chunk, v_chunk, h_chunk)
        if len(out_chunk) > 0:
            processed_blocks.append(out_chunk)
            
    total_processed = np.concatenate(processed_blocks, axis=0)
    
    # Output should have 8 channels (near_ear_expanded)
    assert total_processed.shape[1] == 8
    # 3 seconds @ 64 Hz should yield ~192 samples
    expected_samples = int(3.0 * 64.0)
    assert abs(total_processed.shape[0] - expected_samples) <= 2
    # No NaN or Inf
    assert not np.any(np.isnan(total_processed))
    assert not np.any(np.isinf(total_processed))
