import pytest
import numpy as np
from src.audio.causal_gammatone import StreamingCausalAudioGammatoneExtractor, erb_space
from src.streaming.dual_stream_ingestor import DualStreamIngestionEngine, DualStreamFrame


def test_erb_space_distribution():
    low_freq = 50.0
    high_freq = 7500.0
    num_bands = 28
    
    cfs = erb_space(low_freq=low_freq, high_freq=high_freq, num_bands=num_bands)
    
    assert len(cfs) == num_bands
    # Frequencies must strictly increase
    assert np.all(np.diff(cfs) > 0)
    assert cfs[0] >= low_freq - 1.0
    assert cfs[-1] <= high_freq + 1.0


def test_causal_gammatone_chunk_invariance():
    fs = 16000.0
    duration_sec = 2.0
    n_samples = int(duration_sec * fs)
    t = np.linspace(0, duration_sec, n_samples, endpoint=False)
    
    # 500 Hz tone with harmonics
    sig = np.sin(2 * np.pi * 500.0 * t) + 0.5 * np.sin(2 * np.pi * 1500.0 * t)
    
    # Mode A: Single batch
    extractor_batch = StreamingCausalAudioGammatoneExtractor(audio_fs=fs, target_fs=64.0, power_exponent=0.3)
    out_batch = extractor_batch.process_audio_chunk(sig)
    
    # Mode B: Streamed across 31.25 ms chunks (500 samples per chunk @ 16 kHz)
    extractor_stream = StreamingCausalAudioGammatoneExtractor(audio_fs=fs, target_fs=64.0, power_exponent=0.3)
    chunk_size = 500
    streamed_chunks = []
    for i in range(0, n_samples, chunk_size):
        c = sig[i:i + chunk_size]
        res = extractor_stream.process_audio_chunk(c)
        if len(res) > 0:
            streamed_chunks.append(res)
            
    out_stream = np.concatenate(streamed_chunks, axis=0) if streamed_chunks else np.empty(0)
    
    assert len(out_batch) == len(out_stream)
    max_diff = np.max(np.abs(out_batch - out_stream))
    assert max_diff < 1e-3, f"Streaming output differs from batch output: max diff = {max_diff}"


def test_fractional_phase_decimation_sample_count():
    # 1.0 second of audio at 44.1 kHz = 44100 samples
    fs = 44100.0
    duration_sec = 1.0
    n_samples = int(duration_sec * fs)
    t = np.linspace(0, duration_sec, n_samples, endpoint=False)
    sig = np.sin(2 * np.pi * 400.0 * t)
    
    extractor = StreamingCausalAudioGammatoneExtractor(audio_fs=fs, target_fs=64.0, power_exponent=0.3)
    
    # Stream in ~31.25 ms chunks (1378 samples each)
    chunk_size = 1378
    total_out = 0
    for i in range(0, n_samples, chunk_size):
        c = sig[i:min(n_samples, i + chunk_size)]
        res = extractor.process_audio_chunk(c)
        total_out += len(res)
        
    # At 64 Hz, 1.0 second of audio should produce exactly 64 samples
    assert abs(total_out - 64) <= 1, f"Expected 64 samples at 64 Hz, got {total_out}"


def test_power_exponent_configurations():
    fs = 16000.0
    sig = np.random.randn(3200) * 0.5
    
    # Test p = 0.3 (DTU MATLAB baseline)
    ext_03 = StreamingCausalAudioGammatoneExtractor(audio_fs=fs, power_exponent=0.3)
    out_03 = ext_03.process_audio_chunk(sig)
    assert np.all(np.isfinite(out_03))
    assert np.all(out_03 >= 0.0)
    
    # Test p = 0.6 (MatchNet offline training baseline)
    ext_06 = StreamingCausalAudioGammatoneExtractor(audio_fs=fs, power_exponent=0.6)
    out_06 = ext_06.process_audio_chunk(sig)
    assert np.all(np.isfinite(out_06))
    assert np.all(out_06 >= 0.0)


def test_extractor_reset_functionality():
    fs = 16000.0
    sig = np.sin(np.linspace(0, 100, 16000))
    
    ext = StreamingCausalAudioGammatoneExtractor(audio_fs=fs, target_fs=64.0)
    out_run1 = ext.process_audio_chunk(sig)
    
    # Reset clears internal state
    ext.reset()
    out_run2 = ext.process_audio_chunk(sig)
    
    np.testing.assert_allclose(out_run1, out_run2, atol=1e-6)


def test_dual_stream_ingestion_synchronization():
    raw_eeg_fs = 512.0
    audio_fs = 44100.0
    target_fs = 64.0
    window_sec = 5.0
    hop_sec = 0.5
    
    engine = DualStreamIngestionEngine(
        raw_eeg_fs=raw_eeg_fs,
        audio_fs=audio_fs,
        target_fs=target_fs,
        window_sec=window_sec,
        hop_sec=hop_sec,
        power_exponent=0.3
    )
    
    # Simulate streaming 6.0 seconds (needs > 5.0s window before first evaluation)
    eeg_chunk_size = 16 # 31.25 ms @ 512 Hz
    audio_chunk_size = int(round(audio_fs * (eeg_chunk_size / raw_eeg_fs))) # ~1378 samples
    
    n_ticks = int(6.0 / (eeg_chunk_size / raw_eeg_fs)) # ~192 ticks
    
    eval_triggered_count = 0
    last_frame = None
    
    for t_idx in range(n_ticks):
        fake_eeg = np.random.randn(eeg_chunk_size, 64).astype(np.float32)
        fake_audio_a = np.random.randn(audio_chunk_size).astype(np.float32)
        fake_audio_b = np.random.randn(audio_chunk_size).astype(np.float32)
        
        frame = engine.step(fake_eeg, fake_audio_a, fake_audio_b)
        
        if frame.ready_for_eval:
            eval_triggered_count += 1
            last_frame = frame
            assert frame.eeg_window is not None
            assert frame.audio_a_window is not None
            assert frame.audio_b_window is not None
            
            # Verify window dimensions for CA-TCN
            assert frame.eeg_window.shape == (8, 320), f"Expected (8, 320), got {frame.eeg_window.shape}"
            assert frame.audio_a_window.shape == (1, 320), f"Expected (1, 320), got {frame.audio_a_window.shape}"
            assert frame.audio_b_window.shape == (1, 320), f"Expected (1, 320), got {frame.audio_b_window.shape}"
            
    # In 6.0 seconds with 5.0s window and 0.5s hop:
    # First eval occurs at 5.0s, then 5.5s, 6.0s (at least 2-3 evaluations)
    assert eval_triggered_count >= 2, f"Expected at least 2 eval triggers, got {eval_triggered_count}"
    assert last_frame is not None
    assert "total_dsp_us" in last_frame.dsp_timing_us
