import pytest
import numpy as np
from src.audio.neural_speech_separator import SingleChannelNeuralSeparator


def test_neural_speech_separator_synthetic_mixture():
    """
    Verifies that SingleChannelNeuralSeparator:
      1. Ingests a single-channel mixture.
      2. Separates it into 2 distinct 16 kHz waveforms without NaNs.
      3. Extracts valid 64 Hz broadband Gammatone envelopes.
    """
    fs = 16000
    duration_sec = 1.5
    n_samples = int(fs * duration_sec)
    t = np.linspace(0, duration_sec, n_samples, endpoint=False, dtype=np.float32)

    # 300 Hz tone + 800 Hz tone mixture
    mix = (np.sin(2 * np.pi * 300.0 * t) + np.cos(2 * np.pi * 800.0 * t)).astype(np.float32)

    separator = SingleChannelNeuralSeparator(device="cpu", target_fs=fs)
    s1, s2 = separator.separate_waveform(mix, orig_fs=fs)

    assert len(s1) == n_samples
    assert len(s2) == n_samples
    assert not np.isnan(s1).any()
    assert not np.isnan(s2).any()
    # Confirm two outputs are distinct
    corr = np.corrcoef(s1, s2)[0, 1]
    assert corr < 0.99, f"Outputs should be separated into distinct streams, got correlation {corr:.3f}"

    # Verify envelope extraction
    env1, env2 = separator.extract_neural_envelopes(s1, s2, envelope_fs=64.0)
    assert len(env1) > 0
    assert len(env2) > 0
    assert not np.isnan(env1).any()
    assert not np.isnan(env2).any()
