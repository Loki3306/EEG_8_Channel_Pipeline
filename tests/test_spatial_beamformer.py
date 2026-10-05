import pytest
import numpy as np
from src.audio.spatial_beamformer import ReSpeakerSpatialBeamformer
from src.audio.acoustic_crosstalk_simulator import ReSpeakerAcousticSimulator


def test_beamformer_initialization():
    bf = ReSpeakerSpatialBeamformer(fs=16000, radius=0.0325, theta_a_deg=-45.0, theta_b_deg=45.0)
    assert bf.num_mics == 4
    assert bf.weights_a.shape == (bf.n_bins, 4)
    assert bf.weights_b.shape == (bf.n_bins, 4)


def test_spatial_separation_and_null_steering():
    """
    Verifies that Beam A steered at -45 deg significantly suppresses a source at +45 deg,
    and Beam B steered at +45 deg significantly suppresses a source at -45 deg.
    """
    fs = 16000
    duration_sec = 2.0
    n_samples = int(fs * duration_sec)
    t = np.linspace(0, duration_sec, n_samples, endpoint=False, dtype=np.float32)

    # Source A: 500 Hz pure tone at -45 deg
    sig_a = np.sin(2 * np.pi * 500.0 * t).astype(np.float32)
    # Source B: 1200 Hz pure tone at +45 deg
    sig_b = np.sin(2 * np.pi * 1200.0 * t).astype(np.float32)

    sim = ReSpeakerAcousticSimulator(fs=fs, radius=0.0325, theta_a_deg=-45.0, theta_b_deg=45.0)
    bf = ReSpeakerSpatialBeamformer(fs=fs, radius=0.0325, theta_a_deg=-45.0, theta_b_deg=45.0, mode="lcmv_null")

    # 1. Render only Source A -> measure Beam A vs Beam B energy
    mics_a_only = sim.simulate_4channel_mixture(sig_a, np.zeros_like(sig_a))
    out_a_a, out_b_a = bf.process_continuous_file(mics_a_only)
    power_a_on_beam_a = np.mean(out_a_a ** 2)
    power_a_on_beam_b = np.mean(out_b_a ** 2)
    suppression_a_on_beam_b_db = 10.0 * np.log10((power_a_on_beam_a + 1e-9) / (power_a_on_beam_b + 1e-9))

    # 2. Render only Source B -> measure Beam B vs Beam A energy
    mics_b_only = sim.simulate_4channel_mixture(np.zeros_like(sig_b), sig_b)
    out_a_b, out_b_b = bf.process_continuous_file(mics_b_only)
    power_b_on_beam_b = np.mean(out_b_b ** 2)
    power_b_on_beam_a = np.mean(out_a_b ** 2)
    suppression_b_on_beam_a_db = 10.0 * np.log10((power_b_on_beam_b + 1e-9) / (power_b_on_beam_a + 1e-9))

    # Expect at least 8 dB of spatial attenuation at the competing null
    assert suppression_a_on_beam_b_db > 8.0, f"Expected >8 dB suppression of Source A on Beam B, got {suppression_a_on_beam_b_db:.2f} dB"
    assert suppression_b_on_beam_a_db > 8.0, f"Expected >8 dB suppression of Source B on Beam A, got {suppression_b_on_beam_a_db:.2f} dB"


def test_chunk_streaming_continuity():
    """
    Verifies that streaming chunk-by-chunk produces finite, valid audio matching full length.
    """
    fs = 16000
    n_samples = 3200  # 200 ms
    rng = np.random.RandomState(42)
    dummy_mics = rng.randn(4, n_samples).astype(np.float32) * 0.1

    bf = ReSpeakerSpatialBeamformer(fs=fs, radius=0.0325)
    chunk_size = 500  # 31.25 ms chunks

    out_chunks_a = []
    out_chunks_b = []
    for idx in range(0, n_samples, chunk_size):
        chunk = dummy_mics[:, idx:idx + chunk_size]
        ca, cb = bf.process_chunk(chunk)
        out_chunks_a.append(ca)
        out_chunks_b.append(cb)

    streamed_a = np.concatenate(out_chunks_a)
    streamed_b = np.concatenate(out_chunks_b)

    assert len(streamed_a) == n_samples
    assert len(streamed_b) == n_samples
    assert not np.isnan(streamed_a).any()
    assert not np.isnan(streamed_b).any()
