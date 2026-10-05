import pytest
import numpy as np
from src.audio.respeaker_interface import ReSpeakerDevice


def test_respeaker_interface_simulation_mode():
    """
    Verifies that ReSpeakerDevice operates gracefully in simulation mode when
    no physical ReSpeaker is plugged in, yielding valid 4-channel chunks.
    """
    fs = 16000
    chunk_size = 500  # 31.25 ms
    duration_sec = 1.0
    n_samples = int(fs * duration_sec)
    t = np.linspace(0, duration_sec, n_samples, endpoint=False, dtype=np.float32)

    sig_a = np.sin(2 * np.pi * 440.0 * t).astype(np.float32)
    sig_b = np.cos(2 * np.pi * 880.0 * t).astype(np.float32)

    # Force simulation mode for unit testing
    dev = ReSpeakerDevice(fs=fs, chunk_size=chunk_size, simulation_fallback=True)
    dev.set_simulation_tracks(sig_a, sig_b, crosstalk_sir_db=10.0)

    dev.start_streaming()
    assert dev.is_streaming

    chunks = []
    for _ in range(5):
        chunk = dev.read_chunk()
        assert chunk.shape == (4, chunk_size)
        assert not np.isnan(chunk).any()
        chunks.append(chunk)

    dev.stop_streaming()
    assert not dev.is_streaming
    dev.close()
