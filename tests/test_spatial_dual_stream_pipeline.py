import pytest
import numpy as np
from src.streaming.spatial_dual_stream_pipeline import SpatialDualStreamIngestionEngine


def test_spatial_dual_stream_engine_lockstep_ticks():
    """
    Verifies that SpatialDualStreamIngestionEngine:
      1. Ingests 16 samples of EEG @ 512 Hz and 500 samples of 4-ch ReSpeaker audio @ 16 kHz per tick.
      2. Generates exactly 2 samples at 64 Hz per tick for both EEG and beamformed audio.
      3. Triggers ready_for_eval every 16 ticks (500 ms hop).
      4. Correctly routes audio through the steering DSP.
    """
    engine = SpatialDualStreamIngestionEngine(
        raw_eeg_fs=512.0,
        audio_fs=16000.0,
        target_fs=64.0,
        window_sec=5.0,
        hop_sec=0.5
    )

    n_eeg_samples_per_tick = 16   # 31.25 ms @ 512 Hz
    n_audio_samples_per_tick = 500 # 31.25 ms @ 16 kHz

    # Run for 200 ticks (~6.25 seconds)
    eval_count = 0
    for tick in range(200):
        # 64-channel synthetic EEG packet
        eeg_pkt = np.random.randn(64, n_eeg_samples_per_tick).astype(np.float32)
        # 4-channel ReSpeaker packet
        respeaker_pkt = np.random.randn(4, n_audio_samples_per_tick).astype(np.float32) * 0.1

        frame = engine.process_tick(eeg_pkt, respeaker_pkt)
        assert frame.eeg_samples_generated == 2
        assert frame.audio_samples_generated == 2
        assert frame.raw_beam_a_chunk.shape == (n_audio_samples_per_tick,)
        assert frame.raw_beam_b_chunk.shape == (n_audio_samples_per_tick,)

        if frame.ready_for_eval:
            eval_count += 1
            assert frame.eeg_window.shape == (8, 320)       # 5.0s @ 64 Hz
            assert frame.audio_a_window.shape == (1, 320)   # Beam A envelope
            assert frame.audio_b_window.shape == (1, 320)   # Beam B envelope
            assert not np.isnan(frame.eeg_window).any()
            assert not np.isnan(frame.audio_a_window).any()

            # Test steering DSP output
            left_out, right_out, g_a, g_b = engine.steer_acoustic_output(
                frame.raw_beam_a_chunk, frame.raw_beam_b_chunk, "A", 0.5
            )
            assert len(left_out) == n_audio_samples_per_tick
            assert len(right_out) == n_audio_samples_per_tick
            assert g_a > 0.0  # Amplified

    # After filling the 5.0s buffer (160 ticks), evaluation should trigger every 16 ticks (0.5s)
    assert eval_count >= 2
