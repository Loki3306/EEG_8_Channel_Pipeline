import numpy as np
import pytest
from src.audio.steering_engine import AudioSteeringDSP
from src.audio.metrics import (
    compute_sir_metrics,
    compute_headroom_metrics,
    compute_stoi_intelligibility,
    evaluate_audio_steering_trial,
)

def test_gain_allocation():
    dsp = AudioSteeringDSP(max_boost_db=9.0, max_suppress_db=18.0, threshold_switch=0.35)
    
    # State A, High confidence margin
    g_a, g_b = dsp.compute_target_gains_db("A", 0.70)
    assert abs(g_a - 9.0) < 1e-3
    assert abs(g_b - (-18.0)) < 1e-3
    
    # State B, High confidence margin
    g_a, g_b = dsp.compute_target_gains_db("B", -0.70)
    assert abs(g_a - (-18.0)) < 1e-3
    assert abs(g_b - 9.0) < 1e-3
    
    # State HOLD (neutral pass-through)
    g_a, g_b = dsp.compute_target_gains_db("HOLD", 0.0)
    assert g_a == 0.0
    assert g_b == 0.0

def test_slew_rate_limiter_smoothness():
    dsp = AudioSteeringDSP(fs=16000, tau_ms=60.0)
    
    # Send a sudden +9 dB step to Stream A
    audio_a = np.ones(1600, dtype=np.float32) * 0.1 # 100 ms DC tone
    audio_b = np.zeros(1600, dtype=np.float32)
    
    stereo_out, ga, gb = dsp.process_block(audio_a, audio_b, target_gain_a_db=9.0, target_gain_b_db=-18.0)
    
    # Gain should start at 1.0 (0 dB) and smoothly ramp up towards 10^(9/20) ≈ 2.818
    assert ga[0] < 1.1
    assert ga[-1] > 2.0
    
    # Verify no discontinuous jumps between adjacent samples
    diffs = np.abs(np.diff(ga))
    max_step = np.max(diffs)
    assert max_step < 0.01, f"Gain jump too abrupt: max step = {max_step}"

def test_peak_limiter_ceiling():
    dsp = AudioSteeringDSP(fs=16000, ceiling_dbfs=-0.5)
    # Feed dangerously loud audio (amplitude 5.0)
    audio_a = np.ones(800, dtype=np.float32) * 5.0
    audio_b = np.ones(800, dtype=np.float32) * 5.0
    
    stereo_out, _, _ = dsp.process_block(audio_a, audio_b, 0.0, 0.0)
    
    peak = np.max(np.abs(stereo_out))
    max_allowed = 10.0 ** (-0.5 / 20.0)
    assert peak <= max_allowed + 1e-4

def test_sir_and_headroom_metrics():
    fs = 16000
    T = fs * 2
    audio_a = np.random.randn(T).astype(np.float32) * 0.1
    audio_b = np.random.randn(T).astype(np.float32) * 0.1
    
    # Constant 6 dB boost on A, -12 dB suppression on B
    g_a = np.ones(T, dtype=np.float32) * (10.0 ** (6.0 / 20.0))
    g_b = np.ones(T, dtype=np.float32) * (10.0 ** (-12.0 / 20.0))
    
    sir_res = compute_sir_metrics(audio_a, audio_b, g_a, g_b)
    # Expected improvement is 6 - (-12) = 18 dB
    assert abs(sir_res["delta_sir_db"] - 18.0) < 0.2
    
    stereo_dummy = np.stack([audio_a, audio_b], axis=0)
    headroom = compute_headroom_metrics(stereo_dummy)
    assert headroom["peak_dbfs"] < 0.0
    assert headroom["clipping_rate_pct"] == 0.0

def test_stoi_calculation():
    fs = 16000
    ref = np.random.randn(fs * 2).astype(np.float32)
    # Perfect copy should have STOI ~ 1.0
    stoi_perfect = compute_stoi_intelligibility(ref, ref, fs)
    assert stoi_perfect > 0.90
    
    # Pure noise should have much lower STOI
    noise = np.random.randn(fs * 2).astype(np.float32)
    stoi_noisy = compute_stoi_intelligibility(ref, noise, fs)
    assert stoi_noisy < stoi_perfect
