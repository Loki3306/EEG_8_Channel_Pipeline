import json
import pytest
from pathlib import Path
from src.audio.live_visualizer import build_live_streaming_html, save_live_streaming_dashboard

def create_mock_telemetry():
    frames = []
    for i in range(10):
        t = 5.0 + i * 0.5
        m = 0.5 if i < 8 else -0.4
        dec = "A" if m > 0.35 else ("B" if m < -0.35 else "HOLD")
        frames.append({
            "time_sec": t,
            "raw_margin": m,
            "smoothed_margin": m * 0.9,
            "confidence": 0.85,
            "decision": dec,
            "ground_truth": "A",
            "is_correct": (dec == "A"),
            "gain_a_db": 9.0 if dec == "A" else (-18.0 if dec == "B" else 0.0),
            "gain_b_db": -18.0 if dec == "A" else (9.0 if dec == "B" else 0.0),
            "rms_a": 0.05,
            "rms_b": 0.03,
            "cumulative_accuracy_pct": 90.0,
            "running_delta_sir_db": 17.5,
            "running_stoi": 0.96,
            "switch_count": 0
        })
        
    return {
        "subject": "S8",
        "trial_idx": 15,
        "duration_sec": 50.0,
        "sample_rate": 44100,
        "ground_truth": "A",
        "threshold_switch": 0.35,
        "threshold_maintain": 0.14,
        "max_boost_db": 9.0,
        "max_suppress_db": 18.0,
        "tau_ms": 60.0,
        "headroom_db": 7.8,
        "steered_wav_filename": "S8_trial_15_steered.wav",
        "mixture_wav_filename": "S8_trial_15_mixture.wav",
        "ref_wav_filename": "S8_trial_15_reference.wav",
        "frames": frames
    }

def test_build_live_streaming_html():
    telemetry = create_mock_telemetry()
    html = build_live_streaming_html(telemetry)
    
    # Check essential structural elements
    assert "<!DOCTYPE html>" in html
    assert "AAD Live Streaming Neural Audio Suite" in html
    assert 'id="btn-play"' in html
    assert 'id="card-talker-a"' in html
    assert 'id="card-talker-b"' in html
    assert 'id="beacon"' in html
    assert 'id="margin-pointer"' in html
    assert 'id="audio-steered"' in html
    assert 'id="audio-mixture"' in html
    assert 'id="audio-reference"' in html
    assert "S8_trial_15_steered.wav" in html

def test_save_live_streaming_dashboard(tmp_path: Path):
    telemetry = create_mock_telemetry()
    out_file = tmp_path / "test_player.html"
    res_path = save_live_streaming_dashboard(telemetry, out_file)
    
    assert res_path.exists()
    assert res_path.stat().st_size > 1000
    content = res_path.read_text(encoding="utf-8")
    assert "Speaker A" in content
    assert "Speaker B" in content
