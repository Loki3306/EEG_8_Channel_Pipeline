"""
Unit tests for NeuroSteer Clinical Suite Dashboard Backend.
"""

import pytest
from fastapi.testclient import TestClient
from src.ui.data_provider import StreamDataProvider
from src.ui.session_manager import StreamingSimulationSession
from src.ui.server import app

def test_data_provider_subjects():
    provider = StreamDataProvider()
    subjects = provider.get_subjects_list()
    assert len(subjects) == 18
    assert subjects[0]["subject_id"] == "S1"
    assert subjects[0]["win_rate"] > 70.0
    assert "trials_count" in subjects[0]

def test_data_provider_trials():
    provider = StreamDataProvider()
    trials = provider.get_trials_for_subject("S1")
    assert len(trials) == 57  # Trials 3 to 59
    assert trials[0]["trial_id"] == 3
    assert trials[-1]["trial_id"] == 59
    assert "attended" in trials[0]

def test_data_provider_load_trial():
    provider = StreamDataProvider()
    audio_a, audio_b, eeg_8ch, attended = provider.load_trial_data("S1", 4, duration_sec=5.0)
    assert len(audio_a) == 5 * 16000
    assert len(audio_b) == 5 * 16000
    assert eeg_8ch.shape == (5 * 64, 8)
    assert attended in ["A", "B"]

def test_session_manager_step():
    provider = StreamDataProvider()
    session = StreamingSimulationSession(provider)
    session.load_trial("S1", 4)
    session.is_playing = True
    
    telemetry = session.step_simulation()
    assert telemetry is not None
    assert telemetry["type"] == "tick"
    assert "time_sec" in telemetry
    assert "eeg_sample" in telemetry
    assert len(telemetry["eeg_sample"]) == 8
    assert "gain_a_db" in telemetry
    assert "gain_b_db" in telemetry
    assert "dsp_latency_us" in telemetry
    assert "audio_b64" in telemetry
    assert len(telemetry["audio_b64"]) > 0

def test_session_manager_commands():
    provider = StreamDataProvider()
    session = StreamingSimulationSession(provider)
    
    r_play = session.handle_client_message({"action": "play"})
    assert r_play["state"] == "playing"
    assert session.is_playing is True
    
    r_pause = session.handle_client_message({"action": "pause"})
    assert r_pause["state"] == "paused"
    assert session.is_playing is False
    
    r_mode = session.handle_client_message({"action": "set_listening_mode", "mode": "mixture"})
    assert r_mode["listening_mode"] == "mixture"
    assert session.listening_mode == "mixture"
    
    r_seek = session.handle_client_message({"action": "seek", "time_sec": 12.5})
    assert r_seek["status"] == "ok"

def test_fastapi_rest_endpoints():
    client = TestClient(app)
    
    res_root = client.get("/")
    assert res_root.status_code == 200
    assert "NeuroSteer" in res_root.text
    
    res_subs = client.get("/api/subjects")
    assert res_subs.status_code == 200
    data_subs = res_subs.json()
    assert data_subs["status"] == "ok"
    assert len(data_subs["subjects"]) == 18
    
    res_trials = client.get("/api/trials/S1")
    assert res_trials.status_code == 200
    data_trials = res_trials.json()
    assert data_trials["status"] == "ok"
    assert len(data_trials["trials"]) == 57
    
    res_dev = client.get("/api/device_status")
    assert res_dev.status_code == 200
    assert "respeaker_connected" in res_dev.json()
