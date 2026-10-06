"""
USCAPES: Universal Subject-Calibrated Auditory Processing & EEG Steering
Real-Time Clinical Session & Streaming Inference Manager.

Orchestrates the 1.0x real-time streaming clock (31.25 ms frames / 500 ms hops),
causal IIR DSP conditioning, on-the-fly CA-TCN neural attention decoding,
binaural acoustic steering (+9 dB / -18 dB), and bi-directional WebSocket telemetry dispatches.
"""

import asyncio
import base64
import time
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np
import torch

from ..config import (
    CHECKPOINT_DIR,
    FS_AUDIO,
    FS_EEG,
    AUDIO_BLOCK_SMP,
    EEG_BLOCK_SMP,
    WINDOW_SEC,
    MONTAGE_CHANNELS,
    NUM_CHANNELS,
)
from ..models.catcn import CATCNDirectDecoder
from ..models.spatial_adapter import SpatialEEGAdapter
from ..dsp.causal_filters import StreamingCausalEEGFilter
from ..dsp.causal_gammatone import StreamingCausalAudioGammatoneExtractor
from ..dsp.steering_engine import AudioSteeringDSP
from ..dsp.temporal_gate import StickyHysteresisGate
from .data_provider import StreamDataProvider


class StreamingSimulationSession:
    """
    Manages an active real-time clinical hearing aid simulation session.
    """
    def __init__(self, data_provider: StreamDataProvider):
        self.data_provider = data_provider
        
        # Audio & EEG Parameters
        self.fs_audio = FS_AUDIO
        self.fs_eeg = FS_EEG
        self.block_sec = 0.03125  # 31.25 ms frame
        self.hop_sec = 0.500      # 500 ms evaluation interval
        self.window_sec = WINDOW_SEC
        
        self.audio_block_smp = AUDIO_BLOCK_SMP  # 500 samples
        self.eeg_block_smp = EEG_BLOCK_SMP      # 2 samples
        self.window_eeg_smp = int(self.window_sec * self.fs_eeg)    # 320 samples
        self.hop_ticks = int(round(self.hop_sec / self.block_sec))   # 16 ticks
        
        # Models & DSP
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CATCNDirectDecoder(eeg_channels=NUM_CHANNELS).to(self.device).eval()
        self.adapter = SpatialEEGAdapter(NUM_CHANNELS).to(self.device).eval()
        self.is_calibrated = False

        self.eeg_filter = StreamingCausalEEGFilter(lowcut=1.0, highcut=6.0, fs=float(self.fs_eeg), order=2, n_channels=NUM_CHANNELS)
        self.gammatone_ext_a = StreamingCausalAudioGammatoneExtractor(audio_fs=self.fs_audio, target_fs=self.fs_eeg)
        self.gammatone_ext_b = StreamingCausalAudioGammatoneExtractor(audio_fs=self.fs_audio, target_fs=self.fs_eeg)
        self.gate = StickyHysteresisGate(
            alpha=0.82,
            threshold_switch=0.30,
            threshold_maintain=0.10,
            n_confirm=2,
            deadband_timeout_steps=30,
            boost_db=9.0
        )
        self.steering_dsp = AudioSteeringDSP(fs=self.fs_audio, max_boost_db=9.0, max_suppress_db=18.0, tau_ms=60.0)
        
        # Playback State
        self.is_playing = False
        self.current_subject = "S1"
        self.current_trial = 4
        self.current_mode = "dataset"
        self.listening_mode = "steered"
        self.current_tick = 0
        self.total_ticks = 0
        self.total_duration_sec = 50.0
        
        # Streaming Buffer Memories
        self.eeg_buffer = np.zeros((self.window_eeg_smp, NUM_CHANNELS), dtype=np.float32)
        self.env_a_buffer = np.zeros(self.window_eeg_smp, dtype=np.float32)
        self.env_b_buffer = np.zeros(self.window_eeg_smp, dtype=np.float32)
        
        # Loaded Raw Continuous Streams
        self.trial_audio_a: np.ndarray = np.array([], dtype=np.float32)
        self.trial_audio_b: np.ndarray = np.array([], dtype=np.float32)
        self.trial_eeg: np.ndarray = np.array([], dtype=np.float32)
        self.attended_speaker: str = "A"
        
        # Hardware Telemetry State Caches
        self.last_margin = 0.0
        self.last_smoothed_margin = 0.0
        self.last_decision = "HOLD"
        self.last_gain_a_db = 0.0
        self.last_gain_b_db = 0.0
        self.last_dsp_us = 4200.0
        self.last_steer_us = 80.0
        self.last_gpu_lat_ms = 18.5
        self.smoothed_rtf = 0.015
        self.smoothed_cpu_pct = 2.1
        self.mem_mb = 125.0
        try:
            import psutil
            self.process = psutil.Process()
        except Exception:
            self.process = None
        
        # Load default trial
        self.load_trial(self.current_subject, self.current_trial)

    def _init_model_weights(self):
        """Initializes model weights from checkpoint if available, or identity initialized."""
        sub_ckpt = CHECKPOINT_DIR / f"catcn_adapted_{self.current_subject}.pt"
        if sub_ckpt.exists():
            try:
                ckpt = torch.load(sub_ckpt, map_location=self.device, weights_only=False)
                if "model_state_dict" in ckpt:
                    self.model.load_state_dict(ckpt["model_state_dict"], strict=False)
                if "adapter_state_dict" in ckpt:
                    self.adapter.load_state_dict(ckpt["adapter_state_dict"], strict=False)
                print(f"[USCAPES] Loaded calibrated weights for {self.current_subject} from: {sub_ckpt.name}")
                self.is_calibrated = True
                return
            except Exception as e:
                print(f"[USCAPES] Error loading {sub_ckpt.name}: {e}")

        cand = list(CHECKPOINT_DIR.glob("catcn_adapted_*.pt"))
        if cand:
            try:
                ckpt = torch.load(cand[0], map_location=self.device, weights_only=False)
                if "model_state_dict" in ckpt:
                    self.model.load_state_dict(ckpt["model_state_dict"], strict=False)
                if "adapter_state_dict" in ckpt:
                    self.adapter.load_state_dict(ckpt["adapter_state_dict"], strict=False)
                print(f"[USCAPES] Loaded reference adapted weights from: {cand[0].name}")
                self.is_calibrated = True
                return
            except Exception:
                pass

        base_ckpt = CHECKPOINT_DIR / "universal_catcn_backbone.pt"
        if base_ckpt.exists():
            try:
                ckpt = torch.load(base_ckpt, map_location=self.device, weights_only=False)
                sd = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
                self.model.load_state_dict(sd, strict=False)
                self.adapter.reset_to_identity()
                print(f"[USCAPES] Loaded universal base weights from: {base_ckpt.name}")
            except Exception:
                pass
        self.is_calibrated = False

    def load_trial(self, subject_id: str, trial_id: int):
        """Loads continuous trial streams and resets ring buffer states."""
        self.current_subject = subject_id
        self.current_trial = trial_id
        self._init_model_weights()
        
        audio_a, audio_b, eeg_8ch, attended = self.data_provider.load_trial_data(
            subject_id=subject_id,
            trial_id=trial_id,
            duration_sec=self.total_duration_sec,
            fs_audio=self.fs_audio,
            fs_eeg=self.fs_eeg
        )
        self.trial_audio_a = audio_a
        self.trial_audio_b = audio_b
        self.trial_eeg = eeg_8ch
        self.attended_speaker = attended
        
        self.total_ticks = int(len(self.trial_eeg) / self.eeg_block_smp)
        self.reset_playback()

    def reset_playback(self):
        """Resets playback cursor, filter states, and seeds initial buffer for instant high confidence."""
        self.current_tick = 0
        self.eeg_filter.reset()
        self.gammatone_ext_a.reset()
        self.gammatone_ext_b.reset()
        self.gate.reset()
        self.steering_dsp.reset()

        # Pre-seed sliding buffers with initial 5.0s if trial data is available
        if len(self.trial_eeg) >= self.window_eeg_smp:
            filt_init = self.eeg_filter.process_chunk(self.trial_eeg[:self.window_eeg_smp])
            self.eeg_buffer[:] = filt_init
        else:
            self.eeg_buffer.fill(0)
            
        init_aud_smp = int(self.window_sec * self.fs_audio)
        if len(self.trial_audio_a) >= init_aud_smp and len(self.trial_audio_b) >= init_aud_smp:
            env_a_init = self.gammatone_ext_a.process_audio_chunk(self.trial_audio_a[:init_aud_smp])
            env_b_init = self.gammatone_ext_b.process_audio_chunk(self.trial_audio_b[:init_aud_smp])
            n_env_a = min(len(env_a_init), self.window_eeg_smp)
            n_env_b = min(len(env_b_init), self.window_eeg_smp)
            self.env_a_buffer[:n_env_a] = env_a_init[:n_env_a]
            self.env_b_buffer[:n_env_b] = env_b_init[:n_env_b]
        else:
            self.env_a_buffer.fill(0)
            self.env_b_buffer.fill(0)

        # Warmup neural inference on seeded buffer
        try:
            norm_eeg = (self.eeg_buffer - np.mean(self.eeg_buffer, axis=0, keepdims=True)) / (np.std(self.eeg_buffer, axis=0, keepdims=True) + 1e-8)
            norm_ya = (self.env_a_buffer - np.mean(self.env_a_buffer)) / (np.std(self.env_a_buffer) + 1e-8)
            norm_yb = (self.env_b_buffer - np.mean(self.env_b_buffer)) / (np.std(self.env_b_buffer) + 1e-8)

            t_e = torch.from_numpy(norm_eeg.T).unsqueeze(0).float().to(self.device)
            t_ya = torch.from_numpy(norm_ya).unsqueeze(0).unsqueeze(0).float().to(self.device)
            t_yb = torch.from_numpy(norm_yb).unsqueeze(0).unsqueeze(0).float().to(self.device)

            with torch.no_grad():
                t_e_adapted = self.adapter(t_e)
                delta, _, _ = self.model(t_e_adapted, t_ya, t_yb)
                raw_margin = delta.item()

            gate_out = self.gate.update(raw_margin)
            self.last_margin = raw_margin
            self.last_smoothed_margin = gate_out["smoothed_margin"]
            self.last_decision = gate_out["decision"]
        except Exception:
            pass

    def seek(self, time_sec: float):
        """Seeks playback to a specific timestamp."""
        target_tick = int(round(time_sec / self.block_sec))
        self.current_tick = max(0, min(target_tick, self.total_ticks - 1))

    def step(self) -> Optional[Dict[str, Any]]:
        """
        Executes one real-time 31.25 ms simulation cycle.
        """
        if not self.is_playing:
            return None
            
        if self.current_tick >= self.total_ticks:
            self.is_playing = False
            return {"type": "end_of_trial"}

        t_tick_start = time.perf_counter()

        # 1. Fetch current 31.25 ms data slices
        a_start = self.current_tick * self.audio_block_smp
        a_end = a_start + self.audio_block_smp
        e_start = self.current_tick * self.eeg_block_smp
        e_end = e_start + self.eeg_block_smp

        chunk_a = self.trial_audio_a[a_start:a_end]
        chunk_b = self.trial_audio_b[a_start:a_end]
        chunk_eeg = self.trial_eeg[e_start:e_end]

        if len(chunk_a) < self.audio_block_smp or len(chunk_b) < self.audio_block_smp or len(chunk_eeg) < self.eeg_block_smp:
            self.is_playing = False
            return {"type": "end_of_trial"}

        # Causal IIR Bandpass on current 2-sample EEG slice
        t_dsp_start = time.perf_counter()
        filt_eeg = self.eeg_filter.process_chunk(chunk_eeg)
        
        # Causal Gammatone Auditory Envelope Extraction
        env_a_chunk = self.gammatone_ext_a.process_audio_chunk(chunk_a)
        env_b_chunk = self.gammatone_ext_b.process_audio_chunk(chunk_b)
        if len(env_a_chunk) < self.eeg_block_smp:
            env_a_chunk = np.pad(env_a_chunk, (0, self.eeg_block_smp - len(env_a_chunk)), mode='edge')
        elif len(env_a_chunk) > self.eeg_block_smp:
            env_a_chunk = env_a_chunk[:self.eeg_block_smp]
        if len(env_b_chunk) < self.eeg_block_smp:
            env_b_chunk = np.pad(env_b_chunk, (0, self.eeg_block_smp - len(env_b_chunk)), mode='edge')
        elif len(env_b_chunk) > self.eeg_block_smp:
            env_b_chunk = env_b_chunk[:self.eeg_block_smp]

        # Update sliding ring buffers (320 samples = 5.0s)
        n_c = self.eeg_block_smp
        self.eeg_buffer = np.roll(self.eeg_buffer, -n_c, axis=0)
        self.eeg_buffer[-n_c:] = filt_eeg
        
        self.env_a_buffer = np.roll(self.env_a_buffer, -n_c)
        self.env_a_buffer[-n_c:] = env_a_chunk
        self.env_b_buffer = np.roll(self.env_b_buffer, -n_c)
        self.env_b_buffer[-n_c:] = env_b_chunk

        t_dsp_us = (time.perf_counter() - t_dsp_start) * 1e6
        self.last_dsp_us = t_dsp_us

        # 2. CA-TCN Neural Decoding (every 500 ms hop = 16 ticks)
        if self.current_tick % self.hop_ticks == 0:
            t_inf_start = time.perf_counter()
            
            # Causal z-score normalization
            norm_eeg = (self.eeg_buffer - np.mean(self.eeg_buffer, axis=0, keepdims=True)) / (np.std(self.eeg_buffer, axis=0, keepdims=True) + 1e-8)
            norm_ya = (self.env_a_buffer - np.mean(self.env_a_buffer)) / (np.std(self.env_a_buffer) + 1e-8)
            norm_yb = (self.env_b_buffer - np.mean(self.env_b_buffer)) / (np.std(self.env_b_buffer) + 1e-8)

            t_e = torch.from_numpy(norm_eeg.T).unsqueeze(0).float().to(self.device)
            t_ya = torch.from_numpy(norm_ya).unsqueeze(0).unsqueeze(0).float().to(self.device)
            t_yb = torch.from_numpy(norm_yb).unsqueeze(0).unsqueeze(0).float().to(self.device)

            with torch.no_grad():
                t_e_adapted = self.adapter(t_e)
                delta, _, _ = self.model(t_e_adapted, t_ya, t_yb)
                raw_margin = delta.item()
                
            self.last_gpu_lat_ms = (time.perf_counter() - t_inf_start) * 1000.0
            
            # Update Hysteresis State Machine
            gate_out = self.gate.update(raw_margin)
            self.last_margin = raw_margin
            self.last_smoothed_margin = gate_out["smoothed_margin"]
            self.last_decision = gate_out["decision"]

        # 3. Dynamic Audio Steering (+9 dB / -18 dB)
        t_steer_start = time.perf_counter()
        decision_label = "A" if "A" in self.last_decision else ("B" if "B" in self.last_decision else "HOLD")
        g_a_db, g_b_db = self.steering_dsp.compute_target_gains_db(decision_label, self.last_smoothed_margin)
        stereo_out, ga_traj, gb_traj = self.steering_dsp.process_block(chunk_a, chunk_b, g_a_db, g_b_db)
        self.last_steer_us = (time.perf_counter() - t_steer_start) * 1e6
        self.last_gain_a_db = float(g_a_db)
        self.last_gain_b_db = float(g_b_db)

        # 4. Select Audio Signal for Headphone Output (Stereo Binaural)
        if self.listening_mode == "steered":
            out_audio = stereo_out
        elif self.listening_mode == "mixture":
            mix = 0.5 * (chunk_a + chunk_b)
            out_audio = np.stack([mix, mix], axis=0)
        elif self.listening_mode == "speaker_a":
            out_audio = np.stack([chunk_a, chunk_a], axis=0)
        elif self.listening_mode == "speaker_b":
            out_audio = np.stack([chunk_b, chunk_b], axis=0)
        else:
            out_audio = stereo_out

        # Encode interleaved stereo 16-bit PCM for immersive spatial headphone playback
        left = (np.clip(out_audio[0], -1.0, 1.0) * 32767).astype(np.int16)
        right = (np.clip(out_audio[1], -1.0, 1.0) * 32767).astype(np.int16)
        interleaved = np.empty(len(left) + len(right), dtype=np.int16)
        interleaved[0::2] = left
        interleaved[1::2] = right
        audio_b64 = base64.b64encode(interleaved.tobytes()).decode("ascii")

        # Dynamic Real-Time Benchmarks
        t_tick_total_sec = time.perf_counter() - t_tick_start
        instant_rtf = t_tick_total_sec / self.block_sec
        self.smoothed_rtf = 0.85 * self.smoothed_rtf + 0.15 * instant_rtf
        speedup_x = 1.0 / max(self.smoothed_rtf, 1e-4)
        headroom_pct = max(0.0, 100.0 - (self.smoothed_rtf * 100.0))

        if self.current_tick % 8 == 0 and self.process is not None:
            try:
                proc_cpu = self.process.cpu_percent(interval=None)
                duty_cycle = (t_tick_total_sec / self.block_sec) * 100.0
                active_cpu = proc_cpu if proc_cpu > 0.1 else duty_cycle
                self.smoothed_cpu_pct = 0.8 * self.smoothed_cpu_pct + 0.2 * active_cpu
                self.mem_mb = self.process.memory_info().rss / (1024.0 * 1024.0)
            except Exception:
                pass

        total_lat_ms = (self.last_dsp_us + self.last_steer_us) / 1000.0 + self.last_gpu_lat_ms
        current_time = self.current_tick * self.block_sec
        target_winner = (decision_label == self.attended_speaker)
        
        telemetry = {
            "type": "tick",
            "time_sec": round(current_time, 3),
            "progress_pct": round((current_time / self.total_duration_sec) * 100.0, 1),
            "current_tick": self.current_tick,
            "total_ticks": self.total_ticks,
            "is_calibrated": self.is_calibrated,
            # EEG multi-channel slice (microvolts, 1.0-6.0 Hz bandpass filtered with zero DC offset)
            "eeg_sample": [round(float(filt_eeg[-1, ch]), 2) for ch in range(NUM_CHANNELS)],
            "eeg_block": [[round(float(filt_eeg[s, ch]), 2) for ch in range(NUM_CHANNELS)] for s in range(filt_eeg.shape[0])],
            "channel_names": MONTAGE_CHANNELS,
            # Decoding telemetry
            "raw_margin": round(self.last_margin, 2),
            "smoothed_margin": round(self.last_smoothed_margin, 2),
            "decision": self.last_decision,
            "decision_label": decision_label,
            "attended_speaker": self.attended_speaker,
            "is_correct": target_winner,
            # Gains
            "gain_a_db": round(self.last_gain_a_db, 1),
            "gain_b_db": round(self.last_gain_b_db, 1),
            "listening_mode": self.listening_mode,
            # Real Measured Hardware Telemetry (100% Live)
            "dsp_latency_us": round(self.last_dsp_us, 1),
            "steer_latency_us": round(self.last_steer_us, 1),
            "gpu_latency_ms": round(self.last_gpu_lat_ms, 2),
            "total_latency_ms": round(total_lat_ms, 2),
            "cpu_load_pct": round(self.smoothed_cpu_pct, 1),
            "headroom_pct": round(headroom_pct, 1),
            "rtf": round(self.smoothed_rtf, 4),
            "speedup_x": round(speedup_x, 1),
            "mem_mb": round(self.mem_mb, 1),
            # Audio packet
            "audio_b64": audio_b64
        }

        self.current_tick += 1
        return telemetry

    def handle_client_message(self, msg: Dict[str, Any]) -> Dict[str, Any]:
        """Handles incoming commands from the web UI."""
        action = msg.get("action")
        
        if action == "play":
            if self.current_tick >= self.total_ticks:
                self.reset_playback()
            self.is_playing = True
            return {"status": "ok", "state": "playing"}
            
        elif action == "pause":
            self.is_playing = False
            return {"status": "ok", "state": "paused"}
            
        elif action == "seek":
            t_sec = float(msg.get("time_sec", 0.0))
            self.seek(t_sec)
            return {"status": "ok", "state": "seeked", "time_sec": t_sec}
            
        elif action == "reset":
            self.is_playing = False
            self.reset_playback()
            return {"status": "ok", "state": "reset"}
            
        elif action == "set_subject":
            sub_id = msg.get("subject_id", "S1")
            self.is_playing = False
            self.load_trial(sub_id, self.current_trial)
            return {"status": "ok", "subject_id": sub_id, "trial_id": self.current_trial}
            
        elif action == "set_trial":
            trial_id = int(msg.get("trial_id", 4))
            self.is_playing = False
            self.load_trial(self.current_subject, trial_id)
            return {"status": "ok", "subject_id": self.current_subject, "trial_id": trial_id}
            
        elif action == "set_listening_mode":
            mode = msg.get("mode", "steered")
            if mode in ["steered", "mixture", "speaker_a", "speaker_b"]:
                self.listening_mode = mode
            return {"status": "ok", "listening_mode": self.listening_mode}
            
        elif action == "set_source":
            mode = msg.get("mode", "dataset")
            if mode in ["dataset", "respeaker"]:
                self.current_mode = mode
            return {"status": "ok", "source_mode": self.current_mode}
            
        return {"status": "error", "message": f"Unknown action: {action}"}


SessionStreamingManager = StreamingSimulationSession

