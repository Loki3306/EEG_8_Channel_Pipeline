"""
Hardware Audio Interface for Seeed Studio ReSpeaker Mic Array v2.0.

Provides continuous, low-latency multi-channel ingestion:
  1. Auto-discovers ReSpeaker USB device via PyAudio.
  2. Ingests raw 4-channel MEMS microphone streams at 16 kHz.
  3. Thread-safe non-blocking audio ring buffer.
  4. Automatic fallback to simulated 4-channel acoustic streaming if hardware is disconnected.
"""

from typing import Dict, List, Optional, Tuple, Union
import time
import threading
from collections import deque
import numpy as np

try:
    import pyaudio
    HAS_PYAUDIO = True
except ImportError:
    HAS_PYAUDIO = False

from src.audio.acoustic_crosstalk_simulator import ReSpeakerAcousticSimulator


class ReSpeakerDevice:
    """
    Manages connection and live streaming from Seeed Studio ReSpeaker Mic Array v2.0.
    
    Parameters:
        fs: Target sampling rate (default: 16000 Hz, native ReSpeaker format).
        chunk_size: Samples per chunk (default: 500 samples = 31.25 ms @ 16 kHz).
        device_index: Specific PyAudio device index, or None for auto-discovery.
        simulation_fallback: If True, falls back to acoustic simulation if hardware is missing.
    """
    def __init__(
        self,
        fs: int = 16000,
        chunk_size: int = 500,
        device_index: Optional[int] = None,
        simulation_fallback: bool = True,
    ):
        self.fs = int(fs)
        self.chunk_size = int(chunk_size)
        self.requested_device_index = device_index
        self.simulation_fallback = bool(simulation_fallback)

        self.pyaudio_instance: Optional["pyaudio.PyAudio"] = None
        self.stream: Optional["pyaudio.Stream"] = None
        self.is_streaming = False
        self.is_hardware_connected = False
        self.device_name = "None"
        self.actual_device_index: Optional[int] = None
        self.hardware_num_channels = 0

        # Ring buffer for raw 4-channel audio chunks (shape: (4, chunk_size))
        self.buffer = deque(maxlen=64)  # ~2 seconds of buffer headroom
        self._lock = threading.Lock()

        # Acoustic simulator for offline/fallback mode
        self.simulator = ReSpeakerAcousticSimulator(fs=self.fs)
        self.sim_audio_a: Optional[np.ndarray] = None
        self.sim_audio_b: Optional[np.ndarray] = None
        self.sim_pos = 0

        self._initialize_device()

    def _find_respeaker_device(self) -> Tuple[Optional[int], str, int]:
        """
        Scans all available audio input devices for ReSpeaker or 4-channel USB interfaces.
        """
        if not HAS_PYAUDIO or self.pyaudio_instance is None:
            return None, "PyAudio Unavailable", 0

        count = self.pyaudio_instance.get_device_count()
        candidates = []

        for i in range(count):
            try:
                info = self.pyaudio_instance.get_device_info_by_index(i)
                max_in = info.get("maxInputChannels", 0)
                name = info.get("name", "")
                if max_in >= 4:
                    candidates.append((i, name, max_in))
                    # Prioritize exact ReSpeaker / Seeed matches
                    name_lower = name.lower()
                    if "respeaker" in name_lower or "seeed" in name_lower:
                        return i, name, max_in
            except Exception:
                continue

        # If a candidate with >=4 channels was found, return the best candidate
        if candidates:
            return candidates[0]

        return None, "No 4-channel device found", 0

    def _initialize_device(self):
        """Initializes PyAudio and determines hardware vs simulation mode."""
        if not HAS_PYAUDIO:
            print("  [RESPEAKER] PyAudio not installed. Operating in SIMULATION mode.")
            self.is_hardware_connected = False
            return

        try:
            self.pyaudio_instance = pyaudio.PyAudio()
            if self.requested_device_index is not None:
                info = self.pyaudio_instance.get_device_info_by_index(self.requested_device_index)
                self.actual_device_index = self.requested_device_index
                self.device_name = info.get("name", "Custom Device")
                self.hardware_num_channels = info.get("maxInputChannels", 4)
                self.is_hardware_connected = True
            else:
                idx, name, chans = self._find_respeaker_device()
                if idx is not None:
                    self.actual_device_index = idx
                    self.device_name = name
                    self.hardware_num_channels = chans
                    self.is_hardware_connected = True
                    print(f"  [RESPEAKER] Hardware Detected: '{name}' (Index: {idx}, Channels: {chans})")
                else:
                    self.is_hardware_connected = False
                    if self.simulation_fallback:
                        print(f"  [RESPEAKER] Hardware not connected ({name}). Operating in SIMULATION mode.")
                    else:
                        raise RuntimeError("ReSpeaker hardware not found and simulation fallback disabled.")
        except Exception as e:
            self.is_hardware_connected = False
            print(f"  [RESPEAKER] Hardware init error: {e}. Falling back to SIMULATION mode.")

    def set_simulation_tracks(self, audio_a: np.ndarray, audio_b: np.ndarray, crosstalk_sir_db: Optional[float] = None):
        """
        Supplies ground-truth speech tracks for simulation mode (e.g. from DTU dataset).
        """
        min_len = min(len(audio_a), len(audio_b))
        sim_4ch = self.simulator.simulate_4channel_mixture(
            audio_a[:min_len], audio_b[:min_len], crosstalk_sir_db=crosstalk_sir_db
        )
        self.sim_audio_mixture = sim_4ch
        self.sim_pos = 0

    def _audio_callback(self, in_data, frame_count, time_info, status):
        """PyAudio non-blocking callback to pull raw audio."""
        if in_data is not None:
            # ReSpeaker v2 raw format: int16 interleaved
            audio_int16 = np.frombuffer(in_data, dtype=np.int16)
            # Reshape into (frame_count, num_channels)
            total_samples = len(audio_int16)
            if total_samples % self.hardware_num_channels == 0:
                reshaped = audio_int16.reshape(-1, self.hardware_num_channels).T  # (channels, frame_count)
                # Keep first 4 channels (raw digital MEMS mics) and convert to float32
                raw_4ch = (reshaped[:4, :].astype(np.float32)) / 32768.0
                with self._lock:
                    self.buffer.append(raw_4ch)

        return (None, pyaudio.paContinue)

    def start_streaming(self):
        """Starts live non-blocking audio capture stream."""
        if self.is_streaming:
            return

        if self.is_hardware_connected and self.pyaudio_instance is not None and self.actual_device_index is not None:
            try:
                # Open stream with native channels (typically 6 channels on ReSpeaker v2)
                chans = max(4, self.hardware_num_channels)
                self.stream = self.pyaudio_instance.open(
                    format=pyaudio.paInt16,
                    channels=chans,
                    rate=self.fs,
                    input=True,
                    input_device_index=self.actual_device_index,
                    frames_per_buffer=self.chunk_size,
                    stream_callback=self._audio_callback
                )
                self.stream.start_stream()
                self.is_streaming = True
                print(f"  [RESPEAKER] Live audio streaming started on '{self.device_name}' (16 kHz, 4-ch).")
                return
            except Exception as e:
                print(f"  [RESPEAKER] Failed to open hardware stream: {e}. Switching to SIMULATION mode.")
                self.is_hardware_connected = False

        self.is_streaming = True
        self.sim_pos = 0
        print("  [RESPEAKER] Simulation audio streaming initialized.")

    def read_chunk(self, timeout_sec: float = 0.1) -> np.ndarray:
        """
        Reads the next 4-channel audio chunk of shape (4, chunk_size).
        
        Returns:
            chunk_4ch: 2D numpy array of shape (4, chunk_size), float32 normalized.
        """
        if not self.is_streaming:
            raise RuntimeError("Stream is not running. Call start_streaming() first.")

        if self.is_hardware_connected:
            start_time = time.time()
            while time.time() - start_time < timeout_sec:
                with self._lock:
                    if self.buffer:
                        return self.buffer.popleft()
                time.sleep(0.005)
            # Timeout: return silence chunk
            return np.zeros((4, self.chunk_size), dtype=np.float32)

        # Simulation Mode
        if hasattr(self, "sim_audio_mixture") and self.sim_audio_mixture is not None:
            total_samples = self.sim_audio_mixture.shape[1]
            if self.sim_pos + self.chunk_size <= total_samples:
                chunk = self.sim_audio_mixture[:, self.sim_pos:self.sim_pos + self.chunk_size]
                self.sim_pos += self.chunk_size
                return chunk
            else:
                # Loop simulation or return silence
                rem = total_samples - self.sim_pos
                if rem > 0:
                    chunk = self.sim_audio_mixture[:, self.sim_pos:]
                    self.sim_pos = 0
                    return np.pad(chunk, ((0, 0), (0, self.chunk_size - rem)))
                self.sim_pos = 0
                return self.sim_audio_mixture[:, :self.chunk_size]

        # No simulation tracks provided: return silence
        return np.zeros((4, self.chunk_size), dtype=np.float32)

    def stop_streaming(self):
        """Stops the audio stream and releases hardware resources."""
        if not self.is_streaming:
            return

        if self.stream is not None:
            try:
                self.stream.stop_stream()
                self.stream.close()
            except Exception:
                pass
            self.stream = None

        self.is_streaming = False
        print("  [RESPEAKER] Audio streaming stopped.")

    def close(self):
        """Terminates PyAudio instance."""
        self.stop_streaming()
        if self.pyaudio_instance is not None:
            try:
                self.pyaudio_instance.terminate()
            except Exception:
                pass
            self.pyaudio_instance = None
