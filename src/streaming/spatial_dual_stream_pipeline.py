"""
Spatial Dual-Stream Audio-EEG Synchronization & Beamforming Engine.

Extends the real-time continuous ingestion pipeline with ReSpeaker 4-channel spatial acoustic beamforming:
  1. Ingests raw 4-channel ReSpeaker audio (500 samples @ 16 kHz = 31.25 ms) and raw EEG (16 samples @ 512 Hz = 31.25 ms).
  2. Spatially filters the 4 mic channels via ReSpeakerSpatialBeamformer into Beam A (Left) and Beam B (Right).
  3. Causally extracts Gammatone envelopes at 64 Hz for both beams.
  4. Maintains lock-step sliding ring buffers for EEG (8 ch x 320) and both candidate envelopes (1 ch x 320).
  5. Triggers neural decoding every 500 ms hop, then steers audio amplification (+9 dB / -18 dB).
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, Any, Union
import numpy as np
import time

from src.audio.spatial_beamformer import ReSpeakerSpatialBeamformer
from src.audio.causal_gammatone import StreamingCausalAudioGammatoneExtractor
from src.audio.steering_engine import AudioSteeringDSP
from src.streaming.causal_raw_preprocessor import StreamingCausalRawEEGPreprocessor


@dataclass
class SpatialDualStreamFrame:
    """Represents the output of a single streaming tick from the spatial dual-stream engine."""
    ready_for_eval: bool
    tick_index: int
    eeg_samples_generated: int
    audio_samples_generated: int
    eeg_window: Optional[np.ndarray] = None       # Shape [8, window_samples]
    audio_a_window: Optional[np.ndarray] = None   # Shape [1, window_samples] (Beam A envelope)
    audio_b_window: Optional[np.ndarray] = None   # Shape [1, window_samples] (Beam B envelope)
    raw_beam_a_chunk: Optional[np.ndarray] = None # Time-domain Beam A audio (500 samples @ 16 kHz)
    raw_beam_b_chunk: Optional[np.ndarray] = None # Time-domain Beam B audio (500 samples @ 16 kHz)
    buffered_samples: int = 0
    dsp_timing_us: Dict[str, float] = field(default_factory=dict)


class SpatialDualStreamIngestionEngine:
    """
    Synchronous Multi-Channel Audio-EEG Streaming Engine with ReSpeaker Spatial Beamforming.
    """
    def __init__(
        self,
        raw_eeg_fs: float = 512.0,
        audio_fs: float = 16000.0,
        target_fs: float = 64.0,
        window_sec: float = 5.0,
        hop_sec: float = 0.5,
        theta_a_deg: float = -45.0,
        theta_b_deg: float = 45.0,
        beamformer_mode: str = "lcmv_null",
        montage_name: str = "near_ear_expanded",
        eog_weights: Optional[np.ndarray] = None,
        normalize_audio_window: bool = True,
        max_boost_db: float = 9.0,
        max_suppress_db: float = 18.0,
    ):
        self.raw_eeg_fs = float(raw_eeg_fs)
        self.audio_fs = float(audio_fs)
        self.target_fs = float(target_fs)
        self.window_sec = float(window_sec)
        self.hop_sec = float(hop_sec)
        self.normalize_audio_window = normalize_audio_window

        self.window_samples = int(round(self.window_sec * self.target_fs))
        self.hop_samples = int(round(self.hop_sec * self.target_fs))

        # 1. Causal Raw EEG Preprocessor
        self.eeg_preprocessor = StreamingCausalRawEEGPreprocessor(
            raw_fs=self.raw_eeg_fs,
            target_fs=self.target_fs,
            montage_name=montage_name,
            eog_regression_weights=eog_weights
        )

        # 2. Causal Spatial Beamformer for ReSpeaker 4-Mic Array
        self.beamformer = ReSpeakerSpatialBeamformer(
            fs=int(self.audio_fs),
            radius=0.0325,
            theta_a_deg=theta_a_deg,
            theta_b_deg=theta_b_deg,
            mode=beamformer_mode
        )

        # 3. Dual Causal Gammatone Envelope Extractors (16 kHz -> 64 Hz)
        self.env_extractor_a = StreamingCausalAudioGammatoneExtractor(
            audio_fs=self.audio_fs,
            target_fs=self.target_fs,
            num_bands=28,
            internal_fs=self.audio_fs
        )
        self.env_extractor_b = StreamingCausalAudioGammatoneExtractor(
            audio_fs=self.audio_fs,
            target_fs=self.target_fs,
            num_bands=28,
            internal_fs=self.audio_fs
        )

        # 4. Audio Steering DSP Engine
        self.steering_dsp = AudioSteeringDSP(
            fs=int(self.audio_fs),
            max_boost_db=max_boost_db,
            max_suppress_db=max_suppress_db
        )

        # 5. Sliding Ring Buffers at 64 Hz
        self.eeg_buffer = np.zeros((8, self.window_samples), dtype=np.float32)
        self.audio_a_buffer = np.zeros((1, self.window_samples), dtype=np.float32)
        self.audio_b_buffer = np.zeros((1, self.window_samples), dtype=np.float32)

        self.buffered_samples = 0
        self.samples_since_last_eval = 0
        self.tick_count = 0

    def reset(self):
        """Resets all internal DSP states, beamformer buffers, and ring buffers to zero."""
        self.eeg_preprocessor.reset()
        self.beamformer.reset()
        self.env_extractor_a.reset()
        self.env_extractor_b.reset()
        self.steering_dsp.reset()

        self.eeg_buffer.fill(0.0)
        self.audio_a_buffer.fill(0.0)
        self.audio_b_buffer.fill(0.0)

        self.buffered_samples = 0
        self.samples_since_last_eval = 0
        self.tick_count = 0

    def process_tick(
        self,
        raw_eeg_packet: np.ndarray,
        raw_respeaker_packet: np.ndarray,
        veog_packet: Optional[np.ndarray] = None,
        heog_packet: Optional[np.ndarray] = None
    ) -> SpatialDualStreamFrame:
        """
        Processes a single synchronous time slice (e.g. 31.25 ms).
        
        Parameters:
            raw_eeg_packet: Raw EEG chunk of shape (64, n_eeg_samples) or (n_eeg_samples, 64).
            raw_respeaker_packet: Raw 4-channel ReSpeaker chunk of shape (4, n_audio_samples).
            veog_packet: Optional vertical EOG packet.
            heog_packet: Optional horizontal EOG packet.
            
        Returns:
            SpatialDualStreamFrame containing windowed EEG and envelopes if ready_for_eval is True.
        """
        timing = {}

        # Stage 1: Spatial Beamforming (4 Mics -> Beam A & Beam B)
        t0 = time.perf_counter()
        beam_a, beam_b = self.beamformer.process_chunk(raw_respeaker_packet)
        timing["beamforming_us"] = (time.perf_counter() - t0) * 1e6

        # Stage 2: Causal Gammatone Envelope Extraction (16 kHz -> 64 Hz)
        t0 = time.perf_counter()
        env_a = self.env_extractor_a.process_audio_chunk(beam_a)
        env_b = self.env_extractor_b.process_audio_chunk(beam_b)
        timing["envelope_us"] = (time.perf_counter() - t0) * 1e6

        # Stage 3: Causal EEG Preprocessing (512 Hz -> 64 Hz)
        t0 = time.perf_counter()
        eeg_arr = np.asarray(raw_eeg_packet, dtype=np.float32)
        if eeg_arr.shape[0] == 64 and eeg_arr.shape[1] != 64:
            eeg_arr = eeg_arr.T  # Transpose to [N_samples, 64]
        
        processed_eeg = self.eeg_preprocessor.process_raw_chunk(
            eeg_arr, veog_chunk_512=veog_packet, heog_chunk_512=heog_packet
        )  # [N_out, 8]
        dec_eeg = processed_eeg.T  # Transpose to [8, N_out]
        timing["eeg_dsp_us"] = (time.perf_counter() - t0) * 1e6

        # Enforce temporal synchronization at 64 Hz
        n_eeg = dec_eeg.shape[1]
        n_a = len(env_a)
        n_b = len(env_b)
        n_gen = min(n_eeg, n_a, n_b)

        if n_gen > 0:
            eeg_to_add = dec_eeg[:, :n_gen]
            a_to_add = env_a[:n_gen].reshape(1, -1)
            b_to_add = env_b[:n_gen].reshape(1, -1)

            # Slide ring buffers
            self.eeg_buffer = np.roll(self.eeg_buffer, -n_gen, axis=1)
            self.eeg_buffer[:, -n_gen:] = eeg_to_add

            self.audio_a_buffer = np.roll(self.audio_a_buffer, -n_gen, axis=1)
            self.audio_a_buffer[:, -n_gen:] = a_to_add

            self.audio_b_buffer = np.roll(self.audio_b_buffer, -n_gen, axis=1)
            self.audio_b_buffer[:, -n_gen:] = b_to_add

            self.buffered_samples = min(self.window_samples, self.buffered_samples + n_gen)
            self.samples_since_last_eval += n_gen

        # Check if evaluation frame is ready
        ready = (self.buffered_samples >= self.window_samples) and (self.samples_since_last_eval >= self.hop_samples)

        frame_eeg = None
        frame_a = None
        frame_b = None

        if ready:
            frame_eeg = self.eeg_buffer.copy()
            frame_a = self.audio_a_buffer.copy()
            frame_b = self.audio_b_buffer.copy()

            if self.normalize_audio_window:
                # Per-window standardization matching CA-TCN training
                std_a = np.std(frame_a) + 1e-8
                std_b = np.std(frame_b) + 1e-8
                frame_a = (frame_a - np.mean(frame_a)) / std_a
                frame_b = (frame_b - np.mean(frame_b)) / std_b

            self.samples_since_last_eval = 0

        self.tick_count += 1
        return SpatialDualStreamFrame(
            ready_for_eval=ready,
            tick_index=self.tick_count,
            eeg_samples_generated=n_gen,
            audio_samples_generated=n_gen,
            eeg_window=frame_eeg,
            audio_a_window=frame_a,
            audio_b_window=frame_b,
            raw_beam_a_chunk=beam_a,
            raw_beam_b_chunk=beam_b,
            buffered_samples=self.buffered_samples,
            dsp_timing_us=timing
        )

    def steer_acoustic_output(
        self,
        raw_beam_a: np.ndarray,
        raw_beam_b: np.ndarray,
        decision_state: str,
        neural_margin: float
    ) -> Tuple[np.ndarray, np.ndarray, float, float]:
        """
        Steers the spatial beamformed audio using neural attention decision.
        
        Returns:
            steered_left, steered_right, gain_db_a, gain_db_b
        """
        g_a_db, g_b_db = self.steering_dsp.compute_target_gains_db(decision_state, neural_margin)
        stereo_out, g_a_traj, g_b_traj = self.steering_dsp.process_block(
            raw_beam_a, raw_beam_b, g_a_db, g_b_db
        )
        return stereo_out[0], stereo_out[1], float(g_a_db), float(g_b_db)
