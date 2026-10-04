"""
Real-Time Dual-Stream Audio-EEG Synchronization Engine.

Coordinates incoming raw 512 Hz multi-channel EEG and raw 44.1 kHz / 16 kHz candidate
audio streams in lock-step 31.25 ms packets. Maintains synchronized ring buffers
at 64 Hz and triggers evaluation frames into CA-TCN.
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, Any, Union
import numpy as np
import time

from src.streaming.causal_raw_preprocessor import StreamingCausalRawEEGPreprocessor
from src.audio.causal_gammatone import StreamingCausalAudioGammatoneExtractor


@dataclass
class DualStreamFrame:
    """Represents the output of a single streaming tick from the dual-stream engine."""
    ready_for_eval: bool
    tick_index: int
    eeg_samples_generated: int
    audio_a_samples_generated: int
    audio_b_samples_generated: int
    eeg_window: Optional[np.ndarray] = None       # Shape [8, window_samples]
    audio_a_window: Optional[np.ndarray] = None   # Shape [1, window_samples]
    audio_b_window: Optional[np.ndarray] = None   # Shape [1, window_samples]
    buffered_samples: int = 0
    dsp_timing_us: Dict[str, float] = field(default_factory=dict)


class DualStreamIngestionEngine:
    """
    Synchronous Dual-Stream Ingestion Engine.
    
    Ingests:
      - Raw EEG: 16 samples @ 512 Hz (31.25 ms) with scalp (64 channels) + optional EOG (2 channels).
      - Raw Audio A: 31.25 ms of audio @ audio_fs (e.g. 1378 samples @ 44.1 kHz, or 500 samples @ 16 kHz).
      - Raw Audio B: 31.25 ms of audio @ audio_fs.
      
    Generates:
      - 2 samples at 64 Hz per tick for EEG (8 channels) and both audio streams (broadband envelope).
      - Synchronized ring buffer views of size [8, 320] and [1, 320] (for 5.0s window @ 64 Hz).
      - Frame evaluation trigger every hop_sec (default 0.5s = 16 ticks = 32 samples).
    """
    def __init__(
        self,
        raw_eeg_fs: float = 512.0,
        audio_fs: float = 44100.0,
        target_fs: float = 64.0,
        window_sec: float = 5.0,
        hop_sec: float = 0.5,
        power_exponent: float = 0.3,
        montage_name: str = "near_ear_expanded",
        eog_weights: Optional[np.ndarray] = None,
        normalize_audio_window: bool = True
    ):
        self.raw_eeg_fs = float(raw_eeg_fs)
        self.audio_fs = float(audio_fs)
        self.target_fs = float(target_fs)
        self.window_sec = float(window_sec)
        self.hop_sec = float(hop_sec)
        self.power_exponent = float(power_exponent)
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
        self.n_eeg_channels = self.eeg_preprocessor.n_out_channels
        
        # 2. Causal Raw Audio Gammatone Extractors for Candidate Talkers
        self.audio_extractor_a = StreamingCausalAudioGammatoneExtractor(
            audio_fs=self.audio_fs,
            target_fs=self.target_fs,
            power_exponent=self.power_exponent
        )
        self.audio_extractor_b = StreamingCausalAudioGammatoneExtractor(
            audio_fs=self.audio_fs,
            target_fs=self.target_fs,
            power_exponent=self.power_exponent
        )
        
        # 3. Synchronized Ring Buffers
        self.eeg_buffer = np.zeros((self.window_samples, self.n_eeg_channels), dtype=np.float32)
        self.audio_a_buffer = np.zeros((self.window_samples,), dtype=np.float32)
        self.audio_b_buffer = np.zeros((self.window_samples,), dtype=np.float32)
        
        self.buffered_samples = 0
        self.samples_since_hop = 0
        self.tick_count = 0

    def reset(self):
        """Flushes all filter delay states, phase accumulators, and ring buffers."""
        self.eeg_preprocessor.reset()
        self.audio_extractor_a.reset()
        self.audio_extractor_b.reset()
        
        self.eeg_buffer.fill(0.0)
        self.audio_a_buffer.fill(0.0)
        self.audio_b_buffer.fill(0.0)
        
        self.buffered_samples = 0
        self.samples_since_hop = 0
        self.tick_count = 0

    def step(
        self,
        eeg_chunk: np.ndarray,
        audio_a_chunk: np.ndarray,
        audio_b_chunk: np.ndarray,
        veog_chunk: Optional[np.ndarray] = None,
        heog_chunk: Optional[np.ndarray] = None
    ) -> DualStreamFrame:
        """
        Executes a single synchronized streaming step across EEG and both candidate audio streams.
        
        Parameters:
            eeg_chunk: [N_eeg, 64] raw scalp EEG samples at raw_eeg_fs (typically 16 samples @ 512 Hz).
            audio_a_chunk: 1D or 2D audio samples for Talker A at audio_fs (~1378 samples @ 44.1 kHz).
            audio_b_chunk: 1D or 2D audio samples for Talker B at audio_fs (~1378 samples @ 44.1 kHz).
            veog_chunk: Optional [N_eeg] vertical EOG samples at raw_eeg_fs.
            heog_chunk: Optional [N_eeg] horizontal EOG samples at raw_eeg_fs.
            
        Returns:
            DualStreamFrame containing synchronization flags, window views (if ready), and DSP timing.
        """
        self.tick_count += 1
        
        # 1. Process EEG Chunk
        t0 = time.perf_counter()
        eeg_out_64 = self.eeg_preprocessor.process_raw_chunk(eeg_chunk, veog_chunk, heog_chunk)
        t_eeg_us = (time.perf_counter() - t0) * 1e6
        
        # 2. Process Audio A Chunk
        t1 = time.perf_counter()
        audio_a_out_64 = self.audio_extractor_a.process_audio_chunk(audio_a_chunk)
        t_audio_a_us = (time.perf_counter() - t1) * 1e6
        
        # 3. Process Audio B Chunk
        t2 = time.perf_counter()
        audio_b_out_64 = self.audio_extractor_b.process_audio_chunk(audio_b_chunk)
        t_audio_b_us = (time.perf_counter() - t2) * 1e6
        
        n_eeg_out = len(eeg_out_64)
        n_audio_a_out = len(audio_a_out_64)
        n_audio_b_out = len(audio_b_out_64)
        
        # Number of samples to append in lock-step
        n_samples_step = min(n_eeg_out, n_audio_a_out, n_audio_b_out)
        
        if n_samples_step > 0:
            if n_samples_step >= self.window_samples:
                self.eeg_buffer[:] = eeg_out_64[-self.window_samples:]
                self.audio_a_buffer[:] = audio_a_out_64[-self.window_samples:]
                self.audio_b_buffer[:] = audio_b_out_64[-self.window_samples:]
                self.buffered_samples = self.window_samples
            else:
                self.eeg_buffer[:-n_samples_step] = self.eeg_buffer[n_samples_step:]
                self.eeg_buffer[-n_samples_step:] = eeg_out_64[:n_samples_step]
                
                self.audio_a_buffer[:-n_samples_step] = self.audio_a_buffer[n_samples_step:]
                self.audio_a_buffer[-n_samples_step:] = audio_a_out_64[:n_samples_step]
                
                self.audio_b_buffer[:-n_samples_step] = self.audio_b_buffer[n_samples_step:]
                self.audio_b_buffer[-n_samples_step:] = audio_b_out_64[:n_samples_step]
                
                self.buffered_samples = min(self.window_samples, self.buffered_samples + n_samples_step)
            self.samples_since_hop += n_samples_step
                
        # 4. Check if CA-TCN Evaluation Hop Triggered
        ready = (self.buffered_samples >= self.window_samples) and (self.samples_since_hop >= self.hop_samples)
        
        eeg_win = None
        audio_a_win = None
        audio_b_win = None
        
        if ready:
            self.samples_since_hop = 0
            
            # EEG Window shape: [8, window_samples]
            eeg_win = self.eeg_buffer.T.copy()
            
            # Audio Windows shape: [1, window_samples]
            ya = self.audio_a_buffer.copy()
            yb = self.audio_b_buffer.copy()
            
            if self.normalize_audio_window:
                # Causal window-level z-scoring matching training protocol
                ya = (ya - np.mean(ya)) / (np.std(ya) + 1e-8)
                yb = (yb - np.mean(yb)) / (np.std(yb) + 1e-8)
                
            audio_a_win = np.expand_dims(ya.astype(np.float32), axis=0)
            audio_b_win = np.expand_dims(yb.astype(np.float32), axis=0)
            
        dsp_timing = {
            "eeg_us": t_eeg_us,
            "audio_a_us": t_audio_a_us,
            "audio_b_us": t_audio_b_us,
            "total_dsp_us": t_eeg_us + t_audio_a_us + t_audio_b_us
        }
        
        return DualStreamFrame(
            ready_for_eval=ready,
            tick_index=self.tick_count,
            eeg_samples_generated=n_eeg_out,
            audio_a_samples_generated=n_audio_a_out,
            audio_b_samples_generated=n_audio_b_out,
            eeg_window=eeg_win,
            audio_a_window=audio_a_win,
            audio_b_window=audio_b_win,
            buffered_samples=self.buffered_samples,
            dsp_timing_us=dsp_timing
        )
