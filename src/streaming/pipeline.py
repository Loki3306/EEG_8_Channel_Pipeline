import numpy as np
import torch
import torch.nn as nn
from typing import Optional, Union, Dict, Any

from .causal_filters import StreamingCausalEEGFilter
from .causal_envelope import StreamingCausalEnvelopeExtractor
from .circular_buffer import SynchronizedRingBuffer
from ..deployment.engine import StreamingCATCNEngine
from ..deployment.decision_smoother import EMAHysteresisDecisionLayer

class StreamingAADPipeline:
    """
    Complete, integrated real-time streaming Auditory Attention Decoding (AAD) pipeline.
    
    Coordinates:
      1. Continuous causal EEG IIR filtering (1-6 Hz).
      2. Continuous causal audio envelope extraction / decimation.
      3. Synchronized circular FIFO buffering (window length W).
      4. Periodic rolling-window inference (step S).
      5. EMA confidence smoothing and dual-threshold hysteresis steering.
    """
    def __init__(
        self,
        model: nn.Module,
        n_eeg_channels: int = 8,
        fs: float = 64.0,
        audio_fs: int = 16000,
        raw_audio_input: bool = False,
        window_sec: float = 5.0,
        step_sec: float = 0.5,
        engine_mode: str = "torchscript",
        decision_alpha: float = 0.7,
        decision_threshold: float = 0.25,
        n_confirm: int = 2,
        boost_db: float = 6.0,
    ):
        self.fs = fs
        self.audio_fs = audio_fs
        self.raw_audio_input = raw_audio_input
        self.window_samples = int(window_sec * fs)
        self.step_samples = int(step_sec * fs)
        
        # 1. Causal Preprocessing
        self.eeg_filter = StreamingCausalEEGFilter(
            lowcut=1.0, highcut=6.0, fs=fs, order=2, n_channels=n_eeg_channels
        )
        if raw_audio_input:
            self.env_extractor_a = StreamingCausalEnvelopeExtractor(audio_fs=audio_fs, target_fs=int(fs))
            self.env_extractor_b = StreamingCausalEnvelopeExtractor(audio_fs=audio_fs, target_fs=int(fs))
        else:
            self.env_extractor_a = None
            self.env_extractor_b = None
            
        # 2. Synchronized Ring Buffer
        self.ring_buffer = SynchronizedRingBuffer(
            capacity=self.window_samples, n_eeg_channels=n_eeg_channels
        )
        
        # 3. Optimized Inference Engine
        self.engine = StreamingCATCNEngine(
            model=model, mode=engine_mode, device="cpu", dummy_window_samples=self.window_samples
        )
        
        # 4. Temporal Decision Smoother
        self.decision_layer = EMAHysteresisDecisionLayer(
            alpha=decision_alpha, threshold=decision_threshold, n_confirm=n_confirm, boost_db=boost_db
        )
        
        # Internal step bookkeeping
        self.samples_since_last_step = 0
        self.total_processed_samples = 0

    def reset(self):
        """Resets the entire pipeline state."""
        self.eeg_filter.reset()
        if self.env_extractor_a is not None:
            self.env_extractor_a.reset()
            self.env_extractor_b.reset()
        self.ring_buffer.reset()
        self.decision_layer.reset()
        self.samples_since_last_step = 0
        self.total_processed_samples = 0

    def feed_sample_block(
        self,
        eeg_chunk: np.ndarray,
        audio_a_chunk: np.ndarray,
        audio_b_chunk: np.ndarray
    ) -> Optional[Dict[str, Any]]:
        """
        Feeds a chunk of incoming raw data into the pipeline.
        
        If raw_audio_input is True, audio chunks are expected at audio_fs (e.g. 16 kHz).
        Otherwise, audio chunks are expected already at feature rate fs (64 Hz).
        
        Returns:
            telemetry: dict if an inference step occurred, or None if accumulating samples.
        """
        # 1. Filter EEG
        filtered_eeg = self.eeg_filter.process_chunk(eeg_chunk)
        
        # 2. Extract or pass audio envelopes
        if self.raw_audio_input:
            env_a = self.env_extractor_a.process_raw_audio_chunk(audio_a_chunk)
            env_b = self.env_extractor_b.process_raw_audio_chunk(audio_b_chunk)
        else:
            env_a = np.asarray(audio_a_chunk, dtype=np.float32).reshape(-1)
            env_b = np.asarray(audio_b_chunk, dtype=np.float32).reshape(-1)
            
        n_samples = filtered_eeg.shape[0]
        # Align lengths if decimation yielded slight rounding mismatch
        min_len = min(n_samples, len(env_a), len(env_b))
        if min_len == 0:
            return None
            
        filtered_eeg = filtered_eeg[:min_len]
        env_a = env_a[:min_len]
        env_b = env_b[:min_len]
        
        # 3. Push to ring buffer
        self.ring_buffer.push(filtered_eeg, env_a, env_b)
        self.samples_since_last_step += min_len
        self.total_processed_samples += min_len
        
        # 4. Check if time to trigger inference step
        if self.samples_since_last_step >= self.step_samples and self.ring_buffer.is_ready():
            self.samples_since_last_step = 0
            return self._run_inference_step()
            
        return None

    def _run_inference_step(self) -> Dict[str, Any]:
        """Runs single-window inference from current ring buffer snapshot."""
        eeg_win, a_win, b_win = self.ring_buffer.snapshot()
        
        # Forward pass through model
        inf_result = self.engine.predict(eeg_win, a_win, b_win)
        raw_delta = inf_result["delta"]
        
        # Update decision state machine
        decision = self.decision_layer.update(raw_delta)
        
        timestamp_sec = self.total_processed_samples / self.fs
        return {
            "timestamp_sec": timestamp_sec,
            "raw_delta": raw_delta,
            "logit_a": inf_result["logit_a"],
            "logit_b": inf_result["logit_b"],
            "compute_ms": inf_result["compute_ms"],
            "smoothed_score": decision["smoothed_score"],
            "attended_stream": decision["attended_stream"],
            "confidence": decision["confidence"],
            "gain_a": decision["gain_a"],
            "gain_b": decision["gain_b"],
            "switched": decision["switched"],
        }
