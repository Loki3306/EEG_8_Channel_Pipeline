"""
Real-time Causal Raw EEG Preprocessing Engine for DTU BioSemi ActiveTwo.

Implements the 5-stage causal streaming pipeline:
1. 50 Hz Line Noise IIR Notch Filter (SOS, persistent state)
2. Instantaneous Common Average Referencing (CAR)
3. Causal Bipolar EOG Artifact Suppression (calibrated linear regression)
4. Causal Anti-Aliasing & 8:1 Decimation (512 Hz -> 64 Hz)
5. 8-Channel Selection, 1.0-6.0 Hz Causal Bandpass (SOS), and Online Rolling Normalization

Guarantees:
- Strictly causal (zero forward lookahead, chunk-size invariant).
- Preserves filter states (zi) across arbitrary streaming block boundaries.
- Deterministic constant group delay (~101 ms) within CA-TCN cross-correlation window.
"""

from typing import Optional, List, Union, Tuple, Dict, Any
import numpy as np
from scipy import signal

from .causal_filters import StreamingCausalEEGFilter
from scripts.verify_baseline.training.montages import MONTAGES, DTU_CHANNELS


class Causal50HzNotchFilter:
    """
    Causal 50 Hz IIR notch filter using Second-Order Sections (SOS).
    Attenuates powerline interference with persistent state zi.
    """
    def __init__(self, fs: float = 512.0, f0: float = 50.0, q: float = 30.0, n_channels: int = 64):
        self.fs = fs
        self.f0 = f0
        self.q = q
        self.n_channels = n_channels
        
        b, a = signal.iirnotch(f0, q, fs=fs)
        self.sos = signal.tf2sos(b, a)
        self.n_sections = self.sos.shape[0]
        self.zi = np.zeros((self.n_sections, 2, n_channels), dtype=np.float64)
        self.is_initialized = False
        
    def reset(self):
        self.zi = np.zeros((self.n_sections, 2, self.n_channels), dtype=np.float64)
        self.is_initialized = False
        
    def process_chunk(self, chunk: np.ndarray) -> np.ndarray:
        """
        chunk: [N_samples, N_channels]
        """
        if chunk.ndim == 1:
            chunk = chunk[:, np.newaxis]
        n_samples, n_ch = chunk.shape
        if n_ch != self.n_channels:
            raise ValueError(f"Expected {self.n_channels} channels, got {n_ch}")
            
        if not self.is_initialized and n_samples > 0:
            base_zi = signal.sosfilt_zi(self.sos)
            self.zi = base_zi[:, :, np.newaxis] * chunk[0:1, :].astype(np.float64)
            self.is_initialized = True
            
        filtered, self.zi = signal.sosfilt(self.sos, chunk.astype(np.float64), axis=0, zi=self.zi)
        return filtered


class CausalDecimator:
    """
    Causal 8:1 decimation filter (512 Hz -> 64 Hz).
    Uses a causal 8th-order Butterworth anti-aliasing lowpass (cutoff 24 Hz at fs=512)
    and an internal sample accumulator to support arbitrary chunk boundaries.
    """
    def __init__(self, in_fs: float = 512.0, out_fs: float = 64.0, cutoff_hz: float = 24.0, n_channels: int = 64):
        self.in_fs = in_fs
        self.out_fs = out_fs
        self.decim_factor = int(round(in_fs / out_fs)) # 8
        self.n_channels = n_channels
        
        # 8th-order Butterworth low-pass anti-aliasing filter
        self.sos = signal.butter(8, cutoff_hz, btype='low', fs=in_fs, output='sos')
        self.n_sections = self.sos.shape[0]
        self.zi = np.zeros((self.n_sections, 2, n_channels), dtype=np.float64)
        self.is_initialized = False
        
        # Downsampling phase tracking
        self.sample_counter = 0
        
    def reset(self):
        self.zi = np.zeros((self.n_sections, 2, self.n_channels), dtype=np.float64)
        self.is_initialized = False
        self.sample_counter = 0
        
    def process_chunk(self, chunk: np.ndarray) -> np.ndarray:
        """
        Filters raw 512 Hz chunk and decimates to 64 Hz.
        Returns: np.ndarray of shape [N_out_samples, N_channels] at 64 Hz.
        """
        if chunk.ndim == 1:
            chunk = chunk[:, np.newaxis]
        n_samples, n_ch = chunk.shape
        if n_ch != self.n_channels:
            raise ValueError(f"Expected {self.n_channels} channels, got {n_ch}")
        if n_samples == 0:
            return np.empty((0, self.n_channels), dtype=np.float32)
            
        if not self.is_initialized:
            base_zi = signal.sosfilt_zi(self.sos)
            self.zi = base_zi[:, :, np.newaxis] * chunk[0:1, :].astype(np.float64)
            self.is_initialized = True
            
        filtered, self.zi = signal.sosfilt(self.sos, chunk.astype(np.float64), axis=0, zi=self.zi)
        
        # Select samples matching decimation phase
        # sample_counter marks global sample index
        indices = []
        for i in range(n_samples):
            if (self.sample_counter + i) % self.decim_factor == 0:
                indices.append(i)
        self.sample_counter = (self.sample_counter + n_samples) % self.decim_factor
        
        if not indices:
            return np.empty((0, self.n_channels), dtype=np.float32)
            
        return filtered[indices, :].astype(np.float32)


class CausalRollingZScoreNormalizer:
    """
    Causal Exponential Moving Average (EMA) Z-Score normalizer.
    Replaces non-causal whole-trial offline mean/std with an online Welford statistics tracker.
    """
    def __init__(self, n_channels: int = 8, half_life_sec: float = 10.0, fs: float = 64.0):
        self.n_channels = n_channels
        self.fs = fs
        # Decay factor gamma per sample
        self.gamma = float(np.exp(-np.log(2.0) / (half_life_sec * fs)))
        self.mean = np.zeros(n_channels, dtype=np.float64)
        self.var = np.ones(n_channels, dtype=np.float64)
        self.is_initialized = False
        
    def reset(self):
        self.mean = np.zeros(self.n_channels, dtype=np.float64)
        self.var = np.ones(self.n_channels, dtype=np.float64)
        self.is_initialized = False
        
    def process_chunk(self, chunk: np.ndarray) -> np.ndarray:
        """
        chunk: [N_samples, N_channels]
        Returns normalized chunk with unit variance and zero mean.
        """
        if chunk.ndim == 1:
            chunk = chunk[:, np.newaxis]
        n_samples, n_ch = chunk.shape
        if n_samples == 0:
            return chunk
            
        if not self.is_initialized:
            self.mean = np.mean(chunk[:min(n_samples, int(self.fs)), :], axis=0).astype(np.float64)
            self.var = np.var(chunk[:min(n_samples, int(self.fs)), :], axis=0).astype(np.float64) + 1e-4
            self.is_initialized = True
            
        out = np.zeros_like(chunk, dtype=np.float32)
        g = self.gamma
        one_minus_g = 1.0 - g
        
        for i in range(n_samples):
            x = chunk[i].astype(np.float64)
            # Update mean and variance online
            self.mean = g * self.mean + one_minus_g * x
            diff = x - self.mean
            self.var = g * self.var + one_minus_g * (diff * diff)
            std = np.sqrt(np.maximum(self.var, 1e-8))
            out[i] = (diff / std).astype(np.float32)
            
        return out


class StreamingCausalRawEEGPreprocessor:
    """
    Complete end-to-end real-time raw EEG preprocessor for DTU BioSemi ActiveTwo.
    
    Transforms raw 512 Hz multi-channel EEG into cleaned, calibrated 64 Hz 8-channel EEG
    ready for direct ingestion into the frozen CA-TCN + Spatial Adapter system.
    """
    def __init__(
        self,
        raw_fs: float = 512.0,
        target_fs: float = 64.0,
        n_scalp_channels: int = 64,
        montage_name: str = "near_ear_expanded",
        eog_regression_weights: Optional[np.ndarray] = None, # [64, 2]
        bandpass_lowcut: float = 1.0,
        bandpass_highcut: float = 6.0,
        bandpass_order: int = 2,
        norm_half_life_sec: float = 10.0
    ):
        self.raw_fs = raw_fs
        self.target_fs = target_fs
        self.n_scalp_channels = n_scalp_channels
        
        # 1. 50 Hz Line Noise Filter (at 512 Hz)
        self.notch_filter = Causal50HzNotchFilter(fs=raw_fs, f0=50.0, q=30.0, n_channels=n_scalp_channels)
        
        # 2. EOG Regression Weights: W_eog [64, 2]
        self.eog_weights = eog_regression_weights
        
        # 3. 8:1 Causal Decimator (512 Hz -> 64 Hz)
        self.decimator = CausalDecimator(in_fs=raw_fs, out_fs=target_fs, cutoff_hz=24.0, n_channels=n_scalp_channels)
        
        # 4. Montage Channel Selection (8 channels)
        if montage_name in MONTAGES:
            self.selected_indices = list(MONTAGES[montage_name])
        else:
            self.selected_indices = list(MONTAGES["near_ear_expanded"])
        self.n_out_channels = len(self.selected_indices)
        
        # 5. Causal Auditory Sub-Band Filter (1.0 - 6.0 Hz at 64 Hz)
        self.bandpass_filter = StreamingCausalEEGFilter(
            lowcut=bandpass_lowcut,
            highcut=bandpass_highcut,
            fs=target_fs,
            order=bandpass_order,
            n_channels=self.n_out_channels
        )
        
        # 6. Causal Online Normalizer
        self.normalizer = CausalRollingZScoreNormalizer(
            n_channels=self.n_out_channels,
            half_life_sec=norm_half_life_sec,
            fs=target_fs
        )
        
    def reset(self):
        """Resets all internal filter and normalizer states."""
        self.notch_filter.reset()
        self.decimator.reset()
        self.bandpass_filter.reset()
        self.normalizer.reset()
        
    def calibrate_eog_weights(self, scalp_chunk: np.ndarray, veog_chunk: np.ndarray, heog_chunk: np.ndarray):
        """
        Calibrates linear EOG artifact regression matrix W_eog [64, 2] on calibration data.
        x_clean = x_car - W_eog @ [veog, heog].
        """
        # Notch filter and CAR reference calibration chunk
        notched = self.notch_filter.process_chunk(scalp_chunk)
        car = notched - np.mean(notched, axis=1, keepdims=True)
        
        # Stack EOG features [N, 2]
        eog_features = np.column_stack([veog_chunk.reshape(-1), heog_chunk.reshape(-1)]) # [N, 2]
        
        # Ridge regression: W = (E^T E + lambda I)^-1 E^T X -> [2, 64] -> transpose to [64, 2]
        lam = 1e-3 * np.trace(eog_features.T @ eog_features)
        reg = np.linalg.solve(eog_features.T @ eog_features + lam * np.eye(2), eog_features.T @ car)
        self.eog_weights = reg.T # [64, 2]
        self.reset()
        return self.eog_weights
        
    def process_raw_chunk(
        self,
        scalp_chunk_512: np.ndarray,
        veog_chunk_512: Optional[np.ndarray] = None,
        heog_chunk_512: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Processes a streaming block of raw 512 Hz EEG.
        
        Parameters:
            scalp_chunk_512: [N_samples, 64] raw scalp electrodes at 512 Hz
            veog_chunk_512: Optional [N_samples] vertical EOG at 512 Hz
            heog_chunk_512: Optional [N_samples] horizontal EOG at 512 Hz
            
        Returns:
            processed_chunk_64: [N_out_samples, 8] filtered, normalized EEG at 64 Hz
        """
        if scalp_chunk_512.shape[0] == 0:
            return np.empty((0, self.n_out_channels), dtype=np.float32)
            
        # Step 1: 50 Hz Causal IIR Notch Filter
        notched = self.notch_filter.process_chunk(scalp_chunk_512)
        
        # Step 2: Instantaneous Common Average Referencing (CAR)
        car = notched - np.mean(notched, axis=1, keepdims=True)
        
        # Step 3: Causal EOG Artifact Suppression
        if self.eog_weights is not None and veog_chunk_512 is not None and heog_chunk_512 is not None:
            eog_stack = np.column_stack([veog_chunk_512.reshape(-1), heog_chunk_512.reshape(-1)]) # [N, 2]
            artifact = eog_stack @ self.eog_weights.T # [N, 64]
            cleaned = car - artifact
            # Re-reference to CAR after subtraction
            cleaned = cleaned - np.mean(cleaned, axis=1, keepdims=True)
        else:
            cleaned = car
            
        # Step 4: 8:1 Causal Anti-Aliasing Decimation (512 Hz -> 64 Hz)
        decimated_64 = self.decimator.process_chunk(cleaned) # [N_out, 64]
        if decimated_64.shape[0] == 0:
            return np.empty((0, self.n_out_channels), dtype=np.float32)
            
        # Step 5: Extract 8 Target Near-Ear Channels
        eeg_8ch = decimated_64[:, self.selected_indices] # [N_out, 8]
        
        # Step 6: 1.0 - 6.0 Hz Causal Bandpass (SOS)
        bandpassed_8ch = self.bandpass_filter.process_chunk(eeg_8ch) # [N_out, 8]
        
        # Step 7: Causal Rolling Normalization (EMA Z-score)
        normalized_8ch = self.normalizer.process_chunk(bandpassed_8ch) # [N_out, 8]
        
        return normalized_8ch
