import numpy as np
from scipy import signal
import math

def erb_space(low_freq: float, high_freq: float, num_bands: int) -> np.ndarray:
    """Computes center frequencies according to Glasberg & Moore Equivalent Rectangular Bandwidth (ERB)."""
    erb_low = 21.4 * np.log10(4.37 * low_freq / 1000.0 + 1.0)
    erb_high = 21.4 * np.log10(4.37 * high_freq / 1000.0 + 1.0)
    erb_points = np.linspace(erb_low, erb_high, num_bands)
    cf = (10.0 ** (erb_points / 21.4) - 1.0) / 4.37 * 1000.0
    return cf

class StreamingCausalEnvelopeExtractor:
    """
    Causal, low-latency streaming audio envelope extractor.
    
    Processing flow:
      Raw Audio Chunks (e.g. 16 kHz)
        ↓
      ERB Subband IIR Filterbank (e.g. 8 or 28 bands, Hohmann/Slaney gammatone approximation)
        ↓
      Power Compression: y = |x|^0.6
        ↓
      Causal Low-pass Filter (8 Hz Butterworth SOS with persistent state)
        ↓
      Decimation down to 64 Hz feature rate
        ↓
      Broadband Envelope: Mean across subbands
    """
    def __init__(self, audio_fs: int = 16000, target_fs: int = 64, num_bands: int = 16,
                 low_freq: float = 100.0, high_freq: float = 7500.0):
        self.audio_fs = audio_fs
        self.target_fs = target_fs
        self.num_bands = num_bands
        self.decim_factor = audio_fs // target_fs
        
        # Center frequencies
        self.cfs = erb_space(low_freq, min(high_freq, audio_fs / 2.0 - 100.0), num_bands)
        
        # Design 2nd-order bandpass filters around each center frequency as ERB subband bank
        self.band_sos = []
        self.band_zi = []
        for cf in self.cfs:
            bw = 24.7 * (4.37 * cf / 1000.0 + 1.0)
            f_low = max(20.0, cf - bw / 2.0)
            f_high = min(audio_fs / 2.0 - 10.0, cf + bw / 2.0)
            sos_b = signal.butter(2, [f_low, f_high], btype='bandpass', fs=audio_fs, output='sos')
            self.band_sos.append(sos_b)
            self.band_zi.append(np.zeros((sos_b.shape[0], 2), dtype=np.float64))
            
        # Causal Low-pass smoothing filter (8 Hz Butterworth at audio_fs)
        self.lp_sos = signal.butter(2, 8.0, btype='low', fs=audio_fs, output='sos')
        self.lp_zi = [np.zeros((self.lp_sos.shape[0], 2), dtype=np.float64) for _ in range(num_bands)]
        
        # Residual sample buffer for exact integer decimation
        self.decim_counter = 0

    def reset(self):
        """Resets all internal filter memories."""
        for i in range(self.num_bands):
            self.band_zi[i].fill(0.0)
            self.lp_zi[i].fill(0.0)
        self.decim_counter = 0

    def process_raw_audio_chunk(self, audio_chunk: np.ndarray) -> np.ndarray:
        """
        Processes a raw 1D audio chunk (sampled at audio_fs).
        
        Returns:
            downsampled_env: 1D np.ndarray of envelopes at target_fs (64 Hz).
            May be length 0 if incoming chunk is shorter than remaining decimation interval.
        """
        audio_chunk = np.asarray(audio_chunk, dtype=np.float64).reshape(-1)
        if len(audio_chunk) == 0:
            return np.empty((0,), dtype=np.float32)
            
        band_envelopes = []
        for i in range(self.num_bands):
            # 1. Bandpass filter
            filtered, self.band_zi[i] = signal.sosfilt(self.band_sos[i], audio_chunk, zi=self.band_zi[i])
            # 2. Power compression
            compressed = np.abs(filtered) ** 0.6
            # 3. Causal lowpass smoothing
            env, self.lp_zi[i] = signal.sosfilt(self.lp_sos, compressed, zi=self.lp_zi[i])
            band_envelopes.append(env)
            
        # Broadband envelope: average across bands
        broadband = np.mean(band_envelopes, axis=0) # [len(audio_chunk)]
        
        # Stateful decimation down to target_fs
        indices = np.arange(self.decim_factor - 1 - self.decim_counter, len(broadband), self.decim_factor)
        if len(indices) > 0:
            decimated = broadband[indices]
            self.decim_counter = (self.decim_counter + len(audio_chunk)) % self.decim_factor
        else:
            decimated = np.empty((0,), dtype=np.float64)
            self.decim_counter = (self.decim_counter + len(audio_chunk)) % self.decim_factor
            
        return decimated.astype(np.float32)
