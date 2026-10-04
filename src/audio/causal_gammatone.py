"""
Real-Time Causal Gammatone Auditory Filterbank & Envelope Extractor.

Processes raw continuous audio streams in hardware chunks (e.g., 31.25 ms),
computes 28 ERB-spaced Gammatone auditory subband filters causally,
applies power-law compression (|x|^0.3), smooths with a causal 8 Hz lowpass filter,
and statefully decimates down to the 64 Hz feature rate matching CATCN_AudioEncoder.
"""

from typing import List, Optional, Tuple, Union
import numpy as np
from scipy import signal
import math


def erb_space(low_freq: float = 50.0, high_freq: float = 7500.0, num_bands: int = 28) -> np.ndarray:
    """
    Computes center frequencies along Glasberg & Moore Equivalent Rectangular Bandwidth (ERB) scale.
    
    Parameters:
        low_freq: Lowest center frequency in Hz (default: 50.0 Hz).
        high_freq: Highest center frequency in Hz (default: 7500.0 Hz).
        num_bands: Number of auditory subbands (default: 28).
        
    Returns:
        cf: 1D numpy array of center frequencies in Hz.
    """
    erb_low = 21.4 * np.log10(4.37 * low_freq / 1000.0 + 1.0)
    erb_high = 21.4 * np.log10(4.37 * high_freq / 1000.0 + 1.0)
    erb_points = np.linspace(erb_low, erb_high, num_bands)
    cf = (10.0 ** (erb_points / 21.4) - 1.0) / 4.37 * 1000.0
    return cf


class StreamingCausalAudioGammatoneExtractor:
    """
    100% Causal Streaming Gammatone Envelope Processor.
    
    Processing stages per incoming audio chunk:
      1. Stereo/multi-channel downmix to Mono.
      2. Causal intermediate resampling (if input fs > 16 kHz) via 8th-order anti-aliasing IIR.
      3. 28-band Gammatone auditory IIR filterbank (cascaded SOS) with persistent state registers.
      4. Power-law compression (|x|^power_exponent, default 0.3 matching DTU baseline).
      5. 8 Hz causal Butterworth lowpass envelope smoothing (SOS).
      6. Across-band averaging to produce a 1-channel broadband envelope.
      7. Chunk-invariant fractional phase-accumulator decimation down to 64 Hz.
    """
    def __init__(
        self,
        audio_fs: float = 16000.0,
        target_fs: float = 64.0,
        num_bands: int = 28,
        low_freq: float = 50.0,
        high_freq: float = 7500.0,
        power_exponent: float = 0.3,
        internal_fs: float = 16000.0
    ):
        self.input_fs = float(audio_fs)
        self.target_fs = float(target_fs)
        self.num_bands = num_bands
        self.power_exponent = float(power_exponent)
        
        # Intermediate processing rate:
        # If input fs is high (e.g. 44.1 kHz or 48 kHz), we downsample causally to 16 kHz
        # to maximize DSP execution speed while preserving full auditory bandwidth up to 8 kHz.
        if self.input_fs > 20000.0:
            self.proc_fs = internal_fs
            self.needs_resample = True
            self.input_step = self.proc_fs / self.input_fs
            # 8th-order anti-aliasing lowpass at 0.45 * proc_fs (7.2 kHz)
            fc_aa = min(0.45 * self.proc_fs, 0.45 * self.input_fs)
            self.aa_sos = signal.butter(4, fc_aa, btype='low', fs=self.input_fs, output='sos')
            self.aa_zi = np.zeros((self.aa_sos.shape[0], 2), dtype=np.float64)
            self.input_phase = 0.0
        else:
            self.proc_fs = self.input_fs
            self.needs_resample = False
            self.input_step = 1.0
            
        # Design 28 Gammatone Auditory Filters
        effective_high = min(high_freq, self.proc_fs / 2.0 - 100.0)
        self.cfs = erb_space(low_freq, effective_high, num_bands)
        
        self.band_sos: List[np.ndarray] = []
        self.band_zi: List[np.ndarray] = []
        for cf in self.cfs:
            try:
                b, a = signal.gammatone(cf, 'iir', fs=self.proc_fs)
                sos = signal.tf2sos(b, a)
            except Exception:
                # Fallback 2nd order Butterworth bandpass around cf
                bw = 24.7 * (4.37 * cf / 1000.0 + 1.0)
                f_low = max(20.0, cf - bw / 2.0)
                f_high = min(self.proc_fs / 2.0 - 10.0, cf + bw / 2.0)
                sos = signal.butter(2, [f_low, f_high], btype='bandpass', fs=self.proc_fs, output='sos')
                
            self.band_sos.append(sos)
            self.band_zi.append(np.zeros((sos.shape[0], 2), dtype=np.float64))
            
        # Envelope extraction smoothing filter: 8 Hz 2nd-order Butterworth lowpass at proc_fs
        self.lp_sos = signal.butter(2, 8.0, btype='low', fs=self.proc_fs, output='sos')
        self.lp_zi = [np.zeros((self.lp_sos.shape[0], 2), dtype=np.float64) for _ in range(num_bands)]
        
        # Fractional phase accumulator for exact decimation down to target_fs (64 Hz)
        self.decim_step = self.target_fs / self.proc_fs
        self.decim_phase = 0.0

    def reset(self):
        """Clears all internal filter state registers and phase accumulators."""
        if self.needs_resample:
            self.aa_zi.fill(0.0)
            self.input_phase = 0.0
            
        for i in range(self.num_bands):
            self.band_zi[i].fill(0.0)
            self.lp_zi[i].fill(0.0)
            
        self.decim_phase = 0.0

    def process_audio_chunk(self, audio_chunk: np.ndarray) -> np.ndarray:
        """
        Processes a raw continuous audio chunk.
        
        Parameters:
            audio_chunk: 1D or 2D numpy array of audio samples at input_fs.
            
        Returns:
            envelopes: 1D array of broadband envelope values at 64 Hz.
                       May be length 0 if incoming chunk is shorter than one 64 Hz period (~15.6 ms).
        """
        chunk = np.asarray(audio_chunk, dtype=np.float64)
        if chunk.ndim > 1:
            # Average multi-channel / stereo down to mono
            chunk = np.mean(chunk, axis=-1)
        chunk = chunk.reshape(-1)
        
        if len(chunk) == 0:
            return np.empty((0,), dtype=np.float32)
            
        # 1. Causal intermediate anti-aliasing & resampling if needed
        if self.needs_resample:
            # Apply anti-aliasing lowpass
            chunk_filtered, self.aa_zi = signal.sosfilt(self.aa_sos, chunk, zi=self.aa_zi)
            # Fractional decimation to proc_fs
            k_max = int(np.floor(self.input_phase + len(chunk_filtered) * self.input_step))
            if k_max > 0:
                k_arr = np.arange(1, k_max + 1)
                indices = (k_arr - self.input_phase) / self.input_step
                int_idx = np.clip(np.round(indices).astype(int), 0, len(chunk_filtered) - 1)
                chunk_proc = chunk_filtered[int_idx]
                self.input_phase = (self.input_phase + len(chunk_filtered) * self.input_step) - k_max
            else:
                self.input_phase += len(chunk_filtered) * self.input_step
                return np.empty((0,), dtype=np.float32)
        else:
            chunk_proc = chunk
            
        if len(chunk_proc) == 0:
            return np.empty((0,), dtype=np.float32)
            
        # 2. Filterbank processing across 28 ERB subbands
        band_envelopes = []
        for i in range(self.num_bands):
            # A. Causal Gammatone filter
            filtered, self.band_zi[i] = signal.sosfilt(self.band_sos[i], chunk_proc, zi=self.band_zi[i])
            # B. Power-law compression
            compressed = np.abs(filtered) ** self.power_exponent
            # C. Causal 8 Hz lowpass smoothing
            env_smoothed, self.lp_zi[i] = signal.sosfilt(self.lp_sos, compressed, zi=self.lp_zi[i])
            band_envelopes.append(env_smoothed)
            
        # 3. Across-band averaging -> 1D broadband envelope
        broadband = np.mean(band_envelopes, axis=0) # [len(chunk_proc)]
        
        # 4. Fractional phase-accumulator decimation down to target_fs (64 Hz)
        n_proc = len(broadband)
        out_k_max = int(np.floor(self.decim_phase + n_proc * self.decim_step))
        if out_k_max > 0:
            out_k_arr = np.arange(1, out_k_max + 1)
            out_indices = (out_k_arr - self.decim_phase) / self.decim_step
            out_int_idx = np.clip(np.round(out_indices).astype(int), 0, n_proc - 1)
            decimated_env = broadband[out_int_idx]
            self.decim_phase = (self.decim_phase + n_proc * self.decim_step) - out_k_max
        else:
            decimated_env = np.empty((0,), dtype=np.float64)
            self.decim_phase += n_proc * self.decim_step
            
        return decimated_env.astype(np.float32)
