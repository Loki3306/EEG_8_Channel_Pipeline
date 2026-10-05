"""
Acoustic Room & Cross-Talk Simulator for ReSpeaker 4-Mic Array.

Synthesizes realistic 4-channel microphone recordings from two clean speech sources:
  1. Simulates direct-path acoustic wave propagation delays across the 4 circular microphones.
  2. Injects configurable acoustic cross-talk / spatial interference (SIR in dB).
  3. Injects ambient room background noise (SNR in dB).
  4. Generates standard 4-channel ReSpeaker streams (16 kHz, float32) for offline benchmarking.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from scipy import signal


class ReSpeakerAcousticSimulator:
    """
    Simulates physical acoustic wave propagation onto a circular ReSpeaker 4-Mic array.
    
    Parameters:
        fs: Sampling rate in Hz (default: 16000 Hz).
        radius: Microphone circle radius in meters (default: 0.0325 m = 32.5 mm).
        mic_angles_deg: Microphone azimuth angles (default: [0, 90, 180, 270] degrees).
        theta_a_deg: Azimuth of Speaker A (default: -45.0 degrees, Front-Left).
        theta_b_deg: Azimuth of Speaker B (default: +45.0 degrees, Front-Right).
        speed_of_sound: Speed of sound in m/s (default: 343.0 m/s).
    """
    def __init__(
        self,
        fs: int = 16000,
        radius: float = 0.0325,
        mic_angles_deg: Optional[List[float]] = None,
        theta_a_deg: float = -45.0,
        theta_b_deg: float = 45.0,
        speed_of_sound: float = 343.0,
    ):
        self.fs = int(fs)
        self.radius = float(radius)
        self.speed_of_sound = float(speed_of_sound)
        self.theta_a_deg = float(theta_a_deg)
        self.theta_b_deg = float(theta_b_deg)

        if mic_angles_deg is None:
            self.mic_angles_deg = [0.0, 90.0, 180.0, 270.0]
        else:
            self.mic_angles_deg = [float(a) for a in mic_angles_deg]

        self.num_mics = len(self.mic_angles_deg)
        self.mic_angles_rad = np.radians(self.mic_angles_deg)

        # 2D mic Cartesian positions
        self.mic_pos = np.zeros((self.num_mics, 2), dtype=np.float64)
        for i, phi in enumerate(self.mic_angles_rad):
            self.mic_pos[i, 0] = self.radius * np.cos(phi)
            self.mic_pos[i, 1] = self.radius * np.sin(phi)

    def _fractional_delay(self, audio: np.ndarray, delay_samples: float) -> np.ndarray:
        """
        Applies a high-fidelity fractional sample delay using frequency-domain phase shifting.
        """
        n = len(audio)
        if n == 0 or abs(delay_samples) < 1e-6:
            return audio.copy()

        # FFT -> linear phase shift -> IFFT
        spec = np.fft.rfft(audio)
        freqs = np.fft.rfftfreq(n)
        phase_shift = np.exp(-1j * 2.0 * np.pi * freqs * delay_samples)
        delayed = np.fft.irfft(spec * phase_shift, n=n).astype(np.float32)
        return delayed

    def simulate_4channel_mixture(
        self,
        audio_a: np.ndarray,
        audio_b: np.ndarray,
        crosstalk_sir_db: Optional[float] = None,
        ambient_snr_db: Optional[float] = None,
        seed: int = 42,
    ) -> np.ndarray:
        """
        Renders two clean single-channel audio signals into a realistic 4-channel ReSpeaker recording.
        
        Parameters:
            audio_a: 1D numpy array for Speaker A (attended or candidate 1).
            audio_b: 1D numpy array for Speaker B (competing or candidate 2).
            crosstalk_sir_db: If specified, injects acoustic cross-talk with target Signal-to-Interference Ratio.
                              None = purely geometric free-field propagation.
            ambient_snr_db: If specified, injects diffuse Gaussian ambient room noise with target SNR in dB.
            seed: Random seed for noise generation.
            
        Returns:
            mic_signals: 2D numpy array of shape (4, num_samples) representing raw ReSpeaker mics.
        """
        min_len = min(len(audio_a), len(audio_b))
        sig_a = audio_a[:min_len].astype(np.float32)
        sig_b = audio_b[:min_len].astype(np.float32)

        # Normalize RMS levels
        rms_a = np.sqrt(np.mean(sig_a ** 2) + 1e-9)
        rms_b = np.sqrt(np.mean(sig_b ** 2) + 1e-9)
        sig_a = sig_a / rms_a
        sig_b = sig_b / rms_b

        # Calculate propagation delays (in samples) for each microphone
        # Direction unit vector towards sound source
        rad_a = np.radians(self.theta_a_deg)
        rad_b = np.radians(self.theta_b_deg)
        k_a = np.array([np.cos(rad_a), np.sin(rad_a)])
        k_b = np.array([np.cos(rad_b), np.sin(rad_b)])

        delays_a_sec = -np.dot(self.mic_pos, k_a) / self.speed_of_sound
        delays_b_sec = -np.dot(self.mic_pos, k_b) / self.speed_of_sound

        delays_a_samples = delays_a_sec * self.fs
        delays_b_samples = delays_b_sec * self.fs

        # Render 4 channels
        mic_signals = np.zeros((self.num_mics, min_len), dtype=np.float32)

        # Determine relative cross-talk scale
        if crosstalk_sir_db is not None:
            # Scale Speaker B relative to Speaker A by SIR
            scale_b = 10.0 ** (-crosstalk_sir_db / 20.0)
        else:
            scale_b = 1.0  # Equal power natural acoustic mixture

        for m in range(self.num_mics):
            delayed_a = self._fractional_delay(sig_a, delays_a_samples[m])
            delayed_b = self._fractional_delay(sig_b, delays_b_samples[m])
            mic_signals[m, :] = delayed_a + scale_b * delayed_b

        # Add ambient room noise if requested
        if ambient_snr_db is not None:
            rng = np.random.RandomState(seed)
            sig_power = np.mean(mic_signals ** 2)
            noise_power = sig_power / (10.0 ** (ambient_snr_db / 10.0))
            noise_std = np.sqrt(max(1e-9, noise_power))
            ambient_noise = rng.randn(self.num_mics, min_len).astype(np.float32) * noise_std
            # Spatially correlate noise slightly (diffuse field)
            mic_signals += ambient_noise

        # Global peak normalization to avoid clipping
        peak = np.max(np.abs(mic_signals))
        if peak > 1e-6:
            mic_signals = (mic_signals / peak) * 0.90

        return mic_signals
