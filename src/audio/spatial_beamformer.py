"""
Causal Spatial Acoustic Beamformer for Seeed Studio ReSpeaker 4-Mic Array.

Provides real-time multi-channel spatial beamforming for circular microphone arrays:
  1. Computes physical inter-microphone Time-Difference-of-Arrival (TDOA) based on array geometry.
  2. Dual-beam steering: Left Candidate (-45 deg) with null at Right (+45 deg),
     and Right Candidate (+45 deg) with null at Left (-45 deg).
  3. Causal subband Linearly Constrained Minimum Variance (LCMV) and Delay-and-Sum (DAS) beamforming.
  4. Stateful chunk-by-chunk streaming with zero boundary phase discontinuities.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from scipy import signal


class ReSpeakerSpatialBeamformer:
    """
    Real-Time Causal 4-Channel Spatial Acoustic Beamformer for ReSpeaker Mic Array v2.0.
    
    Array Geometry:
      - 4 omnidirectional digital MEMS microphones on a circular ring.
      - Default radius R = 0.0325 m (32.5 mm, ReSpeaker v2 standard).
      - Mic angles: [0 deg (Right), 90 deg (Front), 180 deg (Left), 270 deg (Back)].
      
    Dual Steering Lobes:
      - Beam A: Steered at theta_a (default: -45 deg / 315 deg, Front-Left), null at theta_b.
      - Beam B: Steered at theta_b (default: +45 deg, Front-Right), null at theta_a.
    """
    def __init__(
        self,
        fs: int = 16000,
        radius: float = 0.0325,
        mic_angles_deg: Optional[List[float]] = None,
        theta_a_deg: float = -45.0,
        theta_b_deg: float = 45.0,
        n_fft: int = 256,
        hop_size: int = 128,
        mode: str = "lcmv_null",  # 'lcmv_null' or 'delay_and_sum'
        speed_of_sound: float = 343.0,
        diagonal_loading: float = 1e-3,
    ):
        self.fs = int(fs)
        self.radius = float(radius)
        self.speed_of_sound = float(speed_of_sound)
        self.theta_a_deg = float(theta_a_deg)
        self.theta_b_deg = float(theta_b_deg)
        self.n_fft = int(n_fft)
        self.hop_size = int(hop_size)
        self.mode = mode.lower()
        self.diagonal_loading = float(diagonal_loading)

        if mic_angles_deg is None:
            # ReSpeaker v2.0 4-mic standard circular arrangement
            self.mic_angles_deg = [0.0, 90.0, 180.0, 270.0]
        else:
            self.mic_angles_deg = [float(a) for a in mic_angles_deg]

        self.num_mics = len(self.mic_angles_deg)
        self.mic_angles_rad = np.radians(self.mic_angles_deg)

        # Microphone Cartesian Coordinates in 2D plane (x = right, y = front)
        # Note: 90 deg is front (+y), 0 deg is right (+x), 180 deg is left (-x)
        self.mic_pos = np.zeros((self.num_mics, 2), dtype=np.float64)
        for i, phi in enumerate(self.mic_angles_rad):
            self.mic_pos[i, 0] = self.radius * np.cos(phi)
            self.mic_pos[i, 1] = self.radius * np.sin(phi)

        # STFT Analysis & Synthesis Windows (Square-root Hann for perfect OLA reconstruction)
        self.win = np.sqrt(np.hanning(self.n_fft).astype(np.float32))
        self.freqs = np.fft.rfftfreq(self.n_fft, d=1.0 / self.fs)
        self.n_bins = len(self.freqs)

        # Precompute Beamformer Frequency-Domain Weights
        self.weights_a, self.weights_b = self._precompute_dual_weights()

        # Streaming Overlap-Add State Buffers
        self.input_overlap_buffer = np.zeros((self.num_mics, self.n_fft - self.hop_size), dtype=np.float32)
        self.out_overlap_buffer_a = np.zeros(self.n_fft - self.hop_size, dtype=np.float32)
        self.out_overlap_buffer_b = np.zeros(self.n_fft - self.hop_size, dtype=np.float32)

    def _compute_steering_vector(self, theta_deg: float) -> np.ndarray:
        """
        Computes the theoretical free-field acoustic steering vector for a plane wave.
        
        Parameters:
            theta_deg: Source azimuth angle in degrees (0 = right, 90 = front, 180 = left, -45 = front-left).
            
        Returns:
            steer_vec: Complex array of shape (n_bins, num_mics)
        """
        theta_rad = np.radians(theta_deg)
        # Direction unit vector towards sound source
        # 90 deg is front (+y), 0 deg is right (+x), 180 deg is left (-x)
        kx = np.cos(theta_rad)
        ky = np.sin(theta_rad)

        # Time delay of arrival at each mic relative to center:
        # tau_m = -(x_m * kx + y_m * ky) / c
        delays = -np.dot(self.mic_pos, np.array([kx, ky])) / self.speed_of_sound  # shape (num_mics,)

        # Complex steering vector per frequency bin: e^(-j * 2 * pi * f * tau)
        # phase: (n_bins, 1) * (1, num_mics) -> (n_bins, num_mics)
        phase = -2.0 * np.pi * self.freqs[:, np.newaxis] * delays[np.newaxis, :]
        steer_vec = np.exp(1j * phase)
        return steer_vec

    def _precompute_dual_weights(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Precomputes spatial filter weight matrices for Beam A and Beam B.
        
        Modes:
          - 'delay_and_sum': Classical phased array Delay-and-Sum.
          - 'lcmv_null': Linearly Constrained Minimum Variance with unity gain at target
            and spatial null constraint at the competing speaker angle.
        """
        steer_a = self._compute_steering_vector(self.theta_a_deg)  # (n_bins, M)
        steer_b = self._compute_steering_vector(self.theta_b_deg)  # (n_bins, M)

        weights_a = np.zeros((self.n_bins, self.num_mics), dtype=np.complex64)
        weights_b = np.zeros((self.n_bins, self.num_mics), dtype=np.complex64)

        if self.mode == "delay_and_sum":
            # w = d / M
            weights_a = (steer_a / self.num_mics).astype(np.complex64)
            weights_b = (steer_b / self.num_mics).astype(np.complex64)
        else:
            # LCMV Dual-Constraint Beamforming:
            # For Beam A: C = [d_a, d_b], g = [1, 0] (pass A, null B)
            # For Beam B: C = [d_b, d_a], g = [1, 0] (pass B, null A)
            eye_m = np.eye(2, dtype=np.complex128) * self.diagonal_loading

            for k in range(self.n_bins):
                da_k = steer_a[k, :, np.newaxis]  # (M, 1)
                db_k = steer_b[k, :, np.newaxis]  # (M, 1)

                # Beam A: Pass A (1), Null B (0)
                C_a = np.hstack([da_k, db_k])  # (M, 2)
                g = np.array([[1.0], [0.0]], dtype=np.complex128)
                # w_a = C * (C^H * C + eps * I)^(-1) * g
                chc_a = np.dot(C_a.conj().T, C_a) + eye_m
                try:
                    inv_chc_a = np.linalg.inv(chc_a)
                    w_a_k = np.dot(np.dot(C_a, inv_chc_a), g)  # (M, 1)
                    weights_a[k, :] = w_a_k.squeeze()
                except np.linalg.LinAlgError:
                    weights_a[k, :] = da_k.squeeze() / self.num_mics

                # Beam B: Pass B (1), Null A (0)
                C_b = np.hstack([db_k, da_k])  # (M, 2)
                chc_b = np.dot(C_b.conj().T, C_b) + eye_m
                try:
                    inv_chc_b = np.linalg.inv(chc_b)
                    w_b_k = np.dot(np.dot(C_b, inv_chc_b), g)  # (M, 1)
                    weights_b[k, :] = w_b_k.squeeze()
                except np.linalg.LinAlgError:
                    weights_b[k, :] = db_k.squeeze() / self.num_mics

        return weights_a, weights_b

    def reset(self):
        """Resets streaming overlap-add buffers to silence."""
        self.input_overlap_buffer.fill(0.0)
        self.out_overlap_buffer_a.fill(0.0)
        self.out_overlap_buffer_b.fill(0.0)

    def process_chunk(self, multi_channel_audio: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Causally processes a streaming chunk of multi-channel audio through dual spatial beamformers.
        
        Parameters:
            multi_channel_audio: 2D numpy array of shape (num_mics, num_samples) or (num_samples, num_mics).
                                 Supports standard 4-channel ReSpeaker streams.
                                 
        Returns:
            beam_a: 1D numpy array of shape (num_samples,) containing Left Beam audio.
            beam_b: 1D numpy array of shape (num_samples,) containing Right Beam audio.
        """
        arr = np.asarray(multi_channel_audio, dtype=np.float32)
        if arr.ndim == 1:
            raise ValueError(f"Spatial beamformer requires multi-channel input (got 1D array of shape {arr.shape})")

        # Ensure shape is (num_mics, num_samples)
        if arr.shape[0] != self.num_mics and arr.shape[1] == self.num_mics:
            arr = arr.T
        elif arr.shape[0] > self.num_mics:
            # Handle 6-channel ReSpeaker USB streams (channels 0..3 are raw mics)
            arr = arr[:self.num_mics, :]

        num_mics, num_samples = arr.shape
        if num_mics != self.num_mics:
            raise ValueError(f"Expected {self.num_mics} microphone channels, received {num_mics}")

        if num_samples == 0:
            return np.zeros(0, dtype=np.float32), np.zeros(0, dtype=np.float32)

        # Prepend input overlap history
        full_input = np.hstack([self.input_overlap_buffer, arr])
        total_len = full_input.shape[1]

        # Calculate number of STFT frames we can extract
        overlap_len = self.n_fft - self.hop_size
        num_frames = (total_len - self.n_fft) // self.hop_size + 1

        if num_frames <= 0:
            # Not enough samples for a complete FFT frame; buffer and return silence
            self.input_overlap_buffer = full_input[:, -overlap_len:]
            return np.zeros(num_samples, dtype=np.float32), np.zeros(num_samples, dtype=np.float32)

        out_a_frames = []
        out_b_frames = []

        # Process frame by frame
        for frame_idx in range(num_frames):
            start = frame_idx * self.hop_size
            end = start + self.n_fft
            chunk_mics = full_input[:, start:end] * self.win[np.newaxis, :]  # (M, N)

            # RFFT across time dimension -> (M, n_bins)
            spec_mics = np.fft.rfft(chunk_mics, n=self.n_fft, axis=1)  # (M, n_bins)

            # Spatial filtering: Y = sum_m (w_m^* * X_m)
            # weights shape is (n_bins, M). Transpose spec_mics to (n_bins, M)
            spec_t = spec_mics.T  # (n_bins, M)

            spec_out_a = np.sum(self.weights_a.conj() * spec_t, axis=1)  # (n_bins,)
            spec_out_b = np.sum(self.weights_b.conj() * spec_t, axis=1)  # (n_bins,)

            # Inverse RFFT + synthesis window
            time_out_a = np.fft.irfft(spec_out_a, n=self.n_fft).astype(np.float32) * self.win
            time_out_b = np.fft.irfft(spec_out_b, n=self.n_fft).astype(np.float32) * self.win

            out_a_frames.append(time_out_a)
            out_b_frames.append(time_out_b)

        # Update input overlap buffer for next chunk
        consumed_samples = num_frames * self.hop_size
        remaining_samples = total_len - consumed_samples
        if remaining_samples >= overlap_len:
            self.input_overlap_buffer = full_input[:, -overlap_len:]
        else:
            self.input_overlap_buffer.fill(0.0)
            self.input_overlap_buffer[:, -remaining_samples:] = full_input[:, -remaining_samples:]

        # Overlap-Add Reconstruction
        synth_len = (num_frames - 1) * self.hop_size + self.n_fft
        recon_a = np.zeros(synth_len, dtype=np.float32)
        recon_b = np.zeros(synth_len, dtype=np.float32)

        for i in range(num_frames):
            idx = i * self.hop_size
            recon_a[idx:idx + self.n_fft] += out_a_frames[i]
            recon_b[idx:idx + self.n_fft] += out_b_frames[i]

        # Add previously carried output overlap
        recon_a[:overlap_len] += self.out_overlap_buffer_a
        recon_b[:overlap_len] += self.out_overlap_buffer_b

        # Extract output corresponding to incoming chunk size
        if synth_len >= num_samples:
            out_chunk_a = recon_a[:num_samples].copy()
            out_chunk_b = recon_b[:num_samples].copy()
            # Store remaining tail in output overlap buffer
            tail_a = recon_a[num_samples:]
            tail_b = recon_b[num_samples:]
            self.out_overlap_buffer_a.fill(0.0)
            self.out_overlap_buffer_b.fill(0.0)
            copy_len = min(overlap_len, len(tail_a))
            self.out_overlap_buffer_a[:copy_len] = tail_a[:copy_len]
            self.out_overlap_buffer_b[:copy_len] = tail_b[:copy_len]
        else:
            out_chunk_a = np.pad(recon_a, (0, num_samples - synth_len))
            out_chunk_b = np.pad(recon_b, (0, num_samples - synth_len))
            self.out_overlap_buffer_a.fill(0.0)
            self.out_overlap_buffer_b.fill(0.0)

        return out_chunk_a, out_chunk_b

    def process_continuous_file(self, multi_channel_audio: np.ndarray, chunk_size: int = 500) -> Tuple[np.ndarray, np.ndarray]:
        """
        Utility for processing full recordings chunk-by-chunk to simulate streaming.
        
        Parameters:
            multi_channel_audio: Shape (num_mics, total_samples) or (total_samples, num_mics)
            chunk_size: Samples per chunk (default: 500 samples @ 16 kHz = 31.25 ms)
            
        Returns:
            beam_a, beam_b: Reconstructed 1D audio streams
        """
        arr = np.asarray(multi_channel_audio, dtype=np.float32)
        if arr.shape[0] != self.num_mics and arr.shape[1] == self.num_mics:
            arr = arr.T
        elif arr.shape[0] > self.num_mics:
            arr = arr[:self.num_mics, :]

        total_samples = arr.shape[1]
        out_a = np.zeros(total_samples, dtype=np.float32)
        out_b = np.zeros(total_samples, dtype=np.float32)

        self.reset()
        for idx in range(0, total_samples, chunk_size):
            chunk = arr[:, idx:idx + chunk_size]
            ba, bb = self.process_chunk(chunk)
            valid_len = min(len(ba), total_samples - idx)
            out_a[idx:idx + valid_len] = ba[:valid_len]
            out_b[idx:idx + valid_len] = bb[:valid_len]

        return out_a, out_b
