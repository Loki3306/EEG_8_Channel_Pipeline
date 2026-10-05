import numpy as np
from typing import Tuple, Dict, Any, List, Optional

class AudioSteeringDSP:
    """
    Real-Time Audio Amplification and Unattended Suppression DSP Engine.
    
    Translates discrete and continuous AAD control decisions (state in {A, B, HOLD}, margin s_t)
    into psychoacoustically smooth, click-free acoustic steering.
    
    Key Features:
      1. Confidence-proportional non-linear gain allocation (+G_boost / -G_suppress).
      2. Continuous sample-level IIR slew-rate limiter (tau = 60 ms) preventing clicks.
      3. Binaural equal-power spatial panning (reproducing physical DTU speaker angles).
      4. Soft-knee peak limiter guaranteeing <= -0.5 dBFS ceiling without clipping.
    """
    def __init__(
        self,
        fs: int = 16000,
        max_boost_db: float = 9.0,
        max_suppress_db: float = 18.0,
        tau_ms: float = 60.0,
        pan_angle_deg: float = 30.0,
        ceiling_dbfs: float = -0.5,
        threshold_switch: float = 0.35,
    ):
        self.fs = fs
        self.max_boost_db = max_boost_db
        self.max_suppress_db = max_suppress_db
        self.tau_ms = tau_ms
        self.ceiling_dbfs = ceiling_dbfs
        self.threshold_switch = threshold_switch
        
        # Slew filter coefficient: alpha = exp(-1 / (tau * fs))
        tau_sec = tau_ms / 1000.0
        self.alpha_slew = float(np.exp(-1.0 / (tau_sec * fs)))
        
        # Panning angles (radians): Speaker A at -pan_angle, Speaker B at +pan_angle
        # Equal power panning: theta in [0, pi/2]
        # Speaker A (left): angle_a = 45 - pan_angle/2 -> pan left
        # Speaker B (right): angle_b = 45 + pan_angle/2 -> pan right
        theta_a = np.radians(45.0 - pan_angle_deg * 0.75)
        theta_b = np.radians(45.0 + pan_angle_deg * 0.75)
        
        self.pan_a_left = float(np.cos(theta_a))
        self.pan_a_right = float(np.sin(theta_a))
        self.pan_b_left = float(np.cos(theta_b))
        self.pan_b_right = float(np.sin(theta_b))
        
        # Peak amplitude ceiling
        self.max_peak_amplitude = 10.0 ** (ceiling_dbfs / 20.0)
        
        # State registers
        self.current_gain_linear_a = 1.0
        self.current_gain_linear_b = 1.0

        # Causal Speech Presence Intelligibility EQ (+2.5 dB at 3.0 kHz)
        f0 = 3000.0
        w0 = 2.0 * np.pi * f0 / fs
        A = 10.0 ** (2.5 / 40.0)
        alpha_eq = np.sin(w0) / (2.0 * 1.0)
        b0 = 1.0 + alpha_eq * A
        b1 = -2.0 * np.cos(w0)
        b2 = 1.0 - alpha_eq * A
        a0 = 1.0 + alpha_eq / A
        a1 = -2.0 * np.cos(w0)
        a2 = 1.0 - alpha_eq / A
        self.b_eq = np.array([b0, b1, b2], dtype=np.float64) / a0
        self.a_eq = np.array([a0, a1, a2], dtype=np.float64) / a0
        self.zi_eq_a = np.zeros(2, dtype=np.float64)
        self.zi_eq_b = np.zeros(2, dtype=np.float64)
        
    def reset(self):
        """Resets smoothed gain registers to neutral unity (0 dB)."""
        self.current_gain_linear_a = 1.0
        self.current_gain_linear_b = 1.0
        self.zi_eq_a.fill(0)
        self.zi_eq_b.fill(0)

    def compute_target_gains_db(self, state: str, margin: float) -> Tuple[float, float]:
        """
        Maps AAD decision state ('A', 'B', 'HOLD', 'LOCKED_A', 'LOCKED_B') and confidence margin s_t
        to target gains in decibels for Stream A and Stream B.
        
        Features:
          - High-confidence ceiling: +9.0 dB boost / -18.0 dB suppression (+27 dB SNR improvement).
          - Firm baseline suppression floor: sustains at least -12 dB suppression / +6 dB boost
            while locked, completely preventing audio from fluttering back to unassisted mixture.
        """
        if state in ["A", "LOCKED_A"]:
            conf = float(np.clip(abs(margin) / (self.threshold_switch + 1e-8), 0.0, 1.0))
            eff_conf = 0.65 + 0.35 * conf
            g_a_db = +self.max_boost_db * eff_conf
            g_b_db = -self.max_suppress_db * eff_conf
        elif state in ["B", "LOCKED_B"]:
            conf = float(np.clip(abs(margin) / (self.threshold_switch + 1e-8), 0.0, 1.0))
            eff_conf = 0.65 + 0.35 * conf
            g_a_db = -self.max_suppress_db * eff_conf
            g_b_db = +self.max_boost_db * eff_conf
        else: # "HOLD", "NEUTRAL_HOLD" or neutral pass-through
            g_a_db = 0.0
            g_b_db = 0.0
            
        return g_a_db, g_b_db

    def process_block(
        self,
        audio_a: np.ndarray,
        audio_b: np.ndarray,
        target_gain_a_db: float,
        target_gain_b_db: float,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Processes an audio block with continuous sample-level slew-rate limiting,
        binaural stereo panning, and soft-knee peak limiting.
        
        audio_a, audio_b: 1D numpy arrays of length N
        Returns:
          stereo_out: [2, N] float32 array (left, right)
          g_a_traj: [N] float32 array of smoothed linear gains
          g_b_traj: [N] float32 array of smoothed linear gains
        """
        N = len(audio_a)
        assert len(audio_b) == N, f"Audio length mismatch: {N} vs {len(audio_b)}"
        
        target_lin_a = 10.0 ** (target_gain_a_db / 20.0)
        target_lin_b = 10.0 ** (target_gain_b_db / 20.0)
        
        g_a_traj = np.zeros(N, dtype=np.float32)
        g_b_traj = np.zeros(N, dtype=np.float32)
        
        curr_a = self.current_gain_linear_a
        curr_b = self.current_gain_linear_b
        alpha = self.alpha_slew
        one_minus_alpha = 1.0 - alpha
        
        # Fast vectorizable or loop slew-rate smoothing
        for n in range(N):
            curr_a = alpha * curr_a + one_minus_alpha * target_lin_a
            curr_b = alpha * curr_b + one_minus_alpha * target_lin_b
            g_a_traj[n] = curr_a
            g_b_traj[n] = curr_b
            
        self.current_gain_linear_a = curr_a
        self.current_gain_linear_b = curr_b
        
        # Apply smoothed gains to speech streams
        steered_a = audio_a * g_a_traj
        steered_b = audio_b * g_b_traj

        # Apply causal presence EQ (+2.5 dB @ 3 kHz) to the boosted stream for vocal clarity
        if curr_a > 1.2:
            from scipy import signal
            steered_a, self.zi_eq_a = signal.lfilter(self.b_eq, self.a_eq, steered_a, zi=self.zi_eq_a)
        if curr_b > 1.2:
            from scipy import signal
            steered_b, self.zi_eq_b = signal.lfilter(self.b_eq, self.a_eq, steered_b, zi=self.zi_eq_b)
        
        # Binaural spatial panner
        left = self.pan_a_left * steered_a + self.pan_b_left * steered_b
        right = self.pan_a_right * steered_a + self.pan_b_right * steered_b
        stereo_out = np.stack([left, right], axis=0).astype(np.float32)
        
        # Soft-knee peak limiter using smooth saturation
        peak = np.max(np.abs(stereo_out))
        if peak > self.max_peak_amplitude:
            stereo_out = np.tanh(stereo_out / self.max_peak_amplitude) * self.max_peak_amplitude
            
        return stereo_out, g_a_traj, g_b_traj

    def process_frame(
        self,
        audio_a: np.ndarray,
        audio_b: np.ndarray,
        decision: str = "HOLD",
        margin: float = 0.0
    ) -> np.ndarray:
        """
        Convenience wrapper to process an audio frame given a discrete decision ('A', 'B', 'HOLD')
        and margin. Returns stereo audio array of shape [2, N].
        """
        state = "A" if "A" in decision else ("B" if "B" in decision else "HOLD")
        g_a_db, g_b_db = self.compute_target_gains_db(state, margin)
        stereo_out, _, _ = self.process_block(audio_a, audio_b, g_a_db, g_b_db)
        return stereo_out

    def render_full_trial(
        self,
        audio_a: np.ndarray,
        audio_b: np.ndarray,
        control_timestamps_sec: np.ndarray,
        control_states: List[str],
        control_margins: np.ndarray,
        ground_truth_attended: str = "A"
    ) -> Dict[str, Any]:
        """
        Renders a full trial (~60 seconds) into synchronized audio streams and gain trajectories.
        
        Returns a dictionary containing:
          - 'steered_binaural': [2, N] float32 (the enhanced hearing aid output)
          - 'raw_mixture': [2, N] float32 (unsteered 50/50 mixture)
          - 'clean_attended_reference': [2, N] float32 (isolated target speech)
          - 'gain_trajectory_a': [N] linear gain
          - 'gain_trajectory_b': [N] linear gain
          - 'gain_db_a': [N] dB gain
          - 'gain_db_b': [N] dB gain
        """
        self.reset()
        total_samples = min(len(audio_a), len(audio_b))
        audio_a = audio_a[:total_samples]
        audio_b = audio_b[:total_samples]
        
        steered_chunks = []
        g_a_chunks = []
        g_b_chunks = []
        
        # Build block-wise schedule corresponding to control update intervals
        # Control updates typically arrive every step_sec (e.g. 0.5s)
        n_controls = len(control_timestamps_sec)
        
        curr_sample = 0
        for i in range(n_controls):
            next_sample = int(control_timestamps_sec[i] * self.fs)
            if i == n_controls - 1:
                next_sample = total_samples
            next_sample = min(next_sample, total_samples)
            
            block_len = next_sample - curr_sample
            if block_len <= 0:
                continue
                
            state = control_states[i]
            margin = control_margins[i]
            g_a_db, g_b_db = self.compute_target_gains_db(state, margin)
            
            chunk_a = audio_a[curr_sample:next_sample]
            chunk_b = audio_b[curr_sample:next_sample]
            
            st_out, ga, gb = self.process_block(chunk_a, chunk_b, g_a_db, g_b_db)
            steered_chunks.append(st_out)
            g_a_chunks.append(ga)
            g_b_chunks.append(gb)
            
            curr_sample = next_sample
            
        if curr_sample < total_samples:
            # Remaining tail processed with last gain
            chunk_a = audio_a[curr_sample:total_samples]
            chunk_b = audio_b[curr_sample:total_samples]
            st_out, ga, gb = self.process_block(chunk_a, chunk_b, g_a_db, g_b_db)
            steered_chunks.append(st_out)
            g_a_chunks.append(ga)
            g_b_chunks.append(gb)
            
        steered_binaural = np.concatenate(steered_chunks, axis=1)
        g_a_full = np.concatenate(g_a_chunks, axis=0)
        g_b_full = np.concatenate(g_b_chunks, axis=0)
        
        # Unsteered Raw Mixture (0 dB on both)
        mix_left = self.pan_a_left * audio_a + self.pan_b_left * audio_b
        mix_right = self.pan_a_right * audio_a + self.pan_b_right * audio_b
        raw_mixture = np.stack([mix_left, mix_right], axis=0).astype(np.float32)
        peak_mix = np.max(np.abs(raw_mixture))
        if peak_mix > self.max_peak_amplitude:
            raw_mixture = raw_mixture * (self.max_peak_amplitude / (peak_mix + 1e-8))
            
        # Clean Attended Reference
        if ground_truth_attended == "A":
            ref_l = self.pan_a_left * audio_a
            ref_r = self.pan_a_right * audio_a
        else:
            ref_l = self.pan_b_left * audio_b
            ref_r = self.pan_b_right * audio_b
        clean_ref = np.stack([ref_l, ref_r], axis=0).astype(np.float32)
        
        return {
            "steered_binaural": steered_binaural,
            "raw_mixture": raw_mixture,
            "clean_attended_reference": clean_ref,
            "gain_trajectory_a": g_a_full,
            "gain_trajectory_b": g_b_full,
            "gain_db_a": 20.0 * np.log10(np.maximum(g_a_full, 1e-6)),
            "gain_db_b": 20.0 * np.log10(np.maximum(g_b_full, 1e-6)),
            "fs": self.fs
        }
