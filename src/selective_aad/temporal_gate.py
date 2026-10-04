import math
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

class SignalQualityMonitor:
    """
    Real-time, compute-negligible signal quality monitor.
    Detects electrode disconnects, flatlines, extreme movement bursts,
    and clipping before neural margins reach the steering controller.
    """
    def __init__(
        self,
        min_variance: float = 1e-4,
        max_amplitude_z: float = 8.0,
        flatline_samples_threshold: int = 10
    ):
        self.min_variance = min_variance
        self.max_amplitude_z = max_amplitude_z
        self.flatline_samples_threshold = flatline_samples_threshold

    def check_eeg_window(self, eeg_window: np.ndarray) -> Dict[str, Any]:
        """
        eeg_window: [channels, samples] or [samples, channels]
        Returns artifact detection dict with is_valid boolean flag.
        """
        arr = np.asarray(eeg_window, dtype=np.float32)
        if arr.ndim == 1:
            arr = np.expand_dims(arr, 0)
        elif arr.shape[0] > arr.shape[1]:
            arr = arr.T  # Ensure [C, T]
            
        # 1. Check for NaNs or Infs
        if not np.all(np.isfinite(arr)):
            return {"is_valid": False, "reason": "non_finite_values", "severity": 1.0}
            
        # 2. Check for channel flatlines (disconnected electrode)
        ch_vars = np.var(arr, axis=-1)
        if np.any(ch_vars < self.min_variance):
            return {"is_valid": False, "reason": "flatline_or_zero_variance", "severity": 0.8}
            
        # 3. Check for severe high-amplitude artifacts (movement/swallow bursts)
        ch_means = np.mean(arr, axis=-1, keepdims=True)
        ch_stds = np.std(arr, axis=-1, keepdims=True) + 1e-8
        z_scores = np.abs((arr - ch_means) / ch_stds)
        if np.max(z_scores) > self.max_amplitude_z:
            return {"is_valid": False, "reason": "extreme_amplitude_burst", "severity": 0.6}
            
        return {"is_valid": True, "reason": "clean", "severity": 0.0}


class StickyHysteresisGate:
    """
    Sticky State-Retention Auditory Gating Controller.
    
    Product Policy:
      1. Hard to Switch, Easy to Maintain:
         Switching speaker focus requires sustained affirmative counter-evidence (|S_t| >= θ_switch for N steps).
      2. Sticky Retention During Pauses:
         When speech pauses or neural confidence drops into the ambiguous deadband (|S_t| < θ_maintain),
         the hearing aid RETAINS the currently boosted speaker rather than neutralizing audio.
      3. Safety Fallback:
         Audio drops to Neutral ONLY on bad electrode contact or prolonged deadband timeout (> 10s).
      4. Click-free Audio Gain Steering:
         Continuous exponential gain slew rate.
    """
    def __init__(
        self,
        alpha: float = 0.85,
        threshold_switch: float = 0.35,
        threshold_maintain: float = 0.15,
        n_confirm: int = 2,
        deadband_timeout_steps: int = 20,  # 20 steps at 0.5s = 10.0s of continuous absence
        boost_db: float = 6.0,
        temperature: float = 1.0
    ):
        if threshold_switch < threshold_maintain:
            raise ValueError("threshold_switch must be >= threshold_maintain")
            
        self.alpha = float(alpha)
        self.threshold_switch = float(threshold_switch)
        self.threshold_maintain = float(threshold_maintain)
        self.n_confirm = int(n_confirm)
        self.deadband_timeout_steps = int(deadband_timeout_steps)
        self.temperature = max(1e-4, float(temperature))
        self.boost_db = float(boost_db)
        self.boost_lin = 10.0 ** (self.boost_db / 20.0)
        
        # State variables
        self.smoothed_margin = 0.0
        self.initialized = False
        self.current_state = "NEUTRAL_HOLD"  # 'LOCKED_A', 'LOCKED_B', 'NEUTRAL_HOLD'
        self.pending_switch_target = None
        self.confirm_counter = 0
        self.deadband_counter = 0
        self.gain_a = 0.5
        self.gain_b = 0.5

    def reset(self):
        self.smoothed_margin = 0.0
        self.initialized = False
        self.current_state = "NEUTRAL_HOLD"
        self.pending_switch_target = None
        self.confirm_counter = 0
        self.deadband_counter = 0
        self.gain_a = 0.5
        self.gain_b = 0.5

    def update(self, raw_margin: float, is_artifact: bool = False) -> Dict[str, Any]:
        """
        Processes single streaming control step (e.g. at 2 Hz).
        """
        m = float(raw_margin)
        
        # 1. Causal Exponential Moving Average
        if not self.initialized:
            self.smoothed_margin = m
            self.initialized = True
        else:
            self.smoothed_margin = self.alpha * self.smoothed_margin + (1.0 - self.alpha) * m
            
        s = self.smoothed_margin
        
        # 2. Temperature Calibrated Confidence
        scaled = np.clip(s / self.temperature, -30.0, 30.0)
        prob_a = float(1.0 / (1.0 + np.exp(-scaled)))
        prob_b = 1.0 - prob_a
        confidence = float(abs(prob_a - prob_b))
        
        # 3. Artifact Safety Check
        if is_artifact:
            # Freeze state; do not accumulate confirm counters on artifact steps
            self.pending_switch_target = None
            self.confirm_counter = 0
            return self._format_output(m, s, prob_a, prob_b, confidence, switched=False, is_artifact=True)
            
        switched = False
        prev_state = self.current_state
        
        # 4. Sticky Finite State Machine Logic
        if self.current_state == "NEUTRAL_HOLD":
            # Initial lock-on requires affirmative evidence
            if s >= self.threshold_switch:
                candidate = "LOCKED_A"
            elif s <= -self.threshold_switch:
                candidate = "LOCKED_B"
            else:
                candidate = "NEUTRAL_HOLD"
                
            if candidate in ["LOCKED_A", "LOCKED_B"]:
                if self.pending_switch_target == candidate:
                    self.confirm_counter += 1
                else:
                    self.pending_switch_target = candidate
                    self.confirm_counter = 1
                    
                if self.confirm_counter >= self.n_confirm:
                    self.current_state = candidate
                    switched = True
                    self.pending_switch_target = None
                    self.confirm_counter = 0
                    self.deadband_counter = 0
            else:
                self.pending_switch_target = None
                self.confirm_counter = 0

        elif self.current_state == "LOCKED_A":
            # In State A: check for affirmative switch to B
            if s <= -self.threshold_switch:
                if self.pending_switch_target == "LOCKED_B":
                    self.confirm_counter += 1
                else:
                    self.pending_switch_target = "LOCKED_B"
                    self.confirm_counter = 1
                    
                if self.confirm_counter >= self.n_confirm:
                    self.current_state = "LOCKED_B"
                    switched = True
                    self.pending_switch_target = None
                    self.confirm_counter = 0
                    self.deadband_counter = 0
            else:
                # No affirmative switch
                self.pending_switch_target = None
                self.confirm_counter = 0
                
                # Check for deadband timeout
                if abs(s) < self.threshold_maintain:
                    self.deadband_counter += 1
                    if self.deadband_counter >= self.deadband_timeout_steps:
                        self.current_state = "NEUTRAL_HOLD"
                        switched = True
                        self.deadband_counter = 0
                else:
                    self.deadband_counter = 0

        elif self.current_state == "LOCKED_B":
            # In State B: check for affirmative switch to A
            if s >= self.threshold_switch:
                if self.pending_switch_target == "LOCKED_A":
                    self.confirm_counter += 1
                else:
                    self.pending_switch_target = "LOCKED_A"
                    self.confirm_counter = 1
                    
                if self.confirm_counter >= self.n_confirm:
                    self.current_state = "LOCKED_A"
                    switched = True
                    self.pending_switch_target = None
                    self.confirm_counter = 0
                    self.deadband_counter = 0
            else:
                # No affirmative switch
                self.pending_switch_target = None
                self.confirm_counter = 0
                
                # Check for deadband timeout
                if abs(s) < self.threshold_maintain:
                    self.deadband_counter += 1
                    if self.deadband_counter >= self.deadband_timeout_steps:
                        self.current_state = "NEUTRAL_HOLD"
                        switched = True
                        self.deadband_counter = 0
                else:
                    self.deadband_counter = 0
                    
        return self._format_output(m, s, prob_a, prob_b, confidence, switched=switched, is_artifact=False)

    def _format_output(self, raw_m, smooth_s, prob_a, prob_b, conf, switched, is_artifact):
        # Continuous click-free audio gain steering
        if self.current_state == "LOCKED_A":
            target_ga, target_gb = 1.0, 1.0 / self.boost_lin
            decision_label = "A"
        elif self.current_state == "LOCKED_B":
            target_ga, target_gb = 1.0 / self.boost_lin, 1.0
            decision_label = "B"
        else:
            target_ga, target_gb = 0.5, 0.5
            decision_label = "HOLD"
            
        # Slew rate exponential smoothing
        self.gain_a = 0.4 * self.gain_a + 0.6 * target_ga
        self.gain_b = 0.4 * self.gain_b + 0.6 * target_gb
        
        return {
            "decision": decision_label,
            "state": self.current_state,
            "confidence": conf,
            "prob_a": prob_a,
            "prob_b": prob_b,
            "raw_margin": raw_m,
            "smoothed_margin": float(smooth_s),
            "gain_a": float(self.gain_a),
            "gain_b": float(self.gain_b),
            "switched": switched,
            "is_hold": (self.current_state == "NEUTRAL_HOLD"),
            "is_artifact": is_artifact,
            "deadband_counter": self.deadband_counter,
        }


class AdvancedStickyGate:
    """
    Sticky Pro: Advanced State-Retention Gating Controller.
    
    Upgrades over 1st-generation Hysteresis:
      1. Asymmetric Temporal Momentum: Fast Attack (alpha=0.65) on reinforcing speech,
         Slow Release (alpha=0.92) during conversational pauses.
      2. Continuous Leaky Evidence Accumulation (SPRT / Drift-Diffusion): Replaces
         brittle integer step counters with continuous confidence accumulation.
      3. Cross-Modal Speech Energy Gating (VAD Freeze): Freezes state during mutual
         speech pauses to avoid decoding undefined auditory-EEG correlation.
      4. Dynamic Gain Slew Limiting: Smooth, click-free equal-power audio crossfading.
    """
    def __init__(
        self,
        alpha_fast: float = 0.65,
        alpha_slow: float = 0.92,
        threshold_switch: float = 0.30,
        threshold_maintain: float = 0.12,
        evidence_threshold: float = 0.35,
        lambda_leak: float = 0.50,
        deadband_timeout_steps: int = 24,
        boost_db: float = 6.0,
        temperature: float = 1.0,
        silence_threshold: float = 0.02,
    ):
        self.alpha_fast = float(alpha_fast)
        self.alpha_slow = float(alpha_slow)
        self.threshold_switch = float(threshold_switch)
        self.threshold_maintain = float(threshold_maintain)
        self.evidence_threshold = float(evidence_threshold)
        self.lambda_leak = float(lambda_leak)
        self.deadband_timeout_steps = int(deadband_timeout_steps)
        self.temperature = max(1e-4, float(temperature))
        self.boost_db = float(boost_db)
        self.boost_lin = 10.0 ** (self.boost_db / 20.0)
        self.silence_threshold = float(silence_threshold)
        
        # State variables
        self.smoothed_margin = 0.0
        self.initialized = False
        self.current_state = "NEUTRAL_HOLD"  # 'LOCKED_A', 'LOCKED_B', 'NEUTRAL_HOLD'
        self.evidence_switch = 0.0
        self.deadband_counter = 0
        self.gain_a = 0.5
        self.gain_b = 0.5

    def reset(self):
        self.smoothed_margin = 0.0
        self.initialized = False
        self.current_state = "NEUTRAL_HOLD"
        self.evidence_switch = 0.0
        self.deadband_counter = 0
        self.gain_a = 0.5
        self.gain_b = 0.5

    def update(
        self,
        raw_margin: float,
        is_artifact: bool = False,
        audio_energy: Optional[float] = None
    ) -> Dict[str, Any]:
        m = float(raw_margin)
        
        # 1. Mutual Silence Check (Cross-Modal VAD Freeze)
        if audio_energy is not None and audio_energy < self.silence_threshold and self.current_state != "NEUTRAL_HOLD":
            scaled = np.clip(self.smoothed_margin / self.temperature, -30.0, 30.0)
            prob_a = float(1.0 / (1.0 + np.exp(-scaled)))
            return self._format_output(m, self.smoothed_margin, prob_a, 1.0 - prob_a, abs(2.0 * prob_a - 1.0), switched=False, is_artifact=False, is_silent=True)
            
        # 2. Artifact Safety Check
        if is_artifact:
            self.evidence_switch = 0.0
            scaled = np.clip(self.smoothed_margin / self.temperature, -30.0, 30.0)
            prob_a = float(1.0 / (1.0 + np.exp(-scaled)))
            return self._format_output(m, self.smoothed_margin, prob_a, 1.0 - prob_a, abs(2.0 * prob_a - 1.0), switched=False, is_artifact=True, is_silent=False)
            
        # 3. Asymmetric Temporal Smoothing (Fast Attack, Slow Release)
        if not self.initialized:
            self.smoothed_margin = m
            self.initialized = True
        else:
            if self.current_state == "LOCKED_A":
                alpha = self.alpha_fast if m > self.smoothed_margin else self.alpha_slow
            elif self.current_state == "LOCKED_B":
                alpha = self.alpha_fast if m < self.smoothed_margin else self.alpha_slow
            else:
                alpha = 0.70
                
            self.smoothed_margin = alpha * self.smoothed_margin + (1.0 - alpha) * m
            
        s = self.smoothed_margin
        
        # 4. Temperature-Calibrated Confidence
        scaled = np.clip(s / self.temperature, -30.0, 30.0)
        prob_a = float(1.0 / (1.0 + np.exp(-scaled)))
        prob_b = 1.0 - prob_a
        confidence = float(abs(prob_a - prob_b))
        
        switched = False
        
        # 5. Continuous Leaky Evidence Integration (SPRT / Drift-Diffusion)
        if self.current_state == "NEUTRAL_HOLD":
            # Fast, decisive initial lock-on once speech begins
            if s >= self.threshold_switch:
                self.current_state = "LOCKED_A"
                switched = True
                self.evidence_switch = 0.0
                self.deadband_counter = 0
            elif s <= -self.threshold_switch:
                self.current_state = "LOCKED_B"
                switched = True
                self.evidence_switch = 0.0
                self.deadband_counter = 0
                
        elif self.current_state == "LOCKED_A":
            if s < -self.threshold_switch or m < -self.threshold_switch:
                # Accumulating counter-evidence for B from both smoothed and raw counter-margin
                counter_val = max(-s, -m)
                delta_e = max(0.0, counter_val - self.threshold_switch)
                self.evidence_switch = self.lambda_leak * self.evidence_switch + delta_e
                if self.evidence_switch >= self.evidence_threshold:
                    self.current_state = "LOCKED_B"
                    switched = True
                    self.evidence_switch = 0.0
                    self.deadband_counter = 0
            else:
                # Reinforcing A or in deadband
                if s > 0.0:
                    self.evidence_switch = 0.0  # Decisive affirmation of A flushes opposing evidence
                else:
                    self.evidence_switch *= self.lambda_leak
                    
                # Check for deadband timeout
                if abs(s) < self.threshold_maintain:
                    self.deadband_counter += 1
                    if self.deadband_counter >= self.deadband_timeout_steps:
                        self.current_state = "NEUTRAL_HOLD"
                        switched = True
                        self.deadband_counter = 0
                else:
                    self.deadband_counter = 0
                    
        elif self.current_state == "LOCKED_B":
            if s > self.threshold_switch or m > self.threshold_switch:
                # Accumulating counter-evidence for A from both smoothed and raw counter-margin
                counter_val = max(s, m)
                delta_e = max(0.0, counter_val - self.threshold_switch)
                self.evidence_switch = self.lambda_leak * self.evidence_switch + delta_e
                if self.evidence_switch >= self.evidence_threshold:
                    self.current_state = "LOCKED_A"
                    switched = True
                    self.evidence_switch = 0.0
                    self.deadband_counter = 0
            else:
                # Reinforcing B or in deadband
                if s < 0.0:
                    self.evidence_switch = 0.0
                else:
                    self.evidence_switch *= self.lambda_leak
                    
                if abs(s) < self.threshold_maintain:
                    self.deadband_counter += 1
                    if self.deadband_counter >= self.deadband_timeout_steps:
                        self.current_state = "NEUTRAL_HOLD"
                        switched = True
                        self.deadband_counter = 0
                else:
                    self.deadband_counter = 0
                    
        return self._format_output(m, s, prob_a, prob_b, confidence, switched=switched, is_artifact=False, is_silent=False)

    def _format_output(self, raw_m, smooth_s, prob_a, prob_b, conf, switched, is_artifact, is_silent=False):
        if self.current_state == "LOCKED_A":
            target_ga, target_gb = 1.0, 1.0 / self.boost_lin
            decision_label = "A"
        elif self.current_state == "LOCKED_B":
            target_ga, target_gb = 1.0 / self.boost_lin, 1.0
            decision_label = "B"
        else:
            target_ga, target_gb = 0.5, 0.5
            decision_label = "HOLD"
            
        self.gain_a = 0.4 * self.gain_a + 0.6 * target_ga
        self.gain_b = 0.4 * self.gain_b + 0.6 * target_gb
        
        return {
            "decision": decision_label,
            "state": self.current_state,
            "confidence": conf,
            "prob_a": prob_a,
            "prob_b": prob_b,
            "raw_margin": raw_m,
            "smoothed_margin": float(smooth_s),
            "gain_a": float(self.gain_a),
            "gain_b": float(self.gain_b),
            "switched": switched,
            "is_hold": (self.current_state == "NEUTRAL_HOLD"),
            "is_artifact": is_artifact,
            "is_silent": is_silent,
            "evidence_switch": float(self.evidence_switch),
            "deadband_counter": self.deadband_counter,
        }


class AnalyticalBayesianGate:
    """
    Closed-Form Analytical Bayesian State-Space Gate (0 Parameters).
    Formulates AAD state tracking as an inertia-weighted Hidden Markov Model.
    
    Inertia Parameter:
      P(S_t = S_{t-1}) = 1.0 - lambda_switch (e.g. 0.98, sticky self-transition)
    Emission Model:
      P(m_t | S_t = A) = N(mu, sigma^2)
      P(m_t | S_t = B) = N(-mu, sigma^2)
    """
    def __init__(
        self,
        mu: float = 0.4,
        sigma: float = 0.5,
        switch_prior: float = 0.02,
        decision_threshold: float = 0.70
    ):
        self.mu = float(mu)
        self.sigma = max(1e-3, float(sigma))
        self.switch_prior = float(switch_prior)
        self.decision_threshold = float(decision_threshold)
        
        # State posterior prior [P(A), P(B)]
        self.belief_a = 0.5
        self.belief_b = 0.5

    def reset(self):
        self.belief_a = 0.5
        self.belief_b = 0.5

    def update(self, margin: float) -> Dict[str, Any]:
        m = float(margin)
        
        # 1. State transition step (Sticky Markov transition)
        p_stay = 1.0 - self.switch_prior
        p_switch = self.switch_prior
        
        prior_a = self.belief_a * p_stay + self.belief_b * p_switch
        prior_b = self.belief_b * p_stay + self.belief_a * p_switch
        
        # 2. Gaussian emission likelihood
        # log P(m | A) = -0.5 * ((m - mu) / sigma)^2
        # log P(m | B) = -0.5 * ((m + mu) / sigma)^2
        # log likelihood ratio = log P(m|A) - log P(m|B) = 2 * m * mu / sigma^2
        llr = (2.0 * m * self.mu) / (self.sigma ** 2)
        llr = np.clip(llr, -30.0, 30.0)
        likelihood_ratio = float(np.exp(llr))
        
        # 3. Posterior update
        unnorm_a = prior_a * likelihood_ratio
        unnorm_b = prior_b
        
        total = unnorm_a + unnorm_b + 1e-12
        self.belief_a = float(unnorm_a / total)
        self.belief_b = 1.0 - self.belief_a
        
        # 4. Decision mapping
        if self.belief_a >= self.decision_threshold:
            decision = "A"
        elif self.belief_b >= self.decision_threshold:
            decision = "B"
        else:
            decision = "HOLD"
            
        return {
            "decision": decision,
            "belief_a": self.belief_a,
            "belief_b": self.belief_b,
            "confidence": float(abs(self.belief_a - self.belief_b)),
            "margin": m,
        }


class TinyTemporalGate(nn.Module):
    """
    Lightweight Causal Time-Series Gating Controller (PyTorch Module).
    Size: ~1,100 parameters.
    
    Consumes a sliding sequence of feature vectors u_t:
      u_t = [m_t, Δm_t, rolling_std_m, streak_t, current_gain]
    Produces:
      - Intent posterior logit (P(A) vs P(B))
      - Switching confidence logit (P(Voluntary Switch))
    """
    def __init__(self, input_dim: int = 5, hidden_dim: int = 16):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Causal 1-layer GRU
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        # Linear prediction heads
        self.head_intent = nn.Linear(hidden_dim, 1)
        self.head_switch = nn.Linear(hidden_dim, 1)
        
    def forward(self, x, h=None):
        # x: [B, T_seq, input_dim]
        out, h_next = self.gru(x, h)
        intent_logit = self.head_intent(out[:, -1, :]).squeeze(-1)
        switch_logit = self.head_switch(out[:, -1, :]).squeeze(-1)
        return intent_logit, switch_logit, h_next
