import numpy as np
from typing import Dict, Any, Optional

class SelectiveStreamingGate:
    """
    Real-time streaming integration gate for Auditory Attention Decoding.
    
    Directly consumes scalar neural margin Delta_t = logit_A - logit_B at each control step (e.g. 2 Hz).
    Implements:
      1. Causal Exponential Moving Average (EMA).
      2. Post-hoc Temperature Scaling Calibration.
      3. Dual-Threshold Hysteresis with Consecutive-Confirmation logic.
      4. Safe HOLD State: maintains neutral audio levels or freezes steering during uncertainty.
      5. Click-free acoustic gain transitions.
    """
    def __init__(
        self,
        alpha: float = 0.7,
        threshold_switch: float = 0.35,
        threshold_maintain: float = 0.15,
        n_confirm: int = 2,
        temperature: float = 1.0,
        boost_db: float = 6.0,
        strict_hold_on_deadband: bool = True
    ):
        self.alpha = float(alpha)
        self.threshold_switch = float(threshold_switch)
        self.threshold_maintain = float(threshold_maintain)
        self.n_confirm = int(n_confirm)
        self.temperature = max(1e-4, float(temperature))
        self.boost_db = float(boost_db)
        self.strict_hold_on_deadband = bool(strict_hold_on_deadband)
        
        # Linear gain multiplier (e.g. +6 dB ≈ 2.0x boost)
        self.boost_lin = 10.0 ** (self.boost_db / 20.0)
        
        # State variables
        self.smoothed_margin = 0.0
        self.initialized = False
        self.current_decision = "HOLD"  # 'A', 'B', 'HOLD'
        self.pending_target = None
        self.confirm_counter = 0
        self.gain_a = 0.5
        self.gain_b = 0.5

    def reset(self):
        """Resets all internal filters and state machines."""
        self.smoothed_margin = 0.0
        self.initialized = False
        self.current_decision = "HOLD"
        self.pending_target = None
        self.confirm_counter = 0
        self.gain_a = 0.5
        self.gain_b = 0.5

    def update(self, raw_margin: float) -> Dict[str, Any]:
        """
        Processes single streaming inference step.
        """
        return self._process_update(raw_margin)

    def update_single(self, raw_margin: float) -> Dict[str, Any]:
        """Alias for update."""
        return self._process_update(raw_margin)

    def _process_update(self, raw_margin: float) -> Dict[str, Any]:
        m = float(raw_margin)
        # 1. Causal EMA smoothing
        if not self.initialized:
            self.smoothed_margin = m
            self.initialized = True
        else:
            self.smoothed_margin = self.alpha * self.smoothed_margin + (1.0 - self.alpha) * m
            
        s = self.smoothed_margin
        
        # 2. Temperature calibration on smoothed score
        scaled = np.clip(s / self.temperature, -30.0, 30.0)
        prob_a = float(1.0 / (1.0 + np.exp(-scaled)))
        prob_b = 1.0 - prob_a
        confidence = float(abs(prob_a - prob_b))
        
        # 3. Hysteresis State Machine
        switched = False
        prev_decision = self.current_decision
        
        # Determine candidate state
        if s >= self.threshold_switch:
            candidate = "A"
        elif s <= -self.threshold_switch:
            candidate = "B"
        elif abs(s) < self.threshold_maintain:
            candidate = "HOLD"
        else:
            candidate = "DEADBAND"
            
        if self.current_decision == "HOLD":
            if candidate in ["A", "B"]:
                if self.pending_target == candidate:
                    self.confirm_counter += 1
                else:
                    self.pending_target = candidate
                    self.confirm_counter = 1
                    
                if self.confirm_counter >= self.n_confirm:
                    self.current_decision = candidate
                    switched = True
                    self.pending_target = None
                    self.confirm_counter = 0
            else:
                self.pending_target = None
                self.confirm_counter = 0
                
        elif self.current_decision == "A":
            if candidate == "B":
                if self.pending_target == "B":
                    self.confirm_counter += 1
                else:
                    self.pending_target = "B"
                    self.confirm_counter = 1
                    
                if self.confirm_counter >= self.n_confirm:
                    self.current_decision = "B"
                    switched = True
                    self.pending_target = None
                    self.confirm_counter = 0
            elif candidate == "HOLD" and self.strict_hold_on_deadband:
                self.current_decision = "HOLD"
                switched = True
                self.pending_target = None
                self.confirm_counter = 0
            else:
                if candidate == "A":
                    self.pending_target = None
                    self.confirm_counter = 0
                    
        elif self.current_decision == "B":
            if candidate == "A":
                if self.pending_target == "A":
                    self.confirm_counter += 1
                else:
                    self.pending_target = "A"
                    self.confirm_counter = 1
                    
                if self.confirm_counter >= self.n_confirm:
                    self.current_decision = "A"
                    switched = True
                    self.pending_target = None
                    self.confirm_counter = 0
            elif candidate == "HOLD" and self.strict_hold_on_deadband:
                self.current_decision = "HOLD"
                switched = True
                self.pending_target = None
                self.confirm_counter = 0
            else:
                if candidate == "B":
                    self.pending_target = None
                    self.confirm_counter = 0
                    
        # 4. Continuous click-free audio gain steering
        if self.current_decision == "A":
            target_ga, target_gb = 1.0, 1.0 / self.boost_lin
        elif self.current_decision == "B":
            target_ga, target_gb = 1.0 / self.boost_lin, 1.0
        else:
            # HOLD state: neutral gain balance
            target_ga, target_gb = 0.5, 0.5
            
        # Exponential smoothing of acoustic gains (60% step)
        self.gain_a = 0.4 * self.gain_a + 0.6 * target_ga
        self.gain_b = 0.4 * self.gain_b + 0.6 * target_gb
        
        return {
            "decision": self.current_decision,
            "attended_stream": self.current_decision,
            "confidence": confidence,
            "prob_a": prob_a,
            "prob_b": prob_b,
            "raw_margin": m,
            "smoothed_margin": float(s),
            "gain_a": float(self.gain_a),
            "gain_b": float(self.gain_b),
            "switched": switched,
            "is_hold": (self.current_decision == "HOLD"),
            "confirm_counter": self.confirm_counter,
        }
