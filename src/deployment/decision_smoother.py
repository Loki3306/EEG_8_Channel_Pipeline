import numpy as np

class EMAHysteresisDecisionLayer:
    """
    Temporal decision smoothing and dual-threshold hysteresis layer.
    
    Prevents volume chatter and rapid speaker flipping at attention decision boundaries.
    
    Mathematical Formulation:
      1. Exponential Moving Average (EMA) of raw model output:
         S_t = alpha * S_{t-1} + (1 - alpha) * Delta_t
         where Delta_t = logit_A - logit_B.
         
      2. Confidence Score:
         C_t = tanh(|S_t| / tau_conf)
         
      3. Dual-Threshold Hysteresis State Machine:
         - Currently Attending A: Switch to B only if S_t < -threshold for n_confirm steps.
         - Currently Attending B: Switch to A only if S_t > +threshold for n_confirm steps.
         
      4. Dynamic Gain Steering:
         Continuous audio gains [g_A, g_B] using smooth crossfade to eliminate acoustic clicks.
    """
    def __init__(self, alpha: float = 0.7, threshold: float = 0.25, n_confirm: int = 2,
                 boost_db: float = 6.0, tau_conf: float = 0.5):
        self.alpha = alpha
        self.threshold = threshold
        self.n_confirm = n_confirm
        self.boost_db = boost_db
        self.tau_conf = tau_conf
        
        # State
        self.smoothed_score = 0.0
        self.current_decision = 0  # 0: Uncertain/Neutral, +1: Attending A, -1: Attending B
        self.pending_decision = 0
        self.confirm_counter = 0
        self.gain_a = 0.5
        self.gain_b = 0.5

    def reset(self):
        """Resets the state machine."""
        self.smoothed_score = 0.0
        self.current_decision = 0
        self.pending_decision = 0
        self.confirm_counter = 0
        self.gain_a = 0.5
        self.gain_b = 0.5

    def update(self, raw_delta: float) -> dict:
        """
        Updates the decision layer with a new inference score Delta_t = logit_A - logit_B.
        
        Returns:
            dict containing:
              - 'smoothed_score': float (S_t)
              - 'attended_stream': str ('A', 'B', or 'UNCERTAIN')
              - 'confidence': float in [0.0, 1.0]
              - 'gain_a': float in [0.0, 1.0]
              - 'gain_b': float in [0.0, 1.0]
              - 'switched': bool (True if a speaker switch was triggered on this step)
        """
        # 1. Update EMA
        self.smoothed_score = self.alpha * self.smoothed_score + (1.0 - self.alpha) * float(raw_delta)
        
        # 2. Confidence estimate
        confidence = float(np.tanh(abs(self.smoothed_score) / self.tau_conf))
        
        # 3. Hysteresis State Machine
        switched = False
        target_decision = 0
        if self.smoothed_score > self.threshold:
            target_decision = +1  # Attending A
        elif self.smoothed_score < -self.threshold:
            target_decision = -1  # Attending B
        else:
            target_decision = self.current_decision  # Within deadband: keep current state
            
        if self.current_decision == 0:
            # Initial lock-on
            if target_decision != 0:
                self.confirm_counter += 1
                if self.confirm_counter >= self.n_confirm:
                    self.current_decision = target_decision
                    switched = True
                    self.confirm_counter = 0
        elif target_decision != self.current_decision and target_decision != 0:
            # Potential switch candidate
            if target_decision == self.pending_decision:
                self.confirm_counter += 1
                if self.confirm_counter >= self.n_confirm:
                    self.current_decision = target_decision
                    switched = True
                    self.confirm_counter = 0
            else:
                self.pending_decision = target_decision
                self.confirm_counter = 1
        else:
            # State is stable
            self.pending_decision = 0
            self.confirm_counter = 0
            
        # 4. Continuous Gain Multipliers with smooth exponential approach
        # Attended speaker gets boosted, unattended attenuated
        boost_lin = 10.0 ** (self.boost_db / 20.0) # e.g. +6 dB ≈ 2.0x
        if self.current_decision == +1:
            target_ga, target_gb = 1.0, 1.0 / boost_lin
            stream_str = 'A'
        elif self.current_decision == -1:
            target_ga, target_gb = 1.0 / boost_lin, 1.0
            stream_str = 'B'
        else:
            target_ga, target_gb = 1.0, 1.0
            stream_str = 'UNCERTAIN'
            
        # Smooth gain progression (60% step)
        self.gain_a = 0.4 * self.gain_a + 0.6 * target_ga
        self.gain_b = 0.4 * self.gain_b + 0.6 * target_gb
        
        return {
            "smoothed_score": float(self.smoothed_score),
            "attended_stream": stream_str,
            "confidence": confidence,
            "gain_a": float(self.gain_a),
            "gain_b": float(self.gain_b),
            "switched": switched,
        }
