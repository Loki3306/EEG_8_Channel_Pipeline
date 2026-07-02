import numpy as np
from collections import deque
from typing import Optional
from .base import EvidenceStrategy

class InfiniteAccumulator(EvidenceStrategy):
    """
    The baseline strategy inherited from classical SPRT.
    Ev = Ev + llr
    Memory is infinite.
    """
    def __init__(self):
        self.evidence = 0.0
        
    def update(self, prob: float, margin: float, llr: float) -> float:
        self.evidence += llr
        return self.evidence
        
    def reset(self):
        self.evidence = 0.0
        
    def get_name(self) -> str:
        return "InfiniteAccumulator"

class HardCapAccumulator(EvidenceStrategy):
    """
    Infinite accumulation but capped at a maximum magnitude to prevent
    the evidence mountain from growing too large.
    """
    def __init__(self, cap: float = 20.0):
        self.evidence = 0.0
        self.cap = cap
        
    def update(self, prob: float, margin: float, llr: float) -> float:
        self.evidence += llr
        self.evidence = np.clip(self.evidence, -self.cap, self.cap)
        return self.evidence
        
    def reset(self):
        self.evidence = 0.0
        
    def get_name(self) -> str:
        return f"HardCapAccumulator(cap={self.cap})"

class ExponentialDecayAccumulator(EvidenceStrategy):
    """
    Applies a constant exponential forgetting factor to past evidence.
    Ev = Ev * decay + llr
    """
    def __init__(self, decay: float = 0.95):
        self.evidence = 0.0
        self.decay = decay
        
    def update(self, prob: float, margin: float, llr: float) -> float:
        self.evidence = self.evidence * self.decay + llr
        return self.evidence
        
    def reset(self):
        self.evidence = 0.0
        
    def get_name(self) -> str:
        return f"ExponentialDecay(λ={self.decay})"

class AsymmetricDecayAccumulator(EvidenceStrategy):
    """
    Only decays evidence when the new evidence opposes the current lock direction.
    Allows strong locking, but fast release.
    """
    def __init__(self, decay: float = 0.85):
        self.evidence = 0.0
        self.decay = decay
        
    def update(self, prob: float, margin: float, llr: float) -> float:
        # If evidence is positive (locked to 1) and llr is negative
        # or evidence is negative (locked to 0) and llr is positive
        if (self.evidence > 0 and llr < 0) or (self.evidence < 0 and llr > 0):
            self.evidence = self.evidence * self.decay + llr
        else:
            self.evidence += llr
        return self.evidence
        
    def reset(self):
        self.evidence = 0.0
        
    def get_name(self) -> str:
        return f"AsymmetricDecay(λ={self.decay})"

class SlidingWindowAccumulator(EvidenceStrategy):
    """
    Sum of LLRs over the last N frames.
    """
    def __init__(self, window_size: int = 64): # ~4 seconds at 16Hz
        self.window_size = window_size
        self.buffer = deque(maxlen=window_size)
        
    def update(self, prob: float, margin: float, llr: float) -> float:
        self.buffer.append(llr)
        return sum(self.buffer)
        
    def reset(self):
        self.buffer.clear()
        
    def get_name(self) -> str:
        return f"SlidingWindow(N={self.window_size})"

class BayesianAccumulator(EvidenceStrategy):
    """
    Recursive Bayesian update that includes a transition probability (p_switch).
    This natively models the fact that attention can switch, preventing infinite build-up.
    Eq: P(t) = P(t-1)*(1-p_switch) + (1-P(t-1))*p_switch
    """
    def __init__(self, p_switch: float = 0.01):
        self.prob_state = 0.5
        self.p_switch = p_switch
        
    def update(self, prob: float, margin: float, llr: float) -> float:
        # Prior predict step (incorporate probability of switching)
        prior = self.prob_state * (1 - self.p_switch) + (1 - self.prob_state) * self.p_switch
        
        # Observation update (Bayes Rule)
        # Using LLR formulation to avoid numerical underflow
        prior_llr = np.log(prior / (1 - prior)) if 0 < prior < 1 else 0.0
        posterior_llr = prior_llr + llr
        
        # Convert back to prob
        self.prob_state = 1.0 / (1.0 + np.exp(-posterior_llr))
        return posterior_llr
        
    def reset(self):
        self.prob_state = 0.5
        
    def get_name(self) -> str:
        return f"BayesianAccumulator(p_switch={self.p_switch})"
