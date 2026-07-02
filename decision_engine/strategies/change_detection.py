import numpy as np
from .base import EvidenceStrategy
from .memory import InfiniteAccumulator

class CUSUMHybrid(EvidenceStrategy):
    """
    Runs a baseline accumulator, but concurrently runs a two-sided CUSUM change detector.
    If CUSUM detects a distribution shift, it forcefully flushes the accumulator.
    """
    def __init__(self, drift: float = 0.5, threshold: float = 5.0, base_strategy=None):
        self.drift = drift
        self.threshold = threshold
        self.base_strategy = base_strategy or InfiniteAccumulator()
        
        # CUSUM states for detecting increase (shift to 1) and decrease (shift to 0)
        self.g_plus = 0.0
        self.g_minus = 0.0
        
    def update(self, prob: float, margin: float, llr: float) -> float:
        # CUSUM updates using the margin signal (active vs competing probability)
        # Margin is positive when Speaker 1 is dominant, negative when Speaker 0 is dominant.
        
        # Detect shift towards 1
        self.g_plus = max(0.0, self.g_plus + margin - self.drift)
        # Detect shift towards 0
        self.g_minus = max(0.0, self.g_minus - margin - self.drift)
        
        # Check for change
        if self.g_plus > self.threshold or self.g_minus > self.threshold:
            # Change detected! Reset both CUSUM and the underlying accumulator
            self.reset()
            # We don't return 0 here; we let the new frame start the new accumulation
            return self.base_strategy.update(prob, margin, llr)
            
        return self.base_strategy.update(prob, margin, llr)
        
    def reset(self):
        self.g_plus = 0.0
        self.g_minus = 0.0
        self.base_strategy.reset()
        
    def get_name(self) -> str:
        return f"CUSUMHybrid(d={self.drift}, h={self.threshold})"

class ShiryaevRobertsHybrid(EvidenceStrategy):
    """
    Shiryaev-Roberts procedure for quickest change-point detection.
    It integrates the likelihood of a change occurring at any past time step.
    When the SR statistic crosses a threshold, the accumulator is reset.
    """
    def __init__(self, threshold: float = 50.0, base_strategy=None):
        self.threshold = threshold
        self.base_strategy = base_strategy or InfiniteAccumulator()
        
        # SR statistic
        self.R = 0.0
        
    def update(self, prob: float, margin: float, llr: float) -> float:
        # Likelihood ratio of the current frame (change vs no-change)
        # We approximate the change likelihood using exp(|llr|) assuming a change 
        # causes the LLR magnitude to increase in the opposite direction.
        
        # We'll use a simplified SR update: R_t = (1 + R_{t-1}) * L_t
        # Where L_t is the likelihood ratio of the new state vs old state.
        
        current_ev = self.base_strategy.evidence if hasattr(self.base_strategy, 'evidence') else 0
        
        # If evidence is strongly positive, a negative LLR indicates a change.
        # If evidence is strongly negative, a positive LLR indicates a change.
        is_contradiction = (current_ev > 0 and llr < 0) or (current_ev < 0 and llr > 0)
        
        if is_contradiction:
            L_t = np.exp(abs(llr))
        else:
            L_t = np.exp(-abs(llr))
            
        self.R = (1 + self.R) * L_t
        
        if self.R > self.threshold:
            self.reset()
            return self.base_strategy.update(prob, margin, llr)
            
        return self.base_strategy.update(prob, margin, llr)
        
    def reset(self):
        self.R = 0.0
        self.base_strategy.reset()
        
    def get_name(self) -> str:
        return f"ShiryaevRobertsHybrid(h={self.threshold})"

class PageHinkleyHybrid(EvidenceStrategy):
    """
    Page-Hinkley test for trend change detection.
    """
    def __init__(self, delta: float = 0.1, threshold: float = 5.0, alpha: float = 0.99, base_strategy=None):
        self.delta = delta
        self.threshold = threshold
        self.alpha = alpha
        self.base_strategy = base_strategy or InfiniteAccumulator()
        
        self.sum = 0.0
        self.x_mean = 0.0
        self.count = 0
        self.min_sum = 0.0
        self.max_sum = 0.0
        
    def update(self, prob: float, margin: float, llr: float) -> float:
        self.count += 1
        # Update running mean with forgetting factor
        self.x_mean = self.alpha * self.x_mean + (1 - self.alpha) * llr
        
        # Update cumulative sums
        self.sum += (llr - self.x_mean - self.delta)
        
        self.min_sum = min(self.min_sum, self.sum)
        self.max_sum = max(self.max_sum, self.sum)
        
        ph_plus = self.sum - self.min_sum
        ph_minus = self.max_sum - self.sum
        
        if ph_plus > self.threshold or ph_minus > self.threshold:
            self.reset()
            return self.base_strategy.update(prob, margin, llr)
            
        return self.base_strategy.update(prob, margin, llr)
        
    def reset(self):
        self.sum = 0.0
        self.count = 0
        self.min_sum = 0.0
        self.max_sum = 0.0
        self.base_strategy.reset()
        
    def get_name(self) -> str:
        return f"PageHinkleyHybrid(h={self.threshold})"
