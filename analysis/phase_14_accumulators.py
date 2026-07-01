import numpy as np
from collections import deque
from enum import Enum

class Decision(Enum):
    ACCEPT_STREAM_1 = 1
    ACCEPT_STREAM_2 = 0
    CONTINUE = -1

class BayesianAccumulator:
    """
    Accumulates evidence over time using Bayesian updates (Log-Odds formulation).
    Assumes conditional independence between sequential EEG windows.
    """
    def __init__(self, prior_prob=0.5):
        self.prior_odds = np.log(prior_prob / (1.0 - prior_prob + 1e-15) + 1e-15)
        self.log_odds = self.prior_odds
        self.history = []

    def reset(self):
        self.log_odds = self.prior_odds
        self.history.clear()

    def update(self, prob_t: float) -> float:
        """
        Updates the posterior probability with a new window probability.
        prob_t: calibrated probability that Stream 1 is attended (p > 0.5 means Stream 1).
        """
        prob_t = np.clip(prob_t, 1e-7, 1 - 1e-7)
        evidence_log_odds = np.log(prob_t / (1.0 - prob_t))
        self.log_odds += evidence_log_odds
        
        posterior = 1.0 / (1.0 + np.exp(-self.log_odds))
        self.history.append(posterior)
        return posterior
        
    def get_probability(self) -> float:
        return 1.0 / (1.0 + np.exp(-self.log_odds))

class SPRTAccumulator:
    """
    Wald's Sequential Probability Ratio Test (SPRT).
    Tracks Log-Likelihood Ratio (LLR) and triggers early stopping.
    """
    def __init__(self, alpha=0.05, beta=0.05):
        self.alpha = alpha
        self.beta = beta
        # A: Upper bound (Accept Stream 1)
        self.bound_A = np.log((1 - beta) / alpha)
        # B: Lower bound (Accept Stream 2)
        self.bound_B = np.log(beta / (1 - alpha))
        
        self.llr = 0.0
        self.history = []

    def reset(self):
        self.llr = 0.0
        self.history.clear()

    def update(self, prob_t: float) -> tuple[float, Decision]:
        """
        Updates the LLR and checks against stopping bounds.
        Returns: (Current Probability Estimate, Decision)
        """
        prob_t = np.clip(prob_t, 1e-7, 1 - 1e-7)
        evidence_llr = np.log(prob_t / (1.0 - prob_t))
        self.llr += evidence_llr
        
        posterior = 1.0 / (1.0 + np.exp(-self.llr))
        self.history.append(posterior)
        
        if self.llr >= self.bound_A:
            return posterior, Decision.ACCEPT_STREAM_1
        elif self.llr <= self.bound_B:
            return posterior, Decision.ACCEPT_STREAM_2
        else:
            return posterior, Decision.CONTINUE
            
    def get_llr(self) -> float:
        return self.llr

class EMAAccumulator:
    """
    Exponential Moving Average accumulator.
    More robust to non-stationarity than Bayesian accumulation.
    """
    def __init__(self, alpha=0.1):
        self.alpha = alpha
        self.ema = 0.5
        self.history = []

    def reset(self):
        self.ema = 0.5
        self.history.clear()

    def update(self, prob_t: float) -> float:
        self.ema = self.alpha * prob_t + (1 - self.alpha) * self.ema
        self.history.append(self.ema)
        return self.ema

    def get_probability(self) -> float:
        return self.ema

class SlidingWindowAccumulator:
    """
    Maintains a rolling average of the last K windows.
    """
    def __init__(self, window_size: int = 5):
        self.window_size = window_size
        self.buffer = deque(maxlen=window_size)
        self.history = []

    def reset(self):
        self.buffer.clear()
        self.history.clear()

    def update(self, prob_t: float) -> float:
        self.buffer.append(prob_t)
        mean_prob = sum(self.buffer) / len(self.buffer)
        self.history.append(mean_prob)
        return mean_prob

    def get_probability(self) -> float:
        if not self.buffer:
            return 0.5
        return sum(self.buffer) / len(self.buffer)
