"""
Window Buffer Infrastructure for Sequential Selective AAD.
Maintains temporal memory of window-level predictions.
This module is strictly memory infrastructure and contains NO decision logic.
"""

from dataclasses import dataclass
from typing import List, Optional, Any
import statistics

@dataclass
class WindowPrediction:
    """
    Data container for a single window's prediction and confidence features.
    """
    window_index: int
    timestamp: float
    trial_id: str
    prediction: int
    confidence: float
    margin: float
    corr_a: float
    corr_b: float
    accepted: bool
    correct: bool
    epistemic: float = 0.0
    latent_reference: Optional[Any] = None

class SequentialWindowBuffer:
    """
    Maintains a sequential memory of WindowPredictions.
    Supports running statistics and history retrieval.
    """
    def __init__(self):
        self._buffer: List[WindowPrediction] = []
        
    def append(self, window: WindowPrediction) -> None:
        """Appends a new window prediction to the buffer."""
        self._buffer.append(window)
        
    def get_last(self, n: int) -> List[WindowPrediction]:
        """Returns the last n window predictions. If n is greater than buffer size, returns all."""
        if n <= 0:
            return []
        return self._buffer[-n:]
        
    def clear(self) -> None:
        """Alias for reset. Clears the buffer."""
        self.reset()
        
    def reset(self) -> None:
        """Resets the buffer to an empty state."""
        self._buffer.clear()
        
    def length(self) -> int:
        """Returns the current number of items in the buffer."""
        return len(self._buffer)
        
    def prediction_history(self) -> List[int]:
        """Returns the chronological history of predictions."""
        return [w.prediction for w in self._buffer]
        
    def confidence_history(self) -> List[float]:
        """Returns the chronological history of confidence scores."""
        return [w.confidence for w in self._buffer]
        
    def margin_history(self) -> List[float]:
        """Returns the chronological history of margins."""
        return [w.margin for w in self._buffer]
        
    def running_mean_confidence(self) -> float:
        """Calculates the mean confidence over the entire buffer."""
        if not self._buffer:
            return 0.0
        return sum(w.confidence for w in self._buffer) / len(self._buffer)
        
    def running_mean_margin(self) -> float:
        """Calculates the mean margin over the entire buffer."""
        if not self._buffer:
            return 0.0
        return sum(w.margin for w in self._buffer) / len(self._buffer)
        
    def running_accuracy(self) -> float:
        """Calculates the ratio of correct predictions in the buffer."""
        if not self._buffer:
            return 0.0
        correct_count = sum(1 for w in self._buffer if w.correct)
        return correct_count / len(self._buffer)

class TemporalEvidenceAccumulator:
    """
    Maintains an exponentially smoothed running average of Dirichlet evidence parameters.
    Used for the Hybrid Evidential-Temporal Confidence (HETC) architecture.
    """
    def __init__(self, num_classes: int = 2, momentum: float = 0.9):
        self.num_classes = num_classes
        self.momentum = momentum
        self.evidence_state = None
        
    def reset(self):
        self.evidence_state = None
        
    def update(self, current_evidence: List[float]) -> List[float]:
        """
        Updates the running evidence state with a new observation.
        Returns the smoothed evidence.
        """
        if self.evidence_state is None:
            self.evidence_state = list(current_evidence)
        else:
            for i in range(self.num_classes):
                self.evidence_state[i] = (self.momentum * self.evidence_state[i]) + ((1.0 - self.momentum) * current_evidence[i])
        return self.evidence_state
        
    def get_probability(self) -> List[float]:
        """
        Returns the expected probability distribution p_k = alpha_k / S 
        where alpha = evidence + 1.
        """
        if self.evidence_state is None:
            return [1.0 / self.num_classes] * self.num_classes
            
        alphas = [e + 1.0 for e in self.evidence_state]
        S = sum(alphas)
        return [a / S for a in alphas]
        
    def get_epistemic_uncertainty(self) -> float:
        """
        Returns the epistemic uncertainty u = K / S
        """
        if self.evidence_state is None:
            return 1.0
            
        alphas = [e + 1.0 for e in self.evidence_state]
        S = sum(alphas)
        return self.num_classes / S
