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
