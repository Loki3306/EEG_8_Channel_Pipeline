from abc import ABC, abstractmethod

class EvidenceStrategy(ABC):
    """
    Abstract base class for temporal decision strategies.
    Every accumulator must implement this interface so the DecisionPolicyEngine
    remains completely agnostic to the underlying memory/detection algorithm.
    """
    
    @abstractmethod
    def update(self, prob: float, margin: float, llr: float) -> float:
        """
        Process a new frame of evidence.
        
        Args:
            prob: The raw probability of the active speaker [0, 1].
            margin: The probability margin (active - competing).
            llr: The log-likelihood ratio for this frame.
            
        Returns:
            The current effective evidence (on the LLR scale) used by the controller
            to evaluate confidence and threshold crossings.
        """
        pass
        
    @abstractmethod
    def reset(self):
        """
        Forcefully reset the internal memory state to zero/uncertainty.
        """
        pass
        
    @abstractmethod
    def get_name(self) -> str:
        """
        Return the name of the strategy for reporting.
        """
        pass
