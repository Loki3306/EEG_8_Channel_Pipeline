"""
Selective AAD (Auditory Attention Decoding) and Confidence Gating Layer.

A modular, strictly causal, compute-negligible confidence and abstention framework
operating post-hoc on top of frozen CA-TCN neural margins.
"""

from .core import (
    RawMarginGate,
    EMAMarginGate,
    HysteresisSelectiveGate,
    TemperatureCalibrator,
    SelectiveRiskCoverageOptimizer,
    ConformalSelectiveGate,
)
from .streaming_gate import SelectiveStreamingGate
from .metrics import (
    calculate_selective_metrics,
    compute_risk_coverage_curve,
    compute_aurc,
    compute_ece,
    compute_brier_score,
    compute_temporal_stability_metrics,
)

__all__ = [
    "RawMarginGate",
    "EMAMarginGate",
    "HysteresisSelectiveGate",
    "TemperatureCalibrator",
    "SelectiveRiskCoverageOptimizer",
    "ConformalSelectiveGate",
    "SelectiveStreamingGate",
    "calculate_selective_metrics",
    "compute_risk_coverage_curve",
    "compute_aurc",
    "compute_ece",
    "compute_brier_score",
    "compute_temporal_stability_metrics",
]
