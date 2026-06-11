from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TrialScore:
    trial_index: int
    corr_a: float
    corr_b: float
    true_stream: str
    predicted_stream: str

    @property
    def is_correct(self) -> bool:
        return self.true_stream == self.predicted_stream

    @property
    def corr_difference(self) -> float:
        if self.true_stream == "A":
            return self.corr_a - self.corr_b
        return self.corr_b - self.corr_a


def safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    if x.size != y.size or x.size < 2:
        return 0.0

    x = x - x.mean()
    y = y - y.mean()
    denom = np.linalg.norm(x) * np.linalg.norm(y)
    if denom == 0:
        return 0.0

    return float(np.dot(x, y) / denom)


def score_trial(predicted: np.ndarray, wav_a: np.ndarray, wav_b: np.ndarray, *, trial_index: int, true_stream: str) -> TrialScore:
    corr_a = safe_corr(predicted, wav_a)
    corr_b = safe_corr(predicted, wav_b)
    predicted_stream = "A" if corr_a > corr_b else "B"
    return TrialScore(
        trial_index=trial_index,
        corr_a=corr_a,
        corr_b=corr_b,
        true_stream=true_stream,
        predicted_stream=predicted_stream,
    )


def summarize_trials(scores: list[TrialScore]) -> dict[str, float]:
    if not scores:
        return {
            "trial_accuracy": 0.0,
            "balanced_accuracy": 0.0,
            "mean_corr_difference": 0.0,
            "mean_corr_a": 0.0,
            "mean_corr_b": 0.0,
        }

    accuracy = float(np.mean([score.is_correct for score in scores]))
    true_a = [score for score in scores if score.true_stream == "A"]
    true_b = [score for score in scores if score.true_stream == "B"]

    tpr = float(np.mean([score.is_correct for score in true_a])) if true_a else 0.0
    tnr = float(np.mean([score.is_correct for score in true_b])) if true_b else 0.0
    balanced_accuracy = 0.5 * (tpr + tnr)

    return {
        "trial_accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "mean_corr_difference": float(np.mean([score.corr_difference for score in scores])),
        "mean_corr_a": float(np.mean([score.corr_a for score in scores])),
        "mean_corr_b": float(np.mean([score.corr_b for score in scores])),
    }


def zero_eeg_baseline(trial_count: int) -> float:
    if trial_count <= 0:
        return 0.0
    return 0.5
