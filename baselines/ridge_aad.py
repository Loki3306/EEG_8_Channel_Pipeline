from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np
from scipy.io import loadmat
from scipy.signal import hilbert, butter, filtfilt


REPO_ROOT = Path(__file__).resolve().parents[1]

import os
if "EEG_DATA_DIR" in os.environ:
    DATA_DIR = Path(os.environ["EEG_DATA_DIR"])
elif Path("/kaggle/input").exists():
    try:
        mat_files = list(Path("/kaggle/input").rglob("S*_data_preproc.mat"))
        if mat_files:
            DATA_DIR = mat_files[0].parent
        else:
            DATA_DIR = Path("/kaggle/input")
    except Exception:
        DATA_DIR = Path("/kaggle/input")
else:
    DATA_DIR = Path(r"C:\Users\lokes\Downloads\archive (2)\DATA_preproc")

sys.path.insert(0, str(REPO_ROOT))

from analysis._common import load_subject_data, trial_labels


@dataclass(frozen=True)
class TrialExample:
    subject: str
    trial_index: int
    eeg: np.ndarray
    wav_a: np.ndarray
    wav_b: np.ndarray
    label: int


def subject_files() -> list[Path]:
    return sorted(DATA_DIR.glob("S*_data_preproc.mat"), key=lambda path: int(path.stem.split("_")[0][1:]))


def load_subject_examples(path: Path) -> list[TrialExample]:
    data = load_subject_data(path)
    labels = np.asarray(trial_labels(data), dtype=int)
    examples: list[TrialExample] = []
    for trial_index in range(data.eeg.shape[1]):
        eeg = np.asarray(data.eeg[0, trial_index], dtype=float)
        wav_a = np.asarray(data.wavA[0, trial_index], dtype=float).ravel()
        wav_b = np.asarray(data.wavB[0, trial_index], dtype=float).ravel()
        examples.append(
            TrialExample(
                subject=path.stem.split("_")[0],
                trial_index=trial_index,
                eeg=eeg,
                wav_a=wav_a,
                wav_b=wav_b,
                label=int(labels[trial_index]),
            )
        )
    return examples


def load_all_examples(paths: list[Path] | None = None) -> list[TrialExample]:
    examples: list[TrialExample] = []
    for path in paths or subject_files():
        examples.extend(load_subject_examples(path))
    return examples


def moving_average(x: np.ndarray, window: int = 64) -> np.ndarray:
    kernel = np.ones(window, dtype=float) / float(window)
    return np.convolve(np.asarray(x, dtype=float), kernel, mode="same")


def butter_lowpass_filter(x: np.ndarray, cutoff: float, fs: int) -> np.ndarray:
    b, a = butter(4, float(cutoff) / (0.5 * fs), btype="low")
    return filtfilt(b, a, x)


def butter_bandpass_filter(x: np.ndarray, lowcut: float, highcut: float, fs: int) -> np.ndarray:
    b, a = butter(4, [float(lowcut) / (0.5 * fs), float(highcut) / (0.5 * fs)], btype="band")
    return filtfilt(b, a, x)


def speech_envelope(
    wav: np.ndarray,
    *,
    smooth_window: int = 64,
    compression: float = 1.0,
    lowpass_hz: float | None = None,
    fs: int = 64,
    normalize: bool = True,
) -> np.ndarray:
    wav = np.asarray(wav, dtype=float).ravel()
    envelope = np.abs(hilbert(wav))
    envelope = moving_average(envelope, window=smooth_window)
    if compression != 1.0:
        envelope = np.sign(envelope) * (np.abs(envelope) ** float(compression))
    if lowpass_hz is not None:
        envelope = butter_lowpass_filter(envelope, cutoff=lowpass_hz, fs=fs)
    if not normalize:
        return envelope
    envelope = envelope - envelope.mean()
    scale = envelope.std() + 1e-12
    return envelope / scale


def normalize_eeg(eeg: np.ndarray) -> np.ndarray:
    eeg = np.asarray(eeg, dtype=float)
    eeg = eeg - eeg.mean(axis=0, keepdims=True)
    scale = eeg.std(axis=0, keepdims=True) + 1e-12
    return eeg / scale


def _lag_samples_from_ms(*, lag_ms: int, lag_step_ms: int, fs: int) -> list[int]:
    if lag_ms < 0:
        raise ValueError("lag_ms must be non-negative")
    if lag_step_ms <= 0:
        raise ValueError("lag_step_ms must be positive")

    max_lag_samples = int(round((float(lag_ms) / 1000.0) * fs))
    step_samples = max(int(round((float(lag_step_ms) / 1000.0) * fs)), 1)
    offsets = list(range(0, max_lag_samples + 1, step_samples))
    if offsets[-1] != max_lag_samples:
        offsets.append(max_lag_samples)
    return sorted(set(offsets))


def lagged_eeg_matrix(
    eeg: np.ndarray,
    lags: int | None = 16,
    *,
    lag_ms: int | None = None,
    lag_step_ms: int = 16,
    fs: int = 64,
) -> np.ndarray:
    eeg = normalize_eeg(eeg)
    samples, channels = eeg.shape
    feature_blocks = []
    if lag_ms is not None:
        lag_offsets = _lag_samples_from_ms(lag_ms=lag_ms, lag_step_ms=lag_step_ms, fs=fs)
    else:
        if lags is None:
            lags = 16
        lag_offsets = list(range(lags))

    for lag in lag_offsets:
        if lag == 0:
            feature_blocks.append(eeg)
        else:
            shifted = np.vstack([np.zeros((lag, channels), dtype=float), eeg[: samples - lag]])
            feature_blocks.append(shifted)
    return np.concatenate(feature_blocks, axis=1)


def feature_statistics(
    examples: list[TrialExample],
    *,
    lags: int | None = 16,
    lag_ms: int | None = None,
    lag_step_ms: int = 16,
    fs: int = 64,
    channel_ids: list[int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    num_channels = len(channel_ids) if channel_ids is not None else examples[0].eeg.shape[1]
    if lag_ms is not None:
        feature_count = num_channels * len(_lag_samples_from_ms(lag_ms=lag_ms, lag_step_ms=lag_step_ms, fs=fs))
    else:
        feature_count = num_channels * (lags if lags is not None else 16)
    total_rows = 0
    feature_sum = np.zeros(feature_count, dtype=float)
    feature_sumsq = np.zeros(feature_count, dtype=float)

    for example in examples:
        eeg = example.eeg
        if channel_ids is not None:
            eeg = eeg[channel_ids, :]
        x = lagged_eeg_matrix(eeg, lags=lags, lag_ms=lag_ms, lag_step_ms=lag_step_ms, fs=fs)
        feature_sum += x.sum(axis=0)
        feature_sumsq += np.square(x).sum(axis=0)
        total_rows += x.shape[0]

    mean = feature_sum / max(total_rows, 1)
    variance = feature_sumsq / max(total_rows, 1) - np.square(mean)
    std = np.sqrt(np.maximum(variance, 1e-12))
    return mean, std


def standardize_features(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (np.asarray(x, dtype=float) - mean) / std


def attended_stream_from_label(label: int, mapping: dict[int, str]) -> str:
    if label not in mapping:
        raise KeyError(f"label {label} missing from mapping")
    return mapping[label]


def target_envelope(example: TrialExample, mapping: dict[int, str], *, compression: float = 1.0, lowpass_hz: float | None = None, fs: int = 64) -> np.ndarray:
    stream = attended_stream_from_label(example.label, mapping)
    if stream == "A":
        return speech_envelope(example.wav_a, compression=compression, lowpass_hz=lowpass_hz, fs=fs)
    if stream == "B":
        return speech_envelope(example.wav_b, compression=compression, lowpass_hz=lowpass_hz, fs=fs)
    raise ValueError(f"Unsupported stream mapping: {stream}")


def target_envelope_raw(
    example: TrialExample,
    mapping: dict[int, str],
    *,
    compression: float = 1.0,
    lowpass_hz: float | None = None,
    fs: int = 64,
) -> np.ndarray:
    stream = attended_stream_from_label(example.label, mapping)
    if stream == "A":
        return speech_envelope(example.wav_a, compression=compression, lowpass_hz=lowpass_hz, fs=fs, normalize=False)
    if stream == "B":
        return speech_envelope(example.wav_b, compression=compression, lowpass_hz=lowpass_hz, fs=fs, normalize=False)
    raise ValueError(f"Unsupported stream mapping: {stream}")


def accumulate_ridge_terms(
    examples: list[TrialExample],
    mapping: dict[int, str],
    *,
    lags: int | None = 16,
    lag_ms: int | None = None,
    lag_step_ms: int = 16,
    fs: int = 64,
    feature_mean: np.ndarray | None = None,
    feature_std: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if lag_ms is not None:
        feature_count = examples[0].eeg.shape[1] * len(_lag_samples_from_ms(lag_ms=lag_ms, lag_step_ms=lag_step_ms, fs=fs))
    else:
        feature_count = examples[0].eeg.shape[1] * (lags if lags is not None else 16)
    xtx = np.zeros((feature_count, feature_count), dtype=float)
    xty = np.zeros(feature_count, dtype=float)

    for example in examples:
        x = lagged_eeg_matrix(example.eeg, lags=lags, lag_ms=lag_ms, lag_step_ms=lag_step_ms, fs=fs)
        if feature_mean is not None and feature_std is not None:
            x = standardize_features(x, feature_mean, feature_std)
        y = target_envelope(example, mapping)
        xtx += x.T @ x
        xty += x.T @ y

    return xtx, xty


def subject_ridge_terms(
    path: Path,
    mapping: dict[int, str],
    *,
    lags: int | None = 16,
    lag_ms: int | None = None,
    lag_step_ms: int = 16,
    fs: int = 64,
    feature_mean: np.ndarray | None = None,
    feature_std: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    return accumulate_ridge_terms(
        load_subject_examples(path),
        mapping,
        lags=lags,
        lag_ms=lag_ms,
        lag_step_ms=lag_step_ms,
        fs=fs,
        feature_mean=feature_mean,
        feature_std=feature_std,
    )


def subject_ridge_terms_from_examples(
    examples: list[TrialExample],
    mapping: dict[int, str],
    *,
    lags: int | None = 16,
    lag_ms: int | None = None,
    lag_step_ms: int = 16,
    fs: int = 64,
    feature_mean: np.ndarray | None = None,
    feature_std: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    return accumulate_ridge_terms(
        examples,
        mapping,
        lags=lags,
        lag_ms=lag_ms,
        lag_step_ms=lag_step_ms,
        fs=fs,
        feature_mean=feature_mean,
        feature_std=feature_std,
    )


def fit_ridge(
    examples: list[TrialExample],
    mapping: dict[int, str],
    *,
    lags: int | None = 16,
    lag_ms: int | None = None,
    lag_step_ms: int = 16,
    fs: int = 64,
    ridge_lambda: float = 1.0,
    feature_mean: np.ndarray | None = None,
    feature_std: np.ndarray | None = None,
) -> np.ndarray:
    xtx, xty = accumulate_ridge_terms(
        examples,
        mapping,
        lags=lags,
        lag_ms=lag_ms,
        lag_step_ms=lag_step_ms,
        fs=fs,
        feature_mean=feature_mean,
        feature_std=feature_std,
    )
    regularized = xtx + ridge_lambda * np.eye(xtx.shape[0], dtype=float)
    return np.linalg.solve(regularized, xty)


def predict_envelope(
    eeg: np.ndarray,
    weights: np.ndarray,
    *,
    lags: int | None = 16,
    lag_ms: int | None = None,
    lag_step_ms: int = 16,
    fs: int = 64,
    feature_mean: np.ndarray | None = None,
    feature_std: np.ndarray | None = None,
) -> np.ndarray:
    x = lagged_eeg_matrix(eeg, lags=lags, lag_ms=lag_ms, lag_step_ms=lag_step_ms, fs=fs)
    if feature_mean is not None and feature_std is not None:
        x = standardize_features(x, feature_mean, feature_std)
    pred = x @ weights
    pred = pred - pred.mean()
    scale = pred.std() + 1e-12
    return pred / scale


def iter_leave_one_subject_out(paths: list[Path] | None = None):
    subject_paths = paths or subject_files()
    for held_out in subject_paths:
        train_paths = [path for path in subject_paths if path != held_out]
        yield held_out, train_paths
