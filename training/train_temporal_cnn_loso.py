from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.signal import butter, filtfilt
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from analysis._common import append_readme_update, ensure_output_dirs, save_json
from baselines.ridge_aad import iter_leave_one_subject_out, load_subject_examples, subject_files, target_envelope
from evaluation.aad_metrics import TrialScore, safe_corr, summarize_trials
from models.temporal_cnn import TemporalCNNAAD, TemporalContrastiveAAD, VLAAILiteAAD, cosine_similarity_matrix, count_parameters, correlation_loss

SUMMARY_PATH = REPO_ROOT / "analysis" / "summaries" / "temporal_cnn_loso_summary.json"


def notify(message: str) -> None:
    print(f"[temporal-cnn] {message}", flush=True)


def label_to_stream_mapping(name: str) -> dict[int, str]:
    mappings = {
        "A-B": {1: "A", 2: "B"},
        "B-A": {1: "B", 2: "A"},
    }
    if name not in mappings:
        raise ValueError(f"Unsupported mapping: {name}")
    return mappings[name]


def select_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)

    if torch.cuda.is_available():
        return torch.device("cuda")

    try:
        import torch_directml  # type: ignore

        return torch_directml.device()
    except Exception:
        return torch.device("cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def zero_eeg_like(example_eeg: np.ndarray, channel_ids: list[int]) -> np.ndarray:
    samples = example_eeg.shape[0]
    return np.zeros((samples, len(channel_ids)), dtype=float)


def channel_matrix(example_eeg: np.ndarray, channel_ids: list[int]) -> np.ndarray:
    return np.asarray(example_eeg[:, channel_ids], dtype=float)


def fit_channel_stats(examples, *, channel_ids: list[int]) -> tuple[np.ndarray, np.ndarray]:
    stacked = np.concatenate([channel_matrix(example.eeg, channel_ids) for example in examples], axis=0)
    mean = stacked.mean(axis=0)
    std = stacked.std(axis=0) + 1e-6
    return mean, std


def fit_channel_stats_robust(examples, *, channel_ids: list[int]) -> tuple[np.ndarray, np.ndarray]:
    stacked = np.concatenate([channel_matrix(example.eeg, channel_ids) for example in examples], axis=0)
    median = np.median(stacked, axis=0)
    mad = np.median(np.abs(stacked - median), axis=0)
    std = (mad * 1.4826) + 1e-6
    return median, std


def standardize_channels(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (x - mean) / std


def standardize_audio(x: np.ndarray, mean: float, std: float) -> np.ndarray:
    return (np.asarray(x, dtype=float).ravel() - mean) / std


def fit_target_stats(
    examples,
    mapping: dict[int, str],
    *,
    compression: float,
    lowpass_hz: float | None,
    fs: int,
) -> tuple[float, float]:
    stacked = np.concatenate(
        [target_envelope(example, mapping, compression=compression, lowpass_hz=lowpass_hz, fs=fs) for example in examples],
        axis=0,
    )
    mean = float(stacked.mean())
    std = float(stacked.std() + 1e-6)
    return mean, std


def fit_audio_stats(
    examples,
    mapping: dict[int, str],
    *,
    compression: float,
    lowpass_hz: float | None,
    fs: int,
) -> tuple[float, float]:
    opposite_mapping = {label: ("B" if stream == "A" else "A") for label, stream in mapping.items()}
    stacked = np.concatenate(
        [
            target_envelope(example, mapping, compression=compression, lowpass_hz=lowpass_hz, fs=fs)
            for example in examples
        ]
        + [
            target_envelope(example, opposite_mapping, compression=compression, lowpass_hz=lowpass_hz, fs=fs)
            for example in examples
        ],
        axis=0,
    )
    mean = float(stacked.mean())
    std = float(stacked.std() + 1e-6)
    return mean, std


def iter_within_subject_folds(examples_by_subject: dict[str, list], *, train_ratio: float = 0.8):
    """
    Split each subject's trials into train/test without LOSO.
    Yields (train_examples, test_examples) for each subject.
    """
    if not (0 < train_ratio < 1):
        raise ValueError(f"train_ratio must be in (0, 1), got {train_ratio}")
    for subject, examples in sorted(examples_by_subject.items()):
        n_train = max(1, int(len(examples) * train_ratio))
        indices = np.arange(len(examples))
        np.random.shuffle(indices)
        train_idx, test_idx = indices[:n_train], indices[n_train:]
        train_examples = [examples[i] for i in train_idx]
        test_examples = [examples[i] for i in test_idx]
        yield subject, train_examples, test_examples


def sample_negative_audio_chunk(
    audio_chunk: np.ndarray,
    all_examples: list,
    example_index: int,
    mapping: dict[int, str],
    start_sample: int,
    chunk_length: int,
    *,
    compression: float,
    lowpass_hz: float | None,
    fs: int,
    negative_mode: str = "random",
    negative_min_shift_sec: float = 0.0,
    negative_max_shift_sec: float = 0.5,
) -> np.ndarray:
    """
    Sample a hard negative audio chunk. Modes:
    - random: opposite stream at same time (current default)
    - nearby: opposite stream at shifted time
    - same_trial: wrong stream from same trial at different time
    - mixed: randomly choose among strategies
    """
    if negative_mode == "random":
        # Original behavior: opposite stream, no shift
        example = all_examples[example_index]
        opposite_mapping = {label: ("B" if stream == "A" else "A") for label, stream in mapping.items()}
        neg_stream = target_envelope(example, opposite_mapping, compression=compression, lowpass_hz=lowpass_hz, fs=fs)
        neg_chunk = neg_stream[start_sample : start_sample + chunk_length]
        if len(neg_chunk) < chunk_length:
            neg_chunk = np.pad(neg_chunk, (0, chunk_length - len(neg_chunk)), mode="edge")
        return neg_chunk

    elif negative_mode == "nearby":
        # Opposite stream with temporal shift
        example = all_examples[example_index]
        opposite_mapping = {label: ("B" if stream == "A" else "A") for label, stream in mapping.items()}
        neg_stream = target_envelope(example, opposite_mapping, compression=compression, lowpass_hz=lowpass_hz, fs=fs)
        min_shift = max(int(negative_min_shift_sec * fs), 0)
        max_shift = min(int(negative_max_shift_sec * fs), len(neg_stream) - chunk_length - 1)
        if max_shift > min_shift:
            shift = np.random.randint(min_shift, max_shift + 1)
        else:
            shift = min_shift
        neg_start = max(0, min(start_sample + shift, len(neg_stream) - chunk_length))
        neg_chunk = neg_stream[neg_start : neg_start + chunk_length]
        if len(neg_chunk) < chunk_length:
            neg_chunk = np.pad(neg_chunk, (0, chunk_length - len(neg_chunk)), mode="edge")
        return neg_chunk

    elif negative_mode == "same_trial":
        # Wrong stream from same trial, different time offset
        example = all_examples[example_index]
        opposite_mapping = {label: ("B" if stream == "A" else "A") for label, stream in mapping.items()}
        neg_stream = target_envelope(example, opposite_mapping, compression=compression, lowpass_hz=lowpass_hz, fs=fs)
        min_shift = max(int(negative_min_shift_sec * fs), 1)
        max_shift = min(int(negative_max_shift_sec * fs), len(neg_stream) - chunk_length - 1)
        if max_shift > min_shift:
            shift = np.random.randint(min_shift, max_shift + 1)
        else:
            shift = min_shift if min_shift < len(neg_stream) - chunk_length else 0
        neg_start = max(0, min(start_sample + shift, len(neg_stream) - chunk_length))
        neg_chunk = neg_stream[neg_start : neg_start + chunk_length]
        if len(neg_chunk) < chunk_length:
            neg_chunk = np.pad(neg_chunk, (0, chunk_length - len(neg_chunk)), mode="edge")
        return neg_chunk

    elif negative_mode == "mixed":
        # Randomly pick a strategy
        strategies = ["random", "nearby", "same_trial"]
        chosen = np.random.choice(strategies)
        return sample_negative_audio_chunk(
            audio_chunk,
            all_examples,
            example_index,
            mapping,
            start_sample,
            chunk_length,
            compression=compression,
            lowpass_hz=lowpass_hz,
            fs=fs,
            negative_mode=chosen,
            negative_min_shift_sec=negative_min_shift_sec,
            negative_max_shift_sec=negative_max_shift_sec,
        )

    else:
        raise ValueError(f"Unsupported negative_mode: {negative_mode}")


def chunk_start_indices(length: int, chunk_length: int, *, step: int | None = None) -> list[int]:
    if chunk_length <= 0:
        raise ValueError("chunk_length must be positive")
    if length <= chunk_length:
        return [0]
    stride = chunk_length if step is None else max(step, 1)
    starts = list(range(0, length - chunk_length + 1, stride))
    if starts[-1] != length - chunk_length:
        starts.append(length - chunk_length)
    return starts


def sample_shift_samples(chunk_length: int, *, lag_ms: int | None, lag_step_ms: int, fs: int, batch_size: int, device: torch.device) -> torch.Tensor:
    if lag_ms is None or lag_ms <= 0:
        return torch.randint(1, max(chunk_length, 2), (batch_size,), device=device)

    max_shift = max(int(round((float(lag_ms) / 1000.0) * fs)), 1)
    step = max(int(round((float(lag_step_ms) / 1000.0) * fs)), 1)
    candidates = torch.arange(step, max_shift + 1, step, device=device)
    if candidates.numel() == 0:
        candidates = torch.tensor([1], device=device)
    indices = torch.randint(0, candidates.numel(), (batch_size,), device=device)
    return candidates[indices]


def slice_batch_2d(batch: torch.Tensor, starts: torch.Tensor, length: int) -> torch.Tensor:
    chunks = [batch[index, start : start + length] for index, start in enumerate(starts.tolist())]
    return torch.stack(chunks, dim=0)


def slice_batch_3d(batch: torch.Tensor, starts: torch.Tensor, length: int) -> torch.Tensor:
    chunks = [batch[index, start : start + length, :] for index, start in enumerate(starts.tolist())]
    return torch.stack(chunks, dim=0)


def butter_bandpass(x: np.ndarray, lowcut: float, highcut: float, fs: int) -> np.ndarray:
    b, a = butter(4, [float(lowcut) / (0.5 * fs), float(highcut) / (0.5 * fs)], btype="band")
    # apply per-channel
    out = np.zeros_like(x)
    for ch in range(x.shape[1]):
        out[:, ch] = filtfilt(b, a, x[:, ch])
    return out


def evaluate_trial_windows(predicted: np.ndarray, wav_a: np.ndarray, wav_b: np.ndarray, *, window_seconds: int, fs: int = 64) -> tuple[float, float]:
    if window_seconds <= 0:
        return safe_corr(predicted, wav_a), safe_corr(predicted, wav_b)

    window_samples = window_seconds * fs
    if window_samples >= predicted.size:
        return safe_corr(predicted, wav_a), safe_corr(predicted, wav_b)

    corr_a_values = []
    corr_b_values = []
    for start in range(0, predicted.size - window_samples + 1, window_samples):
        stop = start + window_samples
        corr_a_values.append(safe_corr(predicted[start:stop], wav_a[start:stop]))
        corr_b_values.append(safe_corr(predicted[start:stop], wav_b[start:stop]))

    return float(np.mean(corr_a_values)), float(np.mean(corr_b_values))


@dataclass(frozen=True)
class TemporalExample:
    eeg: np.ndarray
    trial_index: int
    label: int
    wav_a: np.ndarray
    wav_b: np.ndarray


class TemporalTrialDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(
        self,
        examples,
        *,
        mapping: dict[int, str],
        channel_ids: list[int],
        mean: np.ndarray,
        std: np.ndarray,
        target_mean: float,
        target_std: float,
        target_compression: float,
        target_lowpass_hz: float | None,
        fs: int,
        zero_inputs: bool = False,
    ) -> None:
        self.examples = examples
        self.mapping = mapping
        self.channel_ids = channel_ids
        self.mean = mean
        self.std = std
        self.target_mean = target_mean
        self.target_std = target_std
        self.target_compression = target_compression
        self.target_lowpass_hz = target_lowpass_hz
        self.fs = fs
        self.zero_inputs = zero_inputs

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        example = self.examples[index]
        eeg = zero_eeg_like(example.eeg, self.channel_ids) if self.zero_inputs else channel_matrix(example.eeg, self.channel_ids)
        if not self.zero_inputs:
            eeg = standardize_channels(eeg, self.mean, self.std)
        target = target_envelope(
            example,
            self.mapping,
            compression=self.target_compression,
            lowpass_hz=self.target_lowpass_hz,
            fs=self.fs,
        )
        target = standardize_audio(target, self.target_mean, self.target_std)
        return torch.from_numpy(eeg.astype(np.float32)), torch.from_numpy(target.astype(np.float32))


class ContrastiveTrialDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(
        self,
        examples,
        *,
        mapping: dict[int, str],
        channel_ids: list[int],
        mean: np.ndarray,
        std: np.ndarray,
        target_mean: float,
        target_std: float,
        target_compression: float,
        target_lowpass_hz: float | None,
        fs: int,
        zero_inputs: bool = False,
        negative_mode: str = "random",
        negative_min_shift_sec: float = 0.0,
        negative_max_shift_sec: float = 0.5,
    ) -> None:
        self.examples = examples
        self.mapping = mapping
        self.channel_ids = channel_ids
        self.mean = mean
        self.std = std
        self.target_mean = target_mean
        self.target_std = target_std
        self.target_compression = target_compression
        self.target_lowpass_hz = target_lowpass_hz
        self.fs = fs
        self.zero_inputs = zero_inputs

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        example = self.examples[index]
        eeg = zero_eeg_like(example.eeg, self.channel_ids) if self.zero_inputs else channel_matrix(example.eeg, self.channel_ids)
        if not self.zero_inputs:
            eeg = standardize_channels(eeg, self.mean, self.std)

        positive = target_envelope(
            example,
            self.mapping,
            compression=self.target_compression,
            lowpass_hz=self.target_lowpass_hz,
            fs=self.fs,
        )
        
        positive = standardize_audio(positive, self.target_mean, self.target_std)
        return (
            torch.from_numpy(eeg.astype(np.float32)),
            torch.from_numpy(positive.astype(np.float32)),
        )


def score_predictions(predictions: list[np.ndarray], examples, *, mapping: dict[int, str], window_seconds: int) -> dict[str, object]:
    scores: list[TrialScore] = []
    for predicted, example in zip(predictions, examples):
        corr_a, corr_b = evaluate_trial_windows(predicted, example.wav_a, example.wav_b, window_seconds=window_seconds)
        scores.append(
            TrialScore(
                trial_index=example.trial_index,
                corr_a=corr_a,
                corr_b=corr_b,
                true_stream=mapping[example.label],
                predicted_stream="A" if corr_a > corr_b else "B",
            )
        )
    return summarize_trials(scores)


def train_fold(
    *,
    train_examples,
    val_examples,
    mapping: dict[int, str],
    channel_ids: list[int],
    target_mean: float,
    target_std: float,
    target_compression: float,
    target_lowpass_hz: float | None,
    fs: int,
    device: torch.device,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    patience: int,
    zero_inputs: bool,
    model_type: str = "temporal_cnn",
) -> tuple[torch.nn.Module, np.ndarray, np.ndarray, dict[str, float]]:
    mean = np.zeros(len(channel_ids), dtype=float) if zero_inputs else None
    std = np.ones(len(channel_ids), dtype=float) if zero_inputs else None
    if not zero_inputs:
        if getattr(train_fold, "robust", False):
            mean, std = fit_channel_stats_robust(train_examples, channel_ids=channel_ids)
        else:
            mean, std = fit_channel_stats(train_examples, channel_ids=channel_ids)

    train_dataset = TemporalTrialDataset(
        train_examples,
        mapping=mapping,
        channel_ids=channel_ids,
        mean=mean,
        std=std,
        target_mean=target_mean,
        target_std=target_std,
        target_compression=target_compression,
        target_lowpass_hz=target_lowpass_hz,
        fs=fs,
        zero_inputs=zero_inputs,
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=False)

    if model_type == "vlaai_lite":
        model = VLAAILiteAAD(in_channels=len(channel_ids)).to(device)
    else:
        model = TemporalCNNAAD(in_channels=len(channel_ids)).to(device)
        
    notify(f"Model parameters: {count_parameters(model):,}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_state = None
    best_score = float("-inf")
    best_epoch = 0
    patience_left = patience

    if zero_inputs:
        notify("Zero-EEG mode active: skipping training and using untrained model for chance-level sanity.")
        return model, mean, std, {"best_epoch": 0, "best_loss": 0.0}

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        batch_count = 0

        for eeg, target in train_loader:
            eeg = eeg.to(device)
            target = target.to(device)

            optimizer.zero_grad(set_to_none=True)
            prediction = model(eeg)
            loss = correlation_loss(prediction, target)
            loss.backward()
            optimizer.step()

            running_loss += float(loss.item())
            batch_count += 1

        train_loss = running_loss / max(batch_count, 1)

        model.eval()
        validation_predictions: list[np.ndarray] = []
        with torch.no_grad():
            for example in val_examples:
                eeg = zero_eeg_like(example.eeg, channel_ids) if zero_inputs else channel_matrix(example.eeg, channel_ids)
                if not zero_inputs:
                    eeg = standardize_channels(eeg, mean, std)
                eeg_tensor = torch.from_numpy(eeg.astype(np.float32)).unsqueeze(0).to(device)
                prediction = model(eeg_tensor).squeeze(0).detach().cpu().numpy()
                validation_predictions.append(prediction)

        validation_score = score_predictions(validation_predictions, val_examples, mapping=mapping, window_seconds=10)["trial_accuracy"]
        notify(f"  Epoch {epoch}/{epochs}: train_loss={train_loss:.4f}, val_accuracy@10s={validation_score:.4f}")

        if validation_score > best_score:
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            best_score = validation_score
            best_epoch = epoch
            patience_left = patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                notify(f"  Early stopping at epoch {epoch}; best epoch was {best_epoch} with val_accuracy@10s={best_score:.4f}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, mean, std, {"best_epoch": best_epoch, "best_val_accuracy": best_score}


def evaluate_similarity_windows(
    model: torch.nn.Module,
    example,
    *,
    channel_ids: list[int],
    eeg_mean: np.ndarray,
    eeg_std: np.ndarray,
    audio_mean: float,
    audio_std: float,
    target_compression: float,
    target_lowpass_hz: float | None,
    fs: int,
    window_seconds: int,
    device: torch.device,
    random_eeg: bool = False,
) -> tuple[float, float]:
    if window_seconds <= 0:
        window_seconds = int(round(example.eeg.shape[0] / fs))

    window_samples = window_seconds * fs
    if window_samples >= example.eeg.shape[0]:
        starts = [0]
    else:
        starts = list(range(0, example.eeg.shape[0] - window_samples + 1, window_samples))

    scores_a = []
    scores_b = []
    model.eval()
    with torch.no_grad():
        for start in starts:
            stop = min(start + window_samples, example.eeg.shape[0])
            eeg = channel_matrix(example.eeg[start:stop], channel_ids)
            if random_eeg:
                eeg = np.random.randn(*eeg.shape).astype(np.float32)
            eeg = standardize_channels(eeg, eeg_mean, eeg_std)
            eeg_tensor = torch.from_numpy(eeg.astype(np.float32)).unsqueeze(0).to(device)

            audio_a = standardize_audio(
                target_envelope(example, {1: "A", 2: "A"}, compression=target_compression, lowpass_hz=target_lowpass_hz, fs=fs)[start:stop],
                audio_mean,
                audio_std,
            )
            audio_b = standardize_audio(
                target_envelope(example, {1: "B", 2: "B"}, compression=target_compression, lowpass_hz=target_lowpass_hz, fs=fs)[start:stop],
                audio_mean,
                audio_std,
            )
            audio_a_tensor = torch.from_numpy(audio_a.astype(np.float32)).unsqueeze(0).unsqueeze(-1).to(device)
            audio_b_tensor = torch.from_numpy(audio_b.astype(np.float32)).unsqueeze(0).unsqueeze(-1).to(device)

            eeg_embedding = model.encode_eeg(eeg_tensor)
            audio_a_embedding = model.encode_audio(audio_a_tensor)
            audio_b_embedding = model.encode_audio(audio_b_tensor)
            scores_a.append(float(F.cosine_similarity(eeg_embedding, audio_a_embedding).item()))
            scores_b.append(float(F.cosine_similarity(eeg_embedding, audio_b_embedding).item()))

    return float(np.mean(scores_a)), float(np.mean(scores_b))


def score_similarity_trials(
    model: torch.nn.Module,
    examples,
    *,
    channel_ids: list[int],
    eeg_mean: np.ndarray,
    eeg_std: np.ndarray,
    audio_mean: float,
    audio_std: float,
    target_compression: float,
    target_lowpass_hz: float | None,
    fs: int,
    mapping: dict[int, str],
    window_seconds: int,
    device: torch.device,
    random_eeg: bool = False,
) -> dict[str, object]:
    scores: list[TrialScore] = []
    for example in examples:
        sim_a, sim_b = evaluate_similarity_windows(
            model,
            example,
            channel_ids=channel_ids,
            eeg_mean=eeg_mean,
            eeg_std=eeg_std,
            audio_mean=audio_mean,
            audio_std=audio_std,
            target_compression=target_compression,
            target_lowpass_hz=target_lowpass_hz,
            fs=fs,
            window_seconds=window_seconds,
            device=device,
            random_eeg=random_eeg,
        )
        scores.append(
            TrialScore(
                trial_index=example.trial_index,
                corr_a=sim_a,
                corr_b=sim_b,
                true_stream=mapping[example.label],
                predicted_stream="A" if sim_a > sim_b else "B",
            )
        )
    return summarize_trials(scores)


def train_contrastive_fold(
    *,
    train_examples,
    val_examples,
    mapping: dict[int, str],
    channel_ids: list[int],
    target_mean: float,
    target_std: float,
    target_compression: float,
    target_lowpass_hz: float | None,
    fs: int,
    device: torch.device,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    patience: int,
    chunk_seconds: int,
    temperature: float,
    lag_ms: int | None,
    lag_step_ms: int,
    zero_inputs: bool,
    negative_mode: str = "random",
    negative_min_shift_sec: float = 0.0,
    negative_max_shift_sec: float = 0.5,
    random_eeg: bool = False,
    no_training: bool = False,
    random_audio_pairs: bool = False,
) -> tuple[torch.nn.Module, np.ndarray, np.ndarray, dict[str, float]]:
    mean = np.zeros(len(channel_ids), dtype=float) if zero_inputs else None
    std = np.ones(len(channel_ids), dtype=float) if zero_inputs else None
    if not zero_inputs:
        if getattr(train_fold, "robust", False):
            mean, std = fit_channel_stats_robust(train_examples, channel_ids=channel_ids)
        else:
            mean, std = fit_channel_stats(train_examples, channel_ids=channel_ids)

    dataset = ContrastiveTrialDataset(
        train_examples,
        mapping=mapping,
        channel_ids=channel_ids,
        mean=mean,
        std=std,
        target_mean=target_mean,
        target_std=target_std,
        target_compression=target_compression,
        target_lowpass_hz=target_lowpass_hz,
        fs=fs,
        zero_inputs=zero_inputs,
        negative_mode=negative_mode,
        negative_min_shift_sec=negative_min_shift_sec,
        negative_max_shift_sec=negative_max_shift_sec,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=False)

    model = TemporalContrastiveAAD(eeg_channels=len(channel_ids)).to(device)
    notify(f"Model parameters: {count_parameters(model):,}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_state = None
    best_score = float("-inf")
    best_epoch = 0
    patience_left = patience
    chunk_samples = max(int(round(chunk_seconds * fs)), 1)

    if no_training:
        notify("No-training mode active: skipping optimizer updates and returning initialized model.")
        return model, mean, std, {"best_epoch": 0, "best_val_accuracy": 0.0}

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        batch_count = 0

        for eeg, audio_pos in loader:
            eeg = eeg.to(device)
            audio_pos = audio_pos.to(device)

            if random_eeg:
                eeg = torch.randn_like(eeg)
            
            if random_audio_pairs:
                audio_pos = audio_pos[torch.randperm(audio_pos.size(0))]

            max_start = max(eeg.shape[1] - chunk_samples, 0)
            starts = torch.randint(0, max_start + 1, (eeg.shape[0],), device=device)

            eeg_chunk = slice_batch_3d(eeg, starts, chunk_samples)
            pos_chunk = slice_batch_2d(audio_pos, starts, chunk_samples).unsqueeze(-1)

            optimizer.zero_grad(set_to_none=True)
            eeg_embedding = model.encode_eeg(eeg_chunk)
            pos_embedding = model.encode_audio(pos_chunk)
            
            # Batch-wise InfoNCE: the pool is just the positive embeddings from the batch
            audio_pool = pos_embedding
            logits = cosine_similarity_matrix(eeg_embedding, audio_pool) / max(float(temperature), 1e-6)
            loss = F.cross_entropy(logits, torch.arange(eeg_embedding.shape[0], device=device))
            loss.backward()
            optimizer.step()

            running_loss += float(loss.item())
            batch_count += 1

        train_loss = running_loss / max(batch_count, 1)

        validation_score = score_similarity_trials(
            model,
            val_examples,
            channel_ids=channel_ids,
            eeg_mean=mean,
            eeg_std=std,
            audio_mean=target_mean,
            audio_std=target_std,
            target_compression=target_compression,
            target_lowpass_hz=target_lowpass_hz,
            fs=fs,
            mapping=mapping,
            window_seconds=10,
            device=device,
            random_eeg=random_eeg,
        )["trial_accuracy"]
        notify(f"  Epoch {epoch}/{epochs}: train_loss={train_loss:.4f}, val_accuracy@10s={validation_score:.4f}")

        if validation_score > best_score:
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            best_score = validation_score
            best_epoch = epoch
            patience_left = patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                notify(f"  Early stopping at epoch {epoch}; best epoch was {best_epoch} with val_accuracy@10s={best_score:.4f}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, mean, std, {"best_epoch": best_epoch, "best_val_accuracy": best_score}


def run_fold(
    *,
    train_examples,
    val_examples,
    test_examples,
    mapping: dict[int, str],
    channel_ids: list[int],
    objective: str,
    target_mean: float,
    target_std: float,
    target_compression: float,
    target_lowpass_hz: float | None,
    fs: int,
    chunk_seconds: int,
    temperature: float,
    lag_ms: int | None,
    lag_step_ms: int,
    device: torch.device,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    patience: int,
    window_seconds: list[int],
    zero_inputs: bool,
    shuffle_labels: bool = False,
    negative_mode: str = "random",
    negative_min_shift_sec: float = 0.0,
    negative_max_shift_sec: float = 0.5,
    random_eeg: bool = False,
    no_training: bool = False,
    random_audio_pairs: bool = False,
    model_type: str = "temporal_cnn",
) -> dict[str, object]:
    setattr(train_fold, "robust", getattr(run_fold, "robust", False))

    if objective == "contrastive":
        model, mean, std, training_info = train_contrastive_fold(
            train_examples=train_examples,
            val_examples=val_examples,
            mapping=mapping,
            channel_ids=channel_ids,
            target_mean=target_mean,
            target_std=target_std,
            target_compression=target_compression,
            target_lowpass_hz=target_lowpass_hz,
            fs=fs,
            device=device,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            weight_decay=weight_decay,
            patience=patience,
            chunk_seconds=chunk_seconds,
            temperature=temperature,
            lag_ms=lag_ms,
            lag_step_ms=lag_step_ms,
            zero_inputs=zero_inputs,
            negative_mode=negative_mode,
            negative_min_shift_sec=negative_min_shift_sec,
            negative_max_shift_sec=negative_max_shift_sec,
            random_eeg=random_eeg,
            no_training=no_training,
            random_audio_pairs=random_audio_pairs,
        )
    else:
        model, mean, std, training_info = train_fold(
            train_examples=train_examples,
            val_examples=val_examples,
            mapping=mapping,
            channel_ids=channel_ids,
            target_mean=target_mean,
            target_std=target_std,
            target_compression=target_compression,
            target_lowpass_hz=target_lowpass_hz,
            fs=fs,
            device=device,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            weight_decay=weight_decay,
            patience=patience,
            zero_inputs=zero_inputs,
            shuffle_labels=shuffle_labels,
            model_type=model_type,
        )

    predictions: list[np.ndarray] = []
    if objective == "contrastive":
        predictions = []
    elif zero_inputs:
        predictions = [np.zeros(example.wav_a.shape[0], dtype=float) for example in test_examples]
    else:
        model.eval()
        with torch.no_grad():
            for example in test_examples:
                eeg = channel_matrix(example.eeg, channel_ids)
                eeg = standardize_channels(eeg, mean, std)
                eeg_tensor = torch.from_numpy(eeg.astype(np.float32)).unsqueeze(0).to(device)
                prediction = model(eeg_tensor).squeeze(0).detach().cpu().numpy()
                predictions.append(prediction)

    window_results = []
    for window in window_seconds:
        if objective == "contrastive":
            overall = score_similarity_trials(
                model,
                test_examples,
                channel_ids=channel_ids,
                eeg_mean=mean,
                eeg_std=std,
                audio_mean=target_mean,
                audio_std=target_std,
                target_compression=target_compression,
                target_lowpass_hz=target_lowpass_hz,
                fs=fs,
                mapping=mapping,
                window_seconds=window,
                device=device,
                random_eeg=random_eeg,
            )
        else:
            overall = score_predictions(predictions, test_examples, mapping=mapping, window_seconds=window)

        per_subject = []
        if objective == "contrastive":
            for example in test_examples:
                sim_a, sim_b = evaluate_similarity_windows(
                    model,
                    example,
                    channel_ids=channel_ids,
                    eeg_mean=mean,
                    eeg_std=std,
                    audio_mean=target_mean,
                    audio_std=target_std,
                    target_compression=target_compression,
                    target_lowpass_hz=target_lowpass_hz,
                    fs=fs,
                    window_seconds=window,
                    device=device,
                    random_eeg=random_eeg,
                )
                per_subject.append(
                    TrialScore(
                        trial_index=example.trial_index,
                        corr_a=sim_a,
                        corr_b=sim_b,
                        true_stream=mapping[example.label],
                        predicted_stream="A" if sim_a > sim_b else "B",
                    )
                )
        else:
            for predicted, example in zip(predictions, test_examples):
                corr_a, corr_b = evaluate_trial_windows(predicted, example.wav_a, example.wav_b, window_seconds=window)
                per_subject.append(
                    TrialScore(
                        trial_index=example.trial_index,
                        corr_a=corr_a,
                        corr_b=corr_b,
                        true_stream=mapping[example.label],
                        predicted_stream="A" if corr_a > corr_b else "B",
                    )
                )
        window_results.append(
            {
                "window_seconds": window,
                "overall": overall,
                "per_subject": [
                    {
                        "trial_index": score.trial_index,
                        "corr_a": score.corr_a,
                        "corr_b": score.corr_b,
                        "true_stream": score.true_stream,
                        "predicted_stream": score.predicted_stream,
                    }
                    for score in per_subject
                ],
            }
        )

    return {
        "mapping": mapping,
        "objective": objective,
        "training_info": training_info,
        "window_runs": window_results,
    }


def label_to_stream_mappings() -> list[dict[int, str]]:
    return [{1: "A", 2: "B"}, {1: "B", 2: "A"}]


def aggregate_per_subject_accuracies(window_runs: list[dict], window_seconds: int) -> dict[str, float]:
    """
    Aggregate trial-level scores per subject across all folds.
    Returns {subject_id: accuracy} for a given window size.
    """
    per_subject_scores = {}
    
    for fold_result in window_runs:
        window_data = next(
            (wr for wr in fold_result["window_runs"] if wr["window_seconds"] == window_seconds),
            None
        )
        if not window_data:
            continue
        
        # Group per-subject scores
        for score in window_data["per_subject"]:
            subject_key = f"S{score.get('trial_index', 'unknown')}"
            if subject_key not in per_subject_scores:
                per_subject_scores[subject_key] = {"correct": 0, "total": 0}
            per_subject_scores[subject_key]["total"] += 1
            if score.get("true_stream") == score.get("predicted_stream"):
                per_subject_scores[subject_key]["correct"] += 1
    
    # Compute accuracies
    return {
        subject: scores["correct"] / scores["total"] if scores["total"] > 0 else 0.0
        for subject, scores in per_subject_scores.items()
    }


def choose_best_mapping(*, mapping_mode: str) -> list[dict[int, str]]:
    if mapping_mode == "both":
        return label_to_stream_mappings()
    return [label_to_stream_mapping(mapping_mode)]


def run_evaluation_folds(
    *,
    subject_examples: dict,
    subject_paths: list[Path],
    evaluation_mode: str,
    subject_train_ratio: float,
    mapping: dict[int, str],
    args,
    device,
    zero_inputs: bool,
    notify_fn,
    shuffle_labels: bool = False,
    random_eeg: bool = False,
    no_training: bool = False,
    random_audio_pairs: bool = False,
) -> list[dict]:
    """
    Run evaluation folds (either LOSO or within-subject).
    Returns list of fold results.
    """
    window_runs = []
    
    if evaluation_mode == "loso":
        fold_iterator = enumerate(iter_leave_one_subject_out(subject_paths), start=1)
        def iter_fn():
            for fold_index, (held_out, train_paths) in fold_iterator:
                # Pick a validation subject deterministically that rotates with the fold
                val_idx = fold_index % len(train_paths)
                val_path = train_paths[val_idx]
                real_train_paths = train_paths[:val_idx] + train_paths[val_idx + 1:]
                
                train_examples = [example for path in real_train_paths for example in subject_examples[path]]
                val_examples = subject_examples[val_path]
                test_examples = subject_examples[held_out]
                fold_label = f"Fold {fold_index}/{len(subject_paths)}: held out {held_out.stem} (val {val_path.stem})"
                yield fold_index, len(subject_paths), fold_label, train_examples, val_examples, test_examples
    else:  # within-subject
        examples_by_subject = {}
        for path in subject_paths:
            subject_id = path.stem.split("_")[0]
            examples_by_subject[subject_id] = subject_examples[path]
        
        def iter_fn():
            folds = list(iter_within_subject_folds(examples_by_subject, train_ratio=subject_train_ratio))
            for fold_index, (subject_id, train_examples, test_examples) in enumerate(folds, start=1):
                # Split train into real train and val (e.g. 80/20)
                split_idx = int(len(train_examples) * 0.8)
                real_train_examples = train_examples[:split_idx]
                val_examples = train_examples[split_idx:]
                if not val_examples:
                    val_examples = real_train_examples # Fallback if too small
                fold_label = f"Fold {fold_index}/{len(folds)}: within {subject_id}"
                yield fold_index, len(folds), fold_label, real_train_examples, val_examples, test_examples
    
    for fold_index, num_folds, fold_label, train_examples, val_examples, test_examples in iter_fn():
        notify_fn(f"{fold_label}")
        setattr(run_fold, "robust", args.robust_norm)

        target_compression = args.env_compress
        target_lowpass_hz = None if args.env_lowpass <= 0 else args.env_lowpass
        fs = 64

        if args.objective == "contrastive":
            target_mean, target_std = fit_audio_stats(
                train_examples,
                mapping,
                compression=target_compression,
                lowpass_hz=target_lowpass_hz,
                fs=fs,
            )
        else:
            target_mean, target_std = fit_target_stats(
                train_examples,
                mapping,
                compression=target_compression,
                lowpass_hz=target_lowpass_hz,
                fs=fs,
            )

        # optionally bandpass the training and test EEG per trial
        if args.bp_low is not None and args.bp_high is not None and args.bp_high > args.bp_low:
            train_examples = [
                TemporalExample(
                    eeg=butter_bandpass(ex.eeg, args.bp_low, args.bp_high, fs=64),
                    trial_index=ex.trial_index,
                    label=ex.label,
                    wav_a=ex.wav_a,
                    wav_b=ex.wav_b,
                )
                for ex in train_examples
            ]
            test_examples = [
                TemporalExample(
                    eeg=butter_bandpass(ex.eeg, args.bp_low, args.bp_high, fs=64),
                    trial_index=ex.trial_index,
                    label=ex.label,
                    wav_a=ex.wav_a,
                    wav_b=ex.wav_b,
                )
                for ex in test_examples
            ]

        fold_result = run_fold(
            train_examples=train_examples,
            val_examples=val_examples,
            test_examples=test_examples,
            mapping=mapping,
            channel_ids=args.channel_ids,
            objective=args.objective,
            target_mean=target_mean,
            target_std=target_std,
            target_compression=target_compression,
            target_lowpass_hz=target_lowpass_hz,
            fs=fs,
            chunk_seconds=args.chunk_seconds,
            temperature=args.temperature,
            lag_ms=args.lag_ms,
            lag_step_ms=args.lag_step_ms,
            device=device,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            patience=args.patience,
            window_seconds=args.window_seconds,
            zero_inputs=zero_inputs,
            negative_mode=args.negative_mode,
            negative_min_shift_sec=args.negative_min_shift_sec,
            negative_max_shift_sec=args.negative_max_shift_sec,
            model_type=args.model_type,
        )
        window_runs.append(fold_result)
        notify_fn(f"{fold_label} complete")
    
    return window_runs


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a small temporal CNN for LOSO AAD reconstruction.")
    parser.add_argument("--device", type=str, default="auto", help="auto, cuda, cpu, or directml")
    parser.add_argument("--model-type", choices=["temporal_cnn", "vlaai_lite"], default="temporal_cnn")
    parser.add_argument("--objective", choices=["reconstruction", "contrastive"], default="reconstruction")
    parser.add_argument("--mapping", type=str, default="A-B", choices=["A-B", "B-A", "both"])
    parser.add_argument("--evaluation-mode", choices=["loso", "within-subject"], default="loso", help="LOSO (cross-subject) or within-subject train/test split")
    parser.add_argument("--subject-train-ratio", type=float, default=0.8, help="Train ratio for within-subject mode (ignored in LOSO)")
    parser.add_argument("--subject-limit", type=int, default=None)
    parser.add_argument("--channel-ids", type=int, nargs="+", default=[0, 1], help="EEG channel indices to use")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--window-seconds", type=int, nargs="*", default=[10])
    parser.add_argument("--chunk-seconds", type=int, default=5, help="Contrastive chunk length in seconds")
    parser.add_argument("--temperature", type=float, default=0.1, help="InfoNCE temperature for contrastive training")
    parser.add_argument("--lag-ms", type=int, default=500, help="Maximum negative shift in milliseconds for contrastive training")
    parser.add_argument("--lag-step-ms", type=int, default=16, help="Negative shift step in milliseconds for contrastive training")
    parser.add_argument("--negative-mode", choices=["random", "nearby", "same-trial", "mixed"], default="random", help="Hard negative sampling strategy")
    parser.add_argument("--negative-min-shift-sec", type=float, default=0.0, help="Minimum temporal shift (sec) for hard negatives")
    parser.add_argument("--negative-max-shift-sec", type=float, default=0.5, help="Maximum temporal shift (sec) for hard negatives")
    parser.add_argument("--bp-low", type=float, default=1.0, help="EEG bandpass low cutoff (Hz)")
    parser.add_argument("--bp-high", type=float, default=8.0, help="EEG bandpass high cutoff (Hz)")
    parser.add_argument("--env-compress", type=float, default=1.0, help="Envelope compression exponent (e.g., 0.6)")
    parser.add_argument("--env-lowpass", type=float, default=8.0, help="Envelope lowpass Hz (or 0 for none)")
    parser.add_argument("--robust-norm", action="store_true", help="Use robust median/MAD normalization for EEG")
    parser.add_argument("--sanity", choices=["none", "zero-eeg", "shuffle-labels"], default="none")
    parser.add_argument("--random-eeg", action="store_true", help="Replace EEG with random noise dynamically before training")
    parser.add_argument("--shuffle-labels", action="store_true", help="Shuffle attended/unattended labels in the dataset")
    parser.add_argument("--no-training", action="store_true", help="Skip training and run inference directly")
    parser.add_argument("--random-audio-pairs", action="store_true", help="Break EEG/audio correspondence during training")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json-out", type=Path, default=SUMMARY_PATH)
    parser.add_argument("--no-readme-update", action="store_true")
    args = parser.parse_args()

    ensure_output_dirs()
    set_seed(args.seed)
    device = select_device(args.device)

    notify(f"Using device: {device}")

    subject_paths = subject_files()
    if args.subject_limit is not None:
        subject_paths = subject_paths[: args.subject_limit]
        notify(f"Subject limit active: using first {len(subject_paths)} subjects")

    mapping_options = choose_best_mapping(mapping_mode=args.mapping)
    zero_inputs = args.sanity == "zero-eeg"
    shuffle_labels = args.sanity == "shuffle-labels" or args.shuffle_labels

    if args.random_eeg or shuffle_labels or args.no_training or args.random_audio_pairs:
        notify("--- SANITY TEST FRAMEWORK ACTIVE ---")
        if args.random_eeg: notify("-> Random EEG")
        if shuffle_labels: notify("-> Shuffle Labels")
        if args.no_training: notify("-> No Training")
        if args.random_audio_pairs: notify("-> Random Audio Pairing")
        notify("------------------------------------")

    subject_examples = {path: load_subject_examples(path) for path in subject_paths}
    if shuffle_labels:
        rng = np.random.default_rng(args.seed)
        shuffled_examples = {}
        labels = np.asarray([example.label for examples in subject_examples.values() for example in examples], dtype=int)
        rng.shuffle(labels)
        offset = 0
        for path, examples in subject_examples.items():
            updated = []
            for example in examples:
                updated.append(
                    example.__class__(
                        example.subject,
                        example.trial_index,
                        example.eeg,
                        example.wav_a,
                        example.wav_b,
                        int(labels[offset]),
                    )
                )
                offset += 1
            shuffled_examples[path] = updated
        subject_examples = shuffled_examples

    fold_summaries = []
    for mapping in mapping_options:
        notify(f"Evaluating mapping {mapping}")
        window_runs = run_evaluation_folds(
            subject_examples=subject_examples,
            subject_paths=subject_paths,
            evaluation_mode=args.evaluation_mode,
            subject_train_ratio=args.subject_train_ratio,
            mapping=mapping,
            args=args,
            device=device,
            zero_inputs=zero_inputs,
            notify_fn=notify,
            shuffle_labels=shuffle_labels,
            random_eeg=args.random_eeg,
            no_training=args.no_training,
            random_audio_pairs=args.random_audio_pairs,
        )

        aggregated_windows = []
        for window in args.window_seconds:
            window_scores = []
            for fold_result in window_runs:
                window_score = next(item for item in fold_result["window_runs"] if item["window_seconds"] == window)
                window_scores.extend(window_score["per_subject"])

            # Reconstruct TrialScore objects for summarization.
            scores = [
                TrialScore(
                    trial_index=item["trial_index"],
                    corr_a=item["corr_a"],
                    corr_b=item["corr_b"],
                    true_stream=item["true_stream"],
                    predicted_stream=item["predicted_stream"],
                )
                for item in window_scores
            ]
            
            # Compute per-subject accuracies
            per_subject_accs = aggregate_per_subject_accuracies(window_runs, window)
            
            aggregated_windows.append(
                {
                    "window_seconds": window,
                    "overall": summarize_trials(scores),
                    "per_subject_accuracies": per_subject_accs,
                }
            )

        best_window = max(aggregated_windows, key=lambda item: item["overall"]["trial_accuracy"])
        fold_summaries.append(
            {
                "mapping": mapping,
                "window_runs": aggregated_windows,
                "best": best_window,
            }
        )

    best_mapping_summary = max(fold_summaries, key=lambda item: item["best"]["overall"]["trial_accuracy"])
    result = {
        "mode": f"{args.objective}" if args.sanity == "none" else args.sanity,
        "evaluation_mode": args.evaluation_mode,
        "device": str(device),
        "objective": args.objective,
        "negative_mode": args.negative_mode,
        "negative_min_shift_sec": args.negative_min_shift_sec,
        "negative_max_shift_sec": args.negative_max_shift_sec,
        "channels": args.channel_ids,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "patience": args.patience,
        "window_seconds": args.window_seconds,
        "chunk_seconds": args.chunk_seconds,
        "temperature": args.temperature,
        "lag_ms": args.lag_ms,
        "lag_step_ms": args.lag_step_ms,
        "sanity": {"zero_eeg": zero_inputs, "shuffle_labels": shuffle_labels},
        "mapping_mode": args.mapping,
        "candidates": fold_summaries,
        "best": best_mapping_summary["best"],
    }

    save_json(args.json_out, result)
    notify(f"Wrote summary JSON to {args.json_out}")
    print(json.dumps(result["best"], indent=2))

    if not args.no_readme_update:
        append_readme_update(
            [
                f"Temporal CNN LOSO {args.objective} objective implemented.",
                f"Best trial accuracy={result['best']['overall']['trial_accuracy']:.4f}; balanced accuracy={result['best']['overall']['balanced_accuracy']:.4f}.",
                f"Device={device}; channels={args.channel_ids}; mapping_mode={args.mapping}; sanity={result['sanity']}; objective={args.objective}.",
            ],
            title="train_temporal_cnn_loso.py completed",
        )


if __name__ == "__main__":
    main()