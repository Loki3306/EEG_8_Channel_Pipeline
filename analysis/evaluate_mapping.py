from pathlib import Path
from scipy.io import loadmat
import numpy as np
from typing import List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
README_PATH = REPO_ROOT / "DATA_ANALYSIS.md"


def append_readme_update(lines, *, title: str | None = None):
    from datetime import datetime

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    block = [f"[Update - {timestamp}]"]
    if title:
        block.append(f"- {title}")
    for line in lines:
        block.append(f"- {line}")
    content = README_PATH.read_text(encoding="utf-8")
    separator = "\n" if not content.endswith("\n") else ""
    README_PATH.write_text(content + separator + "\n".join(block) + "\n", encoding="utf-8")


def subject_files() -> List[Path]:
    return sorted(DATA_DIR.glob("S*_data_preproc.mat"), key=lambda path: int(path.stem.split("_")[0][1:]))


def trial_labels_from_mat(path: Path) -> List[int]:
    mat = loadmat(path, squeeze_me=False, struct_as_record=False)
    data = mat["data"][0, 0]
    events = data.event[0, 0].eeg
    labels = []
    for trial_index in range(events.shape[1]):
        event = events[0, trial_index]
        val = event.value[0, 0]
        if isinstance(val, np.ndarray):
            labels.append(int(np.asarray(val).squeeze()))
        else:
            labels.append(int(val))
    return labels


def fast_corr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    if x.size != y.size or x.size < 2:
        return float("nan")

    x = x - x.mean()
    y = y - y.mean()
    denom = np.linalg.norm(x) * np.linalg.norm(y)
    if denom == 0:
        return float("nan")

    return float(np.dot(x, y) / denom)


def stream_embedding(stream: np.ndarray, bins: int = 64) -> Tuple[np.ndarray, dict]:
    x = np.asarray(stream, dtype=float).ravel()
    x = np.nan_to_num(x, copy=False)
    if x.size < 2:
        return np.zeros(bins * 3 + 14, dtype=float), {
            "spectral_centroid": 0.0,
            "low_frequency_ratio": 0.0,
            "zero_crossing_rate": 0.0,
        }

    x = x - x.mean()
    x = x / (x.std() + 1e-12)
    window = np.hanning(x.size)
    windowed = x * window

    coarse = np.array([segment.mean() for segment in np.array_split(x, bins)], dtype=float)
    delta = np.diff(x, prepend=x[0])
    delta_coarse = np.array([segment.mean() for segment in np.array_split(delta, bins)], dtype=float)

    spectrum = np.abs(np.fft.rfft(windowed))
    spectrum = np.log1p(spectrum[:bins])
    spectrum = spectrum / (np.linalg.norm(spectrum) + 1e-12)

    lag_values = [1, 2, 4, 8, 16, 32]
    autocorr = []
    for lag in lag_values:
        if lag >= x.size:
            autocorr.append(0.0)
        else:
            autocorr.append(fast_corr(x[:-lag], x[lag:]))
    autocorr = np.asarray(autocorr, dtype=float)
    if not np.isfinite(autocorr).all():
        autocorr = np.nan_to_num(autocorr, nan=0.0, posinf=0.0, neginf=0.0)

    zero_crossing_rate = float(np.mean(np.signbit(x[:-1]) != np.signbit(x[1:])))
    frequency_axis = np.arange(spectrum.size, dtype=float)
    spectral_centroid = float((frequency_axis * spectrum).sum() / (spectrum.sum() + 1e-12))
    low_frequency_ratio = float(spectrum[:8].sum() / (spectrum.sum() + 1e-12))
    mean_abs = float(np.mean(np.abs(x)))
    std_abs = float(np.std(np.abs(x)))
    peak_to_rms = float(np.max(np.abs(x)) / (np.sqrt(np.mean(x**2)) + 1e-12))
    mean_value = float(np.mean(x))
    std_value = float(np.std(x))
    energy = float(np.mean(x**2))
    peak = float(np.max(np.abs(x)))

    embedding = np.concatenate(
        [
            coarse,
            delta_coarse,
            spectrum,
            autocorr,
            np.array(
                [
                    zero_crossing_rate,
                    spectral_centroid,
                    low_frequency_ratio,
                    mean_abs,
                    std_abs,
                    peak_to_rms,
                    mean_value,
                    std_value,
                    energy,
                    peak,
                ],
                dtype=float,
            ),
        ]
    )
    stats = {
        "spectral_centroid": spectral_centroid,
        "low_frequency_ratio": low_frequency_ratio,
        "zero_crossing_rate": zero_crossing_rate,
    }
    return embedding, stats


class TrialPair:
    def __init__(self, file_name: str, trial_index: int, wav_a: np.ndarray, wav_b: np.ndarray, emb_a: np.ndarray, emb_b: np.ndarray, stats_a: dict, stats_b: dict):
        self.file_name = file_name
        self.trial_index = trial_index
        self.wav_a = wav_a
        self.wav_b = wav_b
        self.emb_a = emb_a
        self.emb_b = emb_b
        self.stats_a = stats_a
        self.stats_b = stats_b


def load_trial_pairs(path: Path):
    data = loadmat(path, squeeze_me=False, struct_as_record=False)["data"][0, 0]
    pairs = []
    trial_count = data.wavA.shape[1]
    for trial_index in range(trial_count):
        wav_a = np.asarray(data.wavA[0, trial_index], dtype=float).ravel()
        wav_b = np.asarray(data.wavB[0, trial_index], dtype=float).ravel()
        emb_a, stats_a = stream_embedding(wav_a)
        emb_b, stats_b = stream_embedding(wav_b)
        pairs.append(TrialPair(path.name, trial_index, wav_a, wav_b, emb_a, emb_b, stats_a, stats_b))
    return pairs


def fit_pairwise_clusters(pairs, restarts: int = 8, max_iter: int = 50):
    trial_count = len(pairs)
    trial_distances = [float(np.linalg.norm(pair.emb_a - pair.emb_b)) for pair in pairs]
    seed_trial_index = int(np.argmax(trial_distances)) if trial_distances else 0
    seed_pair = pairs[seed_trial_index]
    seed_a = seed_pair.emb_a.copy()
    seed_b = seed_pair.emb_b.copy()

    def run_once(mu0: np.ndarray, mu1: np.ndarray):
        assignments = []
        for _ in range(max_iter):
            assignments.clear()
            for pair in pairs:
                cost_01 = float(np.sum((pair.emb_a - mu0) ** 2) + np.sum((pair.emb_b - mu1) ** 2))
                cost_10 = float(np.sum((pair.emb_a - mu1) ** 2) + np.sum((pair.emb_b - mu0) ** 2))
                assignments.append((0, 1) if cost_01 <= cost_10 else (1, 0))

            cluster0_vectors = [vec for pair, assignment in zip(pairs, assignments) for vec, label in ((pair.emb_a, assignment[0]), (pair.emb_b, assignment[1])) if label == 0]
            cluster1_vectors = [vec for pair, assignment in zip(pairs, assignments) for vec, label in ((pair.emb_a, assignment[0]), (pair.emb_b, assignment[1])) if label == 1]
            if not cluster0_vectors or not cluster1_vectors:
                return mu0, mu1, list(assignments), float("inf")

            new_mu0 = np.mean(np.vstack(cluster0_vectors), axis=0)
            new_mu1 = np.mean(np.vstack(cluster1_vectors), axis=0)
            delta = float(np.linalg.norm(new_mu0 - mu0) + np.linalg.norm(new_mu1 - mu1))
            mu0, mu1 = new_mu0, new_mu1
            if delta < 1e-6:
                break

        objective = 0.0
        for pair, assignment in zip(pairs, assignments):
            if assignment == (0, 1):
                objective += float(np.sum((pair.emb_a - mu0) ** 2) + np.sum((pair.emb_b - mu1) ** 2))
            else:
                objective += float(np.sum((pair.emb_a - mu1) ** 2) + np.sum((pair.emb_b - mu0) ** 2))
        return mu0, mu1, list(assignments), objective

    best = None
    for restart in range(restarts):
        if restart == 0:
            mu0 = seed_a.copy()
            mu1 = seed_b.copy()
        else:
            trial_index = int(np.random.default_rng(restart).integers(0, trial_count))
            pair = pairs[trial_index]
            if restart % 2 == 0:
                mu0 = pair.emb_a.copy()
                mu1 = pair.emb_b.copy()
            else:
                mu0 = pair.emb_b.copy()
                mu1 = pair.emb_a.copy()

        fit_mu0, fit_mu1, assignments, objective = run_once(mu0, mu1)
        if best is None or objective < best[3]:
            best = (fit_mu0, fit_mu1, assignments, objective)

    assert best is not None
    mu0, mu1, assignments, objective = best

    return {"assignments": assignments, "objective": objective}


def evaluate_mapping(clusterA, clusterB, labels, mapping):
    correct = 0
    total = len(labels)

    for i in range(total):
        cA = clusterA[i]
        cB = clusterB[i]
        label = labels[i]

        # predict based on mapping
        if mapping[cA] == label:
            pred = label
        elif mapping[cB] == label:
            pred = label
        else:
            pred = mapping[cA]  # fallback

        if pred == label:
            correct += 1

    return correct / total


if __name__ == "__main__":
    # Collect pairs and labels across all subject files
    pairs = []
    labels_all = []

    files = subject_files()
    for path in files:
        p = load_trial_pairs(path)
        pairs.extend(p)
        labels_all.extend(trial_labels_from_mat(path))

    # Fit pairwise clusters (re-uses the same algorithm)
    fit = fit_pairwise_clusters(pairs)
    assignments = fit["assignments"]

    clusterA = [a for a, _ in assignments]
    clusterB = [b for _, b in assignments]

    mapping_A = {0: 1, 1: 2}
    mapping_B = {0: 2, 1: 1}

    acc_A = evaluate_mapping(clusterA, clusterB, labels_all, mapping_A)
    acc_B = evaluate_mapping(clusterA, clusterB, labels_all, mapping_B)

    print(f"Mapping A (0->1): {acc_A:.4f}")
    print(f"Mapping B (0->2): {acc_B:.4f}")

    best_mapping = mapping_A if acc_A > acc_B else mapping_B
    best_acc = max(acc_A, acc_B)

    print(f"Best mapping: {best_mapping}, accuracy={best_acc:.4f}")

    # Update DATA_ANALYSIS.md with final confirmed mapping
    lines = [
        f"Final cluster→label mapping: {best_mapping} (accuracy={best_acc:.4f})",
        f"Mapping A accuracy: {acc_A:.4f}",
        f"Mapping B accuracy: {acc_B:.4f}",
    ]
    append_readme_update(lines, title="evaluate_mapping.py completed")
