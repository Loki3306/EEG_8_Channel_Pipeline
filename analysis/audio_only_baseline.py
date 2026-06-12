"""
Audio-Only Baseline
===================

Answers the critical question:
    How accurately can attended speaker be predicted using AUDIO FEATURES ALONE,
    with NO EEG input?

If this accuracy is significantly above 50%, the acoustic bias in the dataset
is strong enough to be exploited by any model — including EEG models — as a
shortcut. All reported EEG model accuracies must be compared against this.

This script runs two baselines under the EXACT SAME LOSO protocol as the EEG models:

    1. HEURISTIC BASELINES (no training)
       Apply a fixed rule to each trial and measure accuracy across all subjects.
       Four rules, one per audio feature:
         - Rule A: "Lower RMS = attended"
         - Rule B: "Higher envelope variance = attended"
         - Rule C: "Lower spectral centroid = attended"
         - Rule D: "Higher dynamic range = attended"
       (Direction of each rule derived from the acoustic_bias.py analysis.)

    2. LOSO LOGISTIC REGRESSION (trained)
       For each fold:
         - Extract audio feature vector for (A, B) per training trial.
         - Train logistic regression on [feature_A - feature_B] → label.
         - Evaluate on held-out subject.
       This gives the upper bound of what is achievable from audio statistics alone
       under a fair cross-subject evaluation.

Usage:
    python analysis/audio_only_baseline.py

On Kaggle:
    !python analysis/audio_only_baseline.py

Outputs:
    analysis/summaries/audio_only_baseline_summary.json
    Full printed report to stdout

Interpretation:
    ~50%  → acoustic features carry no predictive information. Acoustic bias
             analysis was misleading. EEG results can be trusted at face value.

    ~55%  → moderate acoustic confound. Reconstruction TCNN at 58% contains
             only ~3% genuine EEG signal. Interpret with caution.

    ~80%+ → the evaluation protocol is fundamentally compromised. Every model
             comparison in this project is suspect until the dataset is balanced
             or the shortcut is explicitly controlled for.
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.signal import hilbert

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from baselines.ridge_aad import iter_leave_one_subject_out, load_subject_examples, subject_files


# ---------------------------------------------------------------------------
# Audio feature extraction (same as acoustic_bias.py)
# ---------------------------------------------------------------------------

def rms_energy(wav: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(wav.astype(float)))))


def envelope_variance(wav: np.ndarray) -> float:
    env = np.abs(hilbert(wav.astype(float)))
    return float(np.var(env))


def spectral_centroid(wav: np.ndarray, fs: int = 8000) -> float:
    spectrum = np.abs(np.fft.rfft(wav.astype(float)))
    freqs = np.fft.rfftfreq(len(wav), d=1.0 / fs)
    total = spectrum.sum()
    if total < 1e-12:
        return 0.0
    return float(np.dot(freqs, spectrum) / total) / (fs / 2.0)


def dynamic_range(wav: np.ndarray) -> float:
    abs_wav = np.abs(wav.astype(float))
    return float(np.percentile(abs_wav, 95) - np.percentile(abs_wav, 5))


def extract_features(wav: np.ndarray, fs: int = 8000) -> np.ndarray:
    """Return [rms, env_var, spectral_centroid, dynamic_range]."""
    return np.array([
        rms_energy(wav),
        envelope_variance(wav),
        spectral_centroid(wav, fs=fs),
        dynamic_range(wav),
    ], dtype=float)


FEATURE_NAMES = ["rms", "env_var", "spectral_centroid", "dynamic_range"]


# ---------------------------------------------------------------------------
# Heuristic baselines (no training)
# ---------------------------------------------------------------------------
# Each heuristic is: given features for A and B, predict which is attended.
# Direction comes from acoustic_bias.py: if attended has LOWER RMS, the rule is
# "predict the stream with lower RMS as attended."  We test both directions for
# each feature and report the better one honestly with its direction.

def heuristic_accuracy(examples, mapping: dict[int, str], feature_idx: int, predict_lower: bool, fs: int) -> float:
    """
    Apply a simple threshold-free heuristic: predict stream A as attended
    iff feature_idx(A) < feature_idx(B) (or > if predict_lower=False).
    """
    correct = 0
    total = 0
    for ex in examples:
        feat_a = extract_features(ex.wav_a, fs=fs)[feature_idx]
        feat_b = extract_features(ex.wav_b, fs=fs)[feature_idx]
        if predict_lower:
            predicted_stream = "A" if feat_a < feat_b else "B"
        else:
            predicted_stream = "A" if feat_a > feat_b else "B"
        true_stream = mapping[ex.label]
        if predicted_stream == true_stream:
            correct += 1
        total += 1
    return correct / total if total > 0 else 0.5


# ---------------------------------------------------------------------------
# LOSO logistic regression (trained)
# ---------------------------------------------------------------------------

def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


class LogisticRidgeClassifier:
    """
    Pure numpy binary logistic regression with L2 regularisation.
    Input: difference vector (feat_A - feat_B).
    Label: 1 if A is attended, 0 if B is attended.
    """

    def __init__(self, lr: float = 0.1, l2: float = 1.0, n_iter: int = 500) -> None:
        self.lr = lr
        self.l2 = l2
        self.n_iter = n_iter
        self.w: np.ndarray | None = None
        self.b: float = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        n, d = X.shape
        # Standardise
        self.mean_ = X.mean(axis=0)
        self.std_  = X.std(axis=0) + 1e-8
        X = (X - self.mean_) / self.std_

        self.w = np.zeros(d, dtype=float)
        self.b = 0.0

        for _ in range(self.n_iter):
            logits = X @ self.w + self.b
            probs  = sigmoid(logits)
            err    = probs - y
            grad_w = X.T @ err / n + self.l2 * self.w / n
            grad_b = err.mean()
            self.w -= self.lr * grad_w
            self.b -= self.lr * grad_b

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = (X - self.mean_) / self.std_
        return (sigmoid(X @ self.w + self.b) >= 0.5).astype(int)


def loso_logistic_regression(
    subject_examples: dict[str, list],
    subject_paths: list[Path],
    mapping: dict[int, str],
    fs: int = 8000,
) -> tuple[float, list[dict]]:
    """
    LOSO logistic regression using audio difference features.
    Returns (overall_accuracy, per_fold_results).
    """
    per_fold = []

    for fold_idx, (held_out_path, train_paths) in enumerate(iter_leave_one_subject_out(subject_paths), start=1):
        held_out_key = str(held_out_path)
        train_keys   = [str(p) for p in train_paths]

        train_examples = [ex for k in train_keys for ex in subject_examples[k]]
        test_examples  = subject_examples[held_out_key]
        held_out_id    = held_out_path.stem.split("_")[0]

        if len(train_examples) == 0 or len(test_examples) == 0:
            continue

        # Build training matrix: diff = feat_A - feat_B; label = 1 if A attended
        def build_matrix(examples):
            X_rows, y_rows = [], []
            for ex in examples:
                diff = extract_features(ex.wav_a, fs=fs) - extract_features(ex.wav_b, fs=fs)
                label_a_attended = 1 if mapping[ex.label] == "A" else 0
                X_rows.append(diff)
                y_rows.append(label_a_attended)
            return np.array(X_rows, dtype=float), np.array(y_rows, dtype=float)

        X_train, y_train = build_matrix(train_examples)
        X_test,  y_test  = build_matrix(test_examples)

        clf = LogisticRidgeClassifier(lr=0.1, l2=1.0, n_iter=1000)
        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)

        fold_acc = float((preds == y_test).mean())
        per_fold.append({
            "fold":           fold_idx,
            "held_out":       held_out_id,
            "n_test_trials":  len(test_examples),
            "accuracy":       fold_acc,
            "weights":        clf.w.tolist() if clf.w is not None else [],
        })
        print(f"  Fold {fold_idx:2d}: held out {held_out_id:>5s}  |  test accuracy = {fold_acc:.4f}  ({len(test_examples)} trials)")

    all_accs = [f["accuracy"] for f in per_fold]
    overall  = float(np.mean(all_accs)) if all_accs else 0.0
    return overall, per_fold


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    from analysis._common import ensure_output_dirs, SUMMARY_DIR

    ensure_output_dirs()

    mapping = {1: "A", 2: "B"}   # match --mapping A-B default
    fs = 8000                     # audio sample rate in the .mat files

    subject_paths = subject_files()
    if not subject_paths:
        raise RuntimeError("No subject files found. Check EEG_DATA_DIR.")

    print(f"[audio-only] Loading {len(subject_paths)} subjects...", flush=True)
    subject_examples: dict[str, list] = {
        str(p): load_subject_examples(p) for p in subject_paths
    }
    all_examples = [ex for exs in subject_examples.values() for ex in exs]
    print(f"[audio-only] Total trials: {len(all_examples)}\n")

    # -----------------------------------------------------------------------
    # 1.  Heuristic baselines
    # -----------------------------------------------------------------------
    print("=" * 60)
    print("  HEURISTIC BASELINES (no training, no EEG)")
    print("=" * 60)

    heuristic_results = {}

    heuristics = [
        # (name, feature_idx, predict_lower, rationale)
        ("Lower RMS = attended",          0, True,  "Attended speech has lower RMS?"),
        ("Higher RMS = attended",         0, False, "Attended speech has higher RMS?"),
        ("Lower env_var = attended",      1, True,  "Attended speech has lower envelope variance?"),
        ("Higher env_var = attended",     1, False, "Attended speech has higher envelope variance?"),
        ("Lower spectral_cent = attended",2, True,  "Attended speech has lower spectral centroid?"),
        ("Higher spectral_cent = attended",2, False,"Attended speech has higher spectral centroid?"),
        ("Lower dyn_range = attended",    3, True,  "Attended speech has lower dynamic range?"),
        ("Higher dyn_range = attended",   3, False, "Attended speech has higher dynamic range?"),
    ]

    for name, feat_idx, pred_lower, rationale in heuristics:
        acc = heuristic_accuracy(all_examples, mapping, feat_idx, pred_lower, fs=fs)
        flag = "⚠️ " if acc > 0.55 else ("✅ " if acc < 0.52 else "→  ")
        print(f"  {flag}{name:<40s}  acc = {acc:.4f}")
        heuristic_results[name] = float(acc)

    best_heuristic_acc = max(heuristic_results.values())
    best_heuristic_name = max(heuristic_results, key=heuristic_results.get)

    print(f"\n  Best heuristic: [{best_heuristic_name}] acc = {best_heuristic_acc:.4f}")

    # -----------------------------------------------------------------------
    # 2.  LOSO logistic regression
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  LOSO LOGISTIC REGRESSION (trained, no EEG, 4 audio features)")
    print("=" * 60)
    loso_acc, loso_per_fold = loso_logistic_regression(
        subject_examples, subject_paths, mapping, fs=fs
    )
    print(f"\n  LOSO Logistic Regression accuracy: {loso_acc:.4f}")

    # Print mean learned weights (across folds' fold weights for interpretability)
    if loso_per_fold and loso_per_fold[0]["weights"]:
        weights_mat = np.array([f["weights"] for f in loso_per_fold])
        mean_w = weights_mat.mean(axis=0)
        print("\n  Mean learned weights (feat_A - feat_B):")
        for fname, w in zip(FEATURE_NAMES, mean_w):
            direction = "→ higher in attended" if w > 0 else "→ lower in attended"
            print(f"    {fname:<22s}  w = {w:+.4f}  ({direction})")

    # -----------------------------------------------------------------------
    # 3.  Verdict
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  VERDICT")
    print("=" * 60)
    print(f"\n  Best heuristic accuracy:    {best_heuristic_acc:.4f}")
    print(f"  LOSO logistic regression:   {loso_acc:.4f}")
    print(f"\n  Reference EEG baselines:")
    print(f"    Ridge (8ch):              ~0.560")
    print(f"    TCNN Reconstruction (8ch):~0.580")

    if loso_acc > 0.70:
        verdict = "🚨 CRITICAL: Audio-only model greatly exceeds EEG models. The evaluation protocol is compromised."
    elif loso_acc > 0.58:
        verdict = "🚨 SEVERE: Audio-only model matches or beats EEG models. EEG contributes little beyond acoustic shortcuts."
    elif loso_acc > 0.54:
        verdict = "⚠️  WARNING: Audio-only model is competitive. EEG results must be reported relative to this audio-only ceiling."
    elif loso_acc > 0.51:
        verdict = "→  CAUTION: Mild acoustic confound. EEG models have a genuine, but modest, advantage. Report audio-only as a control."
    else:
        verdict = "✅  CLEAR: Audio-only near chance. Acoustic bias analysis did not translate to a usable shortcut. EEG results can be trusted."

    print(f"\n  {verdict}\n")

    # -----------------------------------------------------------------------
    # 4.  Save
    # -----------------------------------------------------------------------
    summary = {
        "n_subjects":                 len(subject_paths),
        "n_trials":                   len(all_examples),
        "heuristic_baselines":        heuristic_results,
        "best_heuristic_accuracy":    best_heuristic_acc,
        "best_heuristic_name":        best_heuristic_name,
        "loso_logistic_regression":   {"overall_accuracy": loso_acc, "per_fold": loso_per_fold},
        "reference_eeg_baselines":    {"ridge_8ch": 0.560, "tcnn_recon_8ch": 0.580},
        "verdict":                    verdict,
    }
    out_path = SUMMARY_DIR / "audio_only_baseline_summary.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"[audio-only] Summary saved to: {out_path}")


if __name__ == "__main__":
    main()
