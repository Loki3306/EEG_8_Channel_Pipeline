"""
Equalized Benchmark Evaluation
================================

This is the definitive equalization test.

Goal: After per-trial RMS normalization of both audio streams,
does audio-only performance collapse to chance (~50%)?

If YES: The RMS confound was the sole shortcut. Equalized benchmarking is valid.
If NO:  Secondary acoustic confounds exist. The dataset requires deeper analysis.

The previous rms_confound_investigation.py showed a suspicious 0.0333 for the
equalized heuristic. That was because the script switched to testing
P(attended lower env_var) but the confound runs in the OPPOSITE direction —
the attended stream has HIGHER envelope variance after equalization.
This script finds ALL remaining shortcuts after equalization.

Process:
  1. Load raw wavA, wavB
  2. Apply per-trial RMS equalization: wav /= RMS(wav)
  3. Extract features from equalized streams
  4. Run 8 directional heuristics (both directions for each feature)
  5. Run LOSO logistic regression on equalized features
  6. Compare to original (non-equalized) audio-only results
  7. Report residual confound magnitude

Usage:
    python analysis/equalized_benchmark.py

On Kaggle:
    !python analysis/equalized_benchmark.py

Outputs:
    analysis/summaries/equalized_benchmark_summary.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.signal import hilbert

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from baselines.ridge_aad import load_subject_examples, subject_files

try:
    from baselines.ridge_aad import iter_leave_one_subject_out
    HAS_LOSO_ITER = True
except ImportError:
    HAS_LOSO_ITER = False

MAPPING = {1: "A", 2: "B"}
FEATURE_NAMES = ["rms", "env_variance", "spectral_centroid", "dynamic_range"]


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def raw_rms(wav: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(wav.astype(float)))))


def env_variance(wav: np.ndarray) -> float:
    env = np.abs(hilbert(wav.astype(float)))
    return float(np.var(env))


def spectral_centroid(wav: np.ndarray) -> float:
    s = np.abs(np.fft.rfft(wav.astype(float)))
    f = np.fft.rfftfreq(len(wav))
    total = s.sum()
    return float(np.dot(f, s) / total) if total > 1e-12 else 0.0


def dynamic_range(wav: np.ndarray) -> float:
    a = np.abs(wav.astype(float))
    return float(np.percentile(a, 95) - np.percentile(a, 5))


FEATURE_FNS = [raw_rms, env_variance, spectral_centroid, dynamic_range]


def equalize(wav: np.ndarray) -> np.ndarray:
    """Divide by per-signal RMS → unit-RMS waveform."""
    r = raw_rms(wav)
    return wav.astype(float) / (r + 1e-12)


def extract_features(wav: np.ndarray, apply_equalization: bool) -> np.ndarray:
    w = equalize(wav) if apply_equalization else wav.astype(float)
    return np.array([fn(w) for fn in FEATURE_FNS], dtype=float)


def attended_wav(ex) -> np.ndarray:
    return ex.wav_a if MAPPING[ex.label] == "A" else ex.wav_b


def unattended_wav(ex) -> np.ndarray:
    return ex.wav_b if MAPPING[ex.label] == "A" else ex.wav_a


# ---------------------------------------------------------------------------
# Heuristic baselines
# ---------------------------------------------------------------------------

def run_all_heuristics(examples: list, equalized: bool) -> dict[str, float]:
    """
    Test 8 directional heuristics (both directions per feature).
    Returns {heuristic_name: accuracy}.
    """
    n = len(examples)
    counts = {f"{name}_lower": 0 for name in FEATURE_NAMES}
    counts.update({f"{name}_higher": 0 for name in FEATURE_NAMES})

    for ex in examples:
        att_w = attended_wav(ex)
        una_w = unattended_wav(ex)
        feat_att = extract_features(att_w, equalized)
        feat_una = extract_features(una_w, equalized)

        for i, name in enumerate(FEATURE_NAMES):
            if feat_att[i] < feat_una[i]:
                counts[f"{name}_lower"] += 1
            else:
                counts[f"{name}_higher"] += 1

    results = {k: v / n for k, v in counts.items()}
    return results


# ---------------------------------------------------------------------------
# LOSO logistic regression
# ---------------------------------------------------------------------------

def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def logistic_fit(X: np.ndarray, y: np.ndarray, lr: float = 0.1, l2: float = 1.0, n_iter: int = 1000):
    mean_ = X.mean(axis=0)
    std_  = X.std(axis=0) + 1e-8
    Xs = (X - mean_) / std_
    n, d = Xs.shape
    w = np.zeros(d)
    b = 0.0
    for _ in range(n_iter):
        p = sigmoid(Xs @ w + b)
        e = p - y
        w -= lr * (Xs.T @ e / n + l2 * w / n)
        b -= lr * e.mean()
    return w, b, mean_, std_


def logistic_predict(X: np.ndarray, w, b, mean_, std_) -> np.ndarray:
    return (sigmoid((X - mean_) / std_ @ w + b) >= 0.5).astype(int)


def loso_logistic(subject_examples: dict, subject_paths: list, equalized: bool) -> tuple[float, list]:
    if not HAS_LOSO_ITER:
        # Fall back: manual LOSO
        paths = list(subject_paths)
        folds = [(paths[i], [p for j, p in enumerate(paths) if j != i]) for i in range(len(paths))]
    else:
        folds = list(iter_leave_one_subject_out(subject_paths))

    per_fold = []
    for held_out_path, train_paths in folds:
        held_out_key = str(held_out_path)
        train_keys = [str(p) for p in train_paths]

        train_examples = [ex for k in train_keys for ex in subject_examples.get(k, [])]
        test_examples  = subject_examples.get(held_out_key, [])
        held_out_id    = held_out_path.stem.split("_")[0]

        if not train_examples or not test_examples:
            continue

        def build_matrix(examples):
            X_rows, y_rows = [], []
            for ex in examples:
                diff = (extract_features(attended_wav(ex), equalized)
                        - extract_features(unattended_wav(ex), equalized))
                y = 1.0  # by construction: attended - unattended
                X_rows.append(diff)
                # We predict "attended = stream A" if label=1, so:
                # label=1 → y=1 (feat_A > feat_B in a dim that helps)
                # Actually we compute (att-una) and label is always "correct"
                # so y should always be 1. Use (att-una) features, label=1 always.
                y_rows.append(1.0)
            return np.array(X_rows), np.array(y_rows)

        # Better approach: use raw (feat_A - feat_B) with true label
        def build_matrix_v2(examples):
            X_rows, y_rows = [], []
            for ex in examples:
                feat_a = extract_features(ex.wav_a, equalized)
                feat_b = extract_features(ex.wav_b, equalized)
                diff = feat_a - feat_b
                label_a_attended = 1 if MAPPING[ex.label] == "A" else 0
                X_rows.append(diff)
                y_rows.append(float(label_a_attended))
            return np.array(X_rows), np.array(y_rows)

        X_train, y_train = build_matrix_v2(train_examples)
        X_test,  y_test  = build_matrix_v2(test_examples)

        w, b, mean_, std_ = logistic_fit(X_train, y_train)
        preds = logistic_predict(X_test, w, b, mean_, std_)
        fold_acc = float((preds == y_test).mean())
        per_fold.append({
            "held_out": held_out_id,
            "n_test":   len(test_examples),
            "accuracy": fold_acc,
        })

    overall = float(np.mean([f["accuracy"] for f in per_fold])) if per_fold else 0.0
    return overall, per_fold


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    from analysis._common import ensure_output_dirs, SUMMARY_DIR

    ensure_output_dirs()

    subject_paths = subject_files()
    if not subject_paths:
        raise RuntimeError("No subject files found. Check EEG_DATA_DIR.")

    print(f"[equalized-benchmark] Loading {len(subject_paths)} subjects...", flush=True)
    subject_examples = {str(p): load_subject_examples(p) for p in subject_paths}
    all_examples = [ex for exs in subject_examples.values() for ex in exs]
    print(f"[equalized-benchmark] Total trials: {len(all_examples)}\n")

    # -----------------------------------------------------------------------
    # 1. Heuristics: original vs equalized
    # -----------------------------------------------------------------------
    print("=" * 70)
    print("  PART 1 — Directional heuristics (original vs equalized audio)")
    print("=" * 70)

    orig_heuristics = run_all_heuristics(all_examples, equalized=False)
    eq_heuristics   = run_all_heuristics(all_examples, equalized=True)

    print(f"\n  {'Heuristic':<40s}  {'Original':>10}  {'Equalized':>10}  {'Drop':>8}")
    print(f"  {'-'*38:<40s}  {'--------':>10}  {'---------':>10}  {'----':>8}")

    heuristic_summary = {}
    best_orig_acc = 0.0
    best_eq_acc   = 0.0

    for name in FEATURE_NAMES:
        for direction in ["lower", "higher"]:
            key = f"{name}_{direction}"
            o = orig_heuristics[key]
            e = eq_heuristics[key]
            drop = o - e
            orig_flag = "⚠️ " if o > 0.75 else "   "
            eq_flag   = "⚠️ " if e > 0.75 else ("✅ " if e < 0.55 else "→  ")
            print(f"  {orig_flag}{'att '+direction+' '+name:<38s}  {o:>10.4f}  {eq_flag}{e:>7.4f}  {drop:>+8.4f}")
            heuristic_summary[key] = {"original": o, "equalized": e, "drop": drop}
            if o > best_orig_acc:
                best_orig_acc = o
            if e > best_eq_acc:
                best_eq_acc = e

    print(f"\n  Best original heuristic:  {best_orig_acc:.4f}")
    print(f"  Best equalized heuristic: {best_eq_acc:.4f}")

    if best_eq_acc > 0.75:
        heuristic_verdict = (
            f"🚨 SECONDARY CONFOUND — Best heuristic after equalization: {best_eq_acc:.4f}. "
            "RMS equalization alone does NOT eliminate the acoustic shortcut. "
            "Other acoustic features (envelope variance, spectral centroid, dynamic range) "
            "also discriminate attended from unattended."
        )
    elif best_eq_acc > 0.60:
        heuristic_verdict = (
            f"⚠️  RESIDUAL CONFOUND — Best heuristic after equalization: {best_eq_acc:.4f}. "
            "Mild residual shortcut remains after RMS equalization."
        )
    else:
        heuristic_verdict = (
            f"✅  CLEAN — Best heuristic after equalization: {best_eq_acc:.4f}. "
            "RMS equalization collapses audio-only heuristic performance to near chance."
        )
    print(f"\n  Verdict: {heuristic_verdict}")

    # -----------------------------------------------------------------------
    # 2. LOSO logistic regression: original vs equalized
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  PART 2 — LOSO Logistic Regression (original vs equalized audio)")
    print("=" * 70)

    print("\n  [Original audio — no equalization]")
    orig_loso_acc, orig_loso_folds = loso_logistic(subject_examples, subject_paths, equalized=False)
    for f in orig_loso_folds:
        print(f"    {f['held_out']:>5s}: {f['accuracy']:.4f}  ({f['n_test']} trials)")
    print(f"  Overall LOSO accuracy (original): {orig_loso_acc:.4f}")

    print("\n  [Equalized audio — per-trial RMS normalization]")
    eq_loso_acc, eq_loso_folds = loso_logistic(subject_examples, subject_paths, equalized=True)
    for f in eq_loso_folds:
        print(f"    {f['held_out']:>5s}: {f['accuracy']:.4f}  ({f['n_test']} trials)")
    print(f"  Overall LOSO accuracy (equalized): {eq_loso_acc:.4f}")

    loso_drop = orig_loso_acc - eq_loso_acc
    print(f"\n  LOSO drop after equalization: {loso_drop:+.4f}")

    if eq_loso_acc < 0.55:
        loso_verdict = (
            f"✅  EQUALIZATION WORKS — LOSO drops from {orig_loso_acc:.4f} to {eq_loso_acc:.4f}. "
            "After per-trial RMS equalization, audio-only performance collapses to chance. "
            "The shortcut was purely amplitude-based. "
            "Per-trial RMS equalization is a valid fix for this benchmark."
        )
    elif eq_loso_acc < 0.70:
        loso_verdict = (
            f"⚠️  PARTIAL — LOSO drops from {orig_loso_acc:.4f} to {eq_loso_acc:.4f}. "
            "RMS equalization helps but secondary confounds remain. "
            "Spectral or envelope shape differences between speakers persist."
        )
    else:
        loso_verdict = (
            f"🚨 EQUALIZATION INSUFFICIENT — LOSO remains at {eq_loso_acc:.4f} after equalization. "
            "The acoustic confound is structural and multi-dimensional. "
            "Simple per-trial RMS normalization is not sufficient to create a valid benchmark. "
            "The two competing speakers have pervasive acoustic differences beyond amplitude."
        )
    print(f"\n  Verdict: {loso_verdict}")

    # -----------------------------------------------------------------------
    # 3. Final summary and recommendations
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  FINAL ASSESSMENT")
    print("=" * 70)

    if eq_loso_acc < 0.55:
        final = (
            "✅  BENCHMARK SALVAGEABLE with per-trial RMS equalization.\n\n"
            "  Action required:\n"
            "    1. Apply wavA /= RMS(wavA), wavB /= RMS(wavB) before envelope extraction\n"
            "       in all training, evaluation, and baseline scripts.\n"
            "    2. Re-run Ridge and TCNN baselines on equalized data.\n"
            "    3. Any accuracy above 50% on equalized data is genuine AAD signal.\n\n"
            f"  Expected clean audio-only ceiling: ~{eq_loso_acc:.1%}\n"
            f"  EEG models must exceed this to be scientifically valid."
        )
    elif eq_loso_acc < 0.70:
        final = (
            f"⚠️  BENCHMARK PARTIALLY SALVAGEABLE. Equalized audio-only ceiling: {eq_loso_acc:.4f}.\n\n"
            "  Action required:\n"
            "    1. Apply per-trial RMS equalization as minimum fix.\n"
            "    2. Report equalized audio-only ({eq_loso_acc:.4f}) as the control baseline.\n"
            "    3. EEG models must significantly exceed this number to be credible.\n"
            "    4. Consider dataset replacement (KUL, MINDS) for publication."
        )
    else:
        final = (
            f"🚨  BENCHMARK NOT SALVAGEABLE by simple equalization. "
            f"Equalized audio-only ceiling: {eq_loso_acc:.4f}.\n\n"
            "  The two competing speakers have pervasive acoustic differences\n"
            "  that persist after amplitude normalization. The benchmark is structurally\n"
            "  confounded at the level of speaker identity, not just amplitude.\n\n"
            "  Required action:\n"
            "    1. Verify speaker identity (which speaker is in which slot per trial)\n"
            "    2. Use a dataset with balanced speaker characteristics\n"
            "       (KUL AAD or MINDS recommended)\n"
            "    3. OR: Use a matched-pairs evaluation where the same CLIPS are used\n"
            "       in both attend-A and attend-B conditions to cancel speaker effects."
        )

    print(f"\n  {final}\n")

    # -----------------------------------------------------------------------
    # Save
    # -----------------------------------------------------------------------
    summary = {
        "n_subjects":              len(subject_paths),
        "n_trials":                len(all_examples),
        "heuristics": {
            "best_original":       best_orig_acc,
            "best_equalized":      best_eq_acc,
            "by_feature":          heuristic_summary,
            "verdict":             heuristic_verdict,
        },
        "loso_logistic": {
            "original_accuracy":   orig_loso_acc,
            "equalized_accuracy":  eq_loso_acc,
            "drop":                loso_drop,
            "original_per_fold":   orig_loso_folds,
            "equalized_per_fold":  eq_loso_folds,
            "verdict":             loso_verdict,
        },
        "final_assessment":        final,
    }
    out_path = SUMMARY_DIR / "equalized_benchmark_summary.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"[equalized-benchmark] Summary saved to: {out_path}")


if __name__ == "__main__":
    main()
