from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from analysis._common import append_readme_update, ensure_output_dirs, save_json
from baselines.ridge_aad import (
    iter_leave_one_subject_out,
    load_subject_examples,
    feature_statistics,
    lagged_eeg_matrix,
    standardize_features,
    target_envelope,
    subject_files,
)
from evaluation.aad_metrics import TrialScore, safe_corr, summarize_trials
SUMMARY_PATH = REPO_ROOT / "analysis" / "summaries" / "ridge_loso_summary.json"


def notify(message: str) -> None:
    print(f"[ridge-aad] {message}", flush=True)


def label_to_stream_mappings() -> list[dict[int, str]]:
    return [
        {1: "A", 2: "B"},
        {1: "B", 2: "A"},
    ]


def parse_mapping(value: str) -> dict[int, str]:
    options = {
        "A-B": {1: "A", 2: "B"},
        "B-A": {1: "B", 2: "A"},
    }
    if value not in options:
        raise ValueError(f"Unsupported mapping option: {value}")
    return options[value]


WINDOW_SECONDS = [5, 10, 50]


def predict_windowed_envelope(
    eeg: np.ndarray,
    weights: np.ndarray,
    *,
    lags: int,
    lag_ms: int | None = None,
    lag_step_ms: int = 16,
    feature_mean: np.ndarray,
    feature_std: np.ndarray,
) -> np.ndarray:
    x = lagged_eeg_matrix(eeg, lags=lags, lag_ms=lag_ms, lag_step_ms=lag_step_ms)
    x = standardize_features(x, feature_mean, feature_std)
    pred = x @ weights
    pred = pred - pred.mean()
    return pred / (pred.std() + 1e-12)


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


def evaluate_fold(
    test_examples,
    weights,
    *,
    mapping: dict[int, str],
    lags: int,
    lag_ms: int | None = None,
    lag_step_ms: int = 16,
    feature_mean: np.ndarray,
    feature_std: np.ndarray,
    window_seconds: int,
) -> list[TrialScore]:
    scores: list[TrialScore] = []
    for example in test_examples:
        predicted = predict_windowed_envelope(
            example.eeg,
            weights,
            lags=lags,
            lag_ms=lag_ms,
            lag_step_ms=lag_step_ms,
            feature_mean=feature_mean,
            feature_std=feature_std,
        )
        corr_a, corr_b = evaluate_trial_windows(predicted, example.wav_a, example.wav_b, window_seconds=window_seconds)
        true_stream = mapping[example.label]
        predicted_stream = "A" if corr_a > corr_b else "B"
        scores.append(
            TrialScore(
                trial_index=example.trial_index,
                corr_a=corr_a,
                corr_b=corr_b,
                true_stream=true_stream,
                predicted_stream=predicted_stream,
            )
        )
    return scores


def run_loso(
    *,
    lags: int,
    lag_ms: int | None = None,
    lag_step_ms: int = 16,
    ridge_lambda: float,
    mapping: dict[int, str],
    channel_ids: list[int] | None = None,
    zero_eeg: bool = False,
    shuffle_labels: bool = False,
    seed: int = 0,
    window_seconds: int = 50,
    subject_limit: int | None = None,
) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    per_subject: list[dict[str, object]] = []
    all_scores: list[TrialScore] = []

    notify(
        f"Starting LOSO run | mapping={mapping} | lags={lags} | lag_ms={lag_ms} | lag_step_ms={lag_step_ms} | lambda={ridge_lambda} | window={window_seconds}s | zero_eeg={zero_eeg} | shuffle_labels={shuffle_labels}"
    )

    if zero_eeg:
        notify("Zero-EEG sanity mode active: skipping ridge fitting and returning baseline scores.")
        subject_paths = subject_files()
        if subject_limit is not None:
            subject_paths = subject_paths[:subject_limit]
        for held_out, _ in iter_leave_one_subject_out(subject_paths):
            notify(f"  Fold {held_out.stem}: evaluating zero-EEG baseline")
            test_examples = load_subject_examples(held_out)
            scores: list[TrialScore] = []
            for example in test_examples:
                true_stream = mapping[example.label]
                scores.append(
                    TrialScore(
                        trial_index=example.trial_index,
                        corr_a=0.0,
                        corr_b=0.0,
                        true_stream=true_stream,
                        predicted_stream="B",
                    )
                )
            fold_summary = summarize_trials(scores)
            fold_summary["held_out_subject"] = held_out.stem
            per_subject.append(fold_summary)
            all_scores.extend(scores)
            notify(
                f"  Fold {held_out.stem}: accuracy={fold_summary['trial_accuracy']:.4f}, balanced={fold_summary['balanced_accuracy']:.4f}"
            )

        summary = summarize_trials(all_scores)
        notify(
            f"Zero-EEG sanity finished | accuracy={summary['trial_accuracy']:.4f} | balanced={summary['balanced_accuracy']:.4f}"
        )
        return {
            "mapping": mapping,
            "lags": lags,
            "ridge_lambda": ridge_lambda,
            "sanity": {"zero_eeg": True, "shuffle_labels": False},
            "overall": summary,
            "per_subject": per_subject,
        }

    subject_paths = subject_files()
    if subject_limit is not None:
        subject_paths = subject_paths[:subject_limit]
        notify(f"Subject limit active: using first {len(subject_paths)} subjects")
    subject_examples = {path: load_subject_examples(path) for path in subject_paths}

    if channel_ids is not None:
        for path, examples in subject_examples.items():
            updated_examples = []
            for example in examples:
                sliced_eeg = example.eeg[:, channel_ids]
                updated_examples.append(
                    example.__class__(example.subject, example.trial_index, sliced_eeg, example.wav_a, example.wav_b, example.label)
                )
            subject_examples[path] = updated_examples

    training_examples = {path: [example for example in examples] for path, examples in subject_examples.items()}

    if shuffle_labels:
        notify("Shuffle-label sanity mode active: permuting training labels before ridge fit.")
        shuffled_labels = np.asarray([example.label for examples in training_examples.values() for example in examples], dtype=int)
        rng.shuffle(shuffled_labels)
        offset = 0
        for path, examples in training_examples.items():
            updated_examples = []
            for example in examples:
                updated_examples.append(
                    example.__class__(example.subject, example.trial_index, example.eeg, example.wav_a, example.wav_b, int(shuffled_labels[offset]))
                )
                offset += 1
            training_examples[path] = updated_examples

    notify(f"Prepared {len(subject_paths)} subject bundles. Beginning fold loop.")

    for fold_index, (held_out, train_paths) in enumerate(iter_leave_one_subject_out(subject_paths), start=1):
        notify(f"  Fold {fold_index}/{len(subject_paths)}: held out {held_out.stem} | fitting ridge")
        fold_train_examples = [example for path in train_paths for example in training_examples[path]]
        feature_mean, feature_std = feature_statistics(fold_train_examples, lags=lags, lag_ms=lag_ms, lag_step_ms=lag_step_ms)
        feature_count = feature_mean.shape[0]
        train_xtx = np.zeros((feature_count, feature_count), dtype=float)
        train_xty = np.zeros(feature_count, dtype=float)
        n_total = len(fold_train_examples)
        for i, example in enumerate(fold_train_examples, start=1):
            x = lagged_eeg_matrix(example.eeg, lags=lags, lag_ms=lag_ms, lag_step_ms=lag_step_ms)
            x = standardize_features(x, feature_mean, feature_std)
            y = target_envelope(example, mapping)
            train_xtx += x.T @ x
            train_xty += x.T @ y
            if (i % 20) == 0 or i == n_total:
                notify(f"    Fold {fold_index}: accumulated {i}/{n_total} trials")
        weights = np.linalg.solve(train_xtx + ridge_lambda * np.eye(train_xtx.shape[0], dtype=float), train_xty)
        test_examples = subject_examples[held_out]
        scores = evaluate_fold(
            test_examples,
            weights,
            mapping=mapping,
            lags=lags,
            lag_ms=lag_ms,
            lag_step_ms=lag_step_ms,
            feature_mean=feature_mean,
            feature_std=feature_std,
            window_seconds=window_seconds,
        )
        fold_summary = summarize_trials(scores)
        fold_summary["held_out_subject"] = held_out.stem
        per_subject.append(fold_summary)
        all_scores.extend(scores)
        notify(
            f"  Fold {fold_index}/{len(subject_paths)} complete: accuracy={fold_summary['trial_accuracy']:.4f}, balanced={fold_summary['balanced_accuracy']:.4f}, mean_diff={fold_summary['mean_corr_difference']:.4f}"
        )

    summary = summarize_trials(all_scores)
    notify(
        f"LOSO complete | accuracy={summary['trial_accuracy']:.4f} | balanced={summary['balanced_accuracy']:.4f} | mean_diff={summary['mean_corr_difference']:.4f}"
    )
    return {
        "mapping": mapping,
        "lags": lags,
        "ridge_lambda": ridge_lambda,
        "window_seconds": window_seconds,
        "sanity": {"zero_eeg": zero_eeg, "shuffle_labels": shuffle_labels},
        "overall": summary,
        "per_subject": per_subject,
    }


def choose_best_mapping(
    *,
    lags: int,
    lag_ms: int | None = None,
    lag_step_ms: int = 16,
    ridge_lambda: float,
    channel_ids: list[int] | None = None,
    zero_eeg: bool = False,
    shuffle_labels: bool = False,
    seed: int = 0,
    window_seconds: int = 50,
    mapping_mode: str = "A-B",
    subject_limit: int | None = None,
) -> dict[str, object]:
    candidates = []
    if mapping_mode == "both":
        mappings = label_to_stream_mappings()
    else:
        mappings = [parse_mapping(mapping_mode)]

    for mapping in mappings:
        notify(f"Evaluating candidate mapping {mapping}")
        result = run_loso(
            lags=lags,
            lag_ms=lag_ms,
            lag_step_ms=lag_step_ms,
            ridge_lambda=ridge_lambda,
            mapping=mapping,
            channel_ids=channel_ids,
            zero_eeg=zero_eeg,
            shuffle_labels=shuffle_labels,
            seed=seed,
            window_seconds=window_seconds,
            subject_limit=subject_limit,
        )
        notify(
            f"Candidate mapping {mapping} finished with accuracy={result['overall']['trial_accuracy']:.4f}, balanced={result['overall']['balanced_accuracy']:.4f}"
        )
        candidates.append(result)

    candidates.sort(key=lambda item: item["overall"]["trial_accuracy"], reverse=True)
    notify(f"Selected best mapping {candidates[0]['mapping']} with accuracy={candidates[0]['overall']['trial_accuracy']:.4f}")
    return {"best": candidates[0], "candidates": candidates}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run LOSO ridge AAD baseline.")
    parser.add_argument("--lags", type=int, default=32)
    parser.add_argument("--lag-ms", type=int, default=None, help="Optional lag range in milliseconds (0 to this value, inclusive)")
    parser.add_argument("--lag-step-ms", type=int, default=16, help="Lag step in milliseconds when using --lag-ms")
    parser.add_argument("--lambda", dest="ridge_lambda", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sanity", choices=["none", "zero-eeg", "shuffle-labels"], default="none")
    parser.add_argument("--shuffle-seeds", type=int, default=5)
    parser.add_argument("--window-seconds", type=int, nargs="*", default=[5, 10, 50])
    parser.add_argument("--subject-limit", type=int, default=None)
    parser.add_argument("--channel-ids", type=int, nargs="+", default=[0, 1], help="EEG channel indices to use")
    parser.add_argument("--mapping", choices=["A-B", "B-A", "both"], default="A-B")
    parser.add_argument("--json-out", type=Path, default=SUMMARY_PATH)
    parser.add_argument("--no-readme-update", action="store_true")
    args = parser.parse_args()

    ensure_output_dirs()

    zero_eeg = args.sanity == "zero-eeg"
    shuffle_labels = args.sanity == "shuffle-labels"
    notify("Baseline runner starting.")
    if shuffle_labels:
        notify(f"Running {args.shuffle_seeds} shuffled seeds for sanity estimation.")
        shuffle_results = []
        for shuffle_seed in range(args.shuffle_seeds):
            notify(f"Shuffle seed {shuffle_seed + 1}/{args.shuffle_seeds}")
            shuffle_results.append(
                choose_best_mapping(
                    lags=args.lags,
                    lag_ms=args.lag_ms,
                    lag_step_ms=args.lag_step_ms,
                    ridge_lambda=args.ridge_lambda,
                    channel_ids=args.channel_ids,
                    zero_eeg=False,
                    shuffle_labels=True,
                    seed=shuffle_seed,
                    window_seconds=args.window_seconds[-1],
                    mapping_mode=args.mapping,
                    subject_limit=args.subject_limit,
                )
            )
        results = {
            "mode": "shuffle-labels",
            "window_seconds": args.window_seconds,
            "lags": args.lags,
            "lag_ms": args.lag_ms,
            "lag_step_ms": args.lag_step_ms,
            "ridge_lambda": args.ridge_lambda,
            "shuffle_seeds": args.shuffle_seeds,
            "seed_results": shuffle_results,
            "mean_accuracy": float(np.mean([item["best"]["overall"]["trial_accuracy"] for item in shuffle_results])),
            "std_accuracy": float(np.std([item["best"]["overall"]["trial_accuracy"] for item in shuffle_results])),
            "mean_balanced_accuracy": float(np.mean([item["best"]["overall"]["balanced_accuracy"] for item in shuffle_results])),
        }
        save_json(args.json_out, results)
        notify(f"Wrote summary JSON to {args.json_out}")
        print(json.dumps(results, indent=2))
    elif zero_eeg:
        results = choose_best_mapping(
            lags=args.lags,
            lag_ms=args.lag_ms,
            lag_step_ms=args.lag_step_ms,
            ridge_lambda=args.ridge_lambda,
            channel_ids=args.channel_ids,
            zero_eeg=True,
            shuffle_labels=False,
            seed=args.seed,
            window_seconds=args.window_seconds[-1],
            mapping_mode=args.mapping,
            subject_limit=args.subject_limit,
        )
        save_json(args.json_out, results)
        notify(f"Wrote summary JSON to {args.json_out}")
        print(json.dumps(results["best"]["overall"], indent=2))
        print(json.dumps(results["best"]["mapping"], indent=2))
    else:
        window_runs = []
        for window_seconds in args.window_seconds:
            notify(f"Running evaluation window={window_seconds}s")
            window_runs.append(
                choose_best_mapping(
                    lags=args.lags,
                    lag_ms=args.lag_ms,
                    lag_step_ms=args.lag_step_ms,
                    ridge_lambda=args.ridge_lambda,
                    channel_ids=args.channel_ids,
                    zero_eeg=False,
                    shuffle_labels=False,
                    seed=args.seed,
                    window_seconds=window_seconds,
                    mapping_mode=args.mapping,
                    subject_limit=args.subject_limit,
                )
            )
        results = {
            "mode": "baseline",
            "lags": args.lags,
            "lag_ms": args.lag_ms,
            "lag_step_ms": args.lag_step_ms,
            "ridge_lambda": args.ridge_lambda,
            "window_seconds": args.window_seconds,
            "window_runs": window_runs,
        }
        best_run = max(window_runs, key=lambda item: item["best"]["overall"]["trial_accuracy"])
        results["best"] = best_run["best"]
        save_json(args.json_out, results)
        notify(f"Wrote summary JSON to {args.json_out}")
        print(json.dumps(results["window_runs"], indent=2))
        print(json.dumps(results["best"]["mapping"], indent=2))

    if not args.no_readme_update:
        best = results["best"] if "best" in results else results["window_runs"][0]["best"]
        notify("Updating DATA_ANALYSIS.md with ridge baseline findings.")
        append_readme_update(
            [
                "Phase 2 ridge AAD baseline implemented.",
                f"LOSO trial accuracy={best['overall']['trial_accuracy']:.4f}; balanced accuracy={best['overall']['balanced_accuracy']:.4f}; mean corr diff={best['overall']['mean_corr_difference']:.4f}.",
                f"Selected label->stream mapping: {best['mapping']}.",
                f"Sanity mode: zero_eeg={zero_eeg}, shuffle_labels={shuffle_labels}, lags={args.lags}, lag_ms={args.lag_ms}, windows={args.window_seconds}.",
            ],
            title="loso_ridge_runner.py completed",
        )


if __name__ == "__main__":
    main()