from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from _common import (
	SUMMARY_DIR,
	append_readme_update,
	ensure_output_dirs,
	load_subject_data,
	save_json,
	subject_files,
	trial_labels,
)


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


def finite_or_neg_inf(value: float) -> float:
	return value if np.isfinite(value) else float("-inf")


def analyze_subject(path: Path) -> dict[str, object]:
	print(f"Loading {path.name}", flush=True)

	data = load_subject_data(path)
	labels = trial_labels(data)
	eeg_trials = data.eeg
	wav_a_trials = data.wavA
	wav_b_trials = data.wavB

	trial_rows: list[dict[str, object]] = []
	label1_total = 0
	label1_match = 0
	label2_total = 0
	label2_match = 0
	best_lag_limit = 20

	num_trials = eeg_trials.shape[1]

	for trial_index in range(num_trials):
		if trial_index % 10 == 0:
			print(f"  Trial {trial_index}/{num_trials}", flush=True)

		eeg = np.asarray(eeg_trials[0, trial_index], dtype=float)
		wav_a = np.asarray(wav_a_trials[0, trial_index], dtype=float).ravel()
		wav_b = np.asarray(wav_b_trials[0, trial_index], dtype=float).ravel()

		best_a = float("-inf")
		best_b = float("-inf")
		best_a_ch = -1
		best_b_ch = -1
		best_a_lag = -1
		best_b_lag = -1

		for channel_index in range(eeg.shape[1]):
			eeg_ch = eeg[:, channel_index]

			for lag in range(best_lag_limit):
				if lag == 0:
					x = eeg_ch
					a = wav_a
					b = wav_b
				else:
					x = eeg_ch[lag:]
					a = wav_a[:-lag]
					b = wav_b[:-lag]

				if x.size < 2 or a.size != x.size or b.size != x.size:
					continue

				corr_a = fast_corr(x, a)
				corr_b = fast_corr(x, b)

				if corr_a > best_a:
					best_a = corr_a
					best_a_ch = channel_index
					best_a_lag = lag

				if corr_b > best_b:
					best_b = corr_b
					best_b_ch = channel_index
					best_b_lag = lag

		label = int(labels[trial_index])

		corr_a_gt_b = bool(best_a > best_b)
		corr_b_gt_a = bool(best_b > best_a)
		pred = 1 if best_a > best_b else 2

		if label == 1:
			label1_total += 1
			label1_match += int(corr_a_gt_b)
		elif label == 2:
			label2_total += 1
			label2_match += int(corr_b_gt_a)

		trial_rows.append(
			{
				"trial": trial_index,
				"label": label,
				"bestA": best_a,
				"bestB": best_b,
				"bestA_channel": best_a_ch,
				"bestB_channel": best_b_ch,
				"bestA_lag": best_a_lag,
				"bestB_lag": best_b_lag,
				"pred": pred,
				"corrA_gt_corrB": corr_a_gt_b,
				"corrB_gt_corrA": corr_b_gt_a,
			}
		)

	label1_rate = float(label1_match / label1_total) if label1_total else float("nan")
	label2_rate = float(label2_match / label2_total) if label2_total else float("nan")
	overall_trials = label1_total + label2_total
	overall_matches = label1_match + label2_match
	overall_rate = float(overall_matches / overall_trials) if overall_trials else float("nan")

	return {
		"file": path.name,
		"trial_count": int(num_trials),
		"label1_total": label1_total,
		"label1_match": label1_match,
		"label1_rate": label1_rate,
		"label2_total": label2_total,
		"label2_match": label2_match,
		"label2_rate": label2_rate,
		"overall_trials": overall_trials,
		"overall_matches": overall_matches,
		"overall_rate": overall_rate,
		"best_lag_limit": best_lag_limit,
		"trial_rows": trial_rows,
	}


def main() -> None:
	parser = argparse.ArgumentParser(description="Validate label mapping using correlation.")
	parser.add_argument("--json-out", type=Path, default=SUMMARY_DIR / "label_validation.json")
	parser.add_argument("--threshold", type=float, default=0.8)
	parser.add_argument("--limit", type=int, default=None)
	parser.add_argument("--no-readme-update", action="store_true")
	args = parser.parse_args()

	ensure_output_dirs()

	files = list(subject_files())
	if args.limit is not None:
		files = files[: args.limit]

	subjects: list[dict[str, object]] = []

	for i, path in enumerate(files):
		print(f"[{i + 1}/{len(files)}] Processing {path.name}...", flush=True)
		subjects.append(analyze_subject(path))

	overall_label1_total = sum(item["label1_total"] for item in subjects)
	overall_label1_match = sum(item["label1_match"] for item in subjects)
	overall_label2_total = sum(item["label2_total"] for item in subjects)
	overall_label2_match = sum(item["label2_match"] for item in subjects)
	overall_trials = sum(item["overall_trials"] for item in subjects)
	overall_matches = sum(item["overall_matches"] for item in subjects)

	overall_label1_rate = float(overall_label1_match / overall_label1_total) if overall_label1_total else float("nan")
	overall_label2_rate = float(overall_label2_match / overall_label2_total) if overall_label2_total else float("nan")
	overall_rate = float(overall_matches / overall_trials) if overall_trials else float("nan")

	payload = {
		"threshold": args.threshold,
		"overall": {
			"label1_rate": overall_label1_rate,
			"label2_rate": overall_label2_rate,
			"overall_rate": overall_rate,
			"label1_total": overall_label1_total,
			"label2_total": overall_label2_total,
			"overall_trials": overall_trials,
		},
		"subjects": subjects,
	}

	save_json(args.json_out, payload)

	print(json.dumps(payload["overall"], indent=2))

	for item in subjects:
		print(
			f"{item['file']}: label1={item['label1_rate']:.3f} ({item['label1_match']}/{item['label1_total']}), "
			f"label2={item['label2_rate']:.3f} ({item['label2_match']}/{item['label2_total']}), "
			f"overall={item['overall_rate']:.3f}"
		)

	mapping_status = "confirmed" if np.isfinite(overall_rate) and overall_rate >= args.threshold else "unclear"

	if not args.no_readme_update:
		append_readme_update(
			[
				f"Validated label-correlation rule for {len(subjects)} subject file(s) and wrote {args.json_out.name}.",
				f"Overall consistency={overall_rate:.3f}; label=1 rate={overall_label1_rate:.3f}; label=2 rate={overall_label2_rate:.3f}.",
				f"Decision rule threshold={args.threshold:.2f} -> mapping {mapping_status}.",
			],
			title="validate_labels.py completed",
		)


if __name__ == "__main__":
	main()