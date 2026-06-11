from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.io import loadmat

from _common import (
	SUMMARY_DIR,
	append_readme_update,
	ensure_output_dirs,
	save_json,
	subject_files,
	unwrap_singleton,
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


def scalar_text(value: object) -> str | None:
	current = unwrap_singleton(value)
	if isinstance(current, np.ndarray):
		if current.dtype.kind in {"U", "S"}:
			return " ".join(str(item) for item in current.ravel())
		return None
	if isinstance(current, (str, np.str_)):
		return str(current)
	return None


def collect_metadata_tokens(value: object) -> list[str]:
	tokens: list[str] = []
	stack: list[object] = [value]

	while stack:
		current = stack.pop()
		current = unwrap_singleton(current)

		if hasattr(current, "_fieldnames"):
			for field_name in current._fieldnames:
				stack.append(getattr(current, field_name))
			continue

		if isinstance(current, np.ndarray):
			if current.dtype.kind in {"U", "S"}:
				tokens.extend(str(item) for item in current.ravel())
			else:
				stack.extend(list(current.ravel()))
			continue

		text = scalar_text(current)
		if text:
			tokens.append(text)

	return tokens


def stream_embedding(stream: np.ndarray, bins: int = 64) -> tuple[np.ndarray, dict[str, float]]:
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


@dataclass
class TrialPair:
	file_name: str
	trial_index: int
	wav_a: np.ndarray
	wav_b: np.ndarray
	emb_a: np.ndarray
	emb_b: np.ndarray
	stats_a: dict[str, float]
	stats_b: dict[str, float]


def load_trial_pairs(path: Path) -> tuple[list[TrialPair], list[str]]:
	data = loadmat(path, squeeze_me=False, struct_as_record=False)["data"][0, 0]
	pairs: list[TrialPair] = []
	trial_count = data.wavA.shape[1]
	metadata_tokens = collect_metadata_tokens(data.cfg) if hasattr(data, "cfg") else []

	for trial_index in range(trial_count):
		wav_a = np.asarray(data.wavA[0, trial_index], dtype=float).ravel()
		wav_b = np.asarray(data.wavB[0, trial_index], dtype=float).ravel()
		emb_a, stats_a = stream_embedding(wav_a)
		emb_b, stats_b = stream_embedding(wav_b)
		pairs.append(
			TrialPair(
				file_name=path.name,
				trial_index=trial_index,
				wav_a=wav_a,
				wav_b=wav_b,
				emb_a=emb_a,
				emb_b=emb_b,
				stats_a=stats_a,
				stats_b=stats_b,
			)
		)

	return pairs, metadata_tokens


def fit_pairwise_clusters(pairs: list[TrialPair], restarts: int = 8, max_iter: int = 50) -> dict[str, object]:
	trial_count = len(pairs)
	trial_distances = [float(np.linalg.norm(pair.emb_a - pair.emb_b)) for pair in pairs]
	seed_trial_index = int(np.argmax(trial_distances))
	seed_pair = pairs[seed_trial_index]
	seed_a = seed_pair.emb_a.copy()
	seed_b = seed_pair.emb_b.copy()

	def run_once(mu0: np.ndarray, mu1: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[tuple[int, int]], float]:
		assignments: list[tuple[int, int]] = []
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

	best: tuple[np.ndarray, np.ndarray, list[tuple[int, int]], float] | None = None
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

	cluster0_vectors = [vec for pair, assignment in zip(pairs, assignments) for vec, label in ((pair.emb_a, assignment[0]), (pair.emb_b, assignment[1])) if label == 0]
	cluster1_vectors = [vec for pair, assignment in zip(pairs, assignments) for vec, label in ((pair.emb_a, assignment[0]), (pair.emb_b, assignment[1])) if label == 1]
	cluster0_mean = np.mean(np.vstack(cluster0_vectors), axis=0)
	cluster1_mean = np.mean(np.vstack(cluster1_vectors), axis=0)

	mean_margin = []
	for pair in pairs:
		cost_01 = float(np.sum((pair.emb_a - mu0) ** 2) + np.sum((pair.emb_b - mu1) ** 2))
		cost_10 = float(np.sum((pair.emb_a - mu1) ** 2) + np.sum((pair.emb_b - mu0) ** 2))
		mean_margin.append(abs(cost_01 - cost_10))

	return {
		"mu0": mu0,
		"mu1": mu1,
		"assignments": assignments,
		"objective": objective,
		"cluster0_mean": cluster0_mean,
		"cluster1_mean": cluster1_mean,
		"mean_margin": float(np.mean(mean_margin)),
		"median_margin": float(np.median(mean_margin)),
		"centroid_distance": float(np.linalg.norm(mu0 - mu1)),
	}


def cluster_stats_from_assignments(pairs: list[TrialPair], assignments: list[tuple[int, int]]) -> dict[int, dict[str, float]]:
	stats_by_cluster = {
		0: {"spectral_centroid": [], "low_frequency_ratio": [], "zero_crossing_rate": []},
		1: {"spectral_centroid": [], "low_frequency_ratio": [], "zero_crossing_rate": []},
	}
	for pair, assignment in zip(pairs, assignments):
		for stream_index, cluster_index in enumerate(assignment):
			stream_stats = pair.stats_a if stream_index == 0 else pair.stats_b
			for key in stats_by_cluster[cluster_index]:
				stats_by_cluster[cluster_index][key].append(stream_stats[key])

	cluster_summary: dict[int, dict[str, float]] = {}
	for cluster_index, values in stats_by_cluster.items():
		cluster_summary[cluster_index] = {
			key: float(np.mean(items)) if items else float("nan") for key, items in values.items()
		}
	return cluster_summary


def analyze_subject(path: Path, assignments_by_trial: list[tuple[int, int]]) -> dict[str, object]:
	data = loadmat(path, squeeze_me=False, struct_as_record=False)["data"][0, 0]
	trial_count = data.wavA.shape[1]
	wav_a_male = 0
	wav_a_female = 0
	wav_b_male = 0
	wav_b_female = 0

	for trial_index in range(trial_count):
		assignment = assignments_by_trial[trial_index]
		if assignment[0] == 0:
			wav_a_male += 1
			wav_b_female += 1
		else:
			wav_a_female += 1
			wav_b_male += 1

	return {
		"file": path.name,
		"trial_count": int(trial_count),
		"wavA_male": wav_a_male,
		"wavA_female": wav_a_female,
		"wavB_male": wav_b_male,
		"wavB_female": wav_b_female,
		"wavA_male_rate": float(wav_a_male / trial_count) if trial_count else float("nan"),
		"wavB_male_rate": float(wav_b_male / trial_count) if trial_count else float("nan"),
	}


def main() -> None:
	parser = argparse.ArgumentParser(description="Recover male/female speaker identity from trial audio streams.")
	parser.add_argument("--json-out", type=Path, default=SUMMARY_DIR / "speaker_identity.json")
	parser.add_argument("--limit", type=int, default=None)
	parser.add_argument("--no-readme-update", action="store_true")
	args = parser.parse_args()

	ensure_output_dirs()

	files = list(subject_files())
	if args.limit is not None:
		files = files[: args.limit]

	metadata_hits: list[str] = []
	pairs: list[TrialPair] = []
	for path in files:
		file_pairs, file_tokens = load_trial_pairs(path)
		metadata_hits.extend(file_tokens)
		pairs.extend(file_pairs)

	metadata_hits = [token for token in metadata_hits if token and token.lower() not in {"co_preprocessing", "co_resampledata", "co_appenddata", "co_denoise", "co_selectdim", "co_selectevent", "co_splitdata", "co_auditoryfilterbank", "co_dimaverage", "co_squeeze", "no", "13-Mar-2018"}]

	fit = fit_pairwise_clusters(pairs)
	assignments = fit["assignments"]
	cluster_stats = cluster_stats_from_assignments(pairs, assignments)

	# Use the lower-frequency cluster as the male speaker heuristic.
	male_cluster = 0 if cluster_stats[0]["spectral_centroid"] <= cluster_stats[1]["spectral_centroid"] else 1
	female_cluster = 1 - male_cluster

	trial_count = len(pairs)
	trial_distinct_rate = float(sum(int(assignment[0] != assignment[1]) for assignment in assignments) / trial_count) if trial_count else float("nan")

	per_subject: list[dict[str, object]] = []
	assignment_offset = 0
	for path in files:
		data = loadmat(path, squeeze_me=False, struct_as_record=False)["data"][0, 0]
		subject_trial_count = data.wavA.shape[1]
		subject_assignments = assignments[assignment_offset : assignment_offset + subject_trial_count]
		assignment_offset += subject_trial_count
		per_subject.append(analyze_subject(path, subject_assignments))

	payload = {
		"metadata_tokens": metadata_hits,
		"method": "pair_constrained_em",
		"summary": {
			"subject_count": len(files),
			"trial_pair_count": len(pairs),
			"distinct_cluster_rate_within_trial": trial_distinct_rate,
			"objective": float(fit["objective"]),
			"mean_margin": float(fit["mean_margin"]),
			"median_margin": float(fit["median_margin"]),
			"centroid_distance": float(fit["centroid_distance"]),
		},
		"cluster_stats": cluster_stats,
		"gender_map": {
			"male_cluster": male_cluster,
			"female_cluster": female_cluster,
			"male_cluster_rule": "lower_spectral_centroid",
		},
		"per_subject": per_subject,
	}

	save_json(args.json_out, payload)

	print(json.dumps(payload["summary"], indent=2))
	print(json.dumps(payload["cluster_stats"], indent=2))

	if not args.no_readme_update:
		status = "direct metadata found" if metadata_hits else "metadata absent; used pair-constrained clustering"
		append_readme_update(
			[
				f"Recovered speaker-identity candidates for {len(files)} subject file(s) and wrote {args.json_out.name}.",
				f"{status}; trial-distinct rate={trial_distinct_rate:.3f}; centroid distance={fit['centroid_distance']:.3f}.",
				f"Heuristic gender map: lower spectral centroid cluster={male_cluster} (male), other cluster={female_cluster} (female).",
			],
			title="recover_speaker_identity.py completed",
		)


if __name__ == "__main__":
	main()