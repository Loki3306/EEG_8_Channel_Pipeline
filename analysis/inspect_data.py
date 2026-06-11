from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from _common import (
    SUMMARY_DIR,
    append_readme_update,
    channel_labels,
    ensure_output_dirs,
    fsample_values,
    load_subject_data,
    load_subject_mat,
    object_array_to_list,
    scalar_int,
    save_json,
    subject_files,
    trial_event_samples,
    trial_labels,
)


def summarize_subject(path: Path) -> dict[str, object]:
    mat = load_subject_mat(path)
    data = mat["data"][0, 0]
    top_level_keys = [key for key in mat.keys() if not key.startswith("__")]
    eeg = data.eeg
    wav_a = data.wavA
    wav_b = data.wavB
    dim = data.dim[0, 0]
    cfg = data.cfg
    event = data.event[0, 0]

    first_eeg = eeg[0, 0]
    first_wav_a = wav_a[0, 0]
    first_wav_b = wav_b[0, 0]

    trial_count = int(eeg.shape[1])
    event_count = int(event.eeg.shape[1])
    channel_names = channel_labels(data)

    cfg_entries: list[dict[str, object]] = []
    for index in range(cfg.shape[1]):
        cfg_item = cfg[0, index][0, 0]
        entry: dict[str, object] = {"index": index}
        if hasattr(cfg_item, "_fieldnames"):
            entry["fields"] = list(cfg_item._fieldnames)
            if hasattr(cfg_item, "fcn"):
                entry["fcn"] = str(np.asarray(cfg_item.fcn).squeeze())
            if hasattr(cfg_item, "date"):
                entry["date"] = str(np.asarray(cfg_item.date).squeeze())
        cfg_entries.append(entry)

    return {
        "file": path.name,
        "top_level_keys": top_level_keys,
        "data_fields": list(data._fieldnames),
        "trial_count": trial_count,
        "event_count": event_count,
        "eeg_shape": list(first_eeg.shape),
        "wavA_shape": list(first_wav_a.shape),
        "wavB_shape": list(first_wav_b.shape),
        "eeg_dtype": str(first_eeg.dtype),
        "wavA_dtype": str(first_wav_a.dtype),
        "wavB_dtype": str(first_wav_b.dtype),
        "fsample": fsample_values(data),
        "channel_count": len(channel_names),
        "channel_names": channel_names,
        "trial_labels": trial_labels(data),
        "trial_event_samples": trial_event_samples(data),
        "cfg": cfg_entries,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect EEG AAD preprocessed .mat files.")
    parser.add_argument("--json-out", type=Path, default=SUMMARY_DIR / "inspect_summary.json")
    parser.add_argument("--subject", type=str, default=None, help="Optional subject file name, e.g. S1_data_preproc.mat")
    parser.add_argument("--no-readme-update", action="store_true")
    args = parser.parse_args()

    ensure_output_dirs()

    files = subject_files()
    if args.subject:
        files = [path for path in files if path.name == args.subject]
        if not files:
            raise SystemExit(f"Subject file not found: {args.subject}")

    summaries = [summarize_subject(path) for path in files]

    consistency = {
        "subjects": len(summaries),
        "trial_counts": sorted({item["trial_count"] for item in summaries}),
        "event_counts": sorted({item["event_count"] for item in summaries}),
        "eeg_shapes": sorted({tuple(item["eeg_shape"]) for item in summaries}),
        "wavA_shapes": sorted({tuple(item["wavA_shape"]) for item in summaries}),
        "wavB_shapes": sorted({tuple(item["wavB_shape"]) for item in summaries}),
        "fsample_eeg": sorted({item["fsample"]["eeg"] for item in summaries}),
        "fsample_wavA": sorted({item["fsample"]["wavA"] for item in summaries}),
        "fsample_wavB": sorted({item["fsample"]["wavB"] for item in summaries}),
        "channel_counts": sorted({item["channel_count"] for item in summaries}),
        "trial_labels": sorted({label for item in summaries for label in item["trial_labels"]}),
    }

    payload = {"consistency": consistency, "subjects": summaries}
    save_json(args.json_out, payload)

    print(json.dumps(consistency, indent=2))
    for item in summaries:
        print(
            f"{item['file']}: trials={item['trial_count']} eeg={tuple(item['eeg_shape'])} "
            f"wavA={tuple(item['wavA_shape'])} wavB={tuple(item['wavB_shape'])} fs={item['fsample']['eeg']}"
        )

    if not args.no_readme_update:
        append_readme_update(
            [
                f"Inspected {len(summaries)} subject file(s) and wrote {args.json_out.name}.",
                f"Consistency check: trials={consistency['trial_counts']}, eeg_shapes={consistency['eeg_shapes']}, fs={consistency['fsample_eeg']}.",
                f"Verified top-level key set: {summaries[0]['top_level_keys'] if summaries else []}.",
            ],
            title="inspect_data.py completed",
        )


if __name__ == "__main__":
    main()
