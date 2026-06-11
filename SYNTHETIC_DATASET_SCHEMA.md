# Synthetic Dataset Schema - EEG AAD

This file turns the verified data-analysis results into a simple schema you can use to generate synthetic training data.

## Verified source format

Each subject file contains 60 trial-level records. Every trial bundles aligned EEG and two audio streams.

- EEG shape per trial: `3200 x 66`
- wavA shape per trial: `3200 x 1`
- wavB shape per trial: `3200 x 1`
- Sampling rate: `64 Hz`
- Label values: `1` or `2`
- Label meaning: speaker identity, not A/B order

## Recommended synthetic record format

Use one row per trial if you want a tabular synthetic dataset, or one JSON object per trial if you want a nested format.

| column | type | allowed values / shape | example value |
| --- | --- | --- | --- |
| subject_id | integer | `1` to `18` | `1` |
| trial_id | integer | `0` to `59` | `17` |
| label | integer | `1` or `2` | `2` |
| fsample_eeg | integer | `64` | `64` |
| fsample_wavA | integer | `64` | `64` |
| fsample_wavB | integer | `64` | `64` |
| eeg | float matrix | `3200 x 66` | `[[0.12, -0.03, ...], ...]` |
| wavA | float matrix | `3200 x 1` | `[[0.008], [0.011], ...]` |
| wavB | float matrix | `3200 x 1` | `[[0.004], [0.010], ...]` |
| channel_names | string list | `66` channel labels | `['Fp1', 'AF7', ..., 'EXG1', 'EXG2']` |

## Channel layout

The verified EEG channel order is:

`Fp1, AF7, AF3, F1, F3, F5, F7, FT7, FC5, FC3, FC1, C1, C3, C5, T7, TP7, CP5, CP3, CP1, P1, P3, P5, P7, P9, PO7, PO3, O1, Iz, Oz, POz, Pz, CPz, Fpz, Fp2, AF8, AF4, AFz, Fz, F2, F4, F6, F8, FT8, FC6, FC4, FC2, FCz, Cz, C2, C4, C6, T8, TP8, CP6, CP4, CP2, P2, P4, P6, P8, P10, PO8, PO4, O2, EXG1, EXG2`

## Example synthetic row

```json
{
  "subject_id": 1,
  "trial_id": 0,
  "label": 2,
  "fsample_eeg": 64,
  "fsample_wavA": 64,
  "fsample_wavB": 64,
  "eeg": "3200x66 float matrix",
  "wavA": "3200x1 float matrix",
  "wavB": "3200x1 float matrix",
  "channel_names": ["Fp1", "AF7", "AF3", "F1", "F3", "F5", "F7", "FT7", "FC5", "FC3", "FC1", "C1", "C3", "C5", "T7", "TP7", "CP5", "CP3", "CP1", "P1", "P3", "P5", "P7", "P9", "PO7", "PO3", "O1", "Iz", "Oz", "POz", "Pz", "CPz", "Fpz", "Fp2", "AF8", "AF4", "AFz", "Fz", "F2", "F4", "F6", "F8", "FT8", "FC6", "FC4", "FC2", "FCz", "Cz", "C2", "C4", "C6", "T8", "TP8", "CP6", "CP4", "CP2", "P2", "P4", "P6", "P8", "P10", "PO8", "PO4", "O2", "EXG1", "EXG2"]
}
```

## Minimal CSV-style version

If you want a flattened file for metadata only, keep these columns:

| subject_id | trial_id | label | eeg_shape | wavA_shape | wavB_shape | fsample |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 0 | 2 | 3200x66 | 3200x1 | 3200x1 | 64 |

## Notes for synthetic generation

- Keep the same trial length: `50 seconds` per record.
- Keep EEG and audio aligned sample-by-sample.
- Use label values only as `1` or `2` unless you explicitly remap them.
- Do not assume wavA = label 1 or wavB = label 2; the project state confirms that wavA/wavB are not ordered by label.