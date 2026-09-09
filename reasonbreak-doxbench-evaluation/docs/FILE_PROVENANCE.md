# File provenance and cleanup decisions

This repository is a **clean evaluation extension**, not a redistribution of the full ReasonBreak codebase.

## Retained and refactored from the research work

| Research file | Clean repository location | Change |
|---|---|---|
| `prepare_doxbench.py` | `src/geoprivacy_eval/dataset.py` | Added CLI, GT CSV export, validation, optional limit |
| `make_subset_json.py` | `src/geoprivacy_eval/dataset.py` + `scripts/make_subset_json.py` | Parameterized subset size/paths |
| `run_step3_openai.py` | `src/geoprivacy_eval/geolocate.py` | Updated to current Responses API, added district/coordinates, resumable JSONL |
| `prediction_utils.py` | `src/geoprivacy_eval/predictions.py` | Fixed hierarchy mapping and protected/original naming |
| `ocr_utils.py` | `src/geoprivacy_eval/ocr.py` | Standardized OCR column names to match downstream metrics |
| `compute_metrics.py` | `src/geoprivacy_eval/metrics.py` | Fixed OCR merge mismatch; added coordinate-distance evaluation and JSON summary |
| `run_experiment.py` | `src/geoprivacy_eval/pipeline.py` | Rebuilt as a self-contained evaluation pipeline |

## Intentionally not redistributed

The following uploaded files appeared to be upstream ReasonBreak or broader lab utilities rather than the core personal evaluation extension, so they are **not included** in the GitHub-ready source tree:

- `model.py`
- `analyze_privacy.py`
- `clueminer.py`
- `experiment.py`
- `csv_passthrough.py`
- `exif_map.py`
- `full_outer_join_csv.py`
- `get_address.py`
- `image_match_overwrite.py`
- `listdir.py`
- `sample_dataset.py`

ReasonBreak adversarial image generation should be run from the official repository and its released weights. The generated protected-image folder can then be supplied to this repository with `--protected-dir`.
