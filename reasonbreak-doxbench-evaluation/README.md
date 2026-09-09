# Geolocation Privacy Evaluation on DoxBench

> Evaluation extension for comparing **original vs. privacy-protected images** using multimodal geolocation inference and OCR leakage analysis.

This repository organizes the code developed during a SAFERR/SAFER AI Lab geolocation-privacy research project into a reproducible portfolio-ready pipeline. The work **does not claim to reimplement ReasonBreak**. Instead, it evaluates how protected images change geolocation predictions and OCR-readable location cues.

## What this repository contributes

- DoxBench download/export utility
- paired original/protected image geolocation inference
- resumable OpenAI vision inference to JSONL
- Tesseract OCR comparison on matched image pairs
- text similarity, OCR confidence drop, and word-count drop
- hierarchical protection-rate calculation when compatible GT is provided
- latitude/longitude distance-based protection metrics for self-contained evaluation
- one-command evaluation pipeline plus offline smoke demo and tests

## Pipeline

```text
DoxBench images
      │
      ├──────────── original images ────────────┐
      │                                         │
      └─> ReasonBreak (official repo) ─> protected images
                                                │
             ┌──────────────────────────────────┘
             │
             ▼
   Same geolocation model on both sets
             │
             ▼
      paired prediction table
             │
      ┌──────┴────────┐
      ▼               ▼
  OCR comparison   GT comparison
      │               │
      └──────┬────────┘
             ▼
       final metrics
```

## Why ReasonBreak is external

ReasonBreak is an ICLR 2026 adversarial framework for geographic privacy. Its official repository provides the model, checkpoint instructions, and adversarial generation pipeline. This repository focuses on the **evaluation extension** and intentionally does not relabel upstream model code as personal implementation.

- Official implementation: https://github.com/jiamingzhang94/ReasonBreak
- Paper: https://arxiv.org/abs/2512.08503

## Repository structure

```text
.
├── src/geoprivacy_eval/
│   ├── dataset.py       # DoxBench preparation
│   ├── geolocate.py     # image geolocation inference
│   ├── predictions.py   # original/protected result pairing
│   ├── ocr.py           # Tesseract OCR comparison
│   ├── metrics.py       # protection + OCR metrics
│   └── pipeline.py      # end-to-end evaluation
├── scripts/             # thin CLI entry points
├── examples/            # API-free smoke demo
├── tests/
├── docs/
├── data/                # large assets ignored by git
└── outputs/             # generated results ignored by git
```

## Installation

Python 3.10+ is recommended.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\\Scripts\\activate
pip install -e ".[data,dev]"
```

Tesseract is also required for OCR:

```bash
# Ubuntu/Debian
sudo apt-get install tesseract-ocr

# macOS
brew install tesseract
```

On Windows, install Tesseract and set `TESSERACT_CMD` in `.env` if it is not on PATH.

## 1. Configure secrets

```bash
cp .env.example .env
```

Add `OPENAI_API_KEY` only if you want to run live geolocation inference. API keys and `.env` are ignored by Git.

The live inference module uses the OpenAI **Responses API** with image input. The default model in this cleaned repository is `gpt-5.6-luna`; use `--model` to select another vision-capable model. For a historical reproduction, use the exact model recorded in the original experiment when available.

## 2. Prepare DoxBench

```bash
python scripts/prepare_doxbench.py --output-root data
```

For a quick subset:

```bash
python scripts/make_subset_json.py --size 20
```

## 3. Generate protected images with ReasonBreak

Follow the official ReasonBreak repository. Its documented generation-only command is:

```bash
bash run.sh --mode adv --step gen --dox_path /path/to/data/images/dox
```

Keep the generated images outside this repository or copy only the resulting image folder under `data/` (which is Git-ignored).

## 4. Run the evaluation

Assuming image filenames match between the two directories:

```bash
python scripts/run_experiment.py \
  --original-dir data/images/doxbench \
  --protected-dir /path/to/reasonbreak/protected_images \
  --gt-csv data/doxbench_gt.csv \
  --output-dir outputs/run1 \
  --model gpt-5.6-luna
```

Outputs:

```text
outputs/run1/
├── original_predictions.jsonl
├── protected_predictions.jsonl
├── prediction_table.csv
├── ocr_comparison.csv
├── final_results.csv
└── metrics_summary.json
```

### Reuse existing API results

If inference has already been run:

```bash
python scripts/run_experiment.py \
  --original-dir data/images/doxbench \
  --protected-dir /path/to/protected_images \
  --gt-csv data/doxbench_gt.csv \
  --output-dir outputs/reanalysis \
  --skip-inference \
  --orig-jsonl /path/to/original_predictions.jsonl \
  --protected-jsonl /path/to/protected_predictions.jsonl
```

## 5. Run the offline demo

The demo creates synthetic sign images, blurs the protected copies, uses pre-made geolocation predictions, and runs the OCR + metric pipeline without any API key:

```bash
python examples/run_offline_demo.py
```

## 6. Run tests

```bash
pytest -q
```

## Metrics

### OCR leakage

- `ocr_text_similarity`: character-sequence similarity between recognized original/protected text
- `ocr_conf_drop`: mean OCR confidence before minus after protection
- `ocr_word_count_drop`: recognized word count before minus after protection

### Privacy protection

When compatible hierarchy GT is available, the repository reports protection rates at the available region/metro/tract/block levels. With public DoxBench latitude/longitude, it also reports protection rates for configurable geodesic-distance thresholds (default: 1 km, 25 km, 100 km).

The metric conditions on samples that were correct before protection:

```text
PPR = originally-correct samples made incorrect after protection
      ----------------------------------------------------------
                   originally-correct samples
```

## Scope and limitations

This is an **evaluation project**, not a claim that the underlying adversarial generator was authored here. OCR degradation measures reduced readable text, but it should not be interpreted alone as proof that all geolocation attacks are prevented. The text-based hierarchical matcher is an exploratory compatibility layer; use the official DoxBench evaluator for benchmark-comparable results.

See `docs/METHODOLOGY.md` and `docs/FILE_PROVENANCE.md` for details.

## Research stack

Python · pandas · OpenCV · Tesseract OCR · OpenAI vision models · Hugging Face Datasets · DoxBench · ReasonBreak

## Citation

If you use ReasonBreak-generated protected images, cite the original authors:

```bibtex
@inproceedings{zhang2026disrupting,
  title={Disrupting Hierarchical Reasoning: Adversarial Protection for Geographic Privacy in Multimodal Reasoning Models},
  author={Zhang, Jiaming and Wang, Che and Cao, Yang and Huang, Longtao and Lim, Wei Yang Bryan},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2026}
}
```
