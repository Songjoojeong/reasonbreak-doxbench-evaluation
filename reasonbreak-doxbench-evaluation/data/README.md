# Data

Large benchmark images, model checkpoints, and private research artifacts are intentionally not committed.

To download a local DoxBench copy:

```bash
python scripts/prepare_doxbench.py --output-root data
```

This creates:

```text
data/
├── images/doxbench/
├── json/doxbench.json
└── doxbench_gt.csv
```

`doxbench_gt.csv` contains the public fields exported by the dataset loader (`filename`, `address`, `latitude`, `longitude`). If you have the official DoxBench hierarchical annotations/evaluation CSV, you can pass that file instead to compute region/metro/tract/block protection rates.
