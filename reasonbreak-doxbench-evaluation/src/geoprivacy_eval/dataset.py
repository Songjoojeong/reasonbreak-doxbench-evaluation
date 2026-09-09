from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path


def prepare_doxbench(
    output_root: str | Path,
    split: str = "train",
    limit: int | None = None,
    hf_token: str | None = None,
) -> tuple[Path, Path, Path]:
    """Download DoxBench and export images, a manifest JSON, and a GT CSV.

    The Hugging Face `datasets` dependency is imported lazily so the rest of the
    evaluation package can be used without it.
    """
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "DoxBench preparation requires `datasets`. Install with `pip install -e '.[data]'`."
        ) from exc

    output_root = Path(output_root)
    image_dir = output_root / "images" / "doxbench"
    json_dir = output_root / "json"
    image_dir.mkdir(parents=True, exist_ok=True)
    json_dir.mkdir(parents=True, exist_ok=True)

    token = hf_token or os.getenv("HF_TOKEN")
    ds = load_dataset("MomoUchi/DoxBench", token=token)
    data = ds[split]

    manifest = []
    gt_rows = []
    n = len(data) if limit is None else min(limit, len(data))

    for i in range(n):
        item = data[i]
        filename = f"{i}.jpg"
        image_path = image_dir / filename
        item["image"].convert("RGB").save(image_path, quality=95)

        address = item.get("address", "")
        latitude = item.get("latitude", "")
        longitude = item.get("longitude", "")

        manifest.append(
            {
                "filename": str(image_path),
                "ground_truth": {
                    "address": address,
                    "latitude": latitude,
                    "longitude": longitude,
                },
            }
        )
        gt_rows.append(
            {
                "filename": filename,
                "address": address,
                "latitude": latitude,
                "longitude": longitude,
            }
        )

    manifest_path = json_dir / "doxbench.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)

    gt_csv = output_root / "doxbench_gt.csv"
    with gt_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["filename", "address", "latitude", "longitude"]
        )
        writer.writeheader()
        writer.writerows(gt_rows)

    return image_dir, manifest_path, gt_csv


def make_subset_json(input_json: str | Path, output_json: str | Path, size: int = 20) -> Path:
    input_json = Path(input_json)
    output_json = Path(output_json)
    with input_json.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as handle:
        json.dump(data[:size], handle, ensure_ascii=False, indent=2)
    return output_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and prepare DoxBench.")
    parser.add_argument("--output-root", default="data")
    parser.add_argument("--split", default="train")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    image_dir, manifest, gt_csv = prepare_doxbench(args.output_root, args.split, args.limit)
    print(f"Images:   {image_dir}")
    print(f"Manifest: {manifest}")
    print(f"GT CSV:   {gt_csv}")


if __name__ == "__main__":
    main()
