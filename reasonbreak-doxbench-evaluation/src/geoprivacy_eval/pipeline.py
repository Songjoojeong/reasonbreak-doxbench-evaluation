from __future__ import annotations

import argparse
from pathlib import Path

from .geolocate import run_folder
from .metrics import compute_metrics
from .ocr import compare_folders
from .predictions import build_prediction_table


def run_pipeline(
    original_dir: str | Path,
    protected_dir: str | Path,
    gt_csv: str | Path,
    output_dir: str | Path = "outputs",
    model: str = "gpt-5.6-luna",
    skip_inference: bool = False,
    orig_jsonl: str | Path | None = None,
    protected_jsonl: str | Path | None = None,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    orig_jsonl = Path(orig_jsonl) if orig_jsonl else output_dir / "original_predictions.jsonl"
    protected_jsonl = Path(protected_jsonl) if protected_jsonl else output_dir / "protected_predictions.jsonl"
    pred_csv = output_dir / "prediction_table.csv"
    ocr_csv = output_dir / "ocr_comparison.csv"
    final_csv = output_dir / "final_results.csv"
    summary_json = output_dir / "metrics_summary.json"

    if not skip_inference:
        print("[1/5] Geolocation inference: original images")
        run_folder(original_dir, orig_jsonl, model)
        print("[2/5] Geolocation inference: protected images")
        run_folder(protected_dir, protected_jsonl, model)
    else:
        if not orig_jsonl.exists() or not protected_jsonl.exists():
            raise FileNotFoundError("--skip-inference requires existing original/protected JSONL files.")

    print("[3/5] Building paired prediction table")
    build_prediction_table(orig_jsonl, protected_jsonl, pred_csv)

    print("[4/5] Comparing OCR leakage")
    compare_folders(original_dir, protected_dir, ocr_csv)

    print("[5/5] Computing privacy metrics")
    _, summary = compute_metrics(pred_csv, gt_csv, ocr_csv, final_csv, summary_json)

    return {
        "original_predictions": str(orig_jsonl),
        "protected_predictions": str(protected_jsonl),
        "prediction_table": str(pred_csv),
        "ocr_comparison": str(ocr_csv),
        "final_results": str(final_csv),
        "metrics_summary": str(summary_json),
        "summary": summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate geolocation privacy before/after image protection."
    )
    parser.add_argument("--original-dir", required=True)
    parser.add_argument("--protected-dir", required=True)
    parser.add_argument("--gt-csv", required=True)
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--model", default="gpt-5.6-luna")
    parser.add_argument("--skip-inference", action="store_true")
    parser.add_argument("--orig-jsonl", default=None)
    parser.add_argument("--protected-jsonl", default=None)
    args = parser.parse_args()

    outputs = run_pipeline(
        args.original_dir,
        args.protected_dir,
        args.gt_csv,
        args.output_dir,
        args.model,
        args.skip_inference,
        args.orig_jsonl,
        args.protected_jsonl,
    )
    print("\nCompleted.")
    for key in ("final_results", "metrics_summary"):
        print(f"{key}: {outputs[key]}")


if __name__ == "__main__":
    main()
