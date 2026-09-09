from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .common import iter_jsonl, normalize_text


def _as_float(value):
    try:
        return float(value) if value not in (None, "") else None
    except (TypeError, ValueError):
        return None


def load_prediction_jsonl(path: str | Path, prefix: str) -> pd.DataFrame:
    rows = []
    for row in iter_jsonl(path):
        filename = Path(str(row.get("filename", ""))).name
        parsed = row.get("parsed") or {}
        usable = not parsed.get("api_error", False) and not parsed.get("parse_error", False)
        rows.append(
            {
                "filename": filename,
                f"{prefix}_region_pred": normalize_text(parsed.get("state_or_region") or parsed.get("country")),
                f"{prefix}_country_pred": normalize_text(parsed.get("country")),
                f"{prefix}_metro_pred": normalize_text(parsed.get("city_or_metro")),
                f"{prefix}_tract_pred": normalize_text(parsed.get("district_or_neighborhood")),
                f"{prefix}_block_pred": normalize_text(parsed.get("specific_place")),
                f"{prefix}_latitude_pred": _as_float(parsed.get("latitude")),
                f"{prefix}_longitude_pred": _as_float(parsed.get("longitude")),
                f"{prefix}_confidence": _as_float(parsed.get("confidence")),
                f"{prefix}_usable": usable,
            }
        )
    return pd.DataFrame(rows)


def build_prediction_table(orig_jsonl: str | Path, protected_jsonl: str | Path, output_csv: str | Path | None = None):
    orig = load_prediction_jsonl(orig_jsonl, "orig")
    protected = load_prediction_jsonl(protected_jsonl, "protected")
    merged = orig.merge(protected, on="filename", how="inner")
    if output_csv:
        output_csv = Path(output_csv)
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        merged.to_csv(output_csv, index=False, encoding="utf-8-sig")
    return merged


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge original/protected geolocation JSONL results.")
    parser.add_argument("--orig-jsonl", required=True)
    parser.add_argument("--protected-jsonl", required=True)
    parser.add_argument("--output-csv", required=True)
    args = parser.parse_args()
    build_prediction_table(args.orig_jsonl, args.protected_jsonl, args.output_csv)


if __name__ == "__main__":
    main()
