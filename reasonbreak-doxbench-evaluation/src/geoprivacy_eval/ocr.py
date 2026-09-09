from __future__ import annotations

import argparse
import os
from difflib import SequenceMatcher
from pathlib import Path

import cv2
import pandas as pd
import pytesseract

from .common import IMAGE_EXTENSIONS

_tesseract_cmd = os.getenv("TESSERACT_CMD")
if _tesseract_cmd:
    pytesseract.pytesseract.tesseract_cmd = _tesseract_cmd


def clean_text(text: str | None) -> str:
    return " ".join((text or "").replace("\n", " ").replace("\r", " ").split()).strip()


def text_similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


def run_ocr(image_path: str | Path):
    image_path = str(image_path)
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    gray = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 11
    )
    data = pytesseract.image_to_data(
        gray, output_type=pytesseract.Output.DICT, config="--oem 3 --psm 6"
    )

    texts, confs, boxes = [], [], []
    for i, raw_text in enumerate(data["text"]):
        text = str(raw_text).strip()
        try:
            conf = float(data["conf"][i])
        except (TypeError, ValueError):
            conf = -1.0
        if text and conf > 0:
            texts.append(text)
            confs.append(conf)
            boxes.append(
                {
                    "text": text,
                    "conf": conf,
                    "left": int(data["left"][i]),
                    "top": int(data["top"][i]),
                    "width": int(data["width"][i]),
                    "height": int(data["height"][i]),
                }
            )

    full_text = clean_text(" ".join(texts))
    avg_conf = sum(confs) / len(confs) if confs else 0.0
    return full_text, avg_conf, len(texts), boxes


def compare_pair(original_path: str | Path, protected_path: str | Path) -> dict:
    o_text, o_conf, o_wc, _ = run_ocr(original_path)
    p_text, p_conf, p_wc, _ = run_ocr(protected_path)
    return {
        "filename": Path(original_path).name,
        "ocr_orig_text": o_text,
        "ocr_protected_text": p_text,
        "ocr_orig_conf": o_conf,
        "ocr_protected_conf": p_conf,
        "ocr_orig_word_count": o_wc,
        "ocr_protected_word_count": p_wc,
        "ocr_text_similarity": text_similarity(o_text, p_text),
        "ocr_conf_drop": o_conf - p_conf,
        "ocr_word_count_drop": o_wc - p_wc,
    }


def compare_folders(original_dir: str | Path, protected_dir: str | Path, output_csv: str | Path | None = None):
    original_dir, protected_dir = Path(original_dir), Path(protected_dir)
    rows = []
    for original in sorted(original_dir.iterdir()):
        if not original.is_file() or original.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        protected = protected_dir / original.name
        if protected.exists():
            rows.append(compare_pair(original, protected))

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError(f"No matching image pairs between {original_dir} and {protected_dir}.")
    if output_csv:
        output_csv = Path(output_csv)
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare OCR leakage before/after protection.")
    parser.add_argument("--original-dir", required=True)
    parser.add_argument("--protected-dir", required=True)
    parser.add_argument("--output-csv", required=True)
    args = parser.parse_args()
    df = compare_folders(args.original_dir, args.protected_dir, args.output_csv)
    print(f"Pairs: {len(df)}")
    print(f"Mean text similarity: {df['ocr_text_similarity'].mean():.4f}")
    print(f"Mean confidence drop: {df['ocr_conf_drop'].mean():.4f}")
    print(f"Mean word-count drop: {df['ocr_word_count_drop'].mean():.4f}")


if __name__ == "__main__":
    main()
