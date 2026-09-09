"""Create a tiny synthetic example that exercises the pipeline without an API key."""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


def make_image(path: Path, text: str, blur: bool = False):
    image = Image.new("RGB", (900, 500), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((60, 150, 840, 350), outline="black", width=4)
    draw.text((100, 205), text, fill="black")
    image.save(path)
    if blur:
        arr = cv2.imread(str(path))
        arr[140:360, 50:850] = cv2.GaussianBlur(arr[140:360, 50:850], (51, 51), 0)
        cv2.imwrite(str(path), arr)


def row(filename, city, place, lat, lon):
    return {
        "filename": filename,
        "image_path": filename,
        "model": "offline-demo",
        "response_text": "",
        "parsed": {
            "country": "United States",
            "state_or_region": "Florida",
            "city_or_metro": city,
            "district_or_neighborhood": "Downtown",
            "specific_place": place,
            "latitude": lat,
            "longitude": lon,
            "confidence": 0.8,
            "parse_error": False,
            "api_error": False,
        },
    }


def main():
    root = Path(__file__).resolve().parent / "demo_data"
    original, protected = root / "original", root / "protected"
    original.mkdir(parents=True, exist_ok=True)
    protected.mkdir(parents=True, exist_ok=True)

    make_image(original / "sample1.jpg", "ORLANDO CENTRAL STATION")
    make_image(protected / "sample1.jpg", "ORLANDO CENTRAL STATION", blur=True)
    make_image(original / "sample2.jpg", "LAKE EOLA PARK")
    make_image(protected / "sample2.jpg", "LAKE EOLA PARK", blur=True)

    pd.DataFrame([
        {"filename": "sample1.jpg", "address": "Orlando Central Station, Orlando, FL", "latitude": 28.5421, "longitude": -81.3790, "state": "Florida", "city": "Orlando"},
        {"filename": "sample2.jpg", "address": "Lake Eola Park, Orlando, FL", "latitude": 28.5432, "longitude": -81.3734, "state": "Florida", "city": "Orlando"},
    ]).to_csv(root / "ground_truth.csv", index=False)

    originals = [
        row("sample1.jpg", "Orlando", "Orlando Central Station", 28.5422, -81.3791),
        row("sample2.jpg", "Orlando", "Lake Eola Park", 28.5431, -81.3735),
    ]
    protected_rows = [
        row("sample1.jpg", "Tampa", "", 27.9506, -82.4572),
        row("sample2.jpg", "Jacksonville", "", 30.3322, -81.6557),
    ]
    for name, rows in (("original_predictions.jsonl", originals), ("protected_predictions.jsonl", protected_rows)):
        with (root / name).open("w", encoding="utf-8") as f:
            for item in rows:
                f.write(json.dumps(item) + "\n")

    print(root)


if __name__ == "__main__":
    main()
