# Methodology

## Research question

How much does an image-privacy perturbation reduce geolocation-relevant information, and is the change observable both in multimodal geolocation predictions and in OCR-readable text?

## Evaluation flow

1. **Prepare data** — export DoxBench images and public ground truth.
2. **Generate protected images externally** — e.g. with the official ReasonBreak implementation and released checkpoint.
3. **Run the same geolocation model** on original and protected images.
4. **Pair predictions** by image filename.
5. **Run OCR on both versions** and measure text similarity, OCR-confidence drop, and recognized-word-count drop.
6. **Compute privacy metrics**:
   - hierarchical protection rate when compatible region/metro/tract/block GT columns exist;
   - coordinate-distance protection at configurable thresholds when latitude/longitude GT is available.
7. **Link OCR changes to protection outcomes** for exploratory analysis.

## Protection-rate definition

For a correctness criterion `C`, the protection rate is:

```text
(number of samples correct before protection and incorrect after protection)
--------------------------------------------------------------------------
(number of samples correct before protection)
```

This conditions on samples the baseline model could originally geolocate correctly.

## Interpretation limits

- OCR degradation is a proxy for the loss of readable textual cues; it is not by itself proof that every geolocation attack is prevented.
- Text hierarchy matching in this repository is exploratory and does not replace the official DoxBench evaluator.
- Coordinate-distance thresholds are included to make the evaluation reproducible with the public latitude/longitude fields.
