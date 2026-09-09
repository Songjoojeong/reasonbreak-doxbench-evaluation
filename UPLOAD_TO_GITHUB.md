# Upload to GitHub

This ZIP is already arranged as a repository root.

## Recommended repository name

`reasonbreak-doxbench-evaluation`

Alternative: `geolocation-privacy-evaluation`

## Recommended GitHub description

`Evaluation extension for DoxBench geolocation privacy: original vs. protected MLLM predictions, OCR leakage, and protection metrics.`

## Upload

1. Create a new empty GitHub repository.
2. Extract this ZIP.
3. Upload **the contents of the extracted `reasonbreak-doxbench-evaluation` folder** to the repository root.
4. Do not add `.env`, model checkpoints, DoxBench images, API keys, or generated outputs.

Or with Git:

```bash
git init
git add .
git commit -m "Initial geolocation privacy evaluation pipeline"
git branch -M main
git remote add origin <YOUR_REPOSITORY_URL>
git push -u origin main
```

## Suggested topics

`geolocation-privacy` · `multimodal` · `computer-vision` · `ocr` · `doxbench` · `adversarial-robustness` · `privacy` · `python`
