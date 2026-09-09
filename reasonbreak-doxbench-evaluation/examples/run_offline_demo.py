from pathlib import Path
from geoprivacy_eval.pipeline import run_pipeline
from create_offline_demo import main as create_demo

if __name__ == "__main__":
    create_demo()
    root = Path(__file__).resolve().parent / "demo_data"
    outputs = run_pipeline(
        original_dir=root / "original",
        protected_dir=root / "protected",
        gt_csv=root / "ground_truth.csv",
        output_dir=root / "outputs",
        skip_inference=True,
        orig_jsonl=root / "original_predictions.jsonl",
        protected_jsonl=root / "protected_predictions.jsonl",
    )
    print(outputs["summary"])
