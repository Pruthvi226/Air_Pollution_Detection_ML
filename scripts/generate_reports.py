from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate compact AirSense AI report artifacts.")
    parser.add_argument("--run-dir", type=Path, default=Path("outputs/smoke_air_quality_models"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir
    reports_dir = run_dir / "reports"
    metrics_path = reports_dir / "metrics_overall.csv"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Metrics file not found: {metrics_path}")

    metrics = pd.read_csv(metrics_path)
    test_metrics = metrics[metrics["split"] == "test"].copy()
    comparison = (
        test_metrics.groupby("strategy", as_index=False)
        .agg(mean_rmse=("rmse", "mean"), mean_mae=("mae", "mean"), mean_r2=("r2", "mean"))
        .sort_values(["mean_rmse", "mean_mae"], kind="stable")
    )
    comparison.to_csv(run_dir / "model_comparison.csv", index=False)
    predictions_path = run_dir / "predictions" / "test_predictions.csv"
    if predictions_path.exists():
        pd.read_csv(predictions_path, nrows=500).to_csv(run_dir / "predictions.csv", index=False)
    payload = {
        "targets": sorted(test_metrics["target"].unique().tolist()),
        "model_comparison": comparison.to_dict(orient="records"),
        "test_metrics": test_metrics.to_dict(orient="records"),
    }
    (run_dir / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote compact reports to {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
