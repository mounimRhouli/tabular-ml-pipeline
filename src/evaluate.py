"""Evaluate a trained model and write machine-readable metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from src.data_preprocessing import load_and_split_data
from src.utils import DEFAULT_PARAMS_PATH, load_params, resolve_project_path


def evaluate_model(
    params_path: str | Path = DEFAULT_PARAMS_PATH,
    model_path: str | Path | None = None,
    metrics_path: str | Path | None = None,
) -> dict[str, float]:
    """Evaluate the saved pipeline on the configured holdout split."""

    params = load_params(params_path)
    data_params = params["data"]
    artifact_path = resolve_project_path(model_path or params["paths"]["model"])
    output_path = resolve_project_path(metrics_path or params["paths"]["metrics"])

    if not artifact_path.exists():
        raise FileNotFoundError(
            f"Model not found at {artifact_path}. Run `python -m src.train` first."
        )

    artifact = joblib.load(artifact_path)
    split = load_and_split_data(
        test_size=float(data_params["test_size"]),
        random_state=int(data_params["random_state"]),
    )
    predictions = artifact["model"].predict(split.X_test)
    metrics = {
        "accuracy": float(accuracy_score(split.y_test, predictions)),
        "precision": float(precision_score(split.y_test, predictions)),
        "recall": float(recall_score(split.y_test, predictions)),
        "f1": float(f1_score(split.y_test, predictions)),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--params", default=str(DEFAULT_PARAMS_PATH))
    parser.add_argument("--model", default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    metrics = evaluate_model(args.params, args.model, args.output)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
