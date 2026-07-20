"""Prediction helpers and command-line entry point."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import joblib
import numpy as np

from src.utils import PROJECT_ROOT


DEFAULT_MODEL_PATH = PROJECT_ROOT / "model" / "model.joblib"


def load_artifact(model_path: str | Path = DEFAULT_MODEL_PATH) -> dict[str, Any]:
    """Load and validate a trained model artifact."""

    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Model not found at {path}. Run `python -m src.train` first."
        )
    artifact = joblib.load(path)
    required_keys = {"model", "feature_names", "target_names"}
    if not isinstance(artifact, dict) or not required_keys.issubset(artifact):
        raise ValueError(f"Invalid model artifact at {path}")
    return artifact


def predict(features: Sequence[float], artifact: dict[str, Any]) -> dict[str, Any]:
    """Predict the class and positive-class probability for one observation."""

    expected = len(artifact["feature_names"])
    if len(features) != expected:
        raise ValueError(f"Expected {expected} features, received {len(features)}")

    row = np.asarray(features, dtype=float).reshape(1, -1)
    predicted_class = int(artifact["model"].predict(row)[0])
    probability = float(artifact["model"].predict_proba(row)[0, predicted_class])
    return {
        "class_id": predicted_class,
        "class_name": artifact["target_names"][predicted_class],
        "probability": probability,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--features", nargs="+", type=float, required=True)
    args = parser.parse_args()

    result = predict(args.features, load_artifact(args.model))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
