"""Train and persist the classification pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.data_preprocessing import load_and_split_data
from src.utils import DEFAULT_PARAMS_PATH, load_params, resolve_project_path


def train_model(
    params_path: str | Path = DEFAULT_PARAMS_PATH,
    model_path: str | Path | None = None,
) -> tuple[Path, dict[str, Any]]:
    """Fit the configured model and save an artifact with its schema."""

    params = load_params(params_path)
    data_params = params["data"]
    model_params = params["model"]
    output_path = resolve_project_path(model_path or params["paths"]["model"])

    split = load_and_split_data(
        test_size=float(data_params["test_size"]),
        random_state=int(data_params["random_state"]),
    )
    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    C=float(model_params["C"]),
                    max_iter=int(model_params["max_iter"]),
                    random_state=int(data_params["random_state"]),
                ),
            ),
        ]
    )
    pipeline.fit(split.X_train, split.y_train)

    artifact: dict[str, Any] = {
        "model": pipeline,
        "feature_names": split.feature_names,
        "target_names": split.target_names,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, output_path)
    return output_path, artifact


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--params", default=str(DEFAULT_PARAMS_PATH))
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    output_path, _ = train_model(args.params, args.output)
    print(f"Saved trained model to {output_path}")


if __name__ == "__main__":
    main()
