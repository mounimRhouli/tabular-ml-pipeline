"""Dataset loading and deterministic train/test preprocessing."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split


@dataclass(frozen=True)
class DatasetSplit:
    """A train/test split plus the schema needed for inference."""

    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    feature_names: list[str]
    target_names: list[str]


def load_and_split_data(
    test_size: float = 0.2, random_state: int = 42
) -> DatasetSplit:
    """Load the sklearn dataset and return a reproducible stratified split."""

    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1")

    dataset = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(
        dataset.data,
        dataset.target,
        test_size=test_size,
        random_state=random_state,
        stratify=dataset.target,
    )

    return DatasetSplit(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        feature_names=dataset.feature_names.tolist(),
        target_names=dataset.target_names.tolist(),
    )
