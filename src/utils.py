"""Shared configuration helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PARAMS_PATH = PROJECT_ROOT / "params.yaml"


def load_params(path: str | Path = DEFAULT_PARAMS_PATH) -> dict[str, Any]:
    """Load YAML parameters and fail clearly when the file is invalid."""

    params_path = Path(path)
    with params_path.open("r", encoding="utf-8") as stream:
        params = yaml.safe_load(stream)

    if not isinstance(params, dict):
        raise ValueError(f"Expected a YAML mapping in {params_path}")
    return params


def resolve_project_path(path: str | Path) -> Path:
    """Resolve a configured path relative to the project root."""

    value = Path(path)
    return value if value.is_absolute() else PROJECT_ROOT / value
