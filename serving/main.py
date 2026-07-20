"""FastAPI application for model inference."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.predict import DEFAULT_MODEL_PATH, load_artifact, predict


class PredictionRequest(BaseModel):
    features: list[float] = Field(..., min_length=1)


def create_app(model_path: str | Path | None = None) -> FastAPI:
    configured_path = Path(model_path or os.getenv("MODEL_PATH", DEFAULT_MODEL_PATH))
    application = FastAPI(
        title="Tabular ML Pipeline API",
        version="1.0.0",
        description="Binary classification using a trained scikit-learn pipeline.",
    )

    @lru_cache(maxsize=1)
    def get_artifact() -> dict[str, Any]:
        return load_artifact(configured_path)

    @application.get("/health")
    def health() -> dict[str, Any]:
        return {"status": "ok", "model_ready": configured_path.exists()}

    @application.get("/metadata")
    def metadata() -> dict[str, Any]:
        try:
            artifact = get_artifact()
        except (FileNotFoundError, ValueError) as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        return {
            "feature_count": len(artifact["feature_names"]),
            "feature_names": artifact["feature_names"],
            "target_names": artifact["target_names"],
        }

    @application.post("/predict")
    def make_prediction(request: PredictionRequest) -> dict[str, Any]:
        try:
            return predict(request.features, get_artifact())
        except FileNotFoundError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    return application


app = create_app()
