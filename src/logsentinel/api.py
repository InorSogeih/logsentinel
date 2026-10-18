from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .config import DEFAULT_MODEL_PATH
from .predict import Predictor


class LogWindow(BaseModel):
    message: str = Field(min_length=1)
    severity: str = "INFO"
    latency_ms: float = Field(default=120.0, ge=0)
    retry_count: int = Field(default=0, ge=0)
    restart_count: int = Field(default=0, ge=0)
    cpu_usage: float = Field(default=0.3, ge=0, le=1)
    memory_usage: float = Field(default=0.4, ge=0, le=1)
    disk_usage: float = Field(default=0.4, ge=0, le=1)
    error_count_5m: int = Field(default=0, ge=0)
    request_count: int = Field(default=100, ge=1)
    hour: int = Field(default=12, ge=0, le=23)


class PredictionResponse(BaseModel):
    is_anomaly: bool
    anomaly_score: float
    root_cause: str | None = None
    root_cause_confidence: float | None = None
    model_version: str


def create_app(
    predictor: Predictor | None = None,
    model_path: Path | str | None = None,
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if app.state.predictor is None:
            app.state.predictor = Predictor.from_path(model_path or DEFAULT_MODEL_PATH)
        yield

    app = FastAPI(
        title="LogSentinel",
        version="0.1.0",
        description="Log anomaly detection and root-cause classification for AI Infra",
        lifespan=lifespan,
    )
    app.state.predictor = predictor
    app.state.model_path = Path(model_path or DEFAULT_MODEL_PATH)

    def load_predictor() -> Predictor:
        if app.state.predictor is None:
            app.state.predictor = Predictor.from_path(app.state.model_path)
        return app.state.predictor

    @app.get("/health")
    def health() -> dict[str, object]:
        try:
            load_predictor()
        except FileNotFoundError:
            return {"status": "degraded", "model_loaded": False}
        return {"status": "ok", "model_loaded": True}

    @app.post("/predict", response_model=PredictionResponse)
    def predict(payload: LogWindow) -> PredictionResponse:
        try:
            predictor = load_predictor()
        except FileNotFoundError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        if predictor is None:
            raise HTTPException(status_code=503, detail="Model is not loaded")
        frame = pd.DataFrame([payload.model_dump()])
        try:
            result = predictor.predict_frame(frame)[0]
        except Exception as exc:  # pragma: no cover - defensive API boundary
            raise HTTPException(status_code=500, detail=f"Inference failed: {exc}") from exc
        return PredictionResponse(**result)

    return app


app = create_app()
