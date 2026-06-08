from __future__ import annotations

import sys
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from airsense.inference import ModelNotReadyError, load_model_bundle, predict


app = FastAPI(
    title="AirSense AI Prediction API",
    version="1.0.0",
    description="Predict next-step PM2.5, PM10, and SO2 for Raipur monitoring regions.",
)


class PredictionRequest(BaseModel):
    region: Literal["AIIMS", "BHATAGAON", "IGKV", "SILTARA"] = "SILTARA"
    date_time: str | None = Field(default=None, description="Optional ISO timestamp")
    strategy: Literal["best", "multi_output", "single_target"] = "best"
    pm2_5: float = Field(default=38.0, ge=0)
    pm10: float = Field(default=112.0, ge=0)
    so2: float = Field(default=18.0, ge=0)
    temp: float = Field(default=31.0, ge=0, le=60)
    hum: float = Field(default=58.0, ge=0, le=100)
    ws: float = Field(default=3.2, ge=0)
    wd: float = Field(default=180.0, ge=0, le=360)


def get_bundle() -> dict:
    try:
        return load_model_bundle()
    except ModelNotReadyError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get("/health")
def health() -> dict[str, str]:
    bundle = get_bundle()
    return {
        "status": "ok",
        "model_dir": str(bundle.get("model_dir")),
        "granularity": str(bundle.get("metadata", {}).get("granularity", "unknown")),
    }


@app.post("/predict")
def predict_endpoint(payload: PredictionRequest) -> dict:
    bundle = get_bundle()
    if hasattr(payload, "model_dump"):
        readings = payload.model_dump(exclude={"region", "date_time", "strategy"})
    else:
        readings = payload.dict(exclude={"region", "date_time", "strategy"})
    try:
        return predict(
            bundle,
            region=payload.region,
            readings=readings,
            date_time=payload.date_time,
            strategy=payload.strategy,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
