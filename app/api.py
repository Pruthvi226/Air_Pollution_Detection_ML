from __future__ import annotations

import sys
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from airsense.inference import ModelNotReadyError, build_metadata, load_model_bundle, predict


app = FastAPI(
    title="AirSense AI Prediction API",
    version="1.0.0",
    description="Predict next-step PM2.5, PM10, and SO2 for Raipur monitoring regions.",
)


class PredictionRequest(BaseModel):
    region: Literal["AIIMS", "BHATAGAON", "IGKV", "SILTARA"] = "SILTARA"
    date_time: str | None = Field(default=None, description="Optional ISO timestamp")
    timestamp: str | None = Field(default=None, description="Alias for date_time")
    strategy: Literal["best", "multi_output", "single_target"] = "best"
    pm2_5: float | None = Field(default=None, ge=0)
    pm25: float | None = Field(default=None, ge=0)
    pm10: float = Field(default=112.0, ge=0)
    so2: float = Field(default=18.0, ge=0)
    temp: float | None = Field(default=None, ge=0, le=60)
    temperature: float | None = Field(default=None, ge=0, le=60)
    hum: float | None = Field(default=None, ge=0, le=100)
    humidity: float | None = Field(default=None, ge=0, le=100)
    ws: float | None = Field(default=None, ge=0)
    wind_speed: float | None = Field(default=None, ge=0)
    wd: float | None = Field(default=None, ge=0, le=360)
    wind_direction: float | None = Field(default=None, ge=0, le=360)


def get_bundle() -> dict:
    try:
        return load_model_bundle()
    except ModelNotReadyError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get("/health")
def health() -> dict[str, object]:
    bundle = get_bundle()
    return {
        "status": "ok",
        "model_loaded": True,
        "model_dir": str(bundle.get("model_dir")),
        "granularity": str(bundle.get("metadata", {}).get("granularity", "unknown")),
    }


@app.get("/metadata")
def metadata() -> dict[str, object]:
    return build_metadata(get_bundle())


@app.post("/predict")
def predict_endpoint(payload: PredictionRequest) -> dict:
    bundle = get_bundle()
    readings = {
        "pm2_5": payload.pm2_5 if payload.pm2_5 is not None else payload.pm25 if payload.pm25 is not None else 38.0,
        "pm10": payload.pm10,
        "so2": payload.so2,
        "temp": payload.temp if payload.temp is not None else payload.temperature if payload.temperature is not None else 31.0,
        "hum": payload.hum if payload.hum is not None else payload.humidity if payload.humidity is not None else 58.0,
        "ws": payload.ws if payload.ws is not None else payload.wind_speed if payload.wind_speed is not None else 3.2,
        "wd": payload.wd if payload.wd is not None else payload.wind_direction if payload.wind_direction is not None else 180.0,
    }
    timestamp = payload.date_time or payload.timestamp
    try:
        return predict(
            bundle,
            region=payload.region,
            readings=readings,
            date_time=timestamp,
            strategy=payload.strategy,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
