from __future__ import annotations

import json
import math
import os
import re
from pathlib import Path
from typing import Any, Mapping

import joblib
import numpy as np
import pandas as pd

from airsense import aqi
from airsense.anomaly import detect_prediction_spikes
from airsense.config import MODEL_DIR_ENV, PROJECT_ROOT, PROJECT_VERSION, SUPPORTED_REGIONS, TARGET_COLUMNS
from airsense.explainability import explain_prediction


REGIONS = SUPPORTED_REGIONS

INPUT_ALIASES = {
    "pm25": "pm2_5",
    "pm2.5": "pm2_5",
    "pm2_5": "pm2_5",
    "pm10": "pm10",
    "so2": "so2",
    "temperature": "temp",
    "temp": "temp",
    "humidity": "hum",
    "hum": "hum",
    "wind": "ws",
    "wind_speed": "ws",
    "ws": "ws",
    "wind_direction": "wd",
    "wd": "wd",
    "benzene": "benz",
    "benz": "benz",
    "co": "co",
    "nh3": "nh3",
    "no": "no",
    "no2": "no2",
    "nox": "nox",
    "o3": "o3",
    "rain_gauge": "rg",
    "rg": "rg",
    "solar_radiation": "sr",
    "sr": "sr",
}

BASE_DEFAULTS = {
    "pm2_5": 38.0,
    "pm10": 112.0,
    "so2": 18.0,
    "temp": 31.0,
    "hum": 58.0,
    "ws": 3.2,
    "wd": 180.0,
    "benz": np.nan,
    "co": np.nan,
    "nh3": np.nan,
    "no": np.nan,
    "no2": np.nan,
    "nox": np.nan,
    "o3": np.nan,
    "rg": np.nan,
    "sr": np.nan,
}


class ModelNotReadyError(RuntimeError):
    """Raised when the trained inference bundle is missing or incomplete."""


def _as_project_path(path: Path | str) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate
    return candidate


def find_model_dir(model_dir: Path | str | None = None) -> Path:
    if model_dir is not None:
        return _as_project_path(model_dir)

    env_value = os.getenv(MODEL_DIR_ENV)
    if env_value:
        return _as_project_path(env_value)

    candidates = [
        PROJECT_ROOT / "outputs" / "air_quality_models",
        PROJECT_ROOT / "outputs" / "smoke_air_quality_models",
    ]
    for candidate in candidates:
        if _find_bundle_path(candidate).exists():
            return candidate
    return candidates[0]


def _find_bundle_path(model_dir: Path) -> Path:
    candidates = [
        model_dir / "inference_bundle.joblib",
        model_dir / "models" / "inference_bundle.joblib",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[-1]


def load_model_bundle(model_dir: Path | str | None = None) -> dict[str, Any]:
    resolved_dir = find_model_dir(model_dir)
    bundle_path = _find_bundle_path(resolved_dir)
    if not bundle_path.exists():
        raise ModelNotReadyError(
            f"Missing inference bundle at {bundle_path}. Run scripts/train_air_quality_models.py first."
        )

    bundle = joblib.load(bundle_path)
    required_keys = {"models", "feature_columns", "target_columns", "target_transformer"}
    missing = required_keys.difference(bundle)
    if missing:
        raise ModelNotReadyError(f"Inference bundle is missing keys: {', '.join(sorted(missing))}")

    bundle["bundle_path"] = str(bundle_path)
    bundle["model_dir"] = str(resolved_dir)
    return bundle


def normalize_region(region: str) -> str:
    normalized = str(region).strip().upper().replace(" ", "_")
    aliases = {
        "BHATAGAON_DCR": "BHATAGAON",
        "SILTARA_DCR": "SILTARA",
        "DCR_AIIMS": "AIIMS",
        "IGKV_DCR": "IGKV",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in REGIONS:
        raise ValueError(f"Unknown region {region!r}. Expected one of: {', '.join(REGIONS)}")
    return normalized


def coerce_timestamp(value: Any | None) -> pd.Timestamp:
    if value is None or value == "":
        return pd.Timestamp.now().floor("h")
    timestamp = pd.to_datetime(value, errors="coerce")
    if pd.isna(timestamp):
        raise ValueError(f"Could not parse date_time value {value!r}")
    return pd.Timestamp(timestamp)


def _numeric_or_nan(value: Any) -> float:
    if value is None or value == "":
        return float("nan")
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float("nan")
    if math.isfinite(number):
        return number
    return float("nan")


def _base_values(readings: Mapping[str, Any]) -> dict[str, float]:
    values = dict(BASE_DEFAULTS)
    for raw_key, raw_value in readings.items():
        canonical_key = INPUT_ALIASES.get(str(raw_key).lower(), str(raw_key).lower())
        if canonical_key in values:
            values[canonical_key] = _numeric_or_nan(raw_value)
    return values


def _add_time_features(row: dict[str, float], timestamp: pd.Timestamp) -> None:
    day_of_week = int(timestamp.dayofweek)
    day_of_year = int(timestamp.dayofyear)
    hour = int(timestamp.hour)
    minute = int(timestamp.minute)
    month = int(timestamp.month)

    time_values = {
        "year": int(timestamp.year),
        "month": month,
        "day": int(timestamp.day),
        "hour": hour,
        "minute": minute,
        "day_of_week": day_of_week,
        "day_of_year": day_of_year,
        "week_of_year": int(timestamp.isocalendar().week),
        "is_weekend": int(day_of_week >= 5),
        "hour_sin": math.sin(2 * math.pi * hour / 24),
        "hour_cos": math.cos(2 * math.pi * hour / 24),
        "minute_sin": math.sin(2 * math.pi * minute / 60),
        "minute_cos": math.cos(2 * math.pi * minute / 60),
        "day_of_week_sin": math.sin(2 * math.pi * day_of_week / 7),
        "day_of_week_cos": math.cos(2 * math.pi * day_of_week / 7),
        "month_sin": math.sin(2 * math.pi * month / 12),
        "month_cos": math.cos(2 * math.pi * month / 12),
        "day_of_year_sin": math.sin(2 * math.pi * day_of_year / 365.25),
        "day_of_year_cos": math.cos(2 * math.pi * day_of_year / 365.25),
    }
    row.update(time_values)


def _base_from_lag_feature(feature: str) -> str | None:
    if "_lag_" not in feature:
        return None
    return feature.rsplit("_lag_", 1)[0]


def _base_from_rolling_feature(feature: str) -> tuple[str, str] | None:
    match = re.match(r"(.+)_roll_(mean|std)_\d+$", feature)
    if not match:
        return None
    return match.group(1), match.group(2)


def build_feature_frame(
    bundle: Mapping[str, Any],
    region: str,
    readings: Mapping[str, Any],
    date_time: Any | None = None,
) -> pd.DataFrame:
    feature_columns = list(bundle["feature_columns"])
    timestamp = coerce_timestamp(date_time)
    normalized_region = normalize_region(region)
    values = _base_values(readings)

    row = {column: np.nan for column in feature_columns}
    _add_time_features(row, timestamp)

    for column, value in values.items():
        if column in row:
            row[column] = value

    wd = values.get("wd", np.nan)
    if pd.notna(wd):
        row["wd_sin"] = math.sin(math.radians(float(wd)))
        row["wd_cos"] = math.cos(math.radians(float(wd)))

    for column in feature_columns:
        lag_base = _base_from_lag_feature(column)
        if lag_base is not None:
            row[column] = values.get(lag_base, np.nan)
            continue

        rolling_parts = _base_from_rolling_feature(column)
        if rolling_parts is not None:
            base_name, statistic = rolling_parts
            base_value = values.get(base_name, np.nan)
            if statistic == "mean":
                row[column] = base_value
            else:
                row[column] = 0.0 if pd.notna(base_value) else np.nan

    for column in feature_columns:
        if column.startswith("region_"):
            row[column] = int(column == f"region_{normalized_region}")

    return pd.DataFrame([row], columns=feature_columns)


def _inverse_predictions(bundle: Mapping[str, Any], predictions: pd.DataFrame) -> pd.DataFrame:
    transformer = bundle.get("target_transformer")
    if transformer is not None and hasattr(transformer, "inverse_transform_frame"):
        return transformer.inverse_transform_frame(predictions)

    metadata = bundle.get("metadata", {})
    restored = predictions.copy()
    if metadata.get("log_target_transform"):
        restored = np.expm1(restored)
    return restored.clip(lower=0)


def _predict_multi_output(bundle: Mapping[str, Any], feature_frame: pd.DataFrame) -> pd.DataFrame:
    target_columns = list(bundle.get("target_columns", TARGET_COLUMNS))
    model = bundle["models"].get("multi_output")
    if model is None:
        raise ModelNotReadyError("The inference bundle does not contain a multi-output model.")
    predictions = pd.DataFrame(model.predict(feature_frame), columns=target_columns)
    return _inverse_predictions(bundle, predictions)


def _predict_single_target(bundle: Mapping[str, Any], feature_frame: pd.DataFrame) -> pd.DataFrame:
    target_columns = list(bundle.get("target_columns", TARGET_COLUMNS))
    models = bundle["models"].get("single_target") or {}
    missing = [target for target in target_columns if target not in models]
    if missing:
        raise ModelNotReadyError(f"The inference bundle is missing single-target models for: {missing}")

    raw_predictions = {
        target: models[target].predict(feature_frame)
        for target in target_columns
    }
    predictions = pd.DataFrame(raw_predictions, columns=target_columns)
    return _inverse_predictions(bundle, predictions)


def _best_strategy(bundle: Mapping[str, Any], region: str, target: str) -> str:
    rows = bundle.get("best_strategy_by_region_target") or []
    for row in rows:
        if row.get("region") == region and row.get("target") == target:
            strategy = str(row.get("strategy", "multi")).lower()
            return "single_target" if strategy.startswith("single") else "multi_output"
    return "multi_output"


def classify_risk(predictions: Mapping[str, float]) -> dict[str, Any]:
    risk = aqi.calculate_pollutant_risk(
        predictions.get("pm2_5", 0),
        predictions.get("pm10", 0),
        predictions.get("so2", 0),
    )
    return {
        **risk,
        "label": risk["category"],
    }


def build_metadata(bundle: Mapping[str, Any]) -> dict[str, Any]:
    metadata = dict(bundle.get("metadata", {}))
    return {
        "project": "AirSense AI",
        "version": PROJECT_VERSION,
        "targets": list(bundle.get("target_columns", TARGET_COLUMNS)),
        "feature_names": list(bundle.get("feature_columns", [])),
        "feature_count": int(len(bundle.get("feature_columns", []))),
        "supported_regions": REGIONS,
        "trained_at": metadata.get("trained_at", "generated smoke artifact"),
        "granularity": metadata.get("granularity", "unknown"),
        "model_dir": bundle.get("model_dir"),
        "strategies": ["best", "multi_output", "single_target"],
    }


def predict(
    bundle: Mapping[str, Any],
    region: str,
    readings: Mapping[str, Any],
    date_time: Any | None = None,
    strategy: str = "best",
) -> dict[str, Any]:
    normalized_region = normalize_region(region)
    feature_frame = build_feature_frame(
        bundle=bundle,
        region=normalized_region,
        readings=readings,
        date_time=date_time,
    )
    normalized_strategy = strategy.lower().strip()

    if normalized_strategy == "multi_output":
        prediction_frame = _predict_multi_output(bundle, feature_frame)
    elif normalized_strategy == "single_target":
        prediction_frame = _predict_single_target(bundle, feature_frame)
    elif normalized_strategy == "best":
        multi_frame = _predict_multi_output(bundle, feature_frame)
        single_frame = _predict_single_target(bundle, feature_frame)
        prediction_frame = multi_frame.copy()
        for target in bundle.get("target_columns", TARGET_COLUMNS):
            selected_strategy = _best_strategy(bundle, normalized_region, target)
            if selected_strategy == "single_target":
                prediction_frame[target] = single_frame[target]
    else:
        raise ValueError("strategy must be one of: best, multi_output, single_target")

    predictions = {
        target: round(float(max(prediction_frame.iloc[0][target], 0.0)), 3)
        for target in bundle.get("target_columns", TARGET_COLUMNS)
    }
    risk = classify_risk(predictions)
    explanation = explain_prediction(bundle, feature_frame=feature_frame)
    spike_alerts = detect_prediction_spikes(predictions)
    return {
        "region": normalized_region,
        "date_time": str(coerce_timestamp(date_time)),
        "strategy": normalized_strategy,
        "predictions": predictions,
        "risk": risk,
        "aqi": risk,
        "recommendation": risk["recommendation"],
        "summary": aqi.summarize_prediction({"predictions": predictions}),
        "anomaly_alerts": spike_alerts,
        "explanation": explanation,
        "feature_count": int(len(bundle["feature_columns"])),
        "model_dir": bundle.get("model_dir"),
    }


def prediction_to_json(result: Mapping[str, Any]) -> str:
    return json.dumps(result, indent=2)
