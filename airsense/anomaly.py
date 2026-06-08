from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd


DEFAULT_SPIKE_THRESHOLDS = {
    "pm2_5": 90.0,
    "pm10": 250.0,
    "so2": 40.0,
}


def detect_prediction_spikes(
    predictions: Mapping[str, Any],
    thresholds: Mapping[str, float] | None = None,
) -> list[dict[str, Any]]:
    active_thresholds = dict(DEFAULT_SPIKE_THRESHOLDS)
    if thresholds:
        active_thresholds.update(thresholds)

    alerts: list[dict[str, Any]] = []
    for pollutant, threshold in active_thresholds.items():
        value = float(predictions.get(pollutant, 0.0))
        if value >= threshold:
            alerts.append(
                {
                    "pollutant": pollutant,
                    "value": round(value, 3),
                    "threshold": threshold,
                    "severity": "high" if value >= threshold * 1.25 else "elevated",
                    "message": f"{pollutant.upper()} is unusually high for this demo threshold.",
                }
            )
    return alerts


def rolling_zscore_alerts(
    data: pd.DataFrame,
    pollutant: str,
    window: int = 24,
    z_threshold: float = 3.0,
) -> pd.DataFrame:
    if pollutant not in data.columns:
        return pd.DataFrame(columns=["date_time", "region", "pollutant", "value", "z_score"])

    frame = data.copy()
    group_columns = ["region"] if "region" in frame.columns else []
    if "date_time" in frame.columns:
        frame = frame.sort_values(group_columns + ["date_time"], kind="stable")

    def add_group_zscore(group: pd.DataFrame) -> pd.DataFrame:
        shifted = pd.to_numeric(group[pollutant], errors="coerce").shift(1)
        mean = shifted.rolling(window, min_periods=max(3, window // 4)).mean()
        std = shifted.rolling(window, min_periods=max(3, window // 4)).std().replace(0, np.nan)
        group = group.copy()
        group["z_score"] = (pd.to_numeric(group[pollutant], errors="coerce") - mean) / std
        return group

    scored = frame.groupby(group_columns, group_keys=False).apply(add_group_zscore) if group_columns else add_group_zscore(frame)
    alerts = scored[scored["z_score"].abs() >= z_threshold].copy()
    alerts["pollutant"] = pollutant
    alerts["value"] = alerts[pollutant]
    output_columns = [column for column in ["date_time", "region", "pollutant", "value", "z_score"] if column in alerts.columns]
    return alerts[output_columns].reset_index(drop=True)
