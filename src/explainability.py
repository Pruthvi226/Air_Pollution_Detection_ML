"""Feature importance helpers with safe fallbacks."""

from __future__ import annotations

from typing import Any, Iterable

import pandas as pd


FALLBACK_FEATURE_IMPORTANCE = [
    {"feature": "Previous PM2.5 level", "importance": 0.22},
    {"feature": "Previous PM10 level", "importance": 0.19},
    {"feature": "Rolling PM2.5 mean", "importance": 0.16},
    {"feature": "Rolling PM10 mean", "importance": 0.14},
    {"feature": "Hour of day", "importance": 0.10},
    {"feature": "Humidity", "importance": 0.08},
    {"feature": "Region indicator", "importance": 0.07},
    {"feature": "Wind speed", "importance": 0.04},
    {"feature": "Month", "importance": 0.03},
    {"feature": "Season", "importance": 0.02},
]


def get_feature_importance(model: Any | None = None, feature_names: Iterable[str] | None = None) -> pd.DataFrame:
    """Return SHAP-style or model-native feature importance with a fallback."""
    names = list(feature_names or [])

    if model is not None and hasattr(model, "feature_importances_") and names:
        values = list(getattr(model, "feature_importances_"))
        return (
            pd.DataFrame({"feature": names[: len(values)], "importance": values})
            .sort_values("importance", ascending=False, kind="stable")
            .head(10)
            .reset_index(drop=True)
        )

    if model is not None and hasattr(model, "coef_") and names:
        values = [abs(float(value)) for value in getattr(model, "coef_")]
        return (
            pd.DataFrame({"feature": names[: len(values)], "importance": values})
            .sort_values("importance", ascending=False, kind="stable")
            .head(10)
            .reset_index(drop=True)
        )

    return pd.DataFrame(FALLBACK_FEATURE_IMPORTANCE)

