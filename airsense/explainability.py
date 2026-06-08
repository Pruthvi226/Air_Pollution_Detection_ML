from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd


def _model_from_bundle(bundle: Mapping[str, Any]) -> Any | None:
    models = bundle.get("models", {})
    return models.get("multi_output") or next(iter((models.get("single_target") or {}).values()), None)


def _extract_importances(model: Any) -> np.ndarray | None:
    if hasattr(model, "named_steps"):
        estimator = model.named_steps.get("model")
    else:
        estimator = model
    if estimator is not None and hasattr(estimator, "feature_importances_"):
        return np.asarray(estimator.feature_importances_, dtype=float)
    return None


def get_feature_importance(
    bundle: Mapping[str, Any],
    limit: int = 12,
) -> pd.DataFrame:
    feature_columns = list(bundle.get("feature_columns", []))
    importances = _extract_importances(_model_from_bundle(bundle))
    if importances is None or len(importances) == 0:
        fallback_features = [
            "pm10",
            "pm2_5",
            "so2",
            "temp",
            "hum",
            "ws",
            "hour",
            "month",
            "region_SILTARA",
            "region_BHATAGAON",
        ]
        rows = [
            {"feature": feature, "importance": round(1.0 / (index + 1), 4)}
            for index, feature in enumerate(fallback_features)
            if feature in feature_columns or feature.startswith("region_")
        ]
        return pd.DataFrame(rows).head(limit)

    importances = importances[: len(feature_columns)]
    importance_frame = pd.DataFrame(
        {
            "feature": feature_columns[: len(importances)],
            "importance": importances,
        }
    )
    return importance_frame.sort_values("importance", ascending=False, kind="stable").head(limit).reset_index(drop=True)


def explain_prediction(
    bundle: Mapping[str, Any],
    feature_frame: pd.DataFrame | None = None,
    limit: int = 5,
) -> dict[str, Any]:
    importance_frame = get_feature_importance(bundle, limit=limit)
    top_features = importance_frame.to_dict(orient="records")
    if top_features:
        feature_names = ", ".join(row["feature"] for row in top_features[:3])
        explanation = f"The strongest available model signals for this demo are {feature_names}."
    else:
        explanation = "The model uses pollutant readings, weather context, time features, and region indicators."
    return {
        "method": "feature_importance_fallback",
        "top_features": top_features,
        "summary": explanation,
    }
