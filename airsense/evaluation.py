from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def regression_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float]:
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def leaderboard_from_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    if metrics.empty:
        return pd.DataFrame(columns=["strategy", "target", "rmse", "mae", "r2"])
    return (
        metrics.sort_values(["target", "rmse", "mae"], kind="stable")
        .groupby(["target"], as_index=False)
        .first()
        .sort_values(["target"], kind="stable")
    )
