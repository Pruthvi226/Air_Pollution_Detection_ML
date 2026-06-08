from __future__ import annotations

import numpy as np
import pandas as pd

from airsense.config import REGION_COLUMN, TARGET_COLUMNS, TIMESTAMP_COLUMN


def add_cyclic_time_features(frame: pd.DataFrame, timestamp_column: str = TIMESTAMP_COLUMN) -> pd.DataFrame:
    output = frame.copy()
    timestamps = pd.to_datetime(output[timestamp_column], errors="coerce")
    output["hour"] = timestamps.dt.hour
    output["day_of_week"] = timestamps.dt.dayofweek
    output["month"] = timestamps.dt.month
    output["is_weekend"] = (output["day_of_week"] >= 5).astype("Int64")
    output["hour_sin"] = np.sin(2 * np.pi * output["hour"] / 24)
    output["hour_cos"] = np.cos(2 * np.pi * output["hour"] / 24)
    output["month_sin"] = np.sin(2 * np.pi * output["month"] / 12)
    output["month_cos"] = np.cos(2 * np.pi * output["month"] / 12)
    return output


def add_group_lag_features(
    frame: pd.DataFrame,
    columns: list[str] | None = None,
    lags: list[int] | None = None,
    region_column: str = REGION_COLUMN,
    timestamp_column: str = TIMESTAMP_COLUMN,
) -> pd.DataFrame:
    output = frame.sort_values([region_column, timestamp_column], kind="stable").copy()
    active_columns = columns or TARGET_COLUMNS
    active_lags = lags or [1]
    for column in active_columns:
        if column not in output.columns:
            continue
        for lag in active_lags:
            output[f"{column}_lag_{lag}"] = output.groupby(region_column)[column].shift(lag)
    return output


def add_missing_indicators(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    output = frame.copy()
    for column in columns:
        if column in output.columns:
            output[f"is_missing_{column}"] = output[column].isna().astype(int)
    return output
