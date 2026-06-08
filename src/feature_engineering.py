"""Feature engineering helpers for air-quality time series."""

from __future__ import annotations

import numpy as np
import pandas as pd


def add_time_features(frame: pd.DataFrame, date_column: str = "date_time") -> pd.DataFrame:
    """Add leakage-safe calendar and cyclic time features."""
    output = frame.copy()
    timestamp = pd.to_datetime(output[date_column], errors="coerce")
    output["hour"] = timestamp.dt.hour
    output["day"] = timestamp.dt.day
    output["month"] = timestamp.dt.month
    output["day_of_week"] = timestamp.dt.dayofweek
    output["is_weekend"] = (output["day_of_week"] >= 5).astype(int)
    output["is_morning_peak"] = output["hour"].between(7, 10).astype(int)
    output["is_evening_peak"] = output["hour"].between(17, 21).astype(int)
    output["hour_sin"] = np.sin(2 * np.pi * output["hour"] / 24)
    output["hour_cos"] = np.cos(2 * np.pi * output["hour"] / 24)
    output["month_sin"] = np.sin(2 * np.pi * output["month"] / 12)
    output["month_cos"] = np.cos(2 * np.pi * output["month"] / 12)
    return output


def add_lag_and_rolling_features(frame: pd.DataFrame, columns: list[str], group_column: str = "region") -> pd.DataFrame:
    """Create lag and rolling features in chronological order."""
    output = frame.copy()
    group = output.groupby(group_column, group_keys=False) if group_column in output else [(None, output)]
    for column in columns:
        for lag in [1, 3, 6, 12, 24]:
            output[f"{column}_lag_{lag}"] = group[column].shift(lag) if hasattr(group, "__getitem__") else output[column].shift(lag)
        for window in [3, 6, 12, 24]:
            shifted = group[column].shift(1) if hasattr(group, "__getitem__") else output[column].shift(1)
            output[f"{column}_rolling_mean_{window}"] = shifted.rolling(window).mean()
            output[f"{column}_rolling_std_{window}"] = shifted.rolling(window).std()
        shifted = group[column].shift(1) if hasattr(group, "__getitem__") else output[column].shift(1)
        output[f"{column}_rolling_min_24"] = shifted.rolling(24).min()
        output[f"{column}_rolling_max_24"] = shifted.rolling(24).max()
    return output

