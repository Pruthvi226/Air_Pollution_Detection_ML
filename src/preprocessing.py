"""Preprocessing helpers for future production training runs."""

from __future__ import annotations

import pandas as pd


def cap_outliers_iqr(frame: pd.DataFrame, columns: list[str], factor: float = 1.5) -> pd.DataFrame:
    """Cap extreme values with IQR bounds for stable forecasting training."""
    output = frame.copy()
    for column in columns:
        if column not in output:
            continue
        q1 = output[column].quantile(0.25)
        q3 = output[column].quantile(0.75)
        iqr = q3 - q1
        output[column] = output[column].clip(lower=q1 - factor * iqr, upper=q3 + factor * iqr)
    return output

