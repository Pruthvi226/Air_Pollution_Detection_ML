from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


@dataclass
class TargetTransformer:
    """Train-time target transform with a deploy-time inverse transform."""

    target_names: list[str]
    clip_upper_quantile: float | None
    log_transform: bool
    lower_bound: float = 0.0

    def __post_init__(self) -> None:
        self.upper_bounds_: dict[str, float | None] = {}

    def fit(self, y_frame: pd.DataFrame) -> "TargetTransformer":
        for target in self.target_names:
            series = pd.to_numeric(y_frame[target], errors="coerce")
            series = series[series.notna()]
            if series.empty or self.clip_upper_quantile is None:
                self.upper_bounds_[target] = None
            else:
                self.upper_bounds_[target] = float(series.quantile(self.clip_upper_quantile))
        return self

    def transform(self, y_frame: pd.DataFrame) -> pd.DataFrame:
        transformed = y_frame.copy()
        for target in self.target_names:
            series = pd.to_numeric(transformed[target], errors="coerce").clip(lower=self.lower_bound)
            upper_bound = self.upper_bounds_.get(target)
            if upper_bound is not None and pd.notna(upper_bound):
                series = series.clip(upper=upper_bound)
            if self.log_transform:
                series = np.log1p(series)
            transformed[target] = series
        return transformed

    def fit_transform(self, y_frame: pd.DataFrame) -> pd.DataFrame:
        return self.fit(y_frame).transform(y_frame)

    def inverse_transform_frame(self, y_frame: pd.DataFrame) -> pd.DataFrame:
        restored = y_frame.copy()
        for target in [name for name in self.target_names if name in restored.columns]:
            series = pd.to_numeric(restored[target], errors="coerce")
            if self.log_transform:
                series = np.expm1(series)
            series = series.clip(lower=self.lower_bound)
            upper_bound = self.upper_bounds_.get(target)
            if upper_bound is not None and pd.notna(upper_bound):
                series = series.clip(upper=upper_bound)
            restored[target] = series
        return restored


class QuantileClipper(BaseEstimator, TransformerMixin):
    """Scikit-learn transformer that clips feature outliers by train quantiles."""

    def __init__(self, lower_quantile: float = 0.005, upper_quantile: float = 0.995):
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> "QuantileClipper":
        X_array = np.asarray(X, dtype=float)
        self.lower_bounds_ = np.nanquantile(X_array, self.lower_quantile, axis=0)
        self.upper_bounds_ = np.nanquantile(X_array, self.upper_quantile, axis=0)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X_array = np.asarray(X, dtype=float)
        return np.clip(X_array, self.lower_bounds_, self.upper_bounds_)
