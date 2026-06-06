from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = PROJECT_ROOT / "data" / "processed" / "all_regions_combined.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "air_quality_models"
TARGET_COLUMNS = ["pm2_5", "pm10", "so2"]
BASE_NUMERIC_COLUMNS = [
    "benz",
    "co",
    "hum",
    "nh3",
    "no",
    "no2",
    "nox",
    "o3",
    "pm10",
    "pm2_5",
    "rg",
    "so2",
    "sr",
    "temp",
    "wd",
    "ws",
]
ROLLING_COLUMNS = ["pm2_5", "pm10", "so2", "temp", "hum", "ws"]
DEFAULT_LAGS_BY_GRANULARITY = {
    "hourly": [1, 3, 6, 12, 24, 48, 168],
    "quarter_hourly": [1, 4, 12, 24, 96, 192, 672],
}
DEFAULT_WINDOWS_BY_GRANULARITY = {
    "hourly": [3, 6, 24, 72],
    "quarter_hourly": [4, 12, 24, 96],
}
GRANULARITY_TO_FREQ = {
    "hourly": "h",
    "quarter_hourly": "15min",
}
IMPOSSIBLE_VALUE_THRESHOLD = 1_000_000
MODEL_REGION_COLUMN = "region"


def coerce_datetime_series(series: pd.Series) -> pd.Series:
    try:
        return pd.to_datetime(series, errors="coerce", format="mixed")
    except TypeError:
        return pd.to_datetime(series, errors="coerce")


@dataclass(frozen=True)
class SplitConfig:
    train_end: str
    validation_end: str


@dataclass
class TargetTransformer:
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train multi-output and single-target air-pollution forecasting models "
            "for the combined all-region DCR dataset."
        )
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--granularity",
        choices=sorted(GRANULARITY_TO_FREQ),
        default="quarter_hourly",
        help="Use hourly or quarter-hourly records for forecasting.",
    )
    parser.add_argument(
        "--train-end",
        default="2024-12-31 23:45:00",
        help="Inclusive training end timestamp.",
    )
    parser.add_argument(
        "--validation-end",
        default="2025-06-30 23:45:00",
        help="Inclusive validation end timestamp.",
    )
    parser.add_argument(
        "--horizon-steps",
        type=int,
        default=1,
        help="Forecast horizon measured in dataset steps.",
    )
    parser.add_argument(
        "--lags",
        type=int,
        nargs="*",
        default=None,
        help="Lag steps used for feature engineering.",
    )
    parser.add_argument(
        "--rolling-windows",
        type=int,
        nargs="*",
        default=None,
        help="Rolling windows used for mean and std features.",
    )
    parser.add_argument(
        "--sample-points",
        type=int,
        default=96,
        help="Number of final test points to show per region in the time-series plot.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed used in the regressors.",
    )
    parser.add_argument(
        "--feature-clip-lower",
        type=float,
        default=0.005,
        help="Lower quantile used to clip extreme feature outliers.",
    )
    parser.add_argument(
        "--feature-clip-upper",
        type=float,
        default=0.995,
        help="Upper quantile used to clip extreme feature outliers.",
    )
    parser.add_argument(
        "--target-clip-upper",
        type=float,
        default=0.995,
        help="Upper quantile used to clip extreme target outliers during training.",
    )
    parser.add_argument(
        "--disable-log-target-transform",
        action="store_true",
        help="Disable log1p target transformation.",
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=140,
        help="Number of trees in each Random Forest model.",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=18,
        help="Maximum tree depth for the Random Forest models.",
    )
    parser.add_argument(
        "--max-samples",
        type=float,
        default=0.35,
        help="Bootstrap sample fraction used per tree.",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="Parallel workers used by Random Forest. Use -1 in Colab or on an unrestricted machine.",
    )
    return parser.parse_args()


def ensure_output_dirs(output_dir: Path) -> dict[str, Path]:
    paths = {
        "root": output_dir,
        "models": output_dir / "models",
        "plots": output_dir / "plots",
        "reports": output_dir / "reports",
        "predictions": output_dir / "predictions",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def load_base_dataframe(dataset_path: Path, granularity: str) -> pd.DataFrame:
    df = pd.read_csv(dataset_path, parse_dates=["date_time"], low_memory=False)
    df["date_time"] = coerce_datetime_series(df["date_time"])
    df = df[df["date_time"].notna()].copy()
    df = df[df["granularity"] == granularity].copy()
    if df.empty:
        raise RuntimeError(f"No rows found for granularity {granularity!r}.")
    df["date_time"] = df["date_time"].dt.round(GRANULARITY_TO_FREQ[granularity])
    df = df.sort_values([MODEL_REGION_COLUMN, "date_time"], kind="stable")
    numeric_cols = [column for column in BASE_NUMERIC_COLUMNS if column in df.columns]
    for column in numeric_cols:
        df[column] = pd.to_numeric(df[column], errors="coerce")
        df.loc[df[column].abs() >= IMPOSSIBLE_VALUE_THRESHOLD, column] = np.nan
    return df


def summarize_region_coverage(df: pd.DataFrame) -> pd.DataFrame:
    summary_rows: list[dict[str, object]] = []
    for region, region_df in df.groupby(MODEL_REGION_COLUMN, sort=True):
        summary_rows.append(
            {
                "region": region,
                "rows": int(len(region_df)),
                "start_time": region_df["date_time"].min(),
                "end_time": region_df["date_time"].max(),
                "pm2_5_non_null": int(region_df["pm2_5"].notna().sum()),
                "pm10_non_null": int(region_df["pm10"].notna().sum()),
                "so2_non_null": int(region_df["so2"].notna().sum()),
            }
        )
    return pd.DataFrame(summary_rows).sort_values("region", kind="stable")


def build_regular_time_frame(region_df: pd.DataFrame, granularity: str) -> pd.DataFrame:
    deduplicated = region_df.sort_values("date_time", kind="stable").drop_duplicates(
        subset=["date_time"], keep="last"
    )
    freq = GRANULARITY_TO_FREQ[granularity]
    base = deduplicated.set_index("date_time").sort_index()
    full_index = pd.date_range(base.index.min(), base.index.max(), freq=freq)
    frame = base.reindex(full_index)
    frame.index.name = "date_time"
    frame[MODEL_REGION_COLUMN] = region_df[MODEL_REGION_COLUMN].iloc[0]
    frame["observed_row"] = frame["granularity"].notna().astype(int)
    frame["granularity"] = granularity
    if "station_name" in frame.columns:
        frame["station_name"] = frame["station_name"].ffill().bfill()
    return frame


def add_time_features(frame: pd.DataFrame) -> pd.DataFrame:
    index = frame.index
    frame["year"] = index.year
    frame["month"] = index.month
    frame["day"] = index.day
    frame["hour"] = index.hour
    frame["minute"] = index.minute
    frame["day_of_week"] = index.dayofweek
    frame["day_of_year"] = index.dayofyear
    frame["week_of_year"] = index.isocalendar().week.astype(int)
    frame["is_weekend"] = (index.dayofweek >= 5).astype(int)
    frame["hour_sin"] = np.sin(2 * np.pi * frame["hour"] / 24)
    frame["hour_cos"] = np.cos(2 * np.pi * frame["hour"] / 24)
    frame["minute_sin"] = np.sin(2 * np.pi * frame["minute"] / 60)
    frame["minute_cos"] = np.cos(2 * np.pi * frame["minute"] / 60)
    frame["day_of_week_sin"] = np.sin(2 * np.pi * frame["day_of_week"] / 7)
    frame["day_of_week_cos"] = np.cos(2 * np.pi * frame["day_of_week"] / 7)
    frame["month_sin"] = np.sin(2 * np.pi * frame["month"] / 12)
    frame["month_cos"] = np.cos(2 * np.pi * frame["month"] / 12)
    frame["day_of_year_sin"] = np.sin(2 * np.pi * frame["day_of_year"] / 365.25)
    frame["day_of_year_cos"] = np.cos(2 * np.pi * frame["day_of_year"] / 365.25)

    if "wd" in frame.columns:
        frame["wd_sin"] = np.sin(np.deg2rad(frame["wd"]))
        frame["wd_cos"] = np.cos(np.deg2rad(frame["wd"]))
    return frame


def add_lag_features(frame: pd.DataFrame, lag_columns: Iterable[str], lags: Iterable[int]) -> pd.DataFrame:
    lag_data: dict[str, pd.Series] = {}
    for column in lag_columns:
        for lag in lags:
            lag_data[f"{column}_lag_{lag}"] = frame[column].shift(lag)
    return pd.concat([frame, pd.DataFrame(lag_data, index=frame.index)], axis=1)


def add_rolling_features(
    frame: pd.DataFrame, rolling_columns: Iterable[str], windows: Iterable[int]
) -> pd.DataFrame:
    rolling_data: dict[str, pd.Series] = {}
    for column in rolling_columns:
        shifted = frame[column].shift(1)
        for window in windows:
            min_periods = max(2, window // 2)
            rolling_data[f"{column}_roll_mean_{window}"] = shifted.rolling(
                window, min_periods=min_periods
            ).mean()
            rolling_data[f"{column}_roll_std_{window}"] = shifted.rolling(
                window, min_periods=min_periods
            ).std()
    return pd.concat([frame, pd.DataFrame(rolling_data, index=frame.index)], axis=1)


def add_targets(frame: pd.DataFrame, targets: Iterable[str], horizon_steps: int) -> pd.DataFrame:
    target_data: dict[str, pd.Series] = {
        "observed_future": frame["observed_row"].shift(-horizon_steps)
    }
    for target in targets:
        target_data[f"target_{target}"] = frame[target].shift(-horizon_steps)
    return pd.concat([frame, pd.DataFrame(target_data, index=frame.index)], axis=1)


def prepare_region_frame(
    region_df: pd.DataFrame,
    granularity: str,
    lags: list[int],
    rolling_windows: list[int],
    horizon_steps: int,
) -> pd.DataFrame:
    frame = build_regular_time_frame(region_df, granularity=granularity)
    frame = add_time_features(frame)
    lag_columns = [column for column in BASE_NUMERIC_COLUMNS if column in frame.columns]
    frame = add_lag_features(frame, lag_columns=lag_columns, lags=lags)
    frame = add_rolling_features(frame, rolling_columns=ROLLING_COLUMNS, windows=rolling_windows)
    frame = add_targets(frame, targets=TARGET_COLUMNS, horizon_steps=horizon_steps)
    return frame.reset_index().rename(columns={"index": "date_time"})


def prepare_model_frame(
    df: pd.DataFrame,
    granularity: str,
    lags: list[int],
    rolling_windows: list[int],
    horizon_steps: int,
) -> pd.DataFrame:
    region_frames: list[pd.DataFrame] = []
    for _, region_df in df.groupby(MODEL_REGION_COLUMN, sort=True):
        region_frames.append(
            prepare_region_frame(
                region_df=region_df,
                granularity=granularity,
                lags=lags,
                rolling_windows=rolling_windows,
                horizon_steps=horizon_steps,
            )
        )
    model_frame = pd.concat(region_frames, ignore_index=True, sort=False)
    region_dummies = pd.get_dummies(model_frame[MODEL_REGION_COLUMN], prefix="region", dtype=int)
    model_frame = pd.concat([model_frame, region_dummies], axis=1)
    return model_frame.sort_values([MODEL_REGION_COLUMN, "date_time"], kind="stable").reset_index(drop=True)


def get_feature_columns(model_frame: pd.DataFrame) -> list[str]:
    excluded_columns = {
        "date_time",
        "granularity",
        MODEL_REGION_COLUMN,
        "station_name",
        "report_type",
        "time_base",
        "source_workbook",
        "source_relative_path",
        "source_sheet",
        "source_month_folder",
        "source_year",
        "selected_row_non_null_score",
        "source_row_count",
        "source_file_count",
        "observed_row",
        "observed_future",
        *[f"target_{target}" for target in TARGET_COLUMNS],
    }
    return [
        column
        for column in model_frame.columns
        if column not in excluded_columns and pd.api.types.is_numeric_dtype(model_frame[column])
    ]


def build_model_pipeline(
    random_state: int,
    lower_quantile: float | None,
    upper_quantile: float | None,
    n_estimators: int,
    max_depth: int,
    max_samples: float,
    n_jobs: int,
) -> Pipeline:
    estimator = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=2,
        max_features="sqrt",
        bootstrap=True,
        max_samples=max_samples,
        random_state=random_state,
        n_jobs=n_jobs,
    )
    steps: list[tuple[str, object]] = []
    if lower_quantile is not None and upper_quantile is not None:
        steps.append(
            (
                "clipper",
                QuantileClipper(
                    lower_quantile=lower_quantile,
                    upper_quantile=upper_quantile,
                ),
            )
        )
    steps.append(("imputer", SimpleImputer(strategy="median", add_indicator=True)))
    steps.append(("model", estimator))
    return Pipeline(steps=steps)


def apply_feature_quality_filters(model_frame: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    filtered = model_frame.copy()
    for column in feature_columns:
        numeric_series = pd.to_numeric(filtered[column], errors="coerce")
        filtered.loc[numeric_series.abs() >= IMPOSSIBLE_VALUE_THRESHOLD, column] = np.nan
    return filtered


def split_masks(model_frame: pd.DataFrame, split_config: SplitConfig) -> dict[str, pd.Series]:
    common_mask = (
        (model_frame["observed_row"] == 1)
        & (model_frame["observed_future"] == 1)
        & model_frame[[f"target_{target}" for target in TARGET_COLUMNS]].notna().all(axis=1)
    )

    train_end = pd.Timestamp(split_config.train_end)
    validation_end = pd.Timestamp(split_config.validation_end)

    return {
        "train": common_mask & (model_frame["date_time"] <= train_end),
        "validation": common_mask
        & (model_frame["date_time"] > train_end)
        & (model_frame["date_time"] <= validation_end),
        "test": common_mask & (model_frame["date_time"] > validation_end),
    }


def evaluate_predictions(
    y_true: pd.DataFrame, y_pred: pd.DataFrame, strategy: str, split_name: str
) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    for target in TARGET_COLUMNS:
        rows.append(
            {
                "strategy": strategy,
                "split": split_name,
                "target": target,
                "rmse": float(np.sqrt(mean_squared_error(y_true[target], y_pred[target]))),
                "mae": float(mean_absolute_error(y_true[target], y_pred[target])),
                "r2": float(r2_score(y_true[target], y_pred[target])),
            }
        )
    return rows


def evaluate_predictions_by_region(
    predictions_df: pd.DataFrame,
    strategy_prefix: str,
    split_name: str,
) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    for region, region_df in predictions_df.groupby("region", sort=True):
        for target in TARGET_COLUMNS:
            actual_column = f"actual_{target}"
            predicted_column = f"{strategy_prefix}_{target}"
            rows.append(
                {
                    "region": region,
                    "strategy": strategy_prefix.replace("_pred", ""),
                    "split": split_name,
                    "target": target,
                    "rmse": float(
                        np.sqrt(mean_squared_error(region_df[actual_column], region_df[predicted_column]))
                    ),
                    "mae": float(mean_absolute_error(region_df[actual_column], region_df[predicted_column])),
                    "r2": float(r2_score(region_df[actual_column], region_df[predicted_column])),
                    "rows": int(len(region_df)),
                }
            )
    return rows


def plot_metric_comparison(metrics_df: pd.DataFrame, output_path: Path) -> None:
    test_metrics = metrics_df[metrics_df["split"] == "test"].copy()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    sns.barplot(data=test_metrics, x="target", y="rmse", hue="strategy", ax=axes[0])
    sns.barplot(data=test_metrics, x="target", y="mae", hue="strategy", ax=axes[1])
    axes[0].set_title("All-Region Test RMSE by Target")
    axes[1].set_title("All-Region Test MAE by Target")
    for axis in axes:
        axis.set_xlabel("Target")
        axis.set_ylabel(axis.get_ylabel().upper())
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_scatter_comparison(
    actual_df: pd.DataFrame,
    multi_pred_df: pd.DataFrame,
    single_pred_df: pd.DataFrame,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(len(TARGET_COLUMNS), 1, figsize=(7, 16), constrained_layout=True)
    for axis, target in zip(axes, TARGET_COLUMNS):
        axis.scatter(actual_df[target], multi_pred_df[target], alpha=0.2, label="Multi-output")
        axis.scatter(actual_df[target], single_pred_df[target], alpha=0.2, label="Single-target")
        combined_min = min(actual_df[target].min(), multi_pred_df[target].min(), single_pred_df[target].min())
        combined_max = max(actual_df[target].max(), multi_pred_df[target].max(), single_pred_df[target].max())
        axis.plot([combined_min, combined_max], [combined_min, combined_max], linestyle="--", color="black")
        axis.set_title(f"Predicted vs Actual: {target}")
        axis.set_xlabel("Actual")
        axis.set_ylabel("Predicted")
        axis.legend()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_region_time_series(
    predictions_df: pd.DataFrame,
    output_path: Path,
    sample_points: int,
) -> None:
    regions = sorted(predictions_df["region"].unique())
    fig, axes = plt.subplots(
        len(regions),
        len(TARGET_COLUMNS),
        figsize=(18, max(10, 3.4 * len(regions))),
        constrained_layout=True,
        sharex=False,
    )
    if len(regions) == 1:
        axes = np.array([axes])

    for row_index, region in enumerate(regions):
        region_sample = predictions_df[predictions_df["region"] == region].tail(sample_points)
        for column_index, target in enumerate(TARGET_COLUMNS):
            axis = axes[row_index, column_index]
            axis.plot(region_sample["date_time"], region_sample[f"actual_{target}"], label="Actual", linewidth=2)
            axis.plot(
                region_sample["date_time"],
                region_sample[f"multi_pred_{target}"],
                label="Multi-output",
                alpha=0.85,
            )
            axis.plot(
                region_sample["date_time"],
                region_sample[f"single_pred_{target}"],
                label="Single-target",
                alpha=0.85,
            )
            axis.set_title(f"{region}: {target}")
            axis.set_xlabel("Time")
            axis.set_ylabel(target)
            if row_index == 0 and column_index == 0:
                axis.legend()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_best_metric_heatmap(
    best_metrics_df: pd.DataFrame,
    value_column: str,
    title: str,
    output_path: Path,
) -> None:
    pivot_df = best_metrics_df.pivot(index="region", columns="target", values=value_column)
    plt.figure(figsize=(8, 4.8))
    sns.heatmap(pivot_df, annot=True, fmt=".2f", cmap="YlOrRd")
    plt.title(title)
    plt.xlabel("Target")
    plt.ylabel("Region")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()


def build_best_strategy_table(metrics_by_region_df: pd.DataFrame) -> pd.DataFrame:
    test_df = metrics_by_region_df[metrics_by_region_df["split"] == "test"].copy()
    winners = (
        test_df.sort_values(["region", "target", "rmse", "mae"], kind="stable")
        .groupby(["region", "target"], as_index=False)
        .first()
    )
    return winners.sort_values(["region", "target"], kind="stable").reset_index(drop=True)


def write_summary_report(
    output_path: Path,
    dataset_path: Path,
    granularity: str,
    split_config: SplitConfig,
    feature_columns: list[str],
    split_counts: dict[str, int],
    region_coverage_df: pd.DataFrame,
    split_by_region_df: pd.DataFrame,
    overall_metrics_df: pd.DataFrame,
    best_strategy_df: pd.DataFrame,
) -> None:
    test_metrics = overall_metrics_df[overall_metrics_df["split"] == "test"].copy()
    overall_winners = (
        test_metrics.sort_values(["target", "rmse", "mae"], kind="stable")
        .groupby("target", as_index=False)
        .first()[["target", "strategy", "rmse", "mae", "r2"]]
    )

    lines = [
        "# All-Region Air Quality Forecasting Report",
        "",
        "## Configuration",
        f"- Dataset: `{dataset_path}`",
        f"- Granularity used for modeling: `{granularity}`",
        "- Modeling scope: combined cross-region forecasting across AIIMS, Bhatagaon, IGKV, and SILTARA",
        "- Forecast horizon: next time step",
        f"- Train end: `{split_config.train_end}`",
        f"- Validation end: `{split_config.validation_end}`",
        f"- Feature count: `{len(feature_columns)}`",
        "",
        "## Region Coverage Used for Modeling",
        "```text",
        region_coverage_df.to_string(index=False),
        "```",
        "",
        "## Split Sizes",
    ]
    lines.extend(f"- {split_name.title()}: `{count}` rows" for split_name, count in split_counts.items())
    lines.extend(
        [
            "",
            "## Split Sizes by Region",
            "```text",
            split_by_region_df.to_string(index=False),
            "```",
            "",
            "## Overall Test Metrics",
            "```text",
            test_metrics.to_string(index=False),
            "```",
            "",
            "## Best Overall Strategy Per Target",
            "```text",
            overall_winners.to_string(index=False),
            "```",
            "",
            "## Best Strategy by Region and Target",
            "```text",
            best_strategy_df.to_string(index=False),
            "```",
            "",
            "## Interpretation",
            "- `multi_output` uses one shared model to predict PM2.5, PM10, and SO2 together.",
            "- `single_target` uses one separate model per pollutant.",
            "- Lower RMSE and MAE indicate better predictive performance.",
            "- Quarter-hourly data was selected for the final combined model because it provides strong coverage for every region, especially SILTARA.",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    dataset_path = args.dataset.resolve()
    output_dirs = ensure_output_dirs(args.output_dir.resolve())
    split_config = SplitConfig(train_end=args.train_end, validation_end=args.validation_end)
    lags = sorted(set(args.lags or DEFAULT_LAGS_BY_GRANULARITY[args.granularity]))
    rolling_windows = sorted(set(args.rolling_windows or DEFAULT_WINDOWS_BY_GRANULARITY[args.granularity]))

    print("Loading dataset...")
    base_df = load_base_dataframe(dataset_path, granularity=args.granularity)
    region_coverage_df = summarize_region_coverage(base_df)
    region_coverage_df.to_csv(output_dirs["reports"] / "region_coverage.csv", index=False)

    print("Engineering region-specific features...")
    model_frame = prepare_model_frame(
        base_df,
        granularity=args.granularity,
        lags=lags,
        rolling_windows=rolling_windows,
        horizon_steps=args.horizon_steps,
    )
    feature_columns = get_feature_columns(model_frame)
    masks = split_masks(model_frame, split_config=split_config)
    model_frame = apply_feature_quality_filters(model_frame, feature_columns)
    feature_columns = [
        column for column in feature_columns if model_frame.loc[masks["train"], column].notna().any()
    ]

    split_counts = {split_name: int(mask.sum()) for split_name, mask in masks.items()}
    if any(count == 0 for count in split_counts.values()):
        raise RuntimeError(f"At least one split is empty: {split_counts}")

    split_by_region_rows: list[dict[str, object]] = []
    for split_name, mask in masks.items():
        split_region_counts = (
            model_frame.loc[mask, MODEL_REGION_COLUMN].value_counts().sort_index().to_dict()
        )
        for region, count in split_region_counts.items():
            split_by_region_rows.append(
                {
                    "split": split_name,
                    "region": region,
                    "rows": int(count),
                }
            )
    split_by_region_df = pd.DataFrame(split_by_region_rows).sort_values(
        ["split", "region"], kind="stable"
    )
    split_by_region_df.to_csv(output_dirs["reports"] / "split_counts_by_region.csv", index=False)

    X = model_frame[feature_columns]
    y = model_frame[[f"target_{target}" for target in TARGET_COLUMNS]].rename(
        columns={f"target_{target}": target for target in TARGET_COLUMNS}
    )

    X_train = X.loc[masks["train"]]
    y_train = y.loc[masks["train"]]
    X_validation = X.loc[masks["validation"]]
    y_validation = y.loc[masks["validation"]]
    X_test = X.loc[masks["test"]]
    y_test = y.loc[masks["test"]]

    target_transformer = TargetTransformer(
        target_names=TARGET_COLUMNS,
        clip_upper_quantile=args.target_clip_upper,
        log_transform=not args.disable_log_target_transform,
    )
    y_train_transformed = target_transformer.fit_transform(y_train)

    print("Training multi-output model...")
    multi_pipeline = build_model_pipeline(
        random_state=args.random_state,
        lower_quantile=args.feature_clip_lower,
        upper_quantile=args.feature_clip_upper,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        max_samples=args.max_samples,
        n_jobs=args.n_jobs,
    )
    multi_pipeline.fit(X_train, y_train_transformed)

    multi_validation_pred = target_transformer.inverse_transform_frame(
        pd.DataFrame(
            multi_pipeline.predict(X_validation),
            index=X_validation.index,
            columns=TARGET_COLUMNS,
        )
    )
    multi_test_pred = target_transformer.inverse_transform_frame(
        pd.DataFrame(
            multi_pipeline.predict(X_test),
            index=X_test.index,
            columns=TARGET_COLUMNS,
        )
    )

    print("Training single-target models...")
    single_pipelines: dict[str, Pipeline] = {}
    single_validation_pred = pd.DataFrame(index=X_validation.index)
    single_test_pred = pd.DataFrame(index=X_test.index)
    for target in TARGET_COLUMNS:
        pipeline = build_model_pipeline(
            random_state=args.random_state,
            lower_quantile=args.feature_clip_lower,
            upper_quantile=args.feature_clip_upper,
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            max_samples=args.max_samples,
            n_jobs=args.n_jobs,
        )
        transformed_target = y_train_transformed[target]
        pipeline.fit(X_train, transformed_target)
        single_pipelines[target] = pipeline
        single_validation_pred[target] = target_transformer.inverse_transform_frame(
            pd.DataFrame({target: pipeline.predict(X_validation)}, index=X_validation.index)
        )[target]
        single_test_pred[target] = target_transformer.inverse_transform_frame(
            pd.DataFrame({target: pipeline.predict(X_test)}, index=X_test.index)
        )[target]

    print("Saving models...")
    joblib.dump(multi_pipeline, output_dirs["models"] / "multi_output_rf.joblib")
    for target, pipeline in single_pipelines.items():
        joblib.dump(pipeline, output_dirs["models"] / f"single_target_{target}_rf.joblib")

    print("Calculating metrics...")
    overall_metrics_rows: list[dict[str, float | str]] = []
    overall_metrics_rows.extend(
        evaluate_predictions(y_validation, multi_validation_pred, "multi_output", "validation")
    )
    overall_metrics_rows.extend(evaluate_predictions(y_test, multi_test_pred, "multi_output", "test"))
    overall_metrics_rows.extend(
        evaluate_predictions(y_validation, single_validation_pred, "single_target", "validation")
    )
    overall_metrics_rows.extend(
        evaluate_predictions(y_test, single_test_pred, "single_target", "test")
    )
    overall_metrics_df = pd.DataFrame(overall_metrics_rows).sort_values(
        ["split", "target", "strategy"], kind="stable"
    )
    overall_metrics_df.to_csv(output_dirs["reports"] / "metrics_overall.csv", index=False)

    predictions_validation = model_frame.loc[masks["validation"], ["date_time", "region"]].copy()
    predictions_test = model_frame.loc[masks["test"], ["date_time", "region"]].copy()
    for predictions_df, actual_frame, multi_frame, single_frame in [
        (predictions_validation, y_validation, multi_validation_pred, single_validation_pred),
        (predictions_test, y_test, multi_test_pred, single_test_pred),
    ]:
        for target in TARGET_COLUMNS:
            predictions_df[f"actual_{target}"] = actual_frame[target].values
            predictions_df[f"multi_pred_{target}"] = multi_frame[target].values
            predictions_df[f"single_pred_{target}"] = single_frame[target].values

    predictions_validation.to_csv(output_dirs["predictions"] / "validation_predictions.csv", index=False)
    predictions_test.to_csv(output_dirs["predictions"] / "test_predictions.csv", index=False)

    metrics_by_region_rows: list[dict[str, float | str]] = []
    metrics_by_region_rows.extend(
        evaluate_predictions_by_region(predictions_validation, "multi_pred", "validation")
    )
    metrics_by_region_rows.extend(evaluate_predictions_by_region(predictions_test, "multi_pred", "test"))
    metrics_by_region_rows.extend(
        evaluate_predictions_by_region(predictions_validation, "single_pred", "validation")
    )
    metrics_by_region_rows.extend(evaluate_predictions_by_region(predictions_test, "single_pred", "test"))
    metrics_by_region_df = pd.DataFrame(metrics_by_region_rows).sort_values(
        ["split", "region", "target", "strategy"], kind="stable"
    )
    metrics_by_region_df.to_csv(output_dirs["reports"] / "metrics_by_region.csv", index=False)

    best_strategy_df = build_best_strategy_table(metrics_by_region_df)
    best_strategy_df.to_csv(output_dirs["reports"] / "best_strategy_by_region_target.csv", index=False)

    print("Generating plots...")
    sns.set_theme(style="whitegrid")
    plot_metric_comparison(overall_metrics_df, output_dirs["plots"] / "metric_comparison.png")
    plot_scatter_comparison(
        actual_df=predictions_test.rename(columns={f"actual_{target}": target for target in TARGET_COLUMNS})[
            ["date_time", *TARGET_COLUMNS]
        ],
        multi_pred_df=predictions_test.rename(
            columns={f"multi_pred_{target}": target for target in TARGET_COLUMNS}
        )[TARGET_COLUMNS],
        single_pred_df=predictions_test.rename(
            columns={f"single_pred_{target}": target for target in TARGET_COLUMNS}
        )[TARGET_COLUMNS],
        output_path=output_dirs["plots"] / "predicted_vs_actual_scatter.png",
    )
    plot_region_time_series(
        predictions_df=predictions_test,
        output_path=output_dirs["plots"] / "region_prediction_timeseries.png",
        sample_points=args.sample_points,
    )
    plot_best_metric_heatmap(
        best_metrics_df=best_strategy_df,
        value_column="rmse",
        title="Best Test RMSE by Region and Target",
        output_path=output_dirs["plots"] / "best_strategy_rmse_heatmap.png",
    )
    plot_best_metric_heatmap(
        best_metrics_df=best_strategy_df,
        value_column="mae",
        title="Best Test MAE by Region and Target",
        output_path=output_dirs["plots"] / "best_strategy_mae_heatmap.png",
    )

    metadata = {
        "dataset": str(dataset_path),
        "granularity": args.granularity,
        "targets": TARGET_COLUMNS,
        "feature_count": len(feature_columns),
        "features": feature_columns,
        "split_counts": split_counts,
        "train_end": args.train_end,
        "validation_end": args.validation_end,
        "horizon_steps": args.horizon_steps,
        "lags": lags,
        "rolling_windows": rolling_windows,
        "feature_clip_lower": args.feature_clip_lower,
        "feature_clip_upper": args.feature_clip_upper,
        "target_clip_upper": args.target_clip_upper,
        "log_target_transform": not args.disable_log_target_transform,
        "n_estimators": args.n_estimators,
        "max_depth": args.max_depth,
        "max_samples": args.max_samples,
        "n_jobs": args.n_jobs,
        "regions": sorted(base_df["region"].unique().tolist()),
    }
    (output_dirs["reports"] / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    write_summary_report(
        output_path=output_dirs["reports"] / "summary_report.md",
        dataset_path=dataset_path,
        granularity=args.granularity,
        split_config=split_config,
        feature_columns=feature_columns,
        split_counts=split_counts,
        region_coverage_df=region_coverage_df,
        split_by_region_df=split_by_region_df,
        overall_metrics_df=overall_metrics_df,
        best_strategy_df=best_strategy_df,
    )

    print("Done.")
    print(f"Output directory: {output_dirs['root']}")
    print(f"Training rows: {split_counts['train']}")
    print(f"Validation rows: {split_counts['validation']}")
    print(f"Test rows: {split_counts['test']}")
    print(overall_metrics_df[overall_metrics_df["split"] == "test"].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
