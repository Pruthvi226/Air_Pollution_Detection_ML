from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_RUN = PROJECT_ROOT / "outputs" / "smoke_air_quality_models"
MODEL_DIR_ENV = "AIRSENSE_MODEL_DIR"

TARGET_COLUMNS = ["pm2_5", "pm10", "so2"]
SUPPORTED_REGIONS = ["AIIMS", "BHATAGAON", "IGKV", "SILTARA"]

TIMESTAMP_COLUMN = "date_time"
REGION_COLUMN = "region"
GRANULARITY_COLUMN = "granularity"

CORE_INPUT_COLUMNS = [
    "region",
    "date_time",
    "pm2_5",
    "pm10",
    "so2",
    "temp",
    "hum",
    "ws",
    "wd",
]

OPTIONAL_SENSOR_COLUMNS = [
    "benz",
    "co",
    "nh3",
    "no",
    "no2",
    "nox",
    "o3",
    "rg",
    "sr",
]

PROJECT_VERSION = "1.1.0"
