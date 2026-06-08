from __future__ import annotations

import pandas as pd


def normalize_sensor_columns(frame: pd.DataFrame) -> pd.DataFrame:
    rename_map = {
        "pm25": "pm2_5",
        "pm2.5": "pm2_5",
        "temperature": "temp",
        "humidity": "hum",
        "wind_speed": "ws",
        "timestamp": "date_time",
    }
    return frame.rename(columns={column: rename_map.get(str(column).lower(), column) for column in frame.columns})


def validate_required_columns(frame: pd.DataFrame, required_columns: list[str]) -> None:
    missing = [column for column in required_columns if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")
