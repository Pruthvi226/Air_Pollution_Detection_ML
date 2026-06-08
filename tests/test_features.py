from __future__ import annotations

import pandas as pd

from airsense.features import add_cyclic_time_features, add_group_lag_features


def test_lag_features_do_not_mix_regions() -> None:
    frame = pd.DataFrame(
        {
            "region": ["AIIMS", "AIIMS", "SILTARA", "SILTARA"],
            "date_time": pd.to_datetime(
                ["2025-01-01 01:00", "2025-01-01 02:00", "2025-01-01 01:00", "2025-01-01 02:00"]
            ),
            "pm2_5": [10, 20, 100, 110],
        }
    )
    features = add_group_lag_features(frame, columns=["pm2_5"], lags=[1])
    first_siltara = features[features["region"] == "SILTARA"].iloc[0]
    assert pd.isna(first_siltara["pm2_5_lag_1"])


def test_cyclic_features_are_created() -> None:
    frame = pd.DataFrame({"date_time": ["2025-01-01 23:00:00"]})
    features = add_cyclic_time_features(frame)
    for column in ["hour_sin", "hour_cos", "month_sin", "month_cos"]:
        assert column in features.columns
