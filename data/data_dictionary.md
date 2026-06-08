# Data Dictionary

## Core Columns

| Column | Meaning |
|---|---|
| `region` | Monitoring region: AIIMS, BHATAGAON, IGKV, or SILTARA |
| `date_time` | Timestamp parsed from the station DCR workbook |
| `granularity` | `hourly` or `quarter_hourly` sample resolution |
| `pm2_5` | PM2.5 particulate reading |
| `pm10` | PM10 particulate reading |
| `so2` | Sulfur dioxide reading |
| `no`, `no2`, `nox`, `nh3`, `co`, `o3`, `benz` | Additional pollutant readings when present |
| `temp`, `hum`, `ws`, `wd`, `sr`, `rg` | Weather and station environment readings |

## Generated Modeling Features

| Feature family | Examples |
|---|---|
| Time fields | `hour`, `month`, `day_of_week`, `week_of_year` |
| Cyclic encodings | `hour_sin`, `hour_cos`, `month_sin`, `month_cos` |
| Lag readings | `pm2_5_lag_1`, `pm10_lag_24`, `so2_lag_168` |
| Rolling statistics | `pm2_5_roll_mean_24`, `temp_roll_std_72` |
| Region indicators | `region_AIIMS`, `region_BHATAGAON`, `region_IGKV`, `region_SILTARA` |

## Data Quality Rules

- Workbook headers are detected automatically from daily sheets.
- Duplicate timestamps are merged by selecting the row with the most non-null measurements.
- Extremely large numeric sensor values are treated as missing during modeling.
- Chronological splits are used so validation and test data occur after training data.
