# Experiment Report

## Dataset Build

- Total processed rows: `586,431`
- Hourly rows: `119,243`
- Quarter-hourly rows: `467,188`
- Regions: `AIIMS`, `BHATAGAON`, `IGKV`, `SILTARA`

## Current Deployable Run

The local smoke run uses hourly data so the full stack can be tested quickly. It writes:

- `outputs/smoke_air_quality_models/models/inference_bundle.joblib`
- `outputs/smoke_air_quality_models/reports/metrics_overall.csv`
- `outputs/smoke_air_quality_models/plots/*.png`

## Recommended Final Run

Use the quarter-hourly command in `DEPLOYMENT.md` for the strongest portfolio result, then point `AIRSENSE_MODEL_DIR` at `outputs/air_quality_models`.

## Demo Contract

Inputs:

- Region
- Timestamp
- Current PM2.5, PM10, SO2
- Temperature, humidity, wind speed, wind direction

Outputs:

- PM2.5 forecast
- PM10 forecast
- SO2 forecast
- Risk label
