# Model Card: AirSense AI

## Intended Use

AirSense AI forecasts next-step PM2.5, PM10, and SO2 values for a portfolio and internship demonstration using historical station readings from four Raipur monitoring regions.

## Data

The processed dataset combines AIIMS, Bhatagaon, IGKV, and SILTARA DCR workbooks into one time-series table with hourly and quarter-hourly records.

## Model

The current deployable artifact uses boosted tree regressors trained on the quarter-hourly dataset:

- `baseline_median`: median baseline used for comparison in generated metrics.
- `multi_output_lightgbm` / `single_target_lightgbm`: LightGBM boosted tree candidates.
- `multi_output_xgboost` / `single_target_xgboost`: XGBoost boosted tree candidates.
- `multi_output` and `single_target`: inference aliases that point to the best validation-selected boosted candidates.
- `best`: chooses the lower-RMSE strategy per region and target based on test metrics.

## Inputs and Outputs

Inputs include region, timestamp, current PM2.5, PM10, SO2, temperature, humidity, wind speed, and wind direction. Outputs include next-step pollutant predictions, an AQI-style risk category, a health recommendation, anomaly alerts, and a feature-importance explanation.

## Evaluation

Evaluation uses chronological train, validation, and test splits with RMSE, MAE, and R2. Reports are generated under `outputs/<run>/reports/`.

## Safety and Ethics

The AQI layer is a demo-oriented risk interpretation helper. It is not official regulatory AQI, medical advice, or an emergency alerting system.

## Limitations

This is a predictive modeling demo, not a certified environmental compliance or emergency alert system. Prediction quality depends on sensor quality, missing values, station coverage, and spike behavior. The trainer drops spreadsheet-origin timestamp artifacts before 2022-01-01 so invalid dates do not inflate the regular time grid.
