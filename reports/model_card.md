# Model Card: AirSense AI

## Intended Use

AirSense AI forecasts next-step PM2.5, PM10, and SO2 values for a portfolio and internship demonstration using historical station readings from four Raipur monitoring regions.

## Data

The processed dataset combines AIIMS, Bhatagaon, IGKV, and SILTARA DCR workbooks into one time-series table with hourly and quarter-hourly records.

## Model

The current deployable artifact uses Scikit-learn Random Forest regressors:

- `multi_output`: one model predicts all three pollutants together.
- `single_target`: one model per pollutant.
- `best`: chooses the lower-RMSE strategy per region and target based on test metrics.

## Evaluation

Evaluation uses chronological train, validation, and test splits with RMSE, MAE, and R2. Reports are generated under `outputs/<run>/reports/`.

## Limitations

This is a predictive modeling demo, not a certified environmental compliance or emergency alert system. Prediction quality depends on sensor quality, missing values, station coverage, and whether final quarter-hourly training has been run.
