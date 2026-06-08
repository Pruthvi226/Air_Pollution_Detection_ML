# AirSense AI Demo Script

## 60-90 Second Version

This is AirSense AI, an end-to-end machine learning project for air-quality forecasting. The project starts with messy DCR air-quality workbooks from four Raipur monitoring regions. I built a preprocessing pipeline to extract sheets, normalize columns, parse timestamps, handle duplicates, and create a combined time-series dataset.

After cleaning, I engineered forecasting features such as lag values, rolling averages, rolling standard deviation, cyclic time encodings, and region indicators. I trained models to predict PM2.5, PM10, and SO2, then evaluated them using RMSE, MAE, R2, and region-wise metrics with a chronological split to avoid data leakage.

I upgraded the project beyond simple prediction by adding AQI-style risk interpretation, pollution spike detection, explainability, a Streamlit dashboard, a FastAPI prediction endpoint, and a CLI predictor. This makes it closer to a real AI application instead of only a notebook.

## 5-Minute Technical Walkthrough

1. Open the README and explain the architecture.
2. Show `scripts/prepare_combined_dataset.py` for raw DCR ingestion and cleanup.
3. Show `scripts/train_air_quality_models.py` for chronological splitting, feature engineering, training, metrics, and bundle export.
4. Run `python scripts/predict_cli.py --region SILTARA --pm25 78 --pm10 145 --so2 14 --temp 31 --hum 62 --ws 2.1`.
5. Open the Streamlit dashboard, use the Live Prediction tab, and click Run Demo Scenario.
6. Show the AQI risk card, Anomaly Detection tab, Explainability tab, and AI Report tab.
7. Open FastAPI `/metadata` and `/predict` to show the deployment contract.
8. Close with the model card, limitations, and future real-time integration path.

## Key Talking Points

- The train/test split is chronological to reduce time-series leakage.
- The inference bundle is shared by Streamlit, FastAPI, and CLI.
- AQI labels are demo risk categories, not official compliance advice.
- Raw data and full processed datasets stay out of Git because they are large.

## Closing Line

AirSense AI demonstrates the full AI internship workflow: preprocessing, feature engineering, model training, evaluation, explainability, risk interpretation, automation, and deployment packaging.
