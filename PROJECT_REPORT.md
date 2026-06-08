# AirSense AI Project Report

## 1. Project Title

**AirSense AI - Intelligent Air Pollution Forecasting & Risk Analytics Dashboard**

## 2. Executive Summary

AirSense AI is an end-to-end machine learning system for forecasting PM2.5, PM10, and SO2 pollution levels across multiple monitoring regions. The project transforms raw air-quality workbooks into modeling-ready time-series features, evaluates global and region-specific model behavior, explains predictions, detects pollution spikes, and presents the complete workflow through a professional Streamlit dashboard, FastAPI service, CLI, reports, and GitHub Pages website.

## 3. Problem Statement

Air pollution data is noisy, time-dependent, and highly region-specific. Raw monitoring files often contain inconsistent formats, timestamp issues, missing values, and pollutant/weather signals that change by location. A useful AI solution must do more than train one model: it must clean data, engineer time-series signals, compare modeling strategies, interpret risk, and communicate results clearly.

## 4. Project Objective

The objective is to build a deployable AI dashboard that can:

- Predict PM2.5, PM10, and SO2.
- Compare global and region-specific model performance.
- Explain why region-level modeling is important.
- Convert forecasts into simplified AQI-style risk categories.
- Detect abnormal pollution spikes.
- Generate a concise air-quality report for presentation.

## 5. Dataset Details

| Metric | Value |
|---|---:|
| Total cleaned records | 586,431 |
| Quarter-hourly records | 467,188 |
| Hourly modeling records | 119,243 |
| Regions processed | 4 |
| Forecast targets | 3 |
| Engineered features | 201 |
| Test rows | 15,772 |

Regions included in the workflow are AIIMS, Bhatagaon, IGKV, and Siltara.

## 6. Data Preprocessing

The preprocessing pipeline standardizes raw DCR files, parses timestamps, normalizes columns, handles missing values, removes duplicate records, and prepares region-aware pollution records for modeling. The cleaned data is aggregated into hourly records for the current deployable model artifact while preserving quarter-hourly data for future high-resolution training.

## 7. Feature Engineering

The feature layer creates time-series and environmental predictors:

- Current pollutant and weather readings.
- Lag features for recent pollutant history.
- Rolling mean and trend statistics.
- Hour and month features.
- Region indicators.
- Target-specific modeling inputs for PM2.5, PM10, and SO2.

## 8. Modeling Strategy

The project compares multiple regression strategies:

- Baseline models for sanity checking.
- LightGBM boosted tree candidates.
- XGBoost boosted tree candidates.
- Global multi-output and single-target aliases selected from validation metrics.
- Region/target strategy selection based on held-out metrics.

This comparison is important because pollution behavior is not uniform across all locations. The final system uses a hybrid strategy: global models provide broad coverage, and region-specific models are preferred when they show stronger target-level performance.

## 9. Global Model Performance

| Target | Best Strategy | RMSE | MAE | R2 | Interpretation |
|---|---|---:|---:|---:|---|
| PM2.5 | LightGBM boosted | 2.58 | 0.94 | 0.978 | Strong boosted forecasting |
| PM10 | LightGBM boosted | 14.87 | 3.07 | 0.883 | Strong boosted forecasting |
| SO2 | LightGBM boosted | 6.06 | 0.55 | 0.245 | Low MAE; spike-sensitive RMSE |

LightGBM is the strongest aggregate candidate in the boosted quarter-hourly run. PM2.5 and PM10 improve sharply compared with the earlier Random Forest baseline. SO2 has low average error but remains sensitive to rare spikes, so it should be monitored separately.

## 10. Region-Specific Performance

| Region | Target | Best R2 | Performance Level |
|---|---|---:|---|
| AIIMS | PM2.5 | 0.984 | Strong |
| IGKV | PM2.5 | 0.980 | Strong |
| IGKV | PM10 | 0.975 | Strong |
| AIIMS | PM10 | 0.969 | Strong |
| Siltara | PM10 | 0.935 | Strong |
| Bhatagaon | SO2 | 0.836 | Strong |
| Siltara | SO2 | 0.683 | Good |

The region-level results show that the boosted selector captures local short-term pollutant behavior well while still exposing SO2 spike sensitivity.

## 11. Main Modeling Insight

The most important result is not only the highest score. The key insight is that boosted lag/rolling models substantially improve short-term PM2.5 and PM10 forecasting, while SO2 still needs residual monitoring because rare spikes dominate RMSE. This demonstrates model diagnosis, error analysis, and a practical improvement strategy.

## 12. Dashboard Implementation

The Streamlit dashboard includes the following pages:

- Overview
- Live Prediction
- Dataset Summary
- Global Performance
- Region Performance
- Region Analytics
- Explainability
- Anomaly Detection
- AI Report
- Project Details

The dashboard is designed as a monitoring product, not a simple notebook output. It uses sidebar navigation, metric cards, charts, tables, risk panels, report generation, and deployment-ready entry points.

## 13. Explainability, AQI Risk, and Anomaly Detection

The dashboard includes:

- Feature importance and SHAP-style explanation fallback.
- Simplified AQI-style risk classification.
- Health recommendation messages.
- Statistical spike detection using current value, rolling average, and rolling standard deviation.
- Rule-based AI-style report generation.

These interpretation layers are project-level analytics and should be validated against official regulatory methods before operational use.

## 14. Application Architecture

The runtime model artifact is reused across multiple surfaces:

| Surface | File |
|---|---|
| Streamlit dashboard | `app/streamlit_app.py` |
| Streamlit Cloud entry point | `app.py` |
| FastAPI service | `app/api.py` |
| CLI predictor | `scripts/predict_cli.py` |
| Shared inference package | `airsense/inference.py` |
| Utility source layer | `src/` |

The repository also includes Docker, Render configuration, Streamlit Cloud support, tests, reports, and GitHub Pages website files.

## 15. Limitations and Future Scope

Current limitations:

- SO2 RMSE remains spike-sensitive and should be monitored separately from MAE.
- The boosted run filters spreadsheet-origin date artifacts before 2022-01-01.
- AQI interpretation is simplified and not an official regulatory AQI calculation.
- The boosted artifact should be retrained periodically as new monitoring data arrives.

Future improvements:

- Add explicit SO2 spike/residual modeling.
- Add real-time pollution board API integration.
- Add 24-hour forecasting.
- Add geospatial heatmaps.
- Add LLM-based report generation.
- Add email or WhatsApp alerts.
- Add model monitoring and scheduled retraining.

## 16. Interview Demo Script

AirSense AI is an end-to-end air pollution forecasting and risk analytics dashboard. I processed 586,431 cleaned monitoring records from four regions and engineered 202 time-series, weather, lag, rolling, and region features. The system predicts PM2.5, PM10, and SO2, then converts predictions into AQI-style risk insights, anomaly alerts, explainability, and an AI-generated summary report.

The boosted quarter-hourly LightGBM run achieved held-out R2 = 0.978 for PM2.5 and R2 = 0.883 for PM10. SO2 achieved low MAE = 0.55 but remains spike-sensitive, so the final system keeps region/target strategy selection and highlights SO2 monitoring as a future improvement.

The final result is good to present because it shows the complete AI workflow: data cleaning, feature engineering, modeling, model evaluation, model diagnosis, risk interpretation, anomaly detection, dashboard deployment, and professional documentation.
