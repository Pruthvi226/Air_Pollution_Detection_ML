# AirSense AI Project Report

## Project Name

AirSense AI - Intelligent Air Pollution Forecasting & Risk Analytics Dashboard

## Problem

Air pollution monitoring data is messy, time-dependent, and region-specific. Raw station workbooks contain inconsistent formats, timestamps, duplicate records, and pollutant/weather signals that vary by monitoring region. A useful AI system needs more than a single prediction model: it needs preprocessing, feature engineering, evaluation, risk interpretation, explainability, anomaly detection, and a clear dashboard.

## Solution

AirSense AI is an end-to-end machine learning dashboard for air pollution forecasting, AQI risk interpretation, region-wise model evaluation, explainability, and anomaly detection. It converts raw DCR workbooks into a modeling-ready time-series dataset, trains forecasting models for PM2.5, PM10, and SO2, compares global and region-specific performance, and presents results through Streamlit, FastAPI, CLI tooling, and a GitHub Pages website.

## Dataset

| Metric | Value |
|---|---:|
| Total cleaned records | 586,431 |
| Quarter-hourly records | 467,188 |
| Hourly modeling records | 119,243 |
| Regions | 4 |
| Forecast targets | 3 |
| Engineered features | 201 |
| Test rows | 15,772 |

Regions: AIIMS, Bhatagaon, IGKV, and Siltara.

## Modeling

The project uses chronological evaluation and compares baseline, multi-output, and single-target regression strategies. Features include pollutant readings, weather signals, time encodings, lag values, rolling statistics, and region indicators.

## Global Performance

| Target | Best strategy | RMSE | MAE | R2 | Interpretation |
|---|---|---:|---:|---:|---|
| PM10 | Single-target | 31.10 | 18.29 | 0.581 | Good global baseline |
| SO2 | Single-target | 2.39 | 1.26 | 0.431 | Moderate global signal |
| PM2.5 | Multi-output | 76.62 | 6.94 | 0.044 | Requires region-specific modeling |

## Region-Specific Performance

| Region | Target | Best R2 | Performance level |
|---|---|---:|---|
| IGKV | PM2.5 | 0.840 | Strong |
| AIIMS | PM2.5 | 0.819 | Strong |
| IGKV | PM10 | 0.769 | Good |
| AIIMS | PM10 | 0.742 | Good |
| Siltara | PM2.5 | 0.585 | Good baseline |
| AIIMS | SO2 | 0.502 | Moderate |

## Main Modeling Insight

The global PM2.5 model has weak performance because PM2.5 behavior varies strongly across regions. Region-specific modeling improves PM2.5 forecasting significantly, reaching R2 = 0.840 for IGKV and R2 = 0.819 for AIIMS. This shows that localized air pollution forecasting can be more effective than one global model for all regions.

## Dashboard Pages

- Overview
- Live Prediction
- Dataset Summary
- Global Model Performance
- Region-Specific Model Performance
- Region Analytics
- Explainability
- Pollution Spike Detection
- AI Report Generator
- Project Details

## Risk and Explainability

The dashboard includes a simplified AQI-style risk layer, health recommendation messages, feature importance / SHAP-style explanations, and spike detection based on deviations from recent trends. These modules are designed for project-level interpretation and should be validated against official monitoring systems for operational use.

## Future Scope

- Full quarter-hourly model training
- Region-specific production models
- Real-time pollution board API integration
- 24-hour forecasting
- Geospatial heatmaps
- LLM-based report generation
- Email or WhatsApp alerts
- Model monitoring and retraining
