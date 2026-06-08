# AirSense AI Demo Script

## Opening

This is AirSense AI, an end-to-end air-pollution forecasting project. It turns messy station-wise DCR workbooks into a clean multi-region dataset, trains forecasting models, evaluates them chronologically, and serves predictions through a dashboard or API.

## Pipeline

The preprocessing script extracts raw archives, discovers valid Excel sheets, normalizes columns, parses timestamps, handles duplicates, and combines AIIMS, Bhatagaon, IGKV, and SILTARA into one modeling table.

## Modeling

The training script creates time, lag, rolling, and region features, then compares multi-output and single-target Random Forest models for PM2.5, PM10, and SO2.

## Application

The Streamlit dashboard and FastAPI endpoint load the same `inference_bundle.joblib`, so the demo and API use the exact trained artifact.

## Closing

This project demonstrates data engineering, feature engineering, ML training, evaluation, visual reporting, and deployment packaging in one coherent AI internship project.
