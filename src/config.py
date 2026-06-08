"""Project constants for AirSense AI.

The values in this module mirror the current validated boosted quarter-hourly
artifact and known region-level evaluation results. They are used for dashboard
display, README/report generation, and fallback demo behavior.
"""

from __future__ import annotations

from datetime import datetime, time


DATASET_METRICS = [
    {"Metric": "Total cleaned records", "Value": "586,431", "Interpretation": "Large cleaned dataset"},
    {"Metric": "Quarter-hourly records", "Value": "467,188", "Interpretation": "High-resolution raw monitoring data"},
    {"Metric": "Quarter-hourly records used by model", "Value": "467,188", "Interpretation": "High-resolution modeling data"},
    {"Metric": "Regions processed", "Value": "4", "Interpretation": "AIIMS, Bhatagaon, IGKV, Siltara"},
    {"Metric": "Forecast targets", "Value": "3", "Interpretation": "PM2.5, PM10, SO2"},
    {"Metric": "Engineered features", "Value": "202", "Interpretation": "Lag, rolling, time, region features"},
    {"Metric": "Test rows", "Value": "51,226", "Interpretation": "Final evaluation sample size"},
]

GLOBAL_PERFORMANCE = [
    {
        "Target": "PM2.5",
        "Best Strategy": "LightGBM boosted",
        "RMSE": 2.58,
        "MAE": 0.94,
        "R2": 0.978,
        "Status": "Strong",
        "Interpretation": "Strong global boosted forecasting",
    },
    {
        "Target": "PM10",
        "Best Strategy": "LightGBM boosted",
        "RMSE": 14.87,
        "MAE": 3.07,
        "R2": 0.883,
        "Status": "Strong",
        "Interpretation": "Strong global boosted forecasting",
    },
    {
        "Target": "SO2",
        "Best Strategy": "LightGBM boosted",
        "RMSE": 6.06,
        "MAE": 0.55,
        "R2": 0.245,
        "Status": "Spike-sensitive",
        "Interpretation": "Low MAE but high spike sensitivity",
    },
]

REGION_PERFORMANCE = [
    {"Region": "AIIMS", "Target": "PM2.5", "Best R2": 0.984, "Performance Level": "Strong", "Notes": "Boosted PM2.5 forecasting signal"},
    {"Region": "IGKV", "Target": "PM2.5", "Best R2": 0.980, "Performance Level": "Strong", "Notes": "Boosted PM2.5 forecasting signal"},
    {"Region": "IGKV", "Target": "PM10", "Best R2": 0.975, "Performance Level": "Strong", "Notes": "Boosted PM10 forecasting signal"},
    {"Region": "AIIMS", "Target": "PM10", "Best R2": 0.969, "Performance Level": "Strong", "Notes": "Boosted PM10 forecasting signal"},
    {"Region": "SILTARA", "Target": "PM10", "Best R2": 0.935, "Performance Level": "Strong", "Notes": "Industrial-region PM10 behavior"},
    {"Region": "BHATAGAON", "Target": "SO2", "Best R2": 0.836, "Performance Level": "Strong", "Notes": "Boosted SO2 forecasting signal"},
    {"Region": "SILTARA", "Target": "SO2", "Best R2": 0.683, "Performance Level": "Good", "Notes": "XGBoost selected for SO2"},
]

REGION_ANALYTICS = [
    {
        "Region": "AIIMS",
        "Best target": "PM2.5",
        "Best R2": 0.984,
        "Pollution behavior": "Medical/residential region with very strong boosted PM2.5 and PM10 forecasting signals.",
        "Suggested model strategy": "Boosted LightGBM selector",
    },
    {
        "Region": "Bhatagaon",
        "Best target": "PM2.5",
        "Best R2": 0.965,
        "Pollution behavior": "Urban region with strong boosted PM2.5 and useful PM10 forecasting signal.",
        "Suggested model strategy": "Boosted LightGBM selector",
    },
    {
        "Region": "IGKV",
        "Best target": "PM2.5",
        "Best R2": 0.980,
        "Pollution behavior": "Very strong PM2.5 and PM10 boosted performance.",
        "Suggested model strategy": "Boosted LightGBM selector",
    },
    {
        "Region": "Siltara",
        "Best target": "PM2.5",
        "Best R2": 0.941,
        "Pollution behavior": "Industrial-region behavior with strong PM2.5/PM10 forecasts and XGBoost-selected SO2.",
        "Suggested model strategy": "Boosted LightGBM/XGBoost selector",
    },
]

FINAL_SELECTED_STRATEGIES = [
    {
        "target": "PM2.5",
        "strategy": "Boosted LightGBM selector",
        "reason": "LightGBM produced the best validation and held-out PM2.5 metrics.",
    },
    {
        "target": "PM10",
        "strategy": "Boosted LightGBM selector",
        "reason": "LightGBM produced the best validation and held-out PM10 metrics.",
    },
    {
        "target": "SO2",
        "strategy": "Boosted LightGBM/XGBoost selector",
        "reason": "SO2 has low MAE but spike-sensitive RMSE; XGBoost wins selected SO2 regions.",
    },
]

INTERPRETATION_CARDS = [
    {
        "Title": "PM2.5 Global Forecasting",
        "Status": "Strong",
        "R2": "0.978",
        "Explanation": "Boosted LightGBM captures short-term PM2.5 behavior strongly using lag, rolling, time, weather, and region features.",
    },
    {
        "Title": "PM10 Global Forecasting",
        "Status": "Strong",
        "R2": "0.883",
        "Explanation": "Boosted LightGBM substantially improves PM10 forecasting over the previous Random Forest result.",
    },
    {
        "Title": "SO2 Global Forecasting",
        "Status": "Spike-sensitive",
        "R2": "0.245",
        "Explanation": "SO2 MAE is low, but rare spikes still hurt RMSE and should be monitored.",
    },
]

PIPELINE_STEPS = [
    "Raw Pollution Data",
    "Data Cleaning",
    "Feature Engineering",
    "ML Model Training",
    "Evaluation",
    "AQI Risk Engine",
    "Dashboard Insights",
]

DEMO_SCENARIO = {
    "region": "SILTARA",
    "strategy": "best",
    "date": datetime(2026, 6, 8).date(),
    "time": time(9, 0),
    "hour": 9,
    "month": 6,
    "readings": {
        "pm2_5": 78.0,
        "pm10": 145.0,
        "so2": 14.0,
        "temp": 31.0,
        "hum": 62.0,
        "ws": 3.2,
        "wd": 210.0,
    },
}

DEFAULT_SCENARIO = {
    "region": "SILTARA",
    "strategy": "best",
    "date": datetime.now().date(),
    "time": time(datetime.now().hour, 0),
    "hour": datetime.now().hour,
    "month": datetime.now().month,
    "readings": {
        "pm2_5": 38.0,
        "pm10": 112.0,
        "so2": 18.0,
        "temp": 31.0,
        "hum": 58.0,
        "ws": 3.2,
        "wd": 180.0,
    },
}

PREDICTION_STATE_KEYS = {
    "region": "prediction_region",
    "strategy": "prediction_strategy",
    "date": "prediction_date",
    "time": "prediction_time",
    "hour": "prediction_hour",
    "month": "prediction_month",
    "pm2_5": "prediction_pm25",
    "pm10": "prediction_pm10",
    "so2": "prediction_so2",
    "temp": "prediction_temp",
    "hum": "prediction_hum",
    "ws": "prediction_ws",
    "wd": "prediction_wd",
}

POLLUTANT_LABELS = {
    "pm2_5": "PM2.5",
    "pm10": "PM10",
    "so2": "SO2",
}
