# AirSense AI - Multi-Region Air Quality Forecasting

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Engineering-150458?style=for-the-badge&logo=pandas&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML%20Forecasting-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)
![Colab](https://img.shields.io/badge/Google%20Colab-Heavy%20Training-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=black)

AirSense AI is an end-to-end AI/ML project for multi-region air-quality forecasting, AQI-style risk intelligence, pollution spike detection, and explainable pollution analytics.

It converts messy DCR air-quality workbooks from four Raipur monitoring regions into a combined time-series dataset, engineers leakage-safe forecasting features, trains models for `PM2.5`, `PM10`, and `SO2`, explains predictions, and serves results through Streamlit, FastAPI, and a CLI.

This project is designed as a professional ML case study: heavy preprocessing, feature engineering, predictive modeling, evaluation, visual reporting, a Streamlit dashboard, a FastAPI prediction endpoint, and a website-ready project presentation.

## Live Project Website

The portfolio website is inside [`docs/`](docs/).

It includes an interactive forecast preview where you can enter station readings
and instantly view predicted `PM2.5`, `PM10`, and `SO2` values in the browser.

After pushing to GitHub, enable GitHub Pages from the `docs/` folder. The expected public URL will be:

```text
https://pruthvi226.github.io/Air_Pollution_Detection_ML/
```

## Key Features

- DCR zip and workbook ingestion for inconsistent `.xlsx`, `.xls`, and `.xlsb` files.
- Timestamp parsing, header detection, duplicate handling, and multi-region dataset building.
- Leakage-safe time-series features: lags, rolling statistics, cyclic time encodings, and region indicators.
- Multi-output and single-target forecasting for PM2.5, PM10, and SO2.
- AQI-style risk category and health recommendation layer.
- Pollution spike/anomaly alerts for unusually high predicted values.
- Explainability through tree feature importance with graceful optional-SHAP fallback.
- Streamlit dashboard with Overview, Live Prediction, Region Analytics, Model Performance, Explainability, Anomaly Detection, AI Report, and Project Details tabs.
- FastAPI endpoint with `/health`, `/metadata`, and `/predict`.
- CLI predictor, tests, Dockerfile, Render config, model card, and experiment report.

## Deployable App Layer

The trained runtime now uses one portable artifact:

```text
outputs/<run>/models/inference_bundle.joblib
```

That same bundle powers:

- Streamlit dashboard: [`app/streamlit_app.py`](app/streamlit_app.py)
- FastAPI endpoint: [`app/api.py`](app/api.py)
- CLI predictor: [`scripts/predict_cli.py`](scripts/predict_cli.py)
- Reusable inference package: [`airsense/`](airsense/)

Set `AIRSENSE_MODEL_DIR` to choose which trained run to serve. By default, the app looks for `outputs/air_quality_models` and then `outputs/smoke_air_quality_models`.

## What This Project Shows

- Real-world data cleaning from inconsistent `.xlsx`, `.xls`, and `.xlsb` workbooks.
- Automated extraction from station-wise DCR zip archives.
- Header detection, sheet filtering, timestamp parsing, and duplicate timestamp handling.
- Combined multi-region dataset creation with `region` labels.
- Leakage-safe time-series feature engineering.
- Baseline, multi-output, and single-target forecasting reports.
- AQI-style risk intelligence, anomaly alerts, and explainability summaries.
- Metrics, predictions, plots, and markdown/JSON experiment reports.

## Verified Dataset Build

The combined dataset was successfully prepared from the supplied DCR archives.

| Region | Rows | Hourly | Quarter-hourly |
|---|---:|---:|---:|
| AIIMS | 154,109 | 30,000 | 124,109 |
| Bhatagaon | 147,966 | 29,395 | 118,571 |
| IGKV | 151,801 | 29,891 | 121,910 |
| SILTARA | 132,555 | 29,957 | 102,598 |

Overall:

- Total rows: `586,431`
- Quarter-hourly rows: `467,188`
- Hourly rows: `119,243`
- Regions: `AIIMS`, `BHATAGAON`, `IGKV`, `SILTARA`

Large raw and processed data files are intentionally ignored by Git. Recreate them locally or in Colab with the scripts below.

## Project Structure

```text
docs/
  index.html                         # GitHub Pages-ready portfolio website
  styles.css                         # Website styling
  assets/                            # Model plots used by the website

app/
  streamlit_app.py                    # Deployable dashboard
  api.py                              # FastAPI prediction service

airsense/
  aqi.py                              # AQI-style risk labels and recommendations
  anomaly.py                          # Pollution spike detection
  explainability.py                   # Feature-importance explanations
  features.py                         # Feature engineering utilities
  inference.py                        # Shared prediction contract
  modeling.py                         # Portable sklearn custom transformers

notebooks/
  air_pollution_prediction_colab.ipynb

scripts/
  prepare_combined_dataset.py         # Extract zips and build the combined dataset
  train_air_quality_models.py         # Train models and generate reports/plots
  predict_cli.py                      # One-shot prediction smoke check
  generate_reports.py                 # Compact metrics/leaderboard generator
  smoke_test.py                       # Runtime smoke test

data/
  .gitkeep                            # Raw/processed datasets are generated locally
  data_dictionary.md                  # Column and feature documentation
  sample/sample_air_quality.csv       # Tiny sample contract file

outputs/
  .gitkeep                            # Model artifacts and reports are generated locally

reports/
  model_card.md
  experiment_report.md
  limitations_and_future_scope.md

Dockerfile
DEPLOYMENT.md
render.yaml

requirements.txt
README.md
```

## Workflow

```text
Raw DCR zip files
        |
        v
Workbook extraction and sheet discovery
        |
        v
Column normalization and timestamp parsing
        |
        v
Duplicate timestamp merge
        |
        v
All-region combined dataset
        |
        v
Feature engineering
        |
        v
Model training and evaluation
        |
        v
Plots, reports, predictions, and website case study
        |
        v
AQI risk, anomaly alerts, explanations, Streamlit dashboard, FastAPI endpoint, and CLI predictor
```

## Local Setup

```powershell
python -m pip install -r requirements.txt
```

Prepare the combined dataset:

```powershell
python scripts\prepare_combined_dataset.py `
  --zip "C:\Users\pruthviraj\Downloads\DCR AIIMS-20260606T154001Z-3-001.zip" `
  --zip "C:\Users\pruthviraj\Downloads\Bhatagaon DCR-20260606T153956Z-3-001.zip" `
  --zip "C:\Users\pruthviraj\Downloads\IGKV DCR-20260606T154005Z-3-001.zip" `
  --zip "C:\Users\pruthviraj\Downloads\SILTARA DCR-20260606T154006Z-3-001.zip"
```

Run a fast local model check:

```powershell
python scripts\train_air_quality_models.py `
  --granularity hourly `
  --n-estimators 10 `
  --max-depth 8 `
  --max-samples 0.15 `
  --n-jobs 1
```

Run the dashboard:

```powershell
streamlit run app\streamlit_app.py
```

Run the API:

```powershell
uvicorn app.api:app --host 0.0.0.0 --port 8000
```

Run a CLI prediction:

```powershell
python scripts\predict_cli.py --region SILTARA --pm25 78 --pm10 145 --so2 14 --temp 31 --hum 62 --ws 2.1
```

Run tests:

```powershell
pytest -q
```

API request:

```powershell
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/predict -ContentType "application/json" -Body '{"region":"SILTARA","pm25":78,"pm10":145,"so2":14,"temperature":31,"humidity":62,"wind_speed":2.1,"timestamp":"2026-06-08T08:00:00"}'
```

Run the stronger final model in Colab:

```powershell
python scripts\train_air_quality_models.py `
  --granularity quarter_hourly `
  --n-estimators 140 `
  --max-depth 18 `
  --max-samples 0.35 `
  --n-jobs -1
```

## Colab Workflow

Use [`notebooks/air_pollution_prediction_colab.ipynb`](notebooks/air_pollution_prediction_colab.ipynb) for the full data-heavy run.

Recommended flow:

1. Upload the four raw DCR zip files.
2. Run dataset preparation.
3. Train the quarter-hourly model.
4. Export final plots from `outputs/air_quality_models/plots/`.
5. Replace website images in `docs/assets/` with the final Colab plots.

## Modeling Approach

Targets:

- `pm2_5`
- `pm10`
- `so2`

Feature families:

- Pollutant and weather measurements
- Cyclic time encodings
- Lag features
- Rolling mean and standard deviation features
- Region one-hot indicators
- Missing-value indicators through model imputation

Model strategies:

- `baseline_median`: median baseline for comparison.
- `multi_output`: one model predicts all three pollutants together.
- `single_target`: one separate model is trained per pollutant.
- `best`: inference-time selection of the best strategy per target and region.

Evaluation:

- Chronological train/validation/test split
- RMSE
- MAE
- R2
- Overall and region-wise metrics
- Model comparison leaderboard
- Compact `metrics.json`, `model_comparison.csv`, and `predictions.csv` artifacts on training runs

## Capability Mapping

| ML project capability | AirSense AI evidence |
|---|---|
| Data preprocessing | Raw DCR extraction, sheet discovery, timestamp parsing, duplicate handling |
| Feature engineering | Lag features, rolling statistics, cyclic encodings, region indicators |
| ML model training | Baseline and Random Forest forecasting for PM2.5, PM10, and SO2 |
| Evaluation | Chronological split, RMSE, MAE, R2, region-wise metrics |
| Automation | CLI predictor, report generator, Docker, Render config |
| Application layer | Streamlit dashboard and FastAPI service |
| Explainability | Feature-importance summaries and plain-English explanations |

## Website Preview

The project website presents:

- Input form and prediction-result preview
- Verified dataset statistics
- Workflow and technology stack
- Current model plots
- Explainability, anomaly detection, AI report, and project-details sections
- Clear project pitch and technical evidence

Open locally:

```text
docs/index.html
```

## Current Status

Ready:

- Clean repository structure
- Dataset preparation pipeline
- Model training pipeline
- Portable inference bundle contract
- Streamlit dashboard
- FastAPI prediction API
- CLI prediction smoke test
- AQI-style risk intelligence
- Pollution spike alerts
- Explainability summaries
- `/metadata` API endpoint
- Pytest suite
- Docker and Render deployment config
- Colab notebook
- GitHub Pages-ready website
- Modern README

Final polish before public job submission:

- Run the final quarter-hourly model in Colab.
- Replace website plots with final quarter-hourly result plots.
- Publish the website through GitHub Pages.
- Upload or generate the final `inference_bundle.joblib` in the deployment environment.

## Portfolio Pitch

> I built an end-to-end AI system that converts messy multi-region air-quality workbooks into a clean time-series dataset, engineers leakage-safe forecasting features, trains ML models for PM2.5, PM10, and SO2, explains predictions, detects pollution spikes, and serves results through a Streamlit dashboard and FastAPI endpoint.
