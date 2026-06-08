# AirSense AI - Multi-Region Air Pollution Forecasting

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Engineering-150458?style=for-the-badge&logo=pandas&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML%20Forecasting-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)
![Colab](https://img.shields.io/badge/Google%20Colab-Heavy%20Training-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=black)

An end-to-end machine-learning project that converts messy DCR air-quality workbooks from four Raipur monitoring regions into a combined forecasting dataset, then predicts future pollutant levels for `PM2.5`, `PM10`, and `SO2`.

This project is designed as a clean AI/ML internship portfolio project: heavy preprocessing, feature engineering, predictive modeling, evaluation, visual reporting, a Streamlit dashboard, a FastAPI prediction endpoint, and a website-ready case study.

## Live Project Website

The portfolio website is inside [`docs/`](docs/).

It includes an interviewer-ready prediction demo where you can enter station readings
and instantly show predicted `PM2.5`, `PM10`, and `SO2` values in the browser.

After pushing to GitHub, enable GitHub Pages from the `docs/` folder. The expected public URL will be:

```text
https://pruthvi226.github.io/Air_Pollution_Detection_ML/
```

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
- Multi-output and single-target forecasting models.
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
  assets/                            # Demo plots used by the website

app/
  streamlit_app.py                    # Deployable dashboard
  api.py                              # FastAPI prediction service

airsense/
  inference.py                        # Shared prediction contract
  modeling.py                         # Portable sklearn custom transformers

notebooks/
  air_pollution_prediction_colab.ipynb

scripts/
  prepare_combined_dataset.py         # Extract zips and build the combined dataset
  train_air_quality_models.py         # Train models and generate reports/plots
  predict_cli.py                      # One-shot prediction smoke check

data/
  .gitkeep                            # Raw/processed datasets are generated locally
  data_dictionary.md                  # Column and feature documentation

outputs/
  .gitkeep                            # Model artifacts and reports are generated locally

reports/
  model_card.md
  experiment_report.md
  demo_script.md

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
Streamlit dashboard, FastAPI endpoint, and CLI predictor
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
python scripts\predict_cli.py --region SILTARA --pm25 78 --pm10 214 --so2 34 --temp 35 --hum 68 --ws 1.8
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
5. Replace demo images in `docs/assets/` with the final Colab plots.

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

- `multi_output`: one model predicts all three pollutants together.
- `single_target`: one separate model is trained per pollutant.

Evaluation:

- Chronological train/validation/test split
- RMSE
- MAE
- R2
- Overall and region-wise metrics

## Website Preview

The project website presents:

- Input form and prediction-result demo
- Verified dataset statistics
- Workflow and technology stack
- Demo model plots
- Readiness checklist
- Clear recruiter-facing project pitch

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
- Docker and Render deployment config
- Colab notebook
- GitHub Pages-ready website
- Modern README

Final polish before public job submission:

- Run the final quarter-hourly model in Colab.
- Replace demo website plots with final quarter-hourly result plots.
- Publish the website through GitHub Pages.
- Upload or generate the final `inference_bundle.joblib` in the deployment environment.

## Portfolio Pitch

> I built an end-to-end ML pipeline that converts messy environmental monitoring workbooks into a combined time-series dataset, engineers leakage-safe forecasting features, and compares models for pollutant prediction across four regions.
