# AirSense AI Deployment

## 1. Train or refresh the inference bundle

Fast local demo artifact:

```powershell
python scripts\train_air_quality_models.py `
  --granularity hourly `
  --n-estimators 5 `
  --max-depth 6 `
  --max-samples 0.1 `
  --output-dir outputs\smoke_air_quality_models
```

Final stronger artifact:

```powershell
python scripts\train_air_quality_models.py `
  --granularity quarter_hourly `
  --n-estimators 140 `
  --max-depth 18 `
  --max-samples 0.35 `
  --n-jobs -1 `
  --output-dir outputs\air_quality_models
```

Both commands create `models/inference_bundle.joblib`, which is the artifact used by the dashboard, API, and CLI.

## 2. Run locally

Streamlit dashboard:

```powershell
streamlit run app\streamlit_app.py
```

FastAPI service:

```powershell
uvicorn app.api:app --host 0.0.0.0 --port 8000
```

CLI prediction:

```powershell
python scripts\predict_cli.py --region SILTARA --pm25 78 --pm10 214 --so2 34 --temp 35 --hum 68 --ws 1.8
```

## 3. Docker

```powershell
docker build -t airsense-ai .
docker run -p 8501:8501 -e AIRSENSE_MODEL_DIR=/app/outputs/smoke_air_quality_models airsense-ai
```

## 4. Cloud notes

- Streamlit Cloud: set the app entry point to `app/streamlit_app.py`.
- Render: `render.yaml` uses the Dockerfile.
- If model files are not committed, upload or generate `outputs/<run>/models/inference_bundle.joblib` during the deployment workflow and set `AIRSENSE_MODEL_DIR`.
