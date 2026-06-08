from __future__ import annotations

import sys
from datetime import datetime, time
from pathlib import Path

import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from airsense.explainability import get_feature_importance
from airsense.inference import ModelNotReadyError, build_metadata, load_model_bundle, predict


st.set_page_config(page_title="AirSense AI", layout="wide")


@st.cache_resource(show_spinner=False)
def get_bundle() -> dict:
    return load_model_bundle()


def load_csv(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


def image_path(bundle: dict, filename: str) -> Path | None:
    model_image = Path(bundle["model_dir"]) / "plots" / filename
    docs_image = PROJECT_ROOT / "docs" / "assets" / filename
    if model_image.exists():
        return model_image
    if docs_image.exists():
        return docs_image
    return None


def render_metric_card(label: str, value: str, help_text: str | None = None) -> None:
    st.metric(label, value, help=help_text)


def prediction_form(bundle: dict) -> dict:
    region = st.selectbox("Region", ["AIIMS", "BHATAGAON", "IGKV", "SILTARA"], index=3)
    strategy = st.selectbox("Model strategy", ["best", "multi_output", "single_target"], index=0)
    date_col, time_col = st.columns(2)
    selected_date = date_col.date_input("Forecast date", value=datetime.now().date())
    selected_time = time_col.time_input("Forecast time", value=time(datetime.now().hour, 0))
    date_time = datetime.combine(selected_date, selected_time)

    pollutant_col, weather_col = st.columns(2)
    with pollutant_col:
        pm25 = st.number_input("PM2.5 now", min_value=0.0, max_value=600.0, value=78.0, step=1.0)
        pm10 = st.number_input("PM10 now", min_value=0.0, max_value=800.0, value=214.0, step=1.0)
        so2 = st.number_input("SO2 now", min_value=0.0, max_value=150.0, value=34.0, step=0.5)
    with weather_col:
        temp = st.number_input("Temperature", min_value=0.0, max_value=55.0, value=35.0, step=0.5)
        hum = st.number_input("Humidity", min_value=0.0, max_value=100.0, value=68.0, step=1.0)
        ws = st.number_input("Wind speed", min_value=0.0, max_value=40.0, value=1.8, step=0.2)
        wd = st.number_input("Wind direction", min_value=0.0, max_value=360.0, value=210.0, step=5.0)

    readings = {
        "pm2_5": pm25,
        "pm10": pm10,
        "so2": so2,
        "temp": temp,
        "hum": hum,
        "ws": ws,
        "wd": wd,
    }
    return predict(bundle, region=region, readings=readings, date_time=date_time, strategy=strategy)


def render_prediction_result(result: dict) -> None:
    predictions = result["predictions"]
    risk = result["risk"]
    cols = st.columns(4)
    cols[0].metric("PM2.5", f"{predictions['pm2_5']:.1f} ug/m3")
    cols[1].metric("PM10", f"{predictions['pm10']:.1f} ug/m3")
    cols[2].metric("SO2", f"{predictions['so2']:.1f} ug/m3")
    cols[3].metric("Risk", risk["category"])

    st.info(result["summary"])
    if result["anomaly_alerts"]:
        st.warning("Spike alert: " + "; ".join(alert["message"] for alert in result["anomaly_alerts"]))

    chart_frame = pd.DataFrame(
        {
            "pollutant": ["PM2.5", "PM10", "SO2"],
            "prediction": [predictions["pm2_5"], predictions["pm10"], predictions["so2"]],
        }
    )
    st.bar_chart(chart_frame, x="pollutant", y="prediction", use_container_width=True)


def main() -> None:
    try:
        bundle = get_bundle()
    except ModelNotReadyError as exc:
        st.error(str(exc))
        st.code(
            "python scripts/train_air_quality_models.py --granularity hourly "
            "--n-estimators 5 --max-depth 6 --max-samples 0.1 --output-dir outputs/smoke_air_quality_models",
            language="powershell",
        )
        return

    metadata = build_metadata(bundle)
    model_dir = Path(bundle["model_dir"])
    reports_dir = model_dir / "reports"

    st.markdown(
        """
        <style>
        .block-container { padding-top: 2rem; }
        [data-testid="stMetricValue"] { font-size: 1.7rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("AirSense AI")
    st.caption("Multi-Region Air Quality Forecasting, AQI Risk Intelligence, and Explainable Pollution Analytics")

    st.sidebar.subheader("Model Status")
    st.sidebar.write(f"Version: `{metadata['version']}`")
    st.sidebar.write(f"Granularity: `{metadata['granularity']}`")
    st.sidebar.write(f"Features: `{metadata['feature_count']}`")
    st.sidebar.write(f"Artifact: `{Path(str(metadata['model_dir'])).name}`")

    tabs = st.tabs(
        [
            "Overview",
            "Prediction Lab",
            "Analytics",
            "Model Performance",
            "Explainability",
            "Documentation",
        ]
    )

    with tabs[0]:
        cols = st.columns(4)
        cols[0].metric("Regions", "4")
        cols[1].metric("Targets", "3")
        cols[2].metric("Features", str(metadata["feature_count"]))
        cols[3].metric("Runtime", "Streamlit + FastAPI")
        st.markdown(
            "AirSense AI turns messy DCR workbooks into a forecasting workflow for PM2.5, PM10, and SO2 across AIIMS, Bhatagaon, IGKV, and SILTARA."
        )
        st.code(
            "Raw DCR files -> cleaning -> feature engineering -> model training -> AQI risk -> dashboard/API",
            language="text",
        )
        st.subheader("Recruiter Demo Flow")
        st.markdown(
            "Open Prediction Lab, use the SILTARA industrial preset values, read the AQI card, then show Model Performance and Explainability."
        )

    with tabs[1]:
        left, right = st.columns([0.95, 1.05], gap="large")
        with left:
            result = prediction_form(bundle)
        with right:
            render_prediction_result(result)

    with tabs[2]:
        summary = load_csv(PROJECT_ROOT / "data" / "processed" / "all_regions_dataset_summary.csv")
        if summary.empty:
            summary = pd.DataFrame(
                [
                    {"region": "AIIMS", "rows_after_merge": 154109, "hourly_rows": 30000, "quarter_hourly_rows": 124109},
                    {"region": "BHATAGAON", "rows_after_merge": 147966, "hourly_rows": 29395, "quarter_hourly_rows": 118571},
                    {"region": "IGKV", "rows_after_merge": 151801, "hourly_rows": 29891, "quarter_hourly_rows": 121910},
                    {"region": "SILTARA", "rows_after_merge": 132555, "hourly_rows": 29957, "quarter_hourly_rows": 102598},
                ]
            )
        st.dataframe(summary, use_container_width=True)
        st.bar_chart(summary, x="region", y=["hourly_rows", "quarter_hourly_rows"], use_container_width=True)

    with tabs[3]:
        metrics = load_csv(reports_dir / "metrics_overall.csv")
        if not metrics.empty:
            st.dataframe(metrics, use_container_width=True)
        for filename in ["metric_comparison.png", "region_prediction_timeseries.png", "predicted_vs_actual_scatter.png"]:
            path = image_path(bundle, filename)
            if path:
                st.image(str(path), use_container_width=True)

    with tabs[4]:
        importance_frame = get_feature_importance(bundle, limit=12)
        if not importance_frame.empty:
            st.dataframe(importance_frame, use_container_width=True)
            st.bar_chart(importance_frame, x="feature", y="importance", use_container_width=True)
        st.markdown(
            "The deployed explanation uses tree feature importance when available and falls back gracefully when optional SHAP tooling is not installed."
        )

    with tabs[5]:
        st.subheader("Job Description Mapping")
        st.table(
            pd.DataFrame(
                [
                    {"AI Intern need": "Preprocessing", "Project evidence": "DCR workbook extraction, timestamp parsing, duplicate handling"},
                    {"AI Intern need": "ML modeling", "Project evidence": "Random Forest forecasting for PM2.5, PM10, and SO2"},
                    {"AI Intern need": "Evaluation", "Project evidence": "RMSE, MAE, R2, chronological split, region-wise reports"},
                    {"AI Intern need": "Deployment", "Project evidence": "Streamlit dashboard, FastAPI endpoint, Docker, Render config"},
                ]
            )
        )
        st.markdown(Path(PROJECT_ROOT / "reports" / "demo_script.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
