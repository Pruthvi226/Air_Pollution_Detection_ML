from __future__ import annotations

import sys
from datetime import datetime, time
from pathlib import Path

import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from airsense.inference import ModelNotReadyError, load_model_bundle, predict


st.set_page_config(
    page_title="AirSense AI",
    layout="wide",
)


@st.cache_resource(show_spinner=False)
def get_bundle() -> dict:
    return load_model_bundle()


def render_model_status(bundle: dict) -> None:
    metadata = bundle.get("metadata", {})
    model_dir = bundle.get("model_dir", "outputs/air_quality_models")
    st.sidebar.subheader("Model")
    st.sidebar.write(f"Granularity: `{metadata.get('granularity', 'unknown')}`")
    st.sidebar.write(f"Features: `{metadata.get('feature_count', len(bundle.get('feature_columns', [])))}`")
    st.sidebar.write(f"Artifacts: `{Path(model_dir).name}`")
    st.sidebar.write(f"Strategy: best per region-target")


def main() -> None:
    st.title("AirSense AI")
    st.caption("Multi-region PM2.5, PM10, and SO2 forecasting dashboard")

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

    render_model_status(bundle)

    left, right = st.columns([0.88, 1.12], gap="large")

    with left:
        st.subheader("Station Reading")
        region = st.selectbox("Region", ["AIIMS", "BHATAGAON", "IGKV", "SILTARA"], index=3)
        strategy = st.selectbox("Model strategy", ["best", "multi_output", "single_target"], index=0)

        date_col, time_col = st.columns(2)
        selected_date = date_col.date_input("Date", value=datetime.now().date())
        selected_time = time_col.time_input("Time", value=time(datetime.now().hour, 0))
        date_time = datetime.combine(selected_date, selected_time)

        pm_col, weather_col = st.columns(2)
        with pm_col:
            pm25 = st.number_input("PM2.5 now", min_value=0.0, max_value=600.0, value=38.0, step=1.0)
            pm10 = st.number_input("PM10 now", min_value=0.0, max_value=800.0, value=112.0, step=1.0)
            so2 = st.number_input("SO2 now", min_value=0.0, max_value=150.0, value=18.0, step=0.5)
        with weather_col:
            temp = st.number_input("Temperature", min_value=0.0, max_value=55.0, value=31.0, step=0.5)
            hum = st.number_input("Humidity", min_value=0.0, max_value=100.0, value=58.0, step=1.0)
            ws = st.number_input("Wind speed", min_value=0.0, max_value=40.0, value=3.2, step=0.2)
            wd = st.number_input("Wind direction", min_value=0.0, max_value=360.0, value=180.0, step=5.0)

        readings = {
            "pm2_5": pm25,
            "pm10": pm10,
            "so2": so2,
            "temp": temp,
            "hum": hum,
            "ws": ws,
            "wd": wd,
        }
        result = predict(bundle, region=region, readings=readings, date_time=date_time, strategy=strategy)

    with right:
        st.subheader("Next-Step Forecast")
        predictions = result["predictions"]
        risk = result["risk"]
        metric_cols = st.columns(4)
        metric_cols[0].metric("PM2.5", f"{predictions['pm2_5']:.1f} ug/m3")
        metric_cols[1].metric("PM10", f"{predictions['pm10']:.1f} ug/m3")
        metric_cols[2].metric("SO2", f"{predictions['so2']:.1f} ug/m3")
        metric_cols[3].metric("Risk", risk["label"])

        chart_data = {
            "pollutant": ["PM2.5", "PM10", "SO2"],
            "prediction": [predictions["pm2_5"], predictions["pm10"], predictions["so2"]],
        }
        st.bar_chart(chart_data, x="pollutant", y="prediction", use_container_width=True)

        plot_dir = Path(bundle["model_dir"]) / "plots"
        plot_tabs = st.tabs(["Metrics", "Region Trends", "Scatter"])
        with plot_tabs[0]:
            image_path = plot_dir / "metric_comparison.png"
            if image_path.exists():
                st.image(str(image_path), use_container_width=True)
        with plot_tabs[1]:
            image_path = plot_dir / "region_prediction_timeseries.png"
            if image_path.exists():
                st.image(str(image_path), use_container_width=True)
        with plot_tabs[2]:
            image_path = plot_dir / "predicted_vs_actual_scatter.png"
            if image_path.exists():
                st.image(str(image_path), use_container_width=True)


if __name__ == "__main__":
    main()
