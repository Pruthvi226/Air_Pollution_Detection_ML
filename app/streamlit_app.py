from __future__ import annotations

import sys
from datetime import datetime, time
from pathlib import Path

import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from airsense.config import PROJECT_VERSION, SUPPORTED_REGIONS, TARGET_COLUMNS
from airsense.anomaly import detect_prediction_spikes
from airsense.explainability import get_feature_importance
from airsense.inference import ModelNotReadyError, load_model_bundle, predict

try:
    from airsense.inference import build_metadata
except ImportError:
    def build_metadata(bundle: dict) -> dict:
        return {
            "project": "AirSense AI",
            "version": PROJECT_VERSION,
            "targets": list(bundle.get("target_columns", TARGET_COLUMNS)),
            "feature_names": list(bundle.get("feature_columns", [])),
            "feature_count": int(len(bundle.get("feature_columns", []))),
            "supported_regions": SUPPORTED_REGIONS,
            "trained_at": bundle.get("metadata", {}).get("trained_at", "generated smoke artifact"),
            "granularity": bundle.get("metadata", {}).get("granularity", "unknown"),
            "model_dir": bundle.get("model_dir"),
            "strategies": ["best", "multi_output", "single_target"],
        }


st.set_page_config(page_title="AirSense AI", layout="wide")


DEMO_SCENARIO = {
    "region": "SILTARA",
    "strategy": "best",
    "date": datetime(2026, 6, 8).date(),
    "time": time(8, 0),
    "readings": {
        "pm2_5": 78.0,
        "pm10": 145.0,
        "so2": 14.0,
        "temp": 31.0,
        "hum": 62.0,
        "ws": 2.1,
        "wd": 210.0,
    },
}

DEFAULT_SCENARIO = {
    "region": "SILTARA",
    "strategy": "best",
    "date": datetime.now().date(),
    "time": time(datetime.now().hour, 0),
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


def load_prediction_state(scenario: dict) -> None:
    st.session_state[PREDICTION_STATE_KEYS["region"]] = scenario["region"]
    st.session_state[PREDICTION_STATE_KEYS["strategy"]] = scenario["strategy"]
    st.session_state[PREDICTION_STATE_KEYS["date"]] = scenario["date"]
    st.session_state[PREDICTION_STATE_KEYS["time"]] = scenario["time"]
    for reading_key, value in scenario["readings"].items():
        st.session_state[PREDICTION_STATE_KEYS[reading_key]] = value


def ensure_prediction_state() -> None:
    for key, value in {
        PREDICTION_STATE_KEYS["region"]: DEFAULT_SCENARIO["region"],
        PREDICTION_STATE_KEYS["strategy"]: DEFAULT_SCENARIO["strategy"],
        PREDICTION_STATE_KEYS["date"]: DEFAULT_SCENARIO["date"],
        PREDICTION_STATE_KEYS["time"]: DEFAULT_SCENARIO["time"],
    }.items():
        st.session_state.setdefault(key, value)

    for reading_key, value in DEFAULT_SCENARIO["readings"].items():
        st.session_state.setdefault(PREDICTION_STATE_KEYS[reading_key], value)


def demo_prediction(bundle: dict) -> tuple[dict, dict]:
    date_time = datetime.combine(DEMO_SCENARIO["date"], DEMO_SCENARIO["time"])
    readings = dict(DEMO_SCENARIO["readings"])
    result = predict(
        bundle,
        region=DEMO_SCENARIO["region"],
        readings=readings,
        date_time=date_time,
        strategy=DEMO_SCENARIO["strategy"],
    )
    return result, readings


def prediction_form(bundle: dict) -> tuple[dict, dict]:
    ensure_prediction_state()

    button_cols = st.columns(2)
    if button_cols[0].button("Run Demo Scenario", type="primary", use_container_width=True):
        load_prediction_state(DEMO_SCENARIO)
    if button_cols[1].button("Reset Inputs", use_container_width=True):
        load_prediction_state(DEFAULT_SCENARIO)

    st.caption("Demo scenario: SILTARA, 08:00, PM2.5 78, PM10 145, SO2 14, temperature 31, humidity 62, wind 2.1.")

    region = st.selectbox(
        "Region",
        ["AIIMS", "BHATAGAON", "IGKV", "SILTARA"],
        key=PREDICTION_STATE_KEYS["region"],
    )
    strategy = st.selectbox(
        "Model strategy",
        ["best", "multi_output", "single_target"],
        key=PREDICTION_STATE_KEYS["strategy"],
    )
    date_col, time_col = st.columns(2)
    selected_date = date_col.date_input("Forecast date", key=PREDICTION_STATE_KEYS["date"])
    selected_time = time_col.time_input("Forecast time", key=PREDICTION_STATE_KEYS["time"])
    date_time = datetime.combine(selected_date, selected_time)

    pollutant_col, weather_col = st.columns(2)
    with pollutant_col:
        pm25 = st.number_input(
            "PM2.5 now",
            min_value=0.0,
            max_value=600.0,
            step=1.0,
            key=PREDICTION_STATE_KEYS["pm2_5"],
        )
        pm10 = st.number_input(
            "PM10 now",
            min_value=0.0,
            max_value=800.0,
            step=1.0,
            key=PREDICTION_STATE_KEYS["pm10"],
        )
        so2 = st.number_input(
            "SO2 now",
            min_value=0.0,
            max_value=150.0,
            step=0.5,
            key=PREDICTION_STATE_KEYS["so2"],
        )
    with weather_col:
        temp = st.number_input(
            "Temperature",
            min_value=0.0,
            max_value=55.0,
            step=0.5,
            key=PREDICTION_STATE_KEYS["temp"],
        )
        hum = st.number_input(
            "Humidity",
            min_value=0.0,
            max_value=100.0,
            step=1.0,
            key=PREDICTION_STATE_KEYS["hum"],
        )
        ws = st.number_input(
            "Wind speed",
            min_value=0.0,
            max_value=40.0,
            step=0.2,
            key=PREDICTION_STATE_KEYS["ws"],
        )
        wd = st.number_input(
            "Wind direction",
            min_value=0.0,
            max_value=360.0,
            step=5.0,
            key=PREDICTION_STATE_KEYS["wd"],
        )

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
    return result, readings


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


def build_anomaly_events(result: dict, readings: dict) -> list[dict]:
    alerts = list(result.get("anomaly_alerts", []))
    demo_alerts = detect_prediction_spikes(
        {
            "pm2_5": readings.get("pm2_5", 0),
            "pm10": readings.get("pm10", 0),
            "so2": readings.get("so2", 0),
        },
        thresholds={"pm2_5": 75.0, "pm10": 145.0, "so2": 28.0},
    )
    for alert in demo_alerts:
        alert = dict(alert)
        label = POLLUTANT_LABELS.get(str(alert["pollutant"]), str(alert["pollutant"]).upper())
        alert["message"] = f"{label} input is elevated for the recruiter demo threshold."
        if not any(existing.get("pollutant") == alert.get("pollutant") for existing in alerts):
            alerts.append(alert)
    return alerts


def render_anomaly_events(result: dict, readings: dict) -> None:
    alerts = build_anomaly_events(result, readings)
    cols = st.columns(3)
    cols[0].metric("Input PM10", f"{readings['pm10']:.1f} ug/m3")
    cols[1].metric("Predicted PM10", f"{result['predictions']['pm10']:.1f} ug/m3")
    cols[2].metric("Alert count", str(len(alerts)))

    if alerts:
        st.warning("Active anomaly review: " + "; ".join(alert["message"] for alert in alerts))
        st.dataframe(pd.DataFrame(alerts), use_container_width=True)
    else:
        st.success("No pollutant exceeds the active spike thresholds for this scenario.")

    st.markdown(
        "The demo anomaly layer checks both the deployed forecast thresholds and the current station readings so an interviewer can see how sudden PM10/PM2.5 spikes would be flagged before a public-health review."
    )


def build_ai_report(result: dict, readings: dict, metadata: dict) -> str:
    predictions = result["predictions"]
    risk = result["risk"]
    alerts = build_anomaly_events(result, readings)
    alert_summary = "No active spike alert." if not alerts else "; ".join(alert["message"] for alert in alerts)
    drivers = ", ".join(risk.get("drivers", []))
    return f"""AirSense AI Report

Region: {result["region"]}
Timestamp: {result["date_time"]}
Model strategy: {result["strategy"]}
Artifact: {Path(str(metadata["model_dir"])).name}

Current readings:
- PM2.5: {readings["pm2_5"]:.1f} ug/m3
- PM10: {readings["pm10"]:.1f} ug/m3
- SO2: {readings["so2"]:.1f} ug/m3
- Temperature: {readings["temp"]:.1f}
- Humidity: {readings["hum"]:.1f}%
- Wind speed: {readings["ws"]:.1f} m/s

Forecast:
- PM2.5: {predictions["pm2_5"]:.1f} ug/m3
- PM10: {predictions["pm10"]:.1f} ug/m3
- SO2: {predictions["so2"]:.1f} ug/m3

AQI-style risk: {risk["category"]} ({risk["risk_level"]})
Primary drivers: {drivers}
Anomaly review: {alert_summary}
Recommendation: {result["recommendation"]}

Project evidence:
- {metadata["feature_count"]} engineered model features
- Multi-region data from AIIMS, Bhatagaon, IGKV, and SILTARA
- Streamlit dashboard, FastAPI endpoint, CLI predictor, model card, and tests
"""


def render_ai_report(result: dict, readings: dict, metadata: dict) -> None:
    report = build_ai_report(result, readings, metadata)
    st.text_area("Generated report", report, height=420)
    st.download_button(
        "Download AI Report",
        data=report,
        file_name="airsense_ai_demo_report.txt",
        mime="text/plain",
        use_container_width=True,
    )


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
        @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@400;600;700;800&family=Plus+Jakarta+Sans:wght@400;500;600&display=swap');

        /* Global Font and Padding overrides */
        .main .block-container, .main, [data-testid="stSidebar"] {
            font-family: 'Plus Jakarta Sans', sans-serif !important;
        }
        .main .block-container {
            padding-top: 2.5rem;
        }
        h1, h2, h3, h4, h5, h6 {
            font-family: 'Outfit', sans-serif !important;
            letter-spacing: -0.5px !important;
        }

        /* Ambient Cosmic Background */
        .main {
            background-color: #060913 !important;
            background-image: radial-gradient(circle at 80% 20%, rgba(0, 242, 254, 0.05), transparent 45%) !important;
        }

        /* Sidebar Glassmorphism */
        [data-testid="stSidebar"] {
            background-color: #0c0f1d !important;
            border-right: 1px solid rgba(255, 255, 255, 0.06) !important;
        }

        /* Premium Metric Cards Override */
        [data-testid="stMetric"] {
            background: rgba(15, 23, 42, 0.45) !important;
            border: 1px solid rgba(255, 255, 255, 0.06) !important;
            border-radius: 14px !important;
            padding: 16px 20px !important;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2) !important;
            transition: all 0.35s cubic-bezier(0.25, 0.8, 0.25, 1) !important;
        }
        [data-testid="stMetric"]:hover {
            border-color: rgba(0, 242, 254, 0.3) !important;
            background: rgba(23, 32, 59, 0.6) !important;
            transform: translateY(-2px) !important;
            box-shadow: 0 12px 20px rgba(0, 0, 0, 0.35), 0 0 15px rgba(0, 242, 254, 0.1) !important;
        }
        [data-testid="stMetricLabel"] {
            font-size: 0.75rem !important;
            font-weight: 800 !important;
            text-transform: uppercase !important;
            color: #94a3b8 !important;
            letter-spacing: 1px !important;
        }
        [data-testid="stMetricValue"] {
            font-family: 'Outfit', sans-serif !important;
            font-size: 1.7rem !important;
            font-weight: 800 !important;
            color: #f8fafc !important;
            background: linear-gradient(135deg, #f8fafc 40%, #94a3b8) !important;
            -webkit-background-clip: text !important;
            -webkit-text-fill-color: transparent !important;
        }

        /* Custom Tabs Styling */
        button[data-baseweb="tab"] {
            font-family: 'Outfit', sans-serif !important;
            font-size: 0.95rem !important;
            font-weight: 600 !important;
            color: #94a3b8 !important;
            border-bottom: 2px solid transparent !important;
            transition: all 0.3s ease !important;
            padding: 10px 16px !important;
        }
        button[data-baseweb="tab"]:hover {
            color: #f8fafc !important;
        }
        button[data-baseweb="tab"][aria-selected="true"] {
            color: #00f2fe !important;
            border-bottom-color: #00f2fe !important;
        }

        /* Form styling */
        div[data-testid="stForm"] {
            background: rgba(12, 15, 29, 0.45) !important;
            border: 1px solid rgba(255, 255, 255, 0.06) !important;
            border-radius: 16px !important;
            padding: 24px !important;
            box-shadow: 0 10px 15px rgba(0, 0, 0, 0.2) !important;
        }

        /* Accent top decoration bar */
        [data-testid="stDecoration"] {
            background: linear-gradient(90deg, #00f2fe, #4facfe) !important;
        }

        /* Table/DataFrame border-radius and styling overrides */
        .stDataFrame, div[data-testid="stTable"] {
            border: 1px solid rgba(255, 255, 255, 0.06) !important;
            border-radius: 12px !important;
            overflow: hidden !important;
        }

        /* Elegant Alerts styling */
        .stAlert {
            background: rgba(15, 23, 42, 0.45) !important;
            border: 1px solid rgba(255, 255, 255, 0.06) !important;
            border-radius: 12px !important;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1) !important;
        }
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

    active_result, active_readings = demo_prediction(bundle)

    tabs = st.tabs(
        [
            "Overview",
            "Live Prediction",
            "Region Analytics",
            "Model Performance",
            "Explainability",
            "Anomaly Detection",
            "AI Report",
            "Project Details",
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
            "Open Live Prediction, click Run Demo Scenario, read the AQI card and anomaly signal, then show Model Performance, Explainability, and the AI Report."
        )

    with tabs[1]:
        left, right = st.columns([0.95, 1.05], gap="large")
        with left:
            active_result, active_readings = prediction_form(bundle)
        with right:
            render_prediction_result(active_result)

    with tabs[2]:
        st.subheader("Region Analytics")
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
        st.markdown(
            "SILTARA represents the industrial use case, Bhatagaon captures traffic-heavy behavior, AIIMS gives a mixed medical/residential profile, and IGKV provides the cleaner agricultural baseline."
        )

    with tabs[3]:
        st.subheader("Model Performance")
        metrics = load_csv(reports_dir / "metrics_overall.csv")
        if not metrics.empty:
            st.dataframe(metrics, use_container_width=True)
        else:
            st.info("Metrics CSV was not found for this artifact; showing bundled plot evidence when available.")
        for filename in ["metric_comparison.png", "region_prediction_timeseries.png", "predicted_vs_actual_scatter.png"]:
            path = image_path(bundle, filename)
            if path:
                st.image(str(path), use_container_width=True)

    with tabs[4]:
        st.subheader("Explainability")
        st.info(active_result.get("explanation", {}).get("summary", "The model uses pollutant, weather, time, and region features."))
        importance_frame = get_feature_importance(bundle, limit=12)
        if not importance_frame.empty:
            st.dataframe(importance_frame, use_container_width=True)
            st.bar_chart(importance_frame, x="feature", y="importance", use_container_width=True)
        st.markdown(
            "The deployed explanation uses tree feature importance when available and falls back gracefully when optional SHAP tooling is not installed."
        )

    with tabs[5]:
        st.subheader("Anomaly Detection")
        render_anomaly_events(active_result, active_readings)

    with tabs[6]:
        st.subheader("AI Report Generator")
        render_ai_report(active_result, active_readings, metadata)

    with tabs[7]:
        st.subheader("Project Details")
        st.markdown(
            "AirSense AI is packaged as a deployable ML project: shared inference code powers Streamlit, FastAPI, and CLI usage from the same model artifact."
        )
        st.subheader("Job Description Mapping")
        st.table(
            pd.DataFrame(
                [
                    {"AI Intern need": "Preprocessing", "Project evidence": "DCR workbook extraction, timestamp parsing, duplicate handling"},
                    {"AI Intern need": "ML modeling", "Project evidence": "Random Forest forecasting for PM2.5, PM10, and SO2"},
                    {"AI Intern need": "Evaluation", "Project evidence": "RMSE, MAE, R2, chronological split, region-wise reports"},
                    {"AI Intern need": "Deployment", "Project evidence": "Streamlit dashboard, FastAPI endpoint, Docker, Render config"},
                    {"AI Intern need": "Explainability", "Project evidence": "Feature importance summaries and plain-English model explanations"},
                    {"AI Intern need": "Communication", "Project evidence": "AI report generator, portfolio website, demo script, and model card"},
                ]
            )
        )
        st.subheader("Demo Script")
        st.markdown(Path(PROJECT_ROOT / "reports" / "demo_script.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
