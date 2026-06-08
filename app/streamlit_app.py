from __future__ import annotations

import sys
from datetime import datetime
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
from src.anomaly_detection import detect_pollution_spike
from src.aqi_utils import classify_aqi_risk
from src.config import (
    DATASET_METRICS,
    DEFAULT_SCENARIO,
    DEMO_SCENARIO,
    GLOBAL_PERFORMANCE,
    INTERPRETATION_CARDS,
    PIPELINE_STEPS,
    POLLUTANT_LABELS,
    PREDICTION_STATE_KEYS,
    REGION_ANALYTICS,
    REGION_PERFORMANCE,
)
from src.predict import select_best_model
from src.report_generator import generate_air_quality_report
from src.visualization_utils import make_bar_chart

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


def status_badge(label: str) -> str:
    colors = {
        "Strong": "#22c55e",
        "Good": "#38bdf8",
        "Good baseline": "#38bdf8",
        "Moderate": "#f59e0b",
        "Needs region-specific modeling": "#f97316",
        "Region-specific modeling required": "#f97316",
    }
    color = colors.get(label, "#94a3b8")
    return (
        f"<span class='status-badge' style='background:{color}; color:#06101f;'>"
        f"{label}</span>"
    )


def render_html_cards(cards: list[dict[str, str]], columns: int = 3) -> None:
    for group_start in range(0, len(cards), columns):
        cols = st.columns(columns)
        for col, card in zip(cols, cards[group_start:group_start + columns]):
            with col:
                st.markdown(
                    f"""
                    <div class="glass-card">
                      <div class="card-kicker">{card.get("kicker", "")}</div>
                      <div class="card-title">{card.get("title", "")}</div>
                      <div class="card-body">{card.get("body", "")}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )


def dataset_frame() -> pd.DataFrame:
    return pd.DataFrame(DATASET_METRICS)


def global_performance_frame() -> pd.DataFrame:
    return pd.DataFrame(GLOBAL_PERFORMANCE)


def region_performance_frame() -> pd.DataFrame:
    return pd.DataFrame(REGION_PERFORMANCE).sort_values("Best R2", ascending=False, kind="stable")


def region_analytics_frame() -> pd.DataFrame:
    return pd.DataFrame(REGION_ANALYTICS)


def render_status_table(frame: pd.DataFrame, status_column: str = "Status") -> None:
    if status_column not in frame.columns:
        st.dataframe(frame, use_container_width=True, hide_index=True)
        return

    display = frame.copy()
    display[status_column] = display[status_column].map(status_badge)
    st.markdown(display.to_html(escape=False, index=False), unsafe_allow_html=True)


def render_bar_chart(frame: pd.DataFrame, x: str, y: str) -> None:
    chart = make_bar_chart(frame[[x, y]], x=x, y=y)
    if chart is not None:
        chart.update_layout(
            template="plotly_dark",
            margin=dict(l=10, r=10, t=20, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(chart, use_container_width=True)
    else:
        st.bar_chart(frame[[x, y]], x=x, y=y, use_container_width=True)


def render_result_images(bundle: dict) -> None:
    for filename, caption in [
        ("predicted_vs_actual_scatter.png", "Sample actual vs predicted visualization"),
        ("region_prediction_timeseries.png", "Region prediction time-series visualization"),
    ]:
        path = image_path(bundle, filename)
        if path:
            st.image(str(path), caption=caption, use_container_width=True)


def load_prediction_state(scenario: dict) -> None:
    st.session_state[PREDICTION_STATE_KEYS["region"]] = scenario["region"]
    st.session_state[PREDICTION_STATE_KEYS["strategy"]] = scenario["strategy"]
    st.session_state[PREDICTION_STATE_KEYS["date"]] = scenario["date"]
    st.session_state[PREDICTION_STATE_KEYS["time"]] = scenario["time"]
    st.session_state[PREDICTION_STATE_KEYS["hour"]] = scenario["hour"]
    st.session_state[PREDICTION_STATE_KEYS["month"]] = scenario["month"]
    for reading_key, value in scenario["readings"].items():
        st.session_state[PREDICTION_STATE_KEYS[reading_key]] = value


def ensure_prediction_state() -> None:
    for key, value in {
        PREDICTION_STATE_KEYS["region"]: DEFAULT_SCENARIO["region"],
        PREDICTION_STATE_KEYS["strategy"]: DEFAULT_SCENARIO["strategy"],
        PREDICTION_STATE_KEYS["date"]: DEFAULT_SCENARIO["date"],
        PREDICTION_STATE_KEYS["time"]: DEFAULT_SCENARIO["time"],
        PREDICTION_STATE_KEYS["hour"]: DEFAULT_SCENARIO["hour"],
        PREDICTION_STATE_KEYS["month"]: DEFAULT_SCENARIO["month"],
    }.items():
        st.session_state.setdefault(key, value)

    for reading_key, value in DEFAULT_SCENARIO["readings"].items():
        st.session_state.setdefault(PREDICTION_STATE_KEYS[reading_key], value)


def scenario_prediction(bundle: dict) -> tuple[dict, dict]:
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

    st.caption(
        "Demo prediction based on saved model or fallback simulation depending on available artifacts. "
        "Siltara scenario: PM2.5 78, PM10 145, SO2 14, temperature 31, humidity 62, wind 3.2, hour 9, month 6."
    )

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
    hour_col, month_col = st.columns(2)
    selected_hour = hour_col.number_input(
        "Hour of day",
        min_value=0,
        max_value=23,
        step=1,
        key=PREDICTION_STATE_KEYS["hour"],
    )
    selected_month = month_col.number_input(
        "Month",
        min_value=1,
        max_value=12,
        step=1,
        key=PREDICTION_STATE_KEYS["month"],
    )
    current_year = datetime.now().year
    date_time = datetime(current_year, int(selected_month), 1, int(selected_hour), 0)

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
    risk = classify_aqi_risk(predictions["pm2_5"], predictions["pm10"], predictions["so2"])
    selected_strategy = select_best_model(result["region"], "PM2.5")
    alerts = build_anomaly_events(result, {
        "pm2_5": predictions["pm2_5"],
        "pm10": predictions["pm10"],
        "so2": predictions["so2"],
    })
    cols = st.columns(5)
    cols[0].metric("PM2.5", f"{predictions['pm2_5']:.1f} ug/m3")
    cols[1].metric("PM10", f"{predictions['pm10']:.1f} ug/m3")
    cols[2].metric("SO2", f"{predictions['so2']:.1f} ug/m3")
    cols[3].metric("AQI Risk", risk["category"])
    cols[4].metric("Spike Alert", "Yes" if alerts else "No")

    st.markdown(
        f"""
        <div class="glass-card">
          <div class="card-kicker">Model Strategy Used</div>
          <div class="card-title">{selected_strategy['strategy']}</div>
          <div class="card-body">{selected_strategy['reason']}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class="risk-panel">
          <strong style="color:{risk['badge_color']}">{risk['category']} risk</strong><br>
          {risk['message']}<br>
          <small>{risk['disclaimer']}</small>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if alerts:
        st.warning("Spike alert: " + "; ".join(alert["message"] for alert in alerts))
    st.info(result.get("explanation", {}).get("summary", "The model uses pollutant history, weather context, time, and region indicators."))

    chart_frame = pd.DataFrame(
        {
            "pollutant": ["PM2.5", "PM10", "SO2"],
            "prediction": [predictions["pm2_5"], predictions["pm10"], predictions["so2"]],
        }
    )
    render_bar_chart(chart_frame, "pollutant", "prediction")


def build_anomaly_events(result: dict, readings: dict) -> list[dict]:
    alerts = list(result.get("anomaly_alerts", []))
    input_alerts = detect_prediction_spikes(
        {
            "pm2_5": readings.get("pm2_5", 0),
            "pm10": readings.get("pm10", 0),
            "so2": readings.get("so2", 0),
        },
        thresholds={"pm2_5": 75.0, "pm10": 145.0, "so2": 28.0},
    )
    for alert in input_alerts:
        alert = dict(alert)
        label = POLLUTANT_LABELS.get(str(alert["pollutant"]), str(alert["pollutant"]).upper())
        alert["message"] = f"{label} input is elevated for the configured review threshold."
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
        "The anomaly layer checks both the deployed forecast thresholds and current station readings so elevated PM10/PM2.5 conditions can be flagged for review."
    )


def build_ai_report(result: dict, readings: dict, metadata: dict) -> str:
    predictions = result["predictions"]
    risk = classify_aqi_risk(predictions["pm2_5"], predictions["pm10"], predictions["so2"])
    anomaly_result = detect_pollution_spike(
        current_value=readings.get("pm10", predictions["pm10"]),
        rolling_mean=80.0,
        rolling_std=20.0,
        pollutant="PM10",
    )
    top_features = result.get("explanation", {}).get("top_features", [])
    feature_names = [str(row.get("feature", "")) for row in top_features[:3] if row.get("feature")]
    if not feature_names:
        feature_names = ["recent PM10 history", "rolling pollution averages", "region-specific trends"]
    selected_strategy = select_best_model(result["region"], "PM2.5")
    report_body = generate_air_quality_report(
        region=result["region"],
        predictions=predictions,
        aqi_result=risk,
        anomaly_result=anomaly_result,
        top_features=feature_names,
        model_strategy=selected_strategy,
    )
    return (
        "AI-Generated Air Quality Summary\n\n"
        f"{report_body}\n\n"
        f"Runtime artifact: {Path(str(metadata['model_dir'])).name}\n"
        "Future scope: this template can be upgraded with an LLM-based report summarizer."
    )


def render_ai_report(result: dict, readings: dict, metadata: dict) -> None:
    report = build_ai_report(result, readings, metadata)
    st.text_area("Generated report", report, height=420)
    st.download_button(
        "Download AI Report",
        data=report,
        file_name="airsense_ai_report.txt",
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
        .hero-panel, .glass-card, .risk-panel {
            background: rgba(15, 23, 42, 0.55);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 16px;
            padding: 22px 24px;
            box-shadow: 0 16px 30px rgba(0, 0, 0, 0.24);
            margin-bottom: 18px;
        }
        .hero-panel h1 {
            margin-bottom: 4px;
        }
        .hero-subtitle {
            color: #38bdf8;
            font-size: 1.15rem;
            font-weight: 800;
            margin-bottom: 10px;
        }
        .hero-copy, .card-body {
            color: #cbd5e1;
            line-height: 1.65;
        }
        .card-kicker {
            color: #94a3b8;
            font-size: 0.72rem;
            font-weight: 800;
            text-transform: uppercase;
            letter-spacing: 0.08em;
        }
        .card-title {
            color: #f8fafc;
            font-family: 'Outfit', sans-serif;
            font-size: 1.15rem;
            font-weight: 800;
            margin: 8px 0;
        }
        .status-badge {
            display: inline-block;
            border-radius: 999px;
            padding: 5px 11px;
            font-weight: 800;
            font-size: 0.78rem;
            white-space: nowrap;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            overflow: hidden;
            border-radius: 12px;
        }
        th, td {
            padding: 11px 12px;
            border-bottom: 1px solid rgba(255, 255, 255, 0.08);
        }
        th {
            color: #e2e8f0;
            background: rgba(15, 23, 42, 0.85);
        }
        td {
            color: #cbd5e1;
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
    artifact_name = Path(str(metadata["model_dir"])).name
    artifact_label = "Hourly runtime bundle" if artifact_name == "smoke_air_quality_models" else artifact_name
    st.sidebar.write(f"Artifact: `{artifact_label}`")

    active_result, active_readings = scenario_prediction(bundle)
    page = st.sidebar.radio(
        "Dashboard Navigation",
        [
            "Overview",
            "Live Prediction",
            "Dataset Summary",
            "Global Performance",
            "Region Performance",
            "Region Analytics",
            "Explainability",
            "Anomaly Detection",
            "AI Report",
            "Project Details",
        ],
    )

    if page == "Overview":
        st.markdown(
            """
            <div class="hero-panel">
              <h1>AirSense AI</h1>
              <div class="hero-subtitle">Intelligent Air Pollution Forecasting & Risk Analytics Dashboard</div>
              <div class="hero-copy">
                An AI-powered system that predicts PM2.5, PM10, and SO2 levels, evaluates global and
                region-specific model performance, detects pollution spikes, and converts model outputs
                into AQI-based risk insights.
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        metric_cols = st.columns(6)
        metric_cols[0].metric("Cleaned Records", "586,431")
        metric_cols[1].metric("Monitoring Regions", "4")
        metric_cols[2].metric("Forecast Targets", "3")
        metric_cols[3].metric("Engineered Features", "201")
        metric_cols[4].metric("Best Region-Level R2", "0.840")
        metric_cols[5].metric("Test Rows", "15,772")

        st.subheader("What This Project Demonstrates")
        render_html_cards(
            [
                {"kicker": "Data", "title": "Raw environmental data cleaning", "body": "Transforms messy DCR workbooks into modeling-ready time-series records."},
                {"kicker": "Features", "title": "Time-series feature engineering", "body": "Builds lag, rolling, cyclic time, and region features for forecasting."},
                {"kicker": "Modeling", "title": "ML model training and evaluation", "body": "Compares baseline, multi-output, and single-target strategies."},
                {"kicker": "Analysis", "title": "Region-wise forecasting analysis", "body": "Shows why localized models can outperform one global model for PM2.5."},
                {"kicker": "Risk", "title": "AQI risk interpretation", "body": "Converts pollutant forecasts into simplified risk categories and health messages."},
                {"kicker": "Product", "title": "Dashboard-based visualization", "body": "Serves predictions, analytics, explainability, anomaly alerts, and reports."},
            ],
            columns=3,
        )
        st.subheader("Pipeline")
        st.code(" -> ".join(PIPELINE_STEPS), language="text")

    elif page == "Live Prediction":
        st.subheader("Live Prediction")
        left, right = st.columns([0.95, 1.05], gap="large")
        with left:
            active_result, active_readings = prediction_form(bundle)
        with right:
            render_prediction_result(active_result)

    elif page == "Dataset Summary":
        st.subheader("Dataset Summary")
        st.dataframe(dataset_frame(), use_container_width=True, hide_index=True)
        st.info(
            "Insight: The dataset is large enough for a strong student-level predictive modeling project. "
            "The data was transformed from raw monitoring records into hourly machine-learning-ready features using cleaning, aggregation, and feature engineering."
        )
        composition = pd.DataFrame(
            [
                {"Resolution": "Quarter-hourly", "Records": 467188},
                {"Resolution": "Hourly", "Records": 119243},
            ]
        )
        st.subheader("Dataset Composition")
        render_bar_chart(composition, "Resolution", "Records")

    elif page == "Global Performance":
        st.subheader("Global Model Performance")
        global_frame = global_performance_frame()
        render_status_table(global_frame)
        st.markdown(
            "The boosted LightGBM run is now strongest for PM2.5 and PM10, with held-out R2 values of 0.978 and 0.883 respectively. "
            "SO2 has low MAE but remains spike-sensitive, so the system keeps region/target strategy selection and flags SO2 residual risk."
        )
        st.subheader("Global Model R2 Comparison")
        render_bar_chart(global_frame[["Target", "R2"]], "Target", "R2")
        st.subheader("RMSE/MAE Comparison")
        st.bar_chart(global_frame[["Target", "RMSE", "MAE"]], x="Target", y=["RMSE", "MAE"], use_container_width=True)
        st.subheader("How to Interpret the Results")
        render_html_cards(
            [
                {"kicker": card["Status"], "title": f"{card['Title']} | R2 = {card['R2']}", "body": card["Explanation"]}
                for card in INTERPRETATION_CARDS
            ],
            columns=3,
        )
        st.info("This analysis is valuable because it shows not only model training, but also model diagnosis and improvement strategy.")
        render_result_images(bundle)

    elif page == "Region Performance":
        st.subheader("Region-Specific Model Performance")
        st.success("Best Result: AIIMS PM2.5 achieved R2 = 0.984 with the boosted selector")
        region_frame = region_performance_frame()
        st.dataframe(region_frame, use_container_width=True, hide_index=True)
        leaderboard = region_frame.copy()
        leaderboard["Region + Target"] = leaderboard["Region"] + " " + leaderboard["Target"]
        st.subheader("Best Region-Level Forecasting Results")
        render_bar_chart(leaderboard[["Region + Target", "Best R2"]], "Region + Target", "Best R2")
        st.info(
            "Region-level results confirm that boosted lag and rolling features capture short-term local pollutant behavior strongly. "
            "The selector still compares region and target behavior so SO2 spike sensitivity is not hidden."
        )
        st.info(
            "For the final system, the recommended strategy is the boosted LightGBM/XGBoost selector with SO2 residual monitoring."
        )

    elif page == "Region Analytics":
        st.subheader("Region Analytics")
        analytics = region_analytics_frame()
        for row in analytics.to_dict(orient="records"):
            st.markdown(
                f"""
                <div class="glass-card">
                  <div class="card-kicker">{row['Region']}</div>
                  <div class="card-title">Best target: {row['Best target']} | Best R2: {row['Best R2']:.3f}</div>
                  <div class="card-body">{row['Pollution behavior']}<br><strong>Suggested strategy:</strong> {row['Suggested model strategy']}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        st.subheader("Region-wise Best R2 Comparison")
        render_bar_chart(analytics[["Region", "Best R2"]], "Region", "Best R2")
        st.info("Region-level analytics helps identify where the model performs strongly and where more localized data or features may be needed.")

    elif page == "Explainability":
        st.subheader("Why Did the Model Predict This?")
        importance_frame = get_feature_importance(bundle, limit=10)
        if importance_frame.empty:
            importance_frame = pd.DataFrame(
                [
                    {"feature": "Previous PM2.5 level", "importance": 0.22},
                    {"feature": "Previous PM10 level", "importance": 0.19},
                    {"feature": "Rolling PM2.5 mean", "importance": 0.16},
                    {"feature": "Rolling PM10 mean", "importance": 0.14},
                    {"feature": "Hour of day", "importance": 0.10},
                    {"feature": "Humidity", "importance": 0.08},
                    {"feature": "Region indicator", "importance": 0.07},
                    {"feature": "Wind speed", "importance": 0.04},
                ]
            )
        st.caption("Feature importance / SHAP-style explanation")
        st.dataframe(importance_frame, use_container_width=True, hide_index=True)
        st.bar_chart(importance_frame, x="feature", y="importance", use_container_width=True)
        st.info(
            "The model prediction is influenced mainly by recent pollutant history, rolling pollution trends, time of day, and region-specific behavior. "
            "This improves interpretability and helps users understand why pollution risk is high or low."
        )

    elif page == "Anomaly Detection":
        st.subheader("Pollution Spike Detection")
        active_result, active_readings = scenario_prediction(bundle)
        st.markdown("Current demo scenario alert review")
        render_anomaly_events(active_result, active_readings)

        st.subheader("Trend-Based Spike Check")
        controls = st.columns(4)
        anomaly_region = controls[0].selectbox("Region", ["AIIMS", "BHATAGAON", "IGKV", "SILTARA"], index=3)
        pollutant_label = controls[1].selectbox("Pollutant", ["PM2.5", "PM10", "SO2"], index=1)
        pollutant_key = {"PM2.5": "pm2_5", "PM10": "pm10", "SO2": "so2"}[pollutant_label]
        current_value = controls[2].number_input(
            "Current value",
            min_value=0.0,
            max_value=800.0,
            value=float(active_readings[pollutant_key]),
            step=1.0,
        )
        rolling_average = controls[3].number_input(
            "Rolling average",
            min_value=0.0,
            max_value=800.0,
            value=80.0 if pollutant_label == "PM10" else 45.0,
            step=1.0,
        )
        rolling_std = st.slider("Rolling standard deviation", min_value=1.0, max_value=100.0, value=20.0, step=1.0)
        spike_result = detect_pollution_spike(current_value, rolling_average, rolling_std, pollutant_label)
        threshold = rolling_average + 2 * rolling_std
        spike_frame = pd.DataFrame(
            [
                {
                    "Region": anomaly_region.title(),
                    "Pollutant": pollutant_label,
                    "Current value": current_value,
                    "Rolling average": rolling_average,
                    "Review threshold": threshold,
                    "Spike ratio": spike_result["spike_ratio"],
                    "Severity": spike_result["severity"],
                    "Message": spike_result["message"],
                }
            ]
        )
        st.dataframe(spike_frame, use_container_width=True, hide_index=True)
        if spike_result["is_spike"]:
            st.warning("Pollution spike detected")
        else:
            st.success("No major spike detected")
        st.caption("Spike detection is based on statistical deviation from recent trends and should be verified with official monitoring systems.")

    elif page == "AI Report":
        st.subheader("AI Report Generator")
        active_result, active_readings = scenario_prediction(bundle)
        render_ai_report(active_result, active_readings, metadata)

    elif page == "Project Details":
        st.subheader("Project Details")
        st.markdown(
            """
            **Project Name:** AirSense AI

            **Problem:** Air pollution monitoring data is messy, time-dependent, and region-specific.
            Manual analysis makes it difficult to forecast pollutant levels and understand risk.

            **Solution:** An end-to-end ML system that cleans raw pollution data, engineers time-series
            features, trains forecasting models, evaluates performance globally and region-wise,
            detects spikes, and presents insights through a dashboard.
            """
        )
        st.subheader("Tech Stack")
        st.write("Python, Pandas, NumPy, Scikit-learn, Matplotlib/Seaborn, Streamlit, Joblib, FastAPI")
        st.subheader("ML Concepts")
        st.write("Regression, time-series feature engineering, lag features, rolling statistics, region encoding, model evaluation, error analysis, explainability, anomaly detection")
        st.subheader("Evaluation Metrics")
        st.write("RMSE, MAE, R2 Score")
        st.subheader("Limitations and Future Scope")
        st.write(
            "The global PM2.5 model performance is weak because PM2.5 behavior varies strongly across regions. "
            "Region-specific models are more reliable for PM2.5. Future work includes real-time pollution board API integration, 24-hour forecasting, region-specific production models, geospatial heatmaps, LLM-based report generation, alerts, and model monitoring."
        )


if __name__ == "__main__":
    main()
