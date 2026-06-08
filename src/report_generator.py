"""Rule-based AI-style report generation for AirSense AI."""

from __future__ import annotations

from typing import Any, Mapping, Sequence


def generate_air_quality_report(
    region: str,
    predictions: Mapping[str, Any],
    aqi_result: Mapping[str, Any],
    anomaly_result: Mapping[str, Any],
    top_features: Sequence[str],
    model_strategy: Mapping[str, Any] | str,
) -> str:
    """Generate a concise air-quality summary paragraph."""
    if isinstance(model_strategy, Mapping):
        strategy_text = str(model_strategy.get("strategy", "hybrid forecasting strategy"))
    else:
        strategy_text = str(model_strategy)

    feature_text = ", ".join(top_features[:3]) if top_features else "recent pollutant history, rolling pollution averages, and time-of-day patterns"
    anomaly_text = (
        "The anomaly detection module also identified abnormal pollutant behavior compared to recent trends."
        if anomaly_result.get("is_spike") or anomaly_result.get("severity") in {"Medium", "High"}
        else "The anomaly detection module did not flag a major spike for this scenario."
    )
    category = aqi_result.get("category", "Moderate")
    message = aqi_result.get("health_message") or aqi_result.get("message") or aqi_result.get("recommendation", "")

    return (
        "This report is generated using a rule-based NLP-style summarization template.\n\n"
        f"Today's forecast for {str(region).title()} indicates PM2.5 at {float(predictions.get('pm2_5', 0)):.1f} ug/m3, "
        f"PM10 at {float(predictions.get('pm10', 0)):.1f} ug/m3, and SO2 at {float(predictions.get('so2', 0)):.1f} ug/m3. "
        f"The AQI-style risk category is {category}. The hybrid model selected {strategy_text} because local pollution behavior is important for this forecast. "
        f"The prediction is mainly influenced by {feature_text}. {anomaly_text} {message}"
    )

