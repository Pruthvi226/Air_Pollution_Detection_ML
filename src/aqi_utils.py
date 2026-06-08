"""Simplified AQI-style risk utilities for the AirSense AI dashboard."""

from __future__ import annotations

from typing import Any


def _to_float(value: Any) -> float:
    """Convert a value to float, returning zero for invalid inputs."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def classify_aqi_risk(pm25: Any, pm10: Any, so2: Any = 0) -> dict[str, Any]:
    """Classify pollutant values into a simplified project-level AQI risk."""
    pm25_value = _to_float(pm25)
    pm10_value = _to_float(pm10)
    so2_value = _to_float(so2)

    if pm25_value <= 30 and pm10_value <= 50 and so2_value <= 15:
        category, risk_level, color = "Good", "Low", "#22c55e"
        message = "Air quality appears acceptable for normal outdoor activity."
    elif pm25_value <= 60 and pm10_value <= 100 and so2_value <= 30:
        category, risk_level, color = "Moderate", "Elevated", "#facc15"
        message = "Sensitive groups should monitor symptoms during prolonged outdoor activity."
    elif pm25_value <= 90 and pm10_value <= 250 and so2_value <= 40:
        category, risk_level, color = "Poor", "High", "#f97316"
        message = "Sensitive groups should reduce prolonged outdoor exposure."
    elif pm25_value <= 120 and pm10_value <= 350 and so2_value <= 80:
        category, risk_level, color = "Very Poor", "Very High", "#ef4444"
        message = "Limit outdoor exposure and use protective measures when possible."
    else:
        category, risk_level, color = "Severe", "Severe", "#991b1b"
        message = "Avoid prolonged outdoor exposure and verify conditions with official monitoring sources."

    return {
        "category": category,
        "risk_level": risk_level,
        "health_message": message,
        "message": message,
        "recommendation": message,
        "badge_color": color,
        "color": color,
        "explanation": "This AQI interpretation is a simplified project-level risk layer and not an official regulatory AQI calculation.",
        "disclaimer": "This AQI interpretation is a simplified project-level risk layer and not an official regulatory AQI calculation.",
    }
