from __future__ import annotations

from typing import Any, Mapping


AQI_CATEGORIES = [
    {
        "category": "Good",
        "risk_level": "Low",
        "score": 1,
        "color": "#207a58",
        "recommendation": "Air quality appears acceptable for normal outdoor activity.",
    },
    {
        "category": "Moderate",
        "risk_level": "Elevated",
        "score": 2,
        "color": "#0f7c75",
        "recommendation": "Sensitive groups should monitor symptoms during prolonged outdoor activity.",
    },
    {
        "category": "Poor",
        "risk_level": "High",
        "score": 3,
        "color": "#d58c22",
        "recommendation": "Sensitive groups should reduce prolonged outdoor exposure.",
    },
    {
        "category": "Severe",
        "risk_level": "Very High",
        "score": 4,
        "color": "#c2414b",
        "recommendation": "Limit outdoor exposure and consider protective measures for sensitive groups.",
    },
]


def _to_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _risk_score(pm25: float, pm10: float, so2: float) -> int:
    if pm25 > 90 or pm10 > 250 or so2 > 40:
        return 4
    if pm25 > 60 or pm10 > 160 or so2 > 28:
        return 3
    if pm25 > 35 or pm10 > 100 or so2 > 18:
        return 2
    return 1


def get_aqi_category(score: int) -> str:
    bounded_score = min(max(int(score), 1), len(AQI_CATEGORIES))
    return AQI_CATEGORIES[bounded_score - 1]["category"]


def generate_health_recommendation(category: str) -> str:
    normalized = str(category).strip().lower()
    for row in AQI_CATEGORIES:
        if row["category"].lower() == normalized:
            return row["recommendation"]
    return AQI_CATEGORIES[1]["recommendation"]


def classify_aqi_risk(pm25: Any, pm10: Any, so2: Any = 0) -> dict[str, Any]:
    """Simplified project-level AQI-style risk layer for dashboard display."""
    pm25_value = _to_float(pm25)
    pm10_value = _to_float(pm10)
    so2_value = _to_float(so2)

    if pm25_value <= 30 and pm10_value <= 50 and so2_value <= 15:
        category = "Good"
        risk_level = "Low"
        color = "#22c55e"
        message = "Air quality appears acceptable for normal outdoor activity."
    elif pm25_value <= 60 and pm10_value <= 100 and so2_value <= 30:
        category = "Moderate"
        risk_level = "Elevated"
        color = "#facc15"
        message = "Sensitive groups should monitor symptoms during prolonged outdoor activity."
    elif pm25_value <= 90 and pm10_value <= 250 and so2_value <= 40:
        category = "Poor"
        risk_level = "High"
        color = "#f97316"
        message = "Sensitive groups should reduce prolonged outdoor exposure."
    elif pm25_value <= 120 and pm10_value <= 350 and so2_value <= 80:
        category = "Very Poor"
        risk_level = "Very High"
        color = "#ef4444"
        message = "Limit outdoor exposure and use protective measures when possible."
    else:
        category = "Severe"
        risk_level = "Severe"
        color = "#991b1b"
        message = "Avoid prolonged outdoor exposure and verify conditions with official monitoring sources."

    return {
        "category": category,
        "risk_level": risk_level,
        "message": message,
        "recommendation": message,
        "badge_color": color,
        "color": color,
        "disclaimer": "This AQI interpretation is a simplified project-level risk layer and not an official regulatory AQI calculation.",
    }


def calculate_pollutant_risk(pm25: Any, pm10: Any, so2: Any) -> dict[str, Any]:
    pm25_value = _to_float(pm25)
    pm10_value = _to_float(pm10)
    so2_value = _to_float(so2)
    score = _risk_score(pm25_value, pm10_value, so2_value)
    category_row = AQI_CATEGORIES[score - 1]
    return {
        "category": category_row["category"],
        "risk_category": category_row["category"],
        "risk_level": category_row["risk_level"],
        "score": score,
        "level": score,
        "color": category_row["color"],
        "recommendation": category_row["recommendation"],
        "drivers": _risk_drivers(pm25_value, pm10_value, so2_value),
    }


def _risk_drivers(pm25: float, pm10: float, so2: float) -> list[str]:
    drivers: list[str] = []
    if pm25 > 35:
        drivers.append("PM2.5")
    if pm10 > 100:
        drivers.append("PM10")
    if so2 > 18:
        drivers.append("SO2")
    return drivers or ["All tracked pollutants"]


def summarize_prediction(prediction_dict: Mapping[str, Any]) -> str:
    predictions = prediction_dict.get("predictions", prediction_dict)
    pm25 = _to_float(predictions.get("pm2_5", predictions.get("pm25")))
    pm10 = _to_float(predictions.get("pm10"))
    so2 = _to_float(predictions.get("so2"))
    risk = calculate_pollutant_risk(pm25, pm10, so2)
    return (
        f"Predicted PM2.5 is {pm25:.1f}, PM10 is {pm10:.1f}, and SO2 is {so2:.1f}. "
        f"The project AQI-style category is {risk['category']} with {risk['risk_level'].lower()} risk. "
        f"{risk['recommendation']}"
    )
