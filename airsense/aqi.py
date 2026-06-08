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
        f"The demo AQI-style category is {risk['category']} with {risk['risk_level'].lower()} risk. "
        f"{risk['recommendation']}"
    )
