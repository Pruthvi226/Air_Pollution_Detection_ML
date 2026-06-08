from __future__ import annotations

from airsense.aqi import calculate_pollutant_risk, generate_health_recommendation, get_aqi_category


def test_category_generation_for_low_and_high_values() -> None:
    assert calculate_pollutant_risk(18, 55, 8)["category"] == "Good"
    assert calculate_pollutant_risk(45, 120, 22)["category"] == "Moderate"
    assert calculate_pollutant_risk(70, 180, 30)["category"] == "Poor"
    assert calculate_pollutant_risk(95, 260, 45)["category"] == "Severe"


def test_recommendation_is_never_empty() -> None:
    for score in [1, 2, 3, 4]:
        category = get_aqi_category(score)
        assert generate_health_recommendation(category)
