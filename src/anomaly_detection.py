"""Pollution spike detection utilities."""

from __future__ import annotations

from typing import Any


def detect_pollution_spike(
    current_value: float,
    rolling_mean: float,
    rolling_std: float,
    pollutant: str,
) -> dict[str, Any]:
    """Detect whether a pollutant value is above recent trend."""
    current = float(current_value)
    mean = float(rolling_mean)
    std = max(float(rolling_std), 1e-6)
    spike_ratio = current / mean if mean > 0 else 0.0

    if current > mean + 3 * std:
        severity = "High"
        is_spike = True
    elif current > mean + 2 * std:
        severity = "Medium"
        is_spike = True
    else:
        severity = "Normal"
        is_spike = False

    if is_spike:
        message = f"{pollutant} is significantly above recent trend. Monitor exposure and verify sensor readings."
    else:
        message = f"{pollutant} is within the expected recent trend range."

    return {
        "is_spike": is_spike,
        "severity": severity,
        "spike_ratio": round(spike_ratio, 2),
        "message": message,
    }

