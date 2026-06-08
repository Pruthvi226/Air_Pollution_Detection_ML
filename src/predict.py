"""Hybrid model-selection helpers for AirSense AI."""

from __future__ import annotations

from typing import Any

from src.config import GLOBAL_PERFORMANCE, REGION_PERFORMANCE


def _normalize_target(target: str) -> str:
    normalized = str(target).strip().upper().replace("_", ".")
    return {"PM25": "PM2.5", "PM2.5": "PM2.5", "PM10": "PM10", "SO2": "SO2"}.get(normalized, normalized)


def _normalize_region(region: str) -> str:
    return str(region).strip().upper()


def select_best_model(region: str, target: str) -> dict[str, Any]:
    """Return the recommended hybrid strategy for a region and target."""
    normalized_region = _normalize_region(region)
    normalized_target = _normalize_target(target)
    region_rows = [
        row for row in REGION_PERFORMANCE
        if row["Region"].upper() == normalized_region and row["Target"].upper() == normalized_target
    ]
    global_rows = [row for row in GLOBAL_PERFORMANCE if row["Target"].upper() == normalized_target]
    global_r2 = float(global_rows[0]["R2"]) if global_rows else float("-inf")

    if normalized_target == "PM2.5":
        if region_rows:
            return {
                "model": f"{normalized_region}_PM25_Model",
                "strategy": "Region-specific single-target",
                "reason": "Region-specific PM2.5 model performs better than the global PM2.5 model.",
            }
        return {
            "model": "global_pm25_model",
            "strategy": "Global single-target fallback",
            "reason": "Region-specific PM2.5 model is unavailable, so the global PM2.5 model is used with caution.",
        }

    if normalized_target == "PM10":
        if region_rows and float(region_rows[0]["Best R2"]) > global_r2:
            return {
                "model": f"{normalized_region}_PM10_Model",
                "strategy": "Region-specific single-target",
                "reason": "Region-specific PM10 result outperforms the global PM10 model.",
            }
        return {
            "model": "global_pm10_model",
            "strategy": "Global single-target",
            "reason": "Global PM10 model provides the strongest available baseline for this target.",
        }

    if normalized_target == "SO2":
        if region_rows and float(region_rows[0]["Best R2"]) > global_r2:
            return {
                "model": f"{normalized_region}_SO2_Model",
                "strategy": "Region-specific single-target",
                "reason": "Region-specific SO2 result outperforms the global SO2 model.",
            }
        return {
            "model": "global_so2_model",
            "strategy": "Global single-target",
            "reason": "Global SO2 model is the best available strategy for this input.",
        }

    return {
        "model": "best_available_model",
        "strategy": "Best available",
        "reason": "Target-specific strategy was not found, so the best available model is selected.",
    }

