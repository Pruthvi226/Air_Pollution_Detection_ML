from __future__ import annotations

import pytest

from airsense.inference import load_model_bundle, predict


def test_inference_bundle_loads_and_predicts() -> None:
    bundle = load_model_bundle()
    result = predict(
        bundle,
        region="SILTARA",
        readings={"pm2_5": 78, "pm10": 214, "so2": 34, "temp": 35, "hum": 68, "ws": 1.8},
    )
    assert set(result["predictions"]) == {"pm2_5", "pm10", "so2"}
    assert result["risk"]["category"]
    assert result["explanation"]["top_features"]


def test_invalid_region_has_clear_error() -> None:
    bundle = load_model_bundle()
    with pytest.raises(ValueError, match="Unknown region"):
        predict(bundle, region="UNKNOWN", readings={"pm2_5": 1, "pm10": 1, "so2": 1})
