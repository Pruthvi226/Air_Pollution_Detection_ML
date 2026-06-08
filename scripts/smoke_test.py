from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from airsense.inference import load_model_bundle, predict


def main() -> int:
    bundle = load_model_bundle()
    result = predict(
        bundle,
        region="SILTARA",
        readings={"pm2_5": 78, "pm10": 214, "so2": 34, "temp": 35, "hum": 68, "ws": 1.8},
        date_time="2026-06-08T10:00:00",
    )
    required_targets = {"pm2_5", "pm10", "so2"}
    if set(result["predictions"]) != required_targets:
        raise RuntimeError(f"Unexpected prediction keys: {result['predictions'].keys()}")
    print(result["summary"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
