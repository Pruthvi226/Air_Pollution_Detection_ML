from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from airsense.inference import load_model_bundle, prediction_to_json, predict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one AirSense AI next-step prediction.")
    parser.add_argument("--model-dir", type=Path, default=None)
    parser.add_argument("--region", default="SILTARA", choices=["AIIMS", "BHATAGAON", "IGKV", "SILTARA"])
    parser.add_argument("--date-time", default=None)
    parser.add_argument("--strategy", default="best", choices=["best", "multi_output", "single_target"])
    parser.add_argument("--pm25", "--pm2-5", dest="pm2_5", type=float, default=38.0)
    parser.add_argument("--pm10", type=float, default=112.0)
    parser.add_argument("--so2", type=float, default=18.0)
    parser.add_argument("--temp", type=float, default=31.0)
    parser.add_argument("--hum", "--humidity", dest="hum", type=float, default=58.0)
    parser.add_argument("--ws", "--wind-speed", dest="ws", type=float, default=3.2)
    parser.add_argument("--wd", "--wind-direction", dest="wd", type=float, default=180.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    bundle = load_model_bundle(args.model_dir)
    readings = {
        "pm2_5": args.pm2_5,
        "pm10": args.pm10,
        "so2": args.so2,
        "temp": args.temp,
        "hum": args.hum,
        "ws": args.ws,
        "wd": args.wd,
    }
    result = predict(
        bundle,
        region=args.region,
        readings=readings,
        date_time=args.date_time,
        strategy=args.strategy,
    )
    print(prediction_to_json(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
