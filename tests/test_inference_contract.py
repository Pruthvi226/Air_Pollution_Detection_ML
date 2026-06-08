from __future__ import annotations

import unittest

from airsense.inference import build_feature_frame, classify_risk, normalize_region


class InferenceContractTests(unittest.TestCase):
    def test_normalize_region_accepts_known_aliases(self) -> None:
        self.assertEqual(normalize_region("siltara"), "SILTARA")
        self.assertEqual(normalize_region("DCR AIIMS"), "AIIMS")

    def test_feature_frame_uses_model_feature_order(self) -> None:
        bundle = {
            "feature_columns": [
                "year",
                "hour",
                "pm2_5",
                "pm2_5_lag_1",
                "pm2_5_roll_mean_3",
                "pm2_5_roll_std_3",
                "region_AIIMS",
                "region_SILTARA",
            ]
        }
        frame = build_feature_frame(
            bundle=bundle,
            region="SILTARA",
            readings={"pm2_5": 55},
            date_time="2025-11-20 18:00:00",
        )
        self.assertEqual(list(frame.columns), bundle["feature_columns"])
        self.assertEqual(frame.loc[0, "year"], 2025)
        self.assertEqual(frame.loc[0, "hour"], 18)
        self.assertEqual(frame.loc[0, "pm2_5_lag_1"], 55)
        self.assertEqual(frame.loc[0, "pm2_5_roll_mean_3"], 55)
        self.assertEqual(frame.loc[0, "pm2_5_roll_std_3"], 0)
        self.assertEqual(frame.loc[0, "region_SILTARA"], 1)
        self.assertEqual(frame.loc[0, "region_AIIMS"], 0)

    def test_risk_thresholds(self) -> None:
        self.assertEqual(classify_risk({"pm2_5": 20, "pm10": 60, "so2": 8})["label"], "Good")
        self.assertEqual(classify_risk({"pm2_5": 45, "pm10": 110, "so2": 19})["label"], "Moderate")
        self.assertEqual(classify_risk({"pm2_5": 70, "pm10": 180, "so2": 30})["label"], "Poor")
        self.assertEqual(classify_risk({"pm2_5": 95, "pm10": 260, "so2": 45})["label"], "Severe")


if __name__ == "__main__":
    unittest.main()
