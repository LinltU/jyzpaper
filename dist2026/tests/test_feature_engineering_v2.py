import os
import sys
import unittest

import numpy as np
import pandas as pd


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from experiments.evaluate_fmlp_fixed_split import build_all_windows
from experiments.evaluate_humidity_stage_fmlp import routing_features
from core.feature_engineering_v2 import (
    feature_columns_v2,
    parse_location_features,
    prepare_rows_v2,
)


class FeatureEngineeringV2Tests(unittest.TestCase):
    def test_compact_location_fields_are_parsed_by_meaning(self):
        locations = pd.Series([
            "LLowSStr1-220kV-18-P-3-4",
            "RUpBStr2-220kV-18-P-15-16",
            "YZStr1-110kV-9-P-6-7",
            "\u5de6\u8fb9\u4e0b\u76f8\u5c0f\u53f7\u4fa7\u4e321_220kV_18\u7247_\u7b2c\u5341\u4e94~\u5341\u516d\u7247",
        ])

        parsed = parse_location_features(locations)

        self.assertEqual(parsed["Loc_SideCode"].tolist(), [-1.0, 1.0, 0.0, -1.0])
        self.assertEqual(parsed["Loc_PhaseCode"].tolist(), [-1.0, 1.0, 0.0, -1.0])
        self.assertEqual(parsed["Loc_SizeCode"].tolist(), [-1.0, 1.0, 0.0, -1.0])
        self.assertEqual(parsed["Loc_StrNum"].tolist(), [1.0, 2.0, 1.0, 1.0])
        self.assertEqual(parsed["LineVoltageKV"].tolist(), [220.0, 220.0, 110.0, 220.0])
        self.assertEqual(parsed["Loc_PlateCount"].tolist(), [18.0, 18.0, 9.0, 18.0])
        self.assertEqual(parsed["Loc_PlateStart"].tolist(), [3.0, 15.0, 6.0, 15.0])
        self.assertEqual(parsed["Loc_PlateEnd"].tolist(), [4.0, 16.0, 7.0, 16.0])

    def test_features_are_causal_and_do_not_require_battery(self):
        frame = pd.DataFrame({
            "IMEI": ["device-a"] * 4,
            "Location": ["LLowSStr1-220kV-18-P-3-4"] * 4,
            "Timestamp": pd.date_range("2026-01-01", periods=4, freq="12h"),
            "Measure": [0.10, 0.12, 0.16, 0.20],
            "Temp": [10.0, 11.0, 12.0, 13.0],
            "Hum": [50.0, 0.0, 70.0, 82.0],
        })

        prepared = prepare_rows_v2(frame, humidity_fill_value=55.0, median_delta_hours=12.0)

        self.assertEqual(len(feature_columns_v2()), 28)
        self.assertNotIn("Battery", feature_columns_v2())
        self.assertFalse(any(column.startswith("Device_") for column in feature_columns_v2()))
        self.assertEqual(prepared["Hum"].tolist(), [50.0, 50.0, 70.0, 82.0])
        self.assertEqual(prepared["HumInvalid"].tolist(), [0.0, 1.0, 0.0, 0.0])
        self.assertAlmostEqual(prepared.loc[2, "HumRate"], 20.0 / 12.0)
        self.assertAlmostEqual(prepared.loc[2, "HoursAbove65"], 12.0)
        self.assertAlmostEqual(prepared.loc[3, "HoursAbove80"], 12.0)
        self.assertEqual(prepared["DeviceElapsedDays"].tolist(), [0.0, 0.5, 1.0, 1.5])
        self.assertTrue(np.isfinite(prepared[feature_columns_v2()].to_numpy()).all())

    def test_causal_router_uses_last_observed_humidity(self):
        frame = pd.DataFrame({
            "IMEI": ["device-a"] * 4,
            "Location": ["LLowSStr1-220kV-18-P-3-4"] * 4,
            "Timestamp": pd.date_range("2026-01-01", periods=4, freq="12h"),
            "Measure": [0.10, 0.12, 0.16, 0.20],
            "Temp": [10.0, 11.0, 12.0, 13.0],
            "Hum": [45.0, 48.0, 72.0, 85.0],
        })
        prepared = prepare_rows_v2(frame, median_delta_hours=12.0)
        _, _, metadata = build_all_windows(prepared, feature_columns_v2(), seq_len=2)

        target_route = routing_features(metadata, causal=False)
        causal_route = routing_features(metadata, causal=True)

        self.assertEqual(target_route[0].tolist(), [72.0, 24.0])
        self.assertEqual(causal_route[0].tolist(), [48.0, 3.0])


if __name__ == "__main__":
    unittest.main()
