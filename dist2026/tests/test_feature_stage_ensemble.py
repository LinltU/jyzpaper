import os
import sys
import unittest


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from experiments.evaluate_feature_stage_ensemble import combine_predictions


class FeatureStageEnsembleTests(unittest.TestCase):
    def test_high_humidity_rows_use_dynamic_expert(self):
        base = [
            {
                "IMEI": "a",
                "Timestamp": "2026-01-01T00:00:00",
                "Hum": 70.0,
                "Actual": 0.2,
                "P10": 0.1,
                "P50": 0.2,
                "P90": 0.3,
            },
            {
                "IMEI": "a",
                "Timestamp": "2026-01-02T00:00:00",
                "Hum": 85.0,
                "Actual": 0.8,
                "P10": 0.4,
                "P50": 0.5,
                "P90": 0.6,
            },
        ]
        high = [
            dict(base[0], P10=0.05, P50=0.15, P90=0.25),
            dict(base[1], P10=0.7, P50=0.8, P90=0.9),
        ]

        combined = combine_predictions(base, high, humidity_threshold=80.0)

        self.assertEqual(combined[0]["SelectedModel"], "clean_elapsed")
        self.assertEqual(combined[0]["P50"], 0.2)
        self.assertEqual(combined[1]["SelectedModel"], "dynamic_high_expert")
        self.assertEqual(combined[1]["P50"], 0.8)


if __name__ == "__main__":
    unittest.main()
