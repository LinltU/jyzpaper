import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from production.humidity_stage_ensemble_predictor import HumidityStageEnsemblePredictor


class OnlineEnsemblePredictorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.predictor = HumidityStageEnsemblePredictor(
            ROOT / "humidity_stage_feature_ensemble_evaluation" / "ensemble_manifest.json",
            device="cpu",
        )
        data_dir = ROOT / "data" / "raw"
        training = pd.read_csv(data_dir / "training.csv", dtype={"IMEI": str})
        test = pd.read_csv(data_dir / "test.csv", dtype={"IMEI": str})
        cls.combined = pd.concat([training, test], ignore_index=True)
        cls.combined["Timestamp"] = pd.to_datetime(cls.combined["Timestamp"])
        cls.expected = []
        with (
            ROOT / "humidity_stage_feature_ensemble_evaluation" / "predictions.jsonl"
        ).open("r", encoding="utf-8") as handle:
            for line in handle:
                cls.expected.append(json.loads(line))

    def assert_replays_offline_prediction(self, expected):
        imei = str(expected["IMEI"])
        target_timestamp = pd.Timestamp(expected["Timestamp"])
        group = self.combined[self.combined["IMEI"] == imei].copy()
        group = group[group["Timestamp"] <= target_timestamp].sort_values(
            "Timestamp",
            ascending=False,
        )
        records = [
            (
                row.Measure,
                row.Temp,
                row.Hum,
                row.Timestamp,
                row.Battery,
                row.Version,
            )
            for row in group.itertuples()
        ]
        target = group.iloc[0]

        actual = self.predictor.predict_from_records(
            imei=imei,
            location=target["Location"],
            records=records,
            target_temp=target["Temp"],
            target_hum=target["Hum"],
            target_timestamp=target_timestamp,
            actual_measure=target["Measure"],
        )

        self.assertEqual(actual["SelectedModel"], expected["SelectedModel"])
        self.assertAlmostEqual(actual["P10"], expected["P10"], places=5)
        self.assertAlmostEqual(actual["P50"], expected["P50"], places=5)
        self.assertAlmostEqual(actual["P90"], expected["P90"], places=5)

    def test_replays_clean_elapsed_prediction(self):
        expected = next(row for row in self.expected if row["SelectedModel"] == "clean_elapsed")
        self.assert_replays_offline_prediction(expected)

    def test_replays_dynamic_high_prediction(self):
        expected = next(
            row for row in self.expected
            if row["SelectedModel"] == "dynamic_high_expert"
        )
        self.assert_replays_offline_prediction(expected)

    def test_blends_multiple_high_experts_with_normalized_weights(self):
        class FakeStageModel:
            def __init__(self, path, _device, _manifest_dir=None):
                self.path = Path(path)
                self.seq_len = 24

            def predict(self, *_args, **_kwargs):
                offset = 1.0 if self.path.name == "high_a.pth" else 3.0
                return {
                    "P10": offset,
                    "P50": offset + 1.0,
                    "P90": offset + 2.0,
                    "Persistence": offset + 3.0,
                    "HumidityStage": "very_humid",
                    "ForecastDeltaHours": offset + 4.0,
                }

        manifest = {
            "base_checkpoint": "base.pth",
            "high_humidity_checkpoint": "high_a.pth",
            "high_humidity_checkpoints": [
                {"checkpoint": "high_a.pth", "weight": 1.0},
                {"checkpoint": "high_b.pth", "weight": 3.0},
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            manifest_path = Path(directory) / "manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with patch(
                "production.humidity_stage_ensemble_predictor._LoadedStageModel",
                FakeStageModel,
            ):
                predictor = HumidityStageEnsemblePredictor(
                    manifest_path, device="cpu"
                )
                result = predictor._predict_high_blend(
                    pd.DataFrame(), "device", "location", 20.0, 90.0,
                    pd.Timestamp("2026-07-30"),
                )

        self.assertAlmostEqual(result["P10"], 2.5)
        self.assertAlmostEqual(result["P50"], 3.5)
        self.assertAlmostEqual(result["P90"], 4.5)
        self.assertAlmostEqual(result["Persistence"], 5.5)
        self.assertAlmostEqual(result["ForecastDeltaHours"], 6.5)
        self.assertEqual(
            [item["weight"] for item in result["HighExpertBlendWeights"]],
            [0.25, 0.75],
        )


if __name__ == "__main__":
    unittest.main()
