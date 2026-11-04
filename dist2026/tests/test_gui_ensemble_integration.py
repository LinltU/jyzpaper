import json
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PyQt5.QtWidgets import QApplication, QMessageBox

from production.main import MainWindow


class _AcceptedDialog:
    def accept(self):
        pass


class GuiEnsembleIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])
        data_dir = ROOT / "data" / "raw"
        training = pd.read_csv(data_dir / "training.csv", dtype={"IMEI": str})
        test = pd.read_csv(data_dir / "test.csv", dtype={"IMEI": str})
        cls.combined = pd.concat([training, test], ignore_index=True)
        cls.combined["Timestamp"] = pd.to_datetime(cls.combined["Timestamp"])
        with (
            ROOT
            / "humidity_stage_feature_ensemble_evaluation"
            / "predictions.jsonl"
        ).open("r", encoding="utf-8") as handle:
            cls.expected = [json.loads(line) for line in handle]

    def setUp(self):
        self.window = MainWindow()
        self.messages = patch.multiple(
            QMessageBox,
            information=Mock(return_value=QMessageBox.Ok),
            warning=Mock(return_value=QMessageBox.Ok),
            critical=Mock(return_value=QMessageBox.Ok),
        )
        self.messages.start()

    def tearDown(self):
        self.messages.stop()
        self.window.close()

    def _configure_replay(self, selected_model):
        expected = next(
            row for row in self.expected
            if row["SelectedModel"] == selected_model
        )
        imei = str(expected["IMEI"])
        timestamp = pd.Timestamp(expected["Timestamp"])
        group = self.combined[self.combined["IMEI"] == imei].copy()
        group = group[group["Timestamp"] <= timestamp].sort_values(
            "Timestamp",
            ascending=False,
        )
        target = group.iloc[0]
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
        latest = records[0]

        self.window.devices = [imei]
        self.window.device_locations = {imei: str(target["Location"])}
        self.window.device_voltages = {imei: "220kV"}
        self.window.device_string_ids = {imei: "集成测试串"}
        self.window.line_to_devices = {0: [0]}
        self.window.line_names = {0: "集成测试串"}
        self.window.data_fetcher = Mock()
        self.window.data_fetcher.get_all_latest_data.return_value = [latest]
        self.window.data_fetcher.get_last_n_data.return_value = records
        return expected, records

    def test_refresh_routes_both_humidity_ranges(self):
        for selected_model in ("clean_elapsed", "dynamic_high_expert"):
            with self.subTest(selected_model=selected_model):
                expected, records = self._configure_replay(selected_model)
                selected_models = []
                original = self.window.predict_with_stage_ensemble

                def capture_prediction(*args, **kwargs):
                    result = self.window.ensemble_predictor.predict_from_records(
                        imei=str(args[0]),
                        location=args[1],
                        records=args[2],
                        target_temp=args[3],
                        target_hum=args[4],
                        target_timestamp=args[5],
                        actual_measure=args[6],
                    )
                    selected_models.append(result["SelectedModel"])
                    return self.window.ensemble_predictor.legacy_tuple(result)

                self.window.predict_with_stage_ensemble = capture_prediction

                self.window.refresh_data(_AcceptedDialog())

                self.assertEqual(selected_models, [selected_model])
                self.window.data_fetcher.get_last_n_data.assert_called_once_with(
                    expected["IMEI"],
                    self.window.ensemble_predictor.required_history_records,
                )
                self.assertGreaterEqual(len(records), 25)
                self.assertIsNotNone(self.window.status_table2.item(0, 6))
                self.window.predict_with_stage_ensemble = original

    def test_refresh_falls_back_to_legacy_model(self):
        expected, records = self._configure_replay("clean_elapsed")
        self.window.predict_with_stage_ensemble = Mock(
            side_effect=ValueError("simulated ensemble failure"),
        )
        self.window.model_trainer = Mock()
        self.window.model_trainer.predict.return_value = (
            1.0,
            0.1,
            False,
            "绝缘正常",
            90.0,
            88.0,
        )

        self.window.refresh_data(_AcceptedDialog())

        self.window.model_trainer.load_model.assert_called()
        args = self.window.model_trainer.predict.call_args.args
        self.assertEqual(list(args[2]), [records[0][0], records[1][0]])
        self.assertEqual(self.window.status_table2.item(0, 6).text(), "88.0 %")


if __name__ == "__main__":
    unittest.main()
