"""Online inference for the battery-free humidity-stage FMLP ensemble."""

import json
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
if __package__ in (None, ""):
    import sys
    sys.path.insert(0, str(PROJECT_DIR))

import numpy as np
import pandas as pd
import torch

from experiments.evaluate_humidity_stage_fmlp import known_target_covariates, stage_name
from core.feature_engineering_v2 import add_device_one_hot, prepare_rows_v2
from model.humidity_stage_fmlp import HumidityStageFMLP
from core.project_paths import DIST_DIR, resolve_project_path


class _LoadedStageModel:
    def __init__(self, checkpoint_path, device, manifest_dir=None):
        self.path = resolve_project_path(
            checkpoint_path,
            relative_to=manifest_dir or DIST_DIR,
        )
        self.checkpoint = torch.load(str(self.path), map_location=device)
        self.model = HumidityStageFMLP(**self.checkpoint["model_config"]).to(device)
        self.model.load_state_dict(self.checkpoint["state_dict"])
        self.model.eval()
        self.device = device
        self.columns = list(self.checkpoint["feature_columns"])
        self.seq_len = int(self.checkpoint["model_config"]["seq_len"])

    def _prepare_history(self, history, imei, location):
        frame = history.copy()
        frame["IMEI"] = str(imei)
        frame["Location"] = str(location)
        preprocessing = self.checkpoint.get("preprocessing", {})
        prepared = prepare_rows_v2(
            frame,
            humidity_fill_value=preprocessing.get("humidity_fill_value"),
            median_delta_hours=preprocessing.get("median_delta_hours"),
            device_start_times=preprocessing.get("device_start_times") or None,
        )

        device_columns = [
            column[len("Device_"):]
            for column in self.columns
            if column.startswith("Device_")
        ]
        if device_columns:
            prepared = add_device_one_hot(prepared, device_columns)

        missing = [column for column in self.columns if column not in prepared.columns]
        if missing:
            raise ValueError("Checkpoint features are missing after preparation: " + ", ".join(missing))
        if len(prepared) < self.seq_len:
            raise ValueError(
                "At least {} historical rows before the target are required; got {}".format(
                    self.seq_len,
                    len(prepared),
                )
            )

        values = prepared[self.columns].tail(self.seq_len).to_numpy(dtype=np.float32)
        mean = np.asarray(self.checkpoint["x_mean"], dtype=np.float32)
        scale = np.asarray(self.checkpoint["x_scale"], dtype=np.float32)
        values = (values - mean) / np.where(scale == 0.0, 1.0, scale)
        return values, prepared

    def predict(self, history, imei, location, target_temp, target_hum, target_timestamp):
        values, prepared = self._prepare_history(history, imei, location)
        last_row = prepared.iloc[-1]
        previous_row = prepared.iloc[-2] if len(prepared) >= 2 else last_row
        target_timestamp = pd.Timestamp(target_timestamp)
        humidity_trend = float(target_hum - last_row["Hum"])
        forecast_delta = max(
            0.0,
            (target_timestamp - last_row["Timestamp"]).total_seconds() / 3600.0,
        )
        metadata = [{
            "IMEI": str(imei),
            "Temp": float(target_temp),
            "Hum": float(target_hum),
            "HumidityTrend": humidity_trend,
            "LastHum": float(last_row["Hum"]),
            "LastHumidityTrend": float(last_row["Hum"] - previous_row["Hum"]),
            "ForecastDeltaHours": forecast_delta,
        }]

        routing_mode = self.checkpoint.get("routing_mode", "target_humidity")
        if routing_mode == "last_observed_humidity":
            route = np.asarray(
                [[metadata[0]["LastHum"], metadata[0]["LastHumidityTrend"]]],
                dtype=np.float32,
            )
        else:
            route = np.asarray([[target_hum, humidity_trend]], dtype=np.float32)

        model_config = self.checkpoint["model_config"]
        device_tensor = None
        if int(model_config.get("device_embedding_dim", 0)) > 0:
            device_map = self.checkpoint.get("device_to_index", {})
            device_index = device_map.get(str(imei), len(device_map))
            device_tensor = torch.tensor([device_index], dtype=torch.long, device=self.device)

        known_tensor = None
        if int(model_config.get("known_covariate_dim", 0)) > 0:
            known = known_target_covariates(metadata)
            known_tensor = torch.from_numpy(known).float().to(self.device)

        with torch.no_grad():
            output = self.model(
                torch.from_numpy(values[None, :, :]).float().to(self.device),
                torch.from_numpy(route).float().to(self.device),
                device_indices=device_tensor,
                known_covariates=known_tensor,
            )
        scaled = output["quantiles"].squeeze(0).squeeze(0).cpu().numpy()
        y_mean = float(np.asarray(self.checkpoint["y_mean"]).reshape(-1)[0])
        y_scale = float(np.asarray(self.checkpoint["y_scale"]).reshape(-1)[0])
        persistence = float(last_row["Measure"])
        quantiles = scaled * y_scale + y_mean + persistence

        margins = self.checkpoint.get("conformal_margins", {})
        margin = float(margins.get(stage_name(target_hum), margins.get("global", 0.0)))
        quantiles[0] -= margin
        quantiles[2] += margin
        return {
            "P10": float(quantiles[0]),
            "P50": float(quantiles[1]),
            "P90": float(quantiles[2]),
            "Persistence": persistence,
            "HumidityStage": stage_name(target_hum),
            "ForecastDeltaHours": forecast_delta,
        }


class HumidityStageEnsemblePredictor:
    def __init__(self, manifest_path, device=None):
        self.manifest_path = resolve_project_path(
            manifest_path,
            relative_to=Path.cwd(),
        )
        with self.manifest_path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        manifest_dir = self.manifest_path.parent
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.threshold = float(manifest.get("humidity_threshold", 80.0))
        self.base = _LoadedStageModel(
            manifest["base_checkpoint"], self.device, manifest_dir
        )
        high_specs = manifest.get("high_humidity_checkpoints")
        if high_specs:
            raw_weights = np.asarray(
                [float(item["weight"]) for item in high_specs], dtype=np.float64
            )
            if np.any(raw_weights < 0.0) or raw_weights.sum() <= 0.0:
                raise ValueError("High-humidity blend weights must be non-negative")
            normalized = raw_weights / raw_weights.sum()
            self.high_models = [
                (
                    _LoadedStageModel(
                        item["checkpoint"], self.device, manifest_dir
                    ),
                    float(weight),
                )
                for item, weight in zip(high_specs, normalized)
            ]
        else:
            self.high_models = [(
                _LoadedStageModel(
                    manifest["high_humidity_checkpoint"],
                    self.device,
                    manifest_dir,
                ),
                1.0,
            )]
        self.high = self.high_models[0][0]
        self.required_history_records = max(
            self.base.seq_len,
            max(model.seq_len for model, _ in self.high_models) + 12,
        ) + 1

    def _predict_high_blend(
        self, history, imei, location, target_temp, target_hum, target_timestamp
    ):
        predictions = [
            (
                model.predict(
                    history,
                    str(imei),
                    str(location),
                    float(target_temp),
                    float(target_hum),
                    target_timestamp,
                ),
                weight,
            )
            for model, weight in self.high_models
        ]
        if len(predictions) == 1:
            return predictions[0][0]
        result = dict(predictions[0][0])
        for key in ("P10", "P50", "P90", "Persistence", "ForecastDeltaHours"):
            result[key] = float(sum(
                weight * prediction[key]
                for prediction, weight in predictions
            ))
        result["HighExpertBlendWeights"] = [
            {
                "checkpoint": str(model.path),
                "weight": float(weight),
            }
            for model, weight in self.high_models
        ]
        return result

    @staticmethod
    def records_to_frame(records):
        rows = []
        for record in records:
            if len(record) < 4:
                raise ValueError("Each history record must contain Measure, Temp, Hum and Timestamp")
            rows.append({
                "Measure": float(record[0]),
                "Temp": float(record[1]),
                "Hum": float(record[2]),
                "Timestamp": pd.Timestamp(record[3]),
            })
        if not rows:
            raise ValueError("History is empty")
        frame = pd.DataFrame(rows)
        return frame.drop_duplicates("Timestamp", keep="first").sort_values("Timestamp")

    def predict_from_records(
        self,
        imei,
        location,
        records,
        target_temp,
        target_hum,
        target_timestamp,
        actual_measure=None,
    ):
        target_timestamp = pd.Timestamp(target_timestamp)
        history = self.records_to_frame(records)
        history = history[history["Timestamp"] < target_timestamp].copy()
        selected_name = (
            "dynamic_high_expert"
            if float(target_hum) >= self.threshold
            else "clean_elapsed"
        )
        if selected_name == "dynamic_high_expert":
            result = self._predict_high_blend(
                history,
                imei,
                location,
                target_temp,
                target_hum,
                target_timestamp,
            )
        else:
            result = self.base.predict(
                history,
                str(imei),
                str(location),
                float(target_temp),
                float(target_hum),
                target_timestamp,
            )
        result["SelectedModel"] = selected_name
        result["IMEI"] = str(imei)
        result["Timestamp"] = target_timestamp.isoformat()
        if actual_measure is not None:
            result.update(self._assessment(result, float(actual_measure)))
        return result

    @staticmethod
    def _assessment(prediction, actual):
        p10 = prediction["P10"]
        p50 = prediction["P50"]
        p90 = prediction["P90"]
        error = abs(actual - p50)
        inside = p10 <= actual <= p90
        half_width = max((p90 - p10) / 2.0, 0.02)
        if inside:
            insulation = float(np.clip(98.0 - 5.0 * error / half_width, 85.0, 98.0))
            status = "\u7edd\u7f18\u6b63\u5e38"
            confidence = float(np.clip(95.0 - 5.0 * error / half_width, 80.0, 95.0))
        else:
            distance = p10 - actual if actual < p10 else actual - p90
            severity = distance / half_width
            insulation = float(np.clip(80.0 - 15.0 * severity, 50.0, 80.0))
            status = "\u7edd\u7f18\u5f02\u5e38"
            confidence = float(np.clip(80.0 + 10.0 * severity, 80.0, 98.0))
        return {
            "Actual": actual,
            "AbsoluteError": error,
            "IsAnomaly": not inside,
            "StatusDescription": status,
            "Confidence": confidence,
            "InsulationPercentage": insulation,
        }

    @staticmethod
    def legacy_tuple(result):
        return (
            result["P50"],
            result["AbsoluteError"],
            result["IsAnomaly"],
            result["StatusDescription"],
            result["Confidence"],
            result["InsulationPercentage"],
        )


