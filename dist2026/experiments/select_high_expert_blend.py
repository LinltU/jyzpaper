"""Select high-humidity expert blend weights on a real validation tail."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from experiments.evaluate_fmlp_fixed_split import build_all_windows, regression_metrics
from experiments.evaluate_humidity_stage_fmlp import (
    apply_calibration,
    interval_metrics,
    known_target_covariates,
    metadata_device_indices,
    metadata_split_indices,
    predict_quantiles,
    routing_features,
    stage_name,
)
from core.feature_engineering_v2 import prepare_rows_v2
from model.humidity_stage_fmlp import HumidityStageFMLP


def checkpoint_validation_predictions(path, x, metadata, device):
    checkpoint = torch.load(str(path), map_location="cpu")
    model = HumidityStageFMLP(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    persistence = np.asarray([item["Persistence"] for item in metadata])

    x_mean = np.asarray(checkpoint["x_mean"], dtype=np.float32)
    x_scale = np.asarray(checkpoint["x_scale"], dtype=np.float32)
    x_scaled = (x - x_mean) / np.where(x_scale == 0.0, 1.0, x_scale)

    class TargetScaler:
        def __init__(self, mean, scale):
            self.mean_ = np.asarray(mean)
            self.scale_ = np.asarray(scale)

    scaler_y = TargetScaler(checkpoint["y_mean"], checkpoint["y_scale"])
    quantiles, _, _ = predict_quantiles(
        model,
        x_scaled.astype(np.float32),
        routing_features(metadata),
        device,
        scaler_y,
        persistence,
        device_indices=metadata_device_indices(
            metadata, checkpoint["device_to_index"]
        ),
        known_covariates=known_target_covariates(metadata),
    )
    return apply_calibration(
        quantiles, metadata, checkpoint.get("conformal_margins", {})
    )


def simplex_weights(count, step=0.05):
    if count < 2 or count > 3:
        raise ValueError("Only two or three checkpoints are supported")
    units = int(round(1.0 / step))
    if not np.isclose(units * step, 1.0):
        raise ValueError("step must divide one exactly")
    if count == 2:
        for first in range(units + 1):
            yield np.asarray([first, units - first], dtype=np.float64) / units
        return
    for first in range(units + 1):
        for second in range(units - first + 1):
            third = units - first - second
            yield np.asarray(
                [first, second, third], dtype=np.float64
            ) / units


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--training", required=True)
    parser.add_argument("--checkpoint", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--coverage-floor", type=float, default=0.78)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if len(args.checkpoint) not in (2, 3):
        raise ValueError("Two or three --checkpoint values are required")

    paths = [Path(value).resolve() for value in args.checkpoint]
    checkpoints = [torch.load(str(path), map_location="cpu") for path in paths]
    columns = list(checkpoints[0]["feature_columns"])
    seq_len = int(checkpoints[0]["model_config"]["seq_len"])
    if any(list(item["feature_columns"]) != columns for item in checkpoints[1:]):
        raise ValueError("Blend checkpoints use different feature columns")
    if any(int(item["model_config"]["seq_len"]) != seq_len for item in checkpoints[1:]):
        raise ValueError("Blend checkpoints use different sequence lengths")
    if "Battery" in columns:
        raise ValueError("Battery must not enter the blend models")

    preprocessing = checkpoints[0].get("preprocessing", {})
    raw = pd.read_csv(args.training, dtype={"IMEI": str})
    prepared = prepare_rows_v2(
        raw,
        humidity_fill_value=preprocessing.get("humidity_fill_value"),
        median_delta_hours=preprocessing.get("median_delta_hours"),
        device_start_times=preprocessing.get("device_start_times") or None,
    )
    all_x, all_y, all_metadata = build_all_windows(prepared, columns, seq_len)
    _, validation_indices = metadata_split_indices(all_metadata, 0.15)
    x_validation = all_x[validation_indices]
    y_validation = all_y[validation_indices]
    validation_metadata = [all_metadata[index] for index in validation_indices]
    high_mask = np.asarray([
        stage_name(item["Hum"]) == "very_humid"
        for item in validation_metadata
    ])
    if high_mask.sum() == 0:
        raise ValueError("The validation tail has no very-humid samples")

    device = torch.device(args.device)
    predictions = [
        checkpoint_validation_predictions(
            path, x_validation, validation_metadata, device
        )
        for path in paths
    ]
    rows = []
    feasible = []
    for weights in simplex_weights(len(predictions), step=0.05):
        blended = sum(
            weight * prediction
            for weight, prediction in zip(weights, predictions)
        )
        point = regression_metrics(y_validation[high_mask], blended[high_mask, 1])
        interval = interval_metrics(y_validation[high_mask], blended[high_mask])
        row = {
            "weights": weights.tolist(),
            "rmse": point["rmse"],
            "mae": point["mae"],
            "coverage_p10_p90": interval["coverage_p10_p90"],
            "mean_interval_width": interval["mean_interval_width"],
            "selection_score": point["rmse"] + 0.25 * point["mae"],
        }
        rows.append(row)
        if row["coverage_p10_p90"] >= args.coverage_floor:
            feasible.append(row)
    candidates = feasible or rows
    selected = min(candidates, key=lambda item: item["selection_score"])
    output = {
        "protocol": {
            "training": str(Path(args.training).resolve()),
            "selection_split": "real per-device chronological validation tail",
            "validation_sequences": int(len(y_validation)),
            "very_humid_validation_sequences": int(high_mask.sum()),
            "test_labels_used": False,
            "battery_in_model": False,
            "coverage_floor": args.coverage_floor,
            "objective": "very_humid RMSE + 0.25 * MAE",
        },
        "checkpoints": [str(path) for path in paths],
        "selected": selected,
        "grid": rows,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(output["selected"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()


