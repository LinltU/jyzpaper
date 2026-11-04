"""Train and evaluate the four-stage humidity FMLP on fixed CSV splits."""

import argparse
import json
import math
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
if __package__ in (None, ""):
    import sys
    sys.path.insert(0, str(PROJECT_DIR))

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from experiments.evaluate_fmlp_fixed_split import (
    build_all_windows,
    build_test_windows,
    feature_columns,
    prepare_rows,
    regression_metrics,
    set_seed,
    split_train_validation,
)
from core.feature_engineering_v2 import (
    add_device_one_hot,
    feature_columns_clean,
    feature_columns_v2,
    feature_diagnostics,
    prepare_rows_v2,
)
from model.humidity_stage_fmlp import HumidityStageFMLP


STAGE_NAMES = ("dry", "transition", "humid", "very_humid")
QUANTILE_LEVELS = torch.tensor([0.1, 0.5, 0.9], dtype=torch.float32)
SCRIPT_DIR = Path(__file__).resolve().parent
DIST_DIR = SCRIPT_DIR.parent
DATA_DIR = DIST_DIR / "data" / "raw"


def stage_name(humidity):
    if humidity < 50.0:
        return "dry"
    if humidity < 65.0:
        return "transition"
    if humidity < 80.0:
        return "humid"
    return "very_humid"


def metadata_split_indices(metadata, validation_ratio):
    train_indices = []
    validation_indices = []
    devices = sorted({item["IMEI"] for item in metadata})
    for device_id in devices:
        indices = [i for i, item in enumerate(metadata) if item["IMEI"] == device_id]
        validation_count = max(1, int(math.ceil(len(indices) * validation_ratio)))
        train_indices.extend(indices[:-validation_count])
        validation_indices.extend(indices[-validation_count:])
    return train_indices, validation_indices


def routing_features(metadata, causal=False):
    humidity_key = "LastHum" if causal else "Hum"
    trend_key = "LastHumidityTrend" if causal else "HumidityTrend"
    return np.asarray(
        [[item[humidity_key], item[trend_key]] for item in metadata],
        dtype=np.float32,
    )


def metadata_device_indices(metadata, device_to_index):
    unknown_index = len(device_to_index)
    return np.asarray(
        [device_to_index.get(item["IMEI"], unknown_index) for item in metadata],
        dtype=np.int64,
    )


def known_target_covariates(metadata):
    return np.asarray(
        [
            [
                (item["Temp"] - 10.0) / 10.0,
                item["Hum"] / 100.0,
                np.clip(item["HumidityTrend"], -20.0, 20.0) / 20.0,
                np.clip(item.get("ForecastDeltaHours", 12.0), 0.0, 72.0) / 24.0,
            ]
            for item in metadata
        ],
        dtype=np.float32,
    )


def stage_sample_weights(metadata):
    stage_weights = {
        "dry": 1.1,
        "transition": 1.0,
        "humid": 1.2,
        "very_humid": 2.4,
    }
    return np.asarray(
        [stage_weights[stage_name(item["Hum"])] for item in metadata],
        dtype=np.float32,
    )


def pinball_per_sample(predicted_quantiles, target):
    levels = QUANTILE_LEVELS.to(predicted_quantiles.device)
    target = target[:, None, None]
    error = target - predicted_quantiles
    loss = torch.maximum(levels * error, (levels - 1.0) * error)
    return loss.mean(dim=(1, 2))


def weighted_quantile_loss(output, target, weights):
    sample_loss = pinball_per_sample(output["quantiles"], target)
    pinball = (sample_loss * weights).sum() / weights.sum().clamp_min(1e-8)
    expert_penalty = output["expert_residuals"].pow(2).mean()
    routing_penalty = (
        output["gates"]
        * (
            torch.log(output["gates"].clamp_min(1e-8))
            - torch.log(output["stage_priors"].clamp_min(1e-8))
        )
    ).sum(dim=1).mean()
    return pinball + 1e-4 * expert_penalty + 1e-2 * routing_penalty


def train_model(model, train_loader, validation_loader, device, max_epochs=180):
    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=2e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=6, min_lr=1e-6
    )
    best_state = None
    best_loss = float("inf")
    stale_epochs = 0
    history = []

    for epoch in range(1, max_epochs + 1):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            batch_x, batch_y, batch_route, batch_weight = batch[:4]
            offset = 4
            batch_device = None
            batch_known = None
            if model.device_embedding is not None:
                batch_device = batch[offset].to(device)
                offset += 1
            if model.known_covariate_dim > 0:
                batch_known = batch[offset].to(device)
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            batch_route = batch_route.to(device)
            batch_weight = batch_weight.to(device)
            optimizer.zero_grad()
            output = model(
                batch_x,
                batch_route,
                device_indices=batch_device,
                known_covariates=batch_known,
            )
            loss = weighted_quantile_loss(output, batch_y, batch_weight)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        validation_loss = 0.0
        with torch.no_grad():
            for batch in validation_loader:
                batch_x, batch_y, batch_route, batch_weight = batch[:4]
                offset = 4
                batch_device = None
                batch_known = None
                if model.device_embedding is not None:
                    batch_device = batch[offset].to(device)
                    offset += 1
                if model.known_covariate_dim > 0:
                    batch_known = batch[offset].to(device)
                output = model(
                    batch_x.to(device),
                    batch_route.to(device),
                    device_indices=batch_device,
                    known_covariates=batch_known,
                )
                loss = weighted_quantile_loss(
                    output,
                    batch_y.to(device),
                    batch_weight.to(device),
                )
                validation_loss += loss.item()

        train_loss /= max(len(train_loader), 1)
        validation_loss /= max(len(validation_loader), 1)
        scheduler.step(validation_loss)
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "validation_loss": validation_loss,
        })

        if validation_loss < best_loss - 1e-6:
            best_loss = validation_loss
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            stale_epochs = 0
        else:
            stale_epochs += 1

        if epoch == 1 or epoch % 10 == 0:
            print(
                "Epoch {:3d}: train={:.6f} val={:.6f} best={:.6f}".format(
                    epoch, train_loss, validation_loss, best_loss
                ),
                flush=True,
            )
        if stale_epochs >= 24:
            print("Early stopping at epoch {}".format(epoch), flush=True)
            break

    if best_state is None:
        raise RuntimeError("Training did not produce a valid model")
    model.load_state_dict(best_state)
    return model, best_loss, history


def predict_quantiles(
    model,
    x,
    routes,
    device,
    scaler_y,
    persistence,
    device_indices=None,
    known_covariates=None,
):
    model.eval()
    device_tensor = (
        torch.from_numpy(device_indices).long().to(device)
        if device_indices is not None
        else None
    )
    known_tensor = (
        torch.from_numpy(known_covariates).float().to(device)
        if known_covariates is not None
        else None
    )
    with torch.no_grad():
        output = model(
            torch.from_numpy(x).float().to(device),
            torch.from_numpy(routes).float().to(device),
            device_indices=device_tensor,
            known_covariates=known_tensor,
        )
    scaled_quantiles = output["quantiles"].squeeze(1).cpu().numpy()
    residual_quantiles = (
        scaled_quantiles * scaler_y.scale_[0] + scaler_y.mean_[0]
    )
    quantiles = residual_quantiles + persistence[:, None]
    return (
        quantiles,
        output["gates"].cpu().numpy(),
        output["stage_priors"].cpu().numpy(),
    )


def conformal_quantile(scores, coverage):
    if len(scores) == 0:
        return 0.0
    quantile_level = min(1.0, math.ceil((len(scores) + 1) * coverage) / len(scores))
    return float(np.quantile(scores, quantile_level, method="higher"))


def calibration_margins(actual, quantiles, metadata, coverage=0.8):
    scores = np.maximum.reduce([
        quantiles[:, 0] - actual,
        actual - quantiles[:, 2],
        np.zeros_like(actual),
    ])
    global_margin = conformal_quantile(scores, coverage)
    margins = {"global": global_margin}
    for name in STAGE_NAMES:
        mask = np.asarray([stage_name(item["Hum"]) == name for item in metadata])
        stage_coverage = 0.9 if name == "very_humid" else coverage
        margins[name] = (
            conformal_quantile(scores[mask], stage_coverage)
            if mask.sum() >= 20
            else global_margin
        )
    return margins


def apply_calibration(quantiles, metadata, margins):
    calibrated = quantiles.copy()
    for index, item in enumerate(metadata):
        margin = margins[stage_name(item["Hum"])]
        calibrated[index, 0] -= margin
        calibrated[index, 2] += margin
    return calibrated


def interval_metrics(actual, quantiles):
    inside = (actual >= quantiles[:, 0]) & (actual <= quantiles[:, 2])
    return {
        "coverage_p10_p90": float(inside.mean()),
        "mean_interval_width": float(np.mean(quantiles[:, 2] - quantiles[:, 0])),
        "median_interval_width": float(np.median(quantiles[:, 2] - quantiles[:, 0])),
    }


def stage_metrics(actual, quantiles, persistence, gates, metadata):
    output = {}
    for name in STAGE_NAMES:
        mask = np.asarray([stage_name(item["Hum"]) == name for item in metadata])
        if not mask.any():
            continue
        median = quantiles[mask, 1]
        output[name] = {
            "fmlp": regression_metrics(actual[mask], median),
            "persistence": regression_metrics(actual[mask], persistence[mask]),
            "interval": interval_metrics(actual[mask], quantiles[mask]),
            "mean_gate_weights": {
                stage: float(gates[mask, stage_index].mean())
                for stage_index, stage in enumerate(STAGE_NAMES)
            },
        }
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--training", default=str(DATA_DIR / "training.csv"))
    parser.add_argument("--test", default=str(DATA_DIR / "test.csv"))
    parser.add_argument("--output-dir")
    parser.add_argument("--seq-len", type=int, default=24)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-epochs", type=int, default=180)
    parser.add_argument(
        "--feature-set",
        choices=("v1", "clean", "v2"),
        default="v1",
    )
    parser.add_argument(
        "--v2-profile",
        choices=("compact", "dynamic", "dynamic_elapsed"),
        default="dynamic",
    )
    parser.add_argument(
        "--routing-mode",
        choices=("auto", "target", "causal"),
        default="auto",
    )
    parser.add_argument("--device-embedding-dim", type=int, default=0)
    parser.add_argument("--use-target-covariates", action="store_true")
    parser.add_argument(
        "--battery-mode",
        choices=("auto", "include", "exclude"),
        default="auto",
    )
    args = parser.parse_args()

    set_seed(args.seed)
    default_outputs = {
        "v1": "humidity_stage_fmlp_evaluation",
        "clean": "humidity_stage_fmlp_clean_evaluation",
        "v2": "humidity_stage_fmlp_v2_evaluation",
    }
    default_output = default_outputs[args.feature_set]
    output_dir = Path(args.output_dir or (SCRIPT_DIR / default_output))
    if not output_dir.is_absolute():
        output_dir = SCRIPT_DIR / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    training_raw = pd.read_csv(args.training)
    test_raw = pd.read_csv(args.test)
    device_ids = sorted(training_raw["IMEI"].astype(str).unique().tolist())
    training_raw["_split"] = "train"
    test_raw["_split"] = "test"

    causal_routing = (
        args.feature_set == "v2"
        if args.routing_mode == "auto"
        else args.routing_mode == "causal"
    )
    include_battery = (
        args.feature_set == "v1"
        if args.battery_mode == "auto"
        else args.battery_mode == "include"
    )
    humidity_fill_value = None
    median_delta_hours = None
    device_start_times = {}
    if args.feature_set in ("clean", "v2"):
        valid_humidity = training_raw["Hum"].where(
            np.isfinite(training_raw["Hum"])
            & (training_raw["Hum"] > 0.0)
            & (training_raw["Hum"] <= 100.0)
        )
        humidity_fill_value = float(valid_humidity.median())
        timestamp_rows = training_raw[["IMEI", "Timestamp"]].copy()
        timestamp_rows["IMEI"] = timestamp_rows["IMEI"].astype(str)
        timestamp_rows["Timestamp"] = pd.to_datetime(timestamp_rows["Timestamp"])
        timestamp_rows = timestamp_rows.sort_values(["IMEI", "Timestamp"])
        device_start_times = (
            timestamp_rows.groupby("IMEI")["Timestamp"].min().to_dict()
        )
        delta_hours = (
            timestamp_rows.groupby("IMEI")["Timestamp"].diff().dt.total_seconds()
            / 3600.0
        )
        median_delta_hours = float(delta_hours[delta_hours > 0.0].median())
        training = prepare_rows_v2(
            training_raw,
            humidity_fill_value=humidity_fill_value,
            median_delta_hours=median_delta_hours,
            device_start_times=device_start_times,
        )
        combined = prepare_rows_v2(
            pd.concat([training_raw, test_raw], ignore_index=True),
            humidity_fill_value=humidity_fill_value,
            median_delta_hours=median_delta_hours,
            device_start_times=device_start_times,
        )
        if args.feature_set == "clean":
            training = add_device_one_hot(training, device_ids)
            combined = add_device_one_hot(combined, device_ids)
            columns = feature_columns_clean(device_ids)
        else:
            columns = feature_columns_v2(args.v2_profile)
        diagnostics = feature_diagnostics(training, columns)
    else:
        training = prepare_rows(training_raw, device_ids)
        combined = prepare_rows(
            pd.concat([training_raw, test_raw], ignore_index=True),
            device_ids,
        )
        columns = feature_columns(device_ids)
        diagnostics = {
            "row_count": int(len(training)),
            "feature_count": int(len(columns)),
            "feature_columns": columns,
            "constant_features": [
                column for column in columns
                if training[column].nunique(dropna=False) <= 1
            ],
        }

    if include_battery and "Battery" not in columns:
        if "Battery" not in training.columns or "Battery" not in combined.columns:
            raise ValueError("Battery was requested but is missing from the input data")
        columns = columns[:3] + ["Battery"] + columns[3:]
    if not include_battery and "Battery" in columns:
        columns = [column for column in columns if column != "Battery"]
    diagnostics["feature_columns"] = columns
    diagnostics["feature_count"] = len(columns)
    diagnostics["battery_in_model"] = include_battery

    all_x, all_y, all_metadata = build_all_windows(training, columns, args.seq_len)
    x_train, y_train, x_validation, y_validation = split_train_validation(
        all_x, all_y, all_metadata, validation_ratio=0.15
    )
    train_indices, validation_indices = metadata_split_indices(all_metadata, 0.15)
    train_metadata = [all_metadata[index] for index in train_indices]
    validation_metadata = [all_metadata[index] for index in validation_indices]

    x_test, y_test, test_metadata = build_test_windows(combined, columns, args.seq_len)
    if min(len(x_train), len(x_validation), len(x_test)) == 0:
        raise ValueError("Not enough rows to build all sequence splits")

    from sklearn.preprocessing import StandardScaler

    scaler_x = StandardScaler()
    scaler_x.fit(x_train.reshape(-1, len(columns)))
    x_train_scaled = scaler_x.transform(
        x_train.reshape(-1, len(columns))
    ).reshape(x_train.shape)
    x_validation_scaled = scaler_x.transform(
        x_validation.reshape(-1, len(columns))
    ).reshape(x_validation.shape)
    x_test_scaled = scaler_x.transform(
        x_test.reshape(-1, len(columns))
    ).reshape(x_test.shape)

    train_persistence = np.asarray([item["Persistence"] for item in train_metadata])
    validation_persistence = np.asarray([
        item["Persistence"] for item in validation_metadata
    ])
    test_persistence = np.asarray([item["Persistence"] for item in test_metadata])

    scaler_y = StandardScaler()
    y_train_scaled = scaler_y.fit_transform(
        (y_train - train_persistence).reshape(-1, 1)
    ).reshape(-1)
    y_validation_scaled = scaler_y.transform(
        (y_validation - validation_persistence).reshape(-1, 1)
    ).reshape(-1)

    train_routes = routing_features(train_metadata, causal=causal_routing)
    validation_routes = routing_features(validation_metadata, causal=causal_routing)
    test_routes = routing_features(test_metadata, causal=causal_routing)

    if args.device_embedding_dim < 0:
        raise ValueError("device-embedding-dim must be non-negative")
    device_to_index = {device_id: index for index, device_id in enumerate(device_ids)}
    train_device_indices = metadata_device_indices(train_metadata, device_to_index)
    validation_device_indices = metadata_device_indices(
        validation_metadata,
        device_to_index,
    )
    test_device_indices = metadata_device_indices(test_metadata, device_to_index)
    train_known = known_target_covariates(train_metadata)
    validation_known = known_target_covariates(validation_metadata)
    test_known = known_target_covariates(test_metadata)

    train_tensors = [
        torch.from_numpy(x_train_scaled).float(),
        torch.from_numpy(y_train_scaled).float(),
        torch.from_numpy(train_routes).float(),
        torch.from_numpy(stage_sample_weights(train_metadata)).float(),
    ]
    validation_tensors = [
        torch.from_numpy(x_validation_scaled).float(),
        torch.from_numpy(y_validation_scaled).float(),
        torch.from_numpy(validation_routes).float(),
        torch.from_numpy(stage_sample_weights(validation_metadata)).float(),
    ]
    if args.device_embedding_dim > 0:
        train_tensors.append(torch.from_numpy(train_device_indices).long())
        validation_tensors.append(torch.from_numpy(validation_device_indices).long())
    if args.use_target_covariates:
        train_tensors.append(torch.from_numpy(train_known).float())
        validation_tensors.append(torch.from_numpy(validation_known).float())

    train_loader = DataLoader(
        TensorDataset(*train_tensors),
        batch_size=32,
        shuffle=True,
    )
    validation_loader = DataLoader(
        TensorDataset(*validation_tensors),
        batch_size=64,
        shuffle=False,
    )

    model_config = {
        "seq_len": args.seq_len,
        "pred_len": 1,
        "enc_in": len(columns),
        "hidden_dim": 64,
        "expert_hidden_dim": 32,
        "dropout": 0.15,
        "kernel_size": 7,
        "target_index": columns.index("Measure"),
        "humidity_index": columns.index("Hum"),
        "routing_temperature": 2.5,
        "learned_gate_scale": 0.25,
        "num_devices": len(device_ids) if args.device_embedding_dim > 0 else 0,
        "device_embedding_dim": args.device_embedding_dim,
        "known_covariate_dim": 4 if args.use_target_covariates else 0,
    }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HumidityStageFMLP(**model_config).to(device)
    print(
        "Sequences: train={} val={} test={} features={} device={}".format(
            len(x_train), len(x_validation), len(x_test), len(columns), device
        ),
        flush=True,
    )
    model, best_validation_loss, history = train_model(
        model,
        train_loader,
        validation_loader,
        device,
        max_epochs=args.max_epochs,
    )

    validation_quantiles, _, _ = predict_quantiles(
        model,
        x_validation_scaled,
        validation_routes,
        device,
        scaler_y,
        validation_persistence,
        device_indices=(
            validation_device_indices if args.device_embedding_dim > 0 else None
        ),
        known_covariates=validation_known if args.use_target_covariates else None,
    )
    margins = calibration_margins(
        y_validation,
        validation_quantiles,
        validation_metadata,
        coverage=0.8,
    )

    test_quantiles, test_gates, test_priors = predict_quantiles(
        model,
        x_test_scaled,
        test_routes,
        device,
        scaler_y,
        test_persistence,
        device_indices=test_device_indices if args.device_embedding_dim > 0 else None,
        known_covariates=test_known if args.use_target_covariates else None,
    )
    test_quantiles = apply_calibration(test_quantiles, test_metadata, margins)
    median_prediction = test_quantiles[:, 1]

    metrics = {
        "protocol": {
            "training_file": str(Path(args.training).resolve()),
            "test_file": str(Path(args.test).resolve()),
            "seq_len": args.seq_len,
            "pred_len": 1,
            "target_mode": "residual_quantiles_from_last_measure",
            "feature_set": args.feature_set,
            "v2_profile": args.v2_profile if args.feature_set == "v2" else None,
            "routing_mode": "last_observed_humidity" if causal_routing else "target_humidity",
            "device_embedding_dim": args.device_embedding_dim,
            "known_target_covariates": (
                ["Temp", "Hum", "HumidityTrend", "ForecastDeltaHours"]
                if args.use_target_covariates else []
            ),
            "battery_in_model": include_battery,
            "max_epochs": args.max_epochs,
            "humidity_stages": {
                "dry": "Hum < 50",
                "transition": "50 <= Hum < 65",
                "humid": "65 <= Hum < 80",
                "very_humid": "Hum >= 80",
            },
            "feature_count": len(columns),
            "train_sequences": int(len(x_train)),
            "validation_sequences": int(len(x_validation)),
            "test_sequences": int(len(x_test)),
            "best_validation_loss_scaled": float(best_validation_loss),
            "conformal_margins": margins,
            "interval_coverage_targets": {
                "dry": 0.8,
                "transition": 0.8,
                "humid": 0.8,
                "very_humid": 0.9,
            },
        },
        "overall": {
            "fmlp": regression_metrics(y_test, median_prediction),
            "persistence": regression_metrics(y_test, test_persistence),
            "interval": interval_metrics(y_test, test_quantiles),
        },
        "by_humidity_stage": stage_metrics(
            y_test,
            test_quantiles,
            test_persistence,
            test_gates,
            test_metadata,
        ),
    }

    with (output_dir / "predictions.jsonl").open("w", encoding="utf-8") as handle:
        for index, item in enumerate(test_metadata):
            row = dict(item)
            row.update({
                "HumidityStage": stage_name(item["Hum"]),
                "Actual": float(y_test[index]),
                "P10": float(test_quantiles[index, 0]),
                "P50": float(test_quantiles[index, 1]),
                "P90": float(test_quantiles[index, 2]),
                "Persistence": float(test_persistence[index]),
                "GateWeights": {
                    stage: float(test_gates[index, stage_index])
                    for stage_index, stage in enumerate(STAGE_NAMES)
                },
                "StagePriors": {
                    stage: float(test_priors[index, stage_index])
                    for stage_index, stage in enumerate(STAGE_NAMES)
                },
            })
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, ensure_ascii=False, indent=2)
    with (output_dir / "training_history.json").open("w", encoding="utf-8") as handle:
        json.dump(history, handle, ensure_ascii=False, indent=2)
    with (output_dir / "feature_diagnostics.json").open("w", encoding="utf-8") as handle:
        json.dump(diagnostics, handle, ensure_ascii=False, indent=2)

    torch.save(
        {
            "state_dict": model.state_dict(),
            "model_config": model_config,
            "feature_columns": columns,
            "x_mean": scaler_x.mean_,
            "x_scale": scaler_x.scale_,
            "y_mean": scaler_y.mean_,
            "y_scale": scaler_y.scale_,
            "conformal_margins": margins,
            "stage_names": STAGE_NAMES,
            "quantiles": HumidityStageFMLP.QUANTILES,
            "feature_set": args.feature_set,
            "v2_profile": args.v2_profile if args.feature_set == "v2" else None,
            "routing_mode": "last_observed_humidity" if causal_routing else "target_humidity",
            "battery_in_model": include_battery,
            "device_to_index": device_to_index if args.device_embedding_dim > 0 else {},
            "known_covariates": (
                ["Temp", "Hum", "HumidityTrend", "ForecastDeltaHours"]
                if args.use_target_covariates else []
            ),
            "preprocessing": {
                "humidity_fill_value": humidity_fill_value,
                "median_delta_hours": median_delta_hours,
                "device_start_times": {
                    str(device_id): pd.Timestamp(timestamp).isoformat()
                    for device_id, timestamp in device_start_times.items()
                },
            },
            "seed": args.seed,
        },
        output_dir / "humidity_stage_fmlp.pth",
    )

    print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()


