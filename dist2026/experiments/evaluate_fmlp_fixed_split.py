"""Evaluate the optimized FMLP on explicit training/test CSV files.

The supplied files overlap in time and the test set contains too few
consecutive rows per device for a six-step horizon. This runner therefore uses
a rolling one-step protocol: the previous ``seq_len`` observations of a device
predict the next Measure value.
"""

import argparse
import json
import math
import os
import random
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
if __package__ in (None, ""):
    import sys
    sys.path.insert(0, str(PROJECT_DIR))

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from model.fmlp_ltsf import FMLP_LTSF


LOCATION_FEATURES = [
    "Loc_Left",
    "Loc_Right",
    "Loc_Upper",
    "Loc_Lower",
    "Loc_Small",
    "Loc_Big",
    "Loc_StrNum",
    "Loc_220kV",
    "Loc_PlateCount",
]
SCRIPT_DIR = Path(__file__).resolve().parent
DIST_DIR = SCRIPT_DIR.parent
DATA_DIR = DIST_DIR / "data" / "raw"


def absolute_humidity(temp, hum):
    saturation = 6.112 * np.exp(17.67 * temp / (temp + 243.5))
    vapor_pressure = hum / 100.0 * saturation
    return 216.7 * vapor_pressure / (temp + 273.15)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def prepare_rows(df, device_ids):
    required = {"IMEI", "Timestamp", "Measure", "Temp", "Hum", "Battery"}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(missing))

    result = df.copy()
    result["IMEI"] = result["IMEI"].astype(str)
    result["Timestamp"] = pd.to_datetime(result["Timestamp"], errors="raise")
    result = result.sort_values(["IMEI", "Timestamp"]).copy()

    result["AH"] = absolute_humidity(result["Temp"], result["Hum"])
    delta_hours = result.groupby("IMEI")["Timestamp"].diff().dt.total_seconds() / 3600.0
    median_delta = delta_hours[delta_hours > 0].median()
    if not np.isfinite(median_delta):
        median_delta = 12.0
    result["DeltaHours"] = delta_hours.fillna(median_delta).clip(lower=0.0, upper=72.0)

    hour = result["Timestamp"].dt.hour + result["Timestamp"].dt.minute / 60.0
    day_of_week = result["Timestamp"].dt.dayofweek
    month = result["Timestamp"].dt.month
    result["Hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    result["Hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
    result["DayOfWeek_sin"] = np.sin(2 * np.pi * day_of_week / 7.0)
    result["DayOfWeek_cos"] = np.cos(2 * np.pi * day_of_week / 7.0)
    result["Month_sin"] = np.sin(2 * np.pi * month / 12.0)
    result["Month_cos"] = np.cos(2 * np.pi * month / 12.0)

    for column in LOCATION_FEATURES:
        if column not in result.columns:
            result[column] = 0.0

    for device_id in device_ids:
        result["Device_" + device_id] = (result["IMEI"] == device_id).astype(float)

    numeric_columns = result.select_dtypes(include=[np.number]).columns
    result[numeric_columns] = result[numeric_columns].replace([np.inf, -np.inf], np.nan)
    if result[numeric_columns].isna().any().any():
        bad_columns = result[numeric_columns].columns[result[numeric_columns].isna().any()].tolist()
        raise ValueError("Non-finite values after feature preparation: " + ", ".join(bad_columns))
    return result


def feature_columns(device_ids):
    return [
        "Measure",
        "Temp",
        "Hum",
        "Battery",
        "AH",
        "DeltaHours",
        "Hour_sin",
        "Hour_cos",
        "DayOfWeek_sin",
        "DayOfWeek_cos",
        "Month_sin",
        "Month_cos",
    ] + LOCATION_FEATURES + ["Device_" + device_id for device_id in device_ids]


def build_all_windows(df, columns, seq_len):
    windows = []
    targets = []
    metadata = []

    for imei, group in df.groupby("IMEI", sort=False):
        group = group.sort_values("Timestamp")
        values = group[columns].to_numpy(dtype=np.float32)
        measures = group["Measure"].to_numpy(dtype=np.float32)
        for target_pos in range(seq_len, len(group)):
            row = group.iloc[target_pos]
            last_row = group.iloc[target_pos - 1]
            previous_row = group.iloc[target_pos - 2] if target_pos >= 2 else last_row
            windows.append(values[target_pos - seq_len:target_pos])
            targets.append(measures[target_pos])
            metadata.append({
                "IMEI": str(imei),
                "Timestamp": row["Timestamp"].isoformat(),
                "Temp": float(row["Temp"]),
                "Hum": float(row["Hum"]),
                "HumidityTrend": float(row["Hum"] - group.iloc[target_pos - 1]["Hum"]),
                "LastHum": float(last_row["Hum"]),
                "LastHumidityTrend": float(last_row["Hum"] - previous_row["Hum"]),
                "ForecastDeltaHours": float(row.get("DeltaHours", 0.0)),
                "Persistence": float(measures[target_pos - 1]),
            })
    return np.asarray(windows), np.asarray(targets), metadata


def build_test_windows(combined_df, columns, seq_len):
    windows = []
    targets = []
    metadata = []

    for imei, group in combined_df.groupby("IMEI", sort=False):
        group = group.sort_values("Timestamp")
        values = group[columns].to_numpy(dtype=np.float32)
        measures = group["Measure"].to_numpy(dtype=np.float32)
        split_values = group["_split"].to_numpy()
        for target_pos in range(seq_len, len(group)):
            if split_values[target_pos] != "test":
                continue
            row = group.iloc[target_pos]
            last_row = group.iloc[target_pos - 1]
            previous_row = group.iloc[target_pos - 2] if target_pos >= 2 else last_row
            windows.append(values[target_pos - seq_len:target_pos])
            targets.append(measures[target_pos])
            metadata.append({
                "IMEI": str(imei),
                "Timestamp": row["Timestamp"].isoformat(),
                "Temp": float(row["Temp"]),
                "Hum": float(row["Hum"]),
                "HumidityTrend": float(row["Hum"] - group.iloc[target_pos - 1]["Hum"]),
                "LastHum": float(last_row["Hum"]),
                "LastHumidityTrend": float(last_row["Hum"] - previous_row["Hum"]),
                "ForecastDeltaHours": float(row.get("DeltaHours", 0.0)),
                "Persistence": float(measures[target_pos - 1]),
            })
    return np.asarray(windows), np.asarray(targets), metadata


def split_train_validation(windows, targets, metadata, validation_ratio):
    train_indices = []
    validation_indices = []
    devices = sorted({item["IMEI"] for item in metadata})

    for device_id in devices:
        indices = [i for i, item in enumerate(metadata) if item["IMEI"] == device_id]
        validation_count = max(1, int(math.ceil(len(indices) * validation_ratio)))
        train_indices.extend(indices[:-validation_count])
        validation_indices.extend(indices[-validation_count:])

    return (
        windows[train_indices],
        targets[train_indices],
        windows[validation_indices],
        targets[validation_indices],
    )


def sample_weights_from_humidity(metadata):
    weights = []
    for item in metadata:
        hum = item["Hum"]
        if hum >= 85:
            weights.append(2.0)
        elif hum >= 75:
            weights.append(1.5)
        else:
            weights.append(1.0)
    return np.asarray(weights, dtype=np.float32)


def regression_metrics(actual, prediction):
    actual = np.asarray(actual)
    prediction = np.asarray(prediction)
    absolute_error = np.abs(actual - prediction)
    denominator = np.maximum(np.abs(actual), 1e-4)
    smape_denominator = np.maximum(np.abs(actual) + np.abs(prediction), 1e-4)
    return {
        "count": int(len(actual)),
        "mse": float(mean_squared_error(actual, prediction)),
        "rmse": float(np.sqrt(mean_squared_error(actual, prediction))),
        "mae": float(mean_absolute_error(actual, prediction)),
        "r2": float(r2_score(actual, prediction)),
        "mape_percent": float(np.mean(absolute_error / denominator) * 100.0),
        "smape_percent": float(np.mean(2.0 * absolute_error / smape_denominator) * 100.0),
        "normalized_mae": float(mean_absolute_error(actual, prediction) / np.mean(np.abs(actual))),
        "median_absolute_error": float(np.median(absolute_error)),
    }


def train_model(
    x_train,
    y_train,
    x_val,
    y_val,
    train_weights,
    val_weights,
    config,
    device,
):
    model = FMLP_LTSF(**config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-6
    )

    train_loader = DataLoader(
        TensorDataset(
            torch.from_numpy(x_train).float(),
            torch.from_numpy(y_train).float(),
            torch.from_numpy(train_weights).float(),
        ),
        batch_size=32,
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(
            torch.from_numpy(x_val).float(),
            torch.from_numpy(y_val).float(),
            torch.from_numpy(val_weights).float(),
        ),
        batch_size=64,
        shuffle=False,
    )

    best_state = None
    best_loss = float("inf")
    patience = 20
    stale_epochs = 0
    history = []

    for epoch in range(1, 151):
        model.train()
        train_loss = 0.0
        for batch_x, batch_y, batch_w in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            batch_w = batch_w.to(device)
            optimizer.zero_grad()
            prediction = model(batch_x).squeeze(-1).squeeze(-1)
            point_loss = F.smooth_l1_loss(prediction, batch_y, reduction="none")
            loss = (point_loss * batch_w).sum() / batch_w.sum().clamp_min(1e-8)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y, batch_w in val_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                batch_w = batch_w.to(device)
                prediction = model(batch_x).squeeze(-1).squeeze(-1)
                point_loss = F.smooth_l1_loss(prediction, batch_y, reduction="none")
                loss = (point_loss * batch_w).sum() / batch_w.sum().clamp_min(1e-8)
                val_loss += loss.item()

        train_loss /= max(len(train_loader), 1)
        val_loss /= max(len(val_loader), 1)
        scheduler.step(val_loss)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

        if val_loss < best_loss - 1e-6:
            best_loss = val_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1

        if epoch == 1 or epoch % 10 == 0:
            print(
                "Epoch {:3d}: train={:.6f} val={:.6f} best={:.6f}".format(
                    epoch, train_loss, val_loss, best_loss
                ),
                flush=True,
            )
        if stale_epochs >= patience:
            print("Early stopping at epoch {}".format(epoch), flush=True)
            break

    if best_state is None:
        raise RuntimeError("Training did not produce a valid model state")
    model.load_state_dict(best_state)
    return model, best_loss, history


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--training", default=str(DATA_DIR / "training.csv"))
    parser.add_argument("--test", default=str(DATA_DIR / "test.csv"))
    parser.add_argument("--output-dir", default=str(SCRIPT_DIR / "fmlp_evaluation"))
    parser.add_argument("--seq-len", type=int, default=24)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = SCRIPT_DIR / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    training_raw = pd.read_csv(args.training)
    test_raw = pd.read_csv(args.test)
    device_ids = sorted(training_raw["IMEI"].astype(str).unique().tolist())
    unknown_devices = sorted(set(test_raw["IMEI"].astype(str)) - set(device_ids))
    if unknown_devices:
        raise ValueError("Test contains unseen devices: " + ", ".join(unknown_devices))

    training_raw["_split"] = "train"
    test_raw["_split"] = "test"
    training = prepare_rows(training_raw, device_ids)
    combined = prepare_rows(pd.concat([training_raw, test_raw], ignore_index=True), device_ids)
    columns = feature_columns(device_ids)

    all_train_x, all_train_y, all_train_meta = build_all_windows(
        training, columns, args.seq_len
    )
    x_train, y_train, x_val, y_val = split_train_validation(
        all_train_x, all_train_y, all_train_meta, validation_ratio=0.15
    )
    train_count = len(x_train)
    train_meta = all_train_meta[:0]
    validation_meta = all_train_meta[:0]
    train_indices = []
    validation_indices = []
    for device_id in device_ids:
        indices = [i for i, item in enumerate(all_train_meta) if item["IMEI"] == device_id]
        validation_count = max(1, int(math.ceil(len(indices) * 0.15)))
        train_indices.extend(indices[:-validation_count])
        validation_indices.extend(indices[-validation_count:])
    train_meta = [all_train_meta[i] for i in train_indices]
    validation_meta = [all_train_meta[i] for i in validation_indices]

    x_test, y_test, test_meta = build_test_windows(combined, columns, args.seq_len)
    if min(len(x_train), len(x_val), len(x_test)) == 0:
        raise ValueError("Not enough rows to create train/validation/test sequences")

    scaler_x = StandardScaler()
    scaler_x.fit(x_train.reshape(-1, len(columns)))
    x_train_scaled = scaler_x.transform(x_train.reshape(-1, len(columns))).reshape(x_train.shape)
    x_val_scaled = scaler_x.transform(x_val.reshape(-1, len(columns))).reshape(x_val.shape)
    x_test_scaled = scaler_x.transform(x_test.reshape(-1, len(columns))).reshape(x_test.shape)

    train_persistence = np.asarray([item["Persistence"] for item in train_meta])
    validation_persistence = np.asarray([item["Persistence"] for item in validation_meta])
    y_train_residual = y_train - train_persistence
    y_val_residual = y_val - validation_persistence

    scaler_y = StandardScaler()
    y_train_scaled = scaler_y.fit_transform(y_train_residual.reshape(-1, 1)).reshape(-1)
    y_val_scaled = scaler_y.transform(y_val_residual.reshape(-1, 1)).reshape(-1)

    model_config = {
        "seq_len": args.seq_len,
        "pred_len": 1,
        "enc_in": len(columns),
        "hidden_dim": 64,
        "dropout": 0.15,
        "kernel_size": 7,
        "use_mlp": True,
        "target_index": columns.index("Measure"),
    }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        "Sequences: train={} val={} test={} features={} device={}".format(
            len(x_train), len(x_val), len(x_test), len(columns), device
        ),
        flush=True,
    )

    model, best_val_loss, history = train_model(
        x_train_scaled,
        y_train_scaled,
        x_val_scaled,
        y_val_scaled,
        sample_weights_from_humidity(train_meta),
        sample_weights_from_humidity(validation_meta),
        model_config,
        device,
    )

    model.eval()
    with torch.no_grad():
        prediction_scaled = (
            model(torch.from_numpy(x_test_scaled).float().to(device))
            .squeeze(-1)
            .squeeze(-1)
            .cpu()
            .numpy()
        )
    persistence = np.asarray([item["Persistence"] for item in test_meta])
    predicted_residual = scaler_y.inverse_transform(
        prediction_scaled.reshape(-1, 1)
    ).reshape(-1)
    prediction = persistence + predicted_residual

    metrics = {
        "protocol": {
            "training_file": str(Path(args.training).resolve()),
            "test_file": str(Path(args.test).resolve()),
            "seq_len": args.seq_len,
            "pred_len": 1,
            "target_mode": "residual_from_last_measure",
            "seed": args.seed,
            "device": str(device),
            "feature_count": len(columns),
            "train_sequences": int(len(x_train)),
            "validation_sequences": int(len(x_val)),
            "test_sequences": int(len(x_test)),
            "best_validation_loss_scaled": float(best_val_loss),
        },
        "fmlp": regression_metrics(y_test, prediction),
        "persistence_baseline": regression_metrics(y_test, persistence),
    }
    metrics["improvement_over_persistence_percent"] = {
        metric_name: float(
            (metrics["persistence_baseline"][metric_name] - metrics["fmlp"][metric_name])
            / metrics["persistence_baseline"][metric_name]
            * 100.0
        )
        for metric_name in ["mse", "rmse", "mae"]
    }

    high_humidity_mask = np.asarray([item["Hum"] >= 80.0 for item in test_meta])
    if high_humidity_mask.any():
        metrics["fmlp_humidity_ge_80"] = regression_metrics(
            y_test[high_humidity_mask], prediction[high_humidity_mask]
        )
        metrics["persistence_humidity_ge_80"] = regression_metrics(
            y_test[high_humidity_mask], persistence[high_humidity_mask]
        )

    prediction_path = output_dir / "predictions.jsonl"
    with prediction_path.open("w", encoding="utf-8") as handle:
        for item, actual, predicted, baseline in zip(test_meta, y_test, prediction, persistence):
            row = dict(item)
            row.update({
                "Actual": float(actual),
                "FMLP_Prediction": float(predicted),
                "Persistence_Prediction": float(baseline),
                "FMLP_Absolute_Error": float(abs(actual - predicted)),
                "Persistence_Absolute_Error": float(abs(actual - baseline)),
            })
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, ensure_ascii=False, indent=2)
    with (output_dir / "training_history.json").open("w", encoding="utf-8") as handle:
        json.dump(history, handle, ensure_ascii=False, indent=2)

    torch.save(
        {
            "state_dict": model.state_dict(),
            "model_config": model_config,
            "feature_columns": columns,
            "x_mean": scaler_x.mean_,
            "x_scale": scaler_x.scale_,
            "y_mean": scaler_y.mean_,
            "y_scale": scaler_y.scale_,
            "target_mode": "residual_from_last_measure",
            "seed": args.seed,
        },
        output_dir / "optimized_fmlp.pth",
    )

    print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()


