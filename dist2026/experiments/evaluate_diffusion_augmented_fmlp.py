"""Evaluate humidity-conditioned DDPM augmentation for the four-stage FMLP.

The diffusion model only sees the real downstream training split. Synthetic
windows are added to downstream training, while validation and test remain
fully real. This keeps the augmentation utility measurement honest.
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from experiments.evaluate_fmlp_fixed_split import (
    absolute_humidity,
    build_all_windows,
    build_test_windows,
    feature_columns,
    prepare_rows,
    regression_metrics,
    set_seed,
)
from experiments.evaluate_humidity_stage_fmlp import (
    STAGE_NAMES,
    apply_calibration,
    calibration_margins,
    interval_metrics,
    metadata_split_indices,
    predict_quantiles,
    routing_features,
    stage_metrics,
    stage_name,
    stage_sample_weights,
    train_model,
)
SCRIPT_DIR = Path(__file__).resolve().parent
DIST_DIR = SCRIPT_DIR.parent
DATA_DIR = DIST_DIR / "data" / "raw"
from model.humidity_conditional_ddpm import ConditionalDDPM
from model.humidity_stage_fmlp import HumidityStageFMLP


GENERATED_HISTORY_COLUMNS = (
    "Measure",
    "Temp",
    "Hum",
    "Battery",
    "DeltaHours",
)
TARGET_CONTEXT_COLUMNS = ("Temp", "Hum")
STAGE_TO_INDEX = {name: index for index, name in enumerate(STAGE_NAMES)}


def generation_vector_layout(seq_len):
    history_size = seq_len * len(GENERATED_HISTORY_COLUMNS)
    return {
        "history": slice(0, history_size),
        "target_temp": history_size,
        "target_hum": history_size + 1,
        "target_residual": history_size + 2,
        "size": history_size + 3,
    }


def pack_generation_vectors(windows, targets, metadata, columns):
    if not (len(windows) == len(targets) == len(metadata)):
        raise ValueError("windows, targets and metadata must have equal length")
    indices = [columns.index(column) for column in GENERATED_HISTORY_COLUMNS]
    history = windows[:, :, indices].reshape(len(windows), -1)
    target_context = np.asarray(
        [[item[column] for column in TARGET_CONTEXT_COLUMNS] for item in metadata],
        dtype=np.float32,
    )
    persistence = np.asarray(
        [item["Persistence"] for item in metadata], dtype=np.float32
    )
    residual = np.asarray(targets, dtype=np.float32) - persistence
    return np.concatenate(
        [history, target_context, residual[:, None]], axis=1
    ).astype(np.float32)


def condition_indices(metadata, device_to_index):
    stages = np.asarray(
        [STAGE_TO_INDEX[stage_name(item["Hum"])] for item in metadata],
        dtype=np.int64,
    )
    devices = np.asarray(
        [device_to_index[str(item["IMEI"])] for item in metadata],
        dtype=np.int64,
    )
    return stages, devices


def fit_generation_scaler(vectors):
    scaler = StandardScaler()
    scaled = scaler.fit_transform(vectors).astype(np.float32)
    lower = np.quantile(vectors, 0.005, axis=0).astype(np.float32)
    upper = np.quantile(vectors, 0.995, axis=0).astype(np.float32)
    return scaler, scaled, lower, upper


def match_marginal_moments(generated, reference, strength):
    if not 0.0 <= strength <= 1.0:
        raise ValueError("moment matching strength must be in [0, 1]")
    if strength == 0.0:
        return generated.copy()
    generated_mean = generated.mean(axis=0)
    generated_std = generated.std(axis=0)
    reference_mean = reference.mean(axis=0)
    reference_std = reference.std(axis=0)
    safe_generated_std = np.maximum(generated_std, 1e-6)
    matched = (
        (generated - generated_mean) / safe_generated_std
        * reference_std
        + reference_mean
    )
    stable = generated_std >= 1e-6
    matched[:, ~stable] = reference_mean[~stable]
    return ((1.0 - strength) * generated + strength * matched).astype(np.float32)


def train_diffusion(
    model,
    vectors,
    stage_indices,
    device_indices,
    device,
    max_epochs,
    batch_size,
):
    counts = np.bincount(stage_indices, minlength=len(STAGE_NAMES))
    weights = np.asarray(
        [1.0 / max(counts[index], 1) for index in stage_indices],
        dtype=np.float64,
    )
    sampler = WeightedRandomSampler(
        torch.from_numpy(weights),
        num_samples=len(weights),
        replacement=True,
    )
    loader = DataLoader(
        TensorDataset(
            torch.from_numpy(vectors).float(),
            torch.from_numpy(stage_indices).long(),
            torch.from_numpy(device_indices).long(),
        ),
        batch_size=batch_size,
        sampler=sampler,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(max_epochs, 1), eta_min=1e-5
    )
    history = []

    for epoch in range(1, max_epochs + 1):
        model.train()
        total_loss = 0.0
        for batch_vectors, batch_stages, batch_devices in loader:
            optimizer.zero_grad()
            loss = model.loss(
                batch_vectors.to(device),
                batch_stages.to(device),
                batch_devices.to(device),
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()
        epoch_loss = total_loss / max(len(loader), 1)
        history.append({"epoch": epoch, "noise_mse": epoch_loss})
        if epoch == 1 or epoch % 10 == 0 or epoch == max_epochs:
            print(
                "DDPM epoch {:3d}: noise_mse={:.6f}".format(epoch, epoch_loss),
                flush=True,
            )
    return history


def _rebuild_synthetic_windows(
    generated_vectors,
    anchor_windows,
    anchor_metadata,
    columns,
    seq_len,
    target_bounds,
):
    layout = generation_vector_layout(seq_len)
    history = generated_vectors[:, layout["history"]].reshape(
        len(generated_vectors), seq_len, len(GENERATED_HISTORY_COLUMNS)
    )
    synthetic_x = np.asarray(anchor_windows, dtype=np.float32).copy()
    for generated_index, column in enumerate(GENERATED_HISTORY_COLUMNS):
        synthetic_x[:, :, columns.index(column)] = history[:, :, generated_index]

    temp = synthetic_x[:, :, columns.index("Temp")]
    hum = synthetic_x[:, :, columns.index("Hum")]
    synthetic_x[:, :, columns.index("AH")] = absolute_humidity(temp, hum)

    persistence = synthetic_x[:, -1, columns.index("Measure")]
    residual = generated_vectors[:, layout["target_residual"]]
    synthetic_y = np.clip(persistence + residual, target_bounds[0], target_bounds[1])
    synthetic_metadata = []
    for index, anchor in enumerate(anchor_metadata):
        item = dict(anchor)
        item.update({
            "Temp": float(generated_vectors[index, layout["target_temp"]]),
            "Hum": float(generated_vectors[index, layout["target_hum"]]),
            "HumidityTrend": float(
                generated_vectors[index, layout["target_hum"]]
                - synthetic_x[index, -1, columns.index("Hum")]
            ),
            "Persistence": float(persistence[index]),
            "Synthetic": True,
            "SyntheticID": "synthetic_{:05d}".format(index),
        })
        synthetic_metadata.append(item)
    return synthetic_x, synthetic_y.astype(np.float32), synthetic_metadata


def generate_very_humid_windows(
    model,
    scaler,
    lower_bounds,
    upper_bounds,
    real_windows,
    real_targets,
    real_metadata,
    columns,
    device_to_index,
    count,
    seq_len,
    seed,
    moment_match_strength,
):
    if count <= 0:
        raise ValueError("synthetic count must be positive")
    candidates = np.asarray([
        index
        for index, item in enumerate(real_metadata)
        if stage_name(item["Hum"]) == "very_humid"
    ])
    if len(candidates) == 0:
        raise ValueError("no very-humid real windows are available for conditioning")

    rng = np.random.RandomState(seed)
    accepted_vectors = []
    accepted_anchor_indices = []
    layout = generation_vector_layout(seq_len)
    attempts = 0
    requested = 0
    while sum(len(batch) for batch in accepted_vectors) < count and attempts < 12:
        remaining = count - sum(len(batch) for batch in accepted_vectors)
        batch_size = max(remaining * 2, min(64, count))
        anchor_indices = rng.choice(candidates, size=batch_size, replace=True)
        stage_conditions = torch.full(
            (batch_size,), STAGE_TO_INDEX["very_humid"], dtype=torch.long
        )
        device_conditions = torch.tensor([
            device_to_index[str(real_metadata[index]["IMEI"])]
            for index in anchor_indices
        ], dtype=torch.long)
        generated_scaled = model.sample(
            stage_conditions, device_conditions, clip_denoised=4.5
        ).cpu().numpy()
        generated = scaler.inverse_transform(generated_scaled)
        generated = np.clip(generated, lower_bounds, upper_bounds)
        generated[:, layout["target_hum"]] = np.clip(
            generated[:, layout["target_hum"]], 0.0, 100.0
        )
        valid = generated[:, layout["target_hum"]] >= 80.0
        if valid.any():
            accepted_vectors.append(generated[valid][:remaining])
            accepted_anchor_indices.append(anchor_indices[valid][:remaining])
        attempts += 1
        requested += batch_size

    accepted_count = sum(len(batch) for batch in accepted_vectors)
    if accepted_count < count:
        raise RuntimeError(
            "DDPM generated only {} valid very-humid samples out of {} requested; "
            "increase training epochs before using augmentation".format(
                accepted_count, count
            )
        )
    generated_vectors = np.concatenate(accepted_vectors, axis=0)[:count]
    anchor_indices = np.concatenate(accepted_anchor_indices, axis=0)[:count]
    real_stage_vectors = pack_generation_vectors(
        real_windows[candidates],
        real_targets[candidates],
        [real_metadata[index] for index in candidates],
        columns,
    )
    generated_vectors = match_marginal_moments(
        generated_vectors, real_stage_vectors, moment_match_strength
    )
    generated_vectors = np.clip(generated_vectors, lower_bounds, upper_bounds)
    generated_vectors[:, layout["target_hum"]] = np.clip(
        generated_vectors[:, layout["target_hum"]], 80.0, 100.0
    )
    anchors = real_windows[anchor_indices]
    anchor_metadata = [real_metadata[index] for index in anchor_indices]
    very_humid_targets = np.asarray([
        target
        for target, item in zip(real_targets, real_metadata)
        if stage_name(item["Hum"]) == "very_humid"
    ])
    target_bounds = (
        float(np.quantile(very_humid_targets, 0.005)),
        float(np.quantile(very_humid_targets, 0.995)),
    )
    synthetic_x, synthetic_y, synthetic_metadata = _rebuild_synthetic_windows(
        generated_vectors,
        anchors,
        anchor_metadata,
        columns,
        seq_len,
        target_bounds,
    )
    diagnostics = {
        "sampling_attempts": attempts,
        "raw_samples_requested": requested,
        "accepted_samples": int(count),
        "raw_stage_acceptance_rate": float(count / max(requested, 1)),
        "moment_match_strength": float(moment_match_strength),
    }
    return (
        synthetic_x,
        synthetic_y,
        synthetic_metadata,
        generated_vectors.astype(np.float32),
        diagnostics,
    )


def generation_quality_metrics(
    real_vectors,
    synthetic_vectors,
    real_targets,
    synthetic_targets,
    scaler,
    seq_len,
):
    real_scaled = scaler.transform(real_vectors)
    synthetic_scaled = scaler.transform(synthetic_vectors)
    synthetic_to_real = pairwise_distances(synthetic_scaled, real_scaled).min(axis=1)
    real_distances = pairwise_distances(real_scaled, real_scaled)
    np.fill_diagonal(real_distances, np.inf)
    real_to_real = real_distances.min(axis=1)
    layout = generation_vector_layout(seq_len)
    real_history = real_vectors[:, layout["history"]].reshape(
        len(real_vectors), seq_len, len(GENERATED_HISTORY_COLUMNS)
    )
    synthetic_history = synthetic_vectors[:, layout["history"]].reshape(
        len(synthetic_vectors), seq_len, len(GENERATED_HISTORY_COLUMNS)
    )

    def correlation(values):
        measure = values[:, :, GENERATED_HISTORY_COLUMNS.index("Measure")].reshape(-1)
        humidity = values[:, :, GENERATED_HISTORY_COLUMNS.index("Hum")].reshape(-1)
        if np.std(measure) < 1e-8 or np.std(humidity) < 1e-8:
            return 0.0
        return float(np.corrcoef(measure, humidity)[0, 1])

    mean_shift = np.mean(
        np.abs(synthetic_scaled.mean(axis=0) - real_scaled.mean(axis=0))
    )
    std_ratio = synthetic_scaled.std(axis=0) / np.maximum(
        real_scaled.std(axis=0), 1e-6
    )
    return {
        "real_count": int(len(real_vectors)),
        "synthetic_count": int(len(synthetic_vectors)),
        "mean_absolute_standardized_mean_shift": float(mean_shift),
        "median_standardized_std_ratio": float(np.median(std_ratio)),
        "nearest_neighbor": {
            "synthetic_to_real_median": float(np.median(synthetic_to_real)),
            "real_to_real_median": float(np.median(real_to_real)),
            "distance_ratio": float(
                np.median(synthetic_to_real) / max(np.median(real_to_real), 1e-8)
            ),
            "synthetic_below_10pct_real_nn_rate": float(
                np.mean(synthetic_to_real < 0.1 * np.median(real_to_real))
            ),
        },
        "history_measure_humidity_correlation": {
            "real": correlation(real_history),
            "synthetic": correlation(synthetic_history),
        },
        "target": {
            "real_mean": float(np.mean(real_targets)),
            "real_std": float(np.std(real_targets)),
            "synthetic_mean": float(np.mean(synthetic_targets)),
            "synthetic_std": float(np.std(synthetic_targets)),
        },
        "target_humidity": {
            "real_mean": float(np.mean(real_vectors[:, layout["target_hum"]])),
            "real_std": float(np.std(real_vectors[:, layout["target_hum"]])),
            "synthetic_mean": float(
                np.mean(synthetic_vectors[:, layout["target_hum"]])
            ),
            "synthetic_std": float(
                np.std(synthetic_vectors[:, layout["target_hum"]])
            ),
        },
        "target_residual": {
            "real_mean": float(
                np.mean(real_vectors[:, layout["target_residual"]])
            ),
            "real_std": float(
                np.std(real_vectors[:, layout["target_residual"]])
            ),
            "synthetic_mean": float(
                np.mean(synthetic_vectors[:, layout["target_residual"]])
            ),
            "synthetic_std": float(
                np.std(synthetic_vectors[:, layout["target_residual"]])
            ),
        },
    }


def build_fmlp_scalers(x_train, y_train, metadata, feature_count):
    scaler_x = StandardScaler()
    scaler_x.fit(x_train.reshape(-1, feature_count))
    persistence = np.asarray([item["Persistence"] for item in metadata])
    scaler_y = StandardScaler()
    scaler_y.fit((y_train - persistence).reshape(-1, 1))
    return scaler_x, scaler_y


def downstream_sample_weights(metadata, synthetic_loss_weight):
    if not 0.0 < synthetic_loss_weight <= 1.0:
        raise ValueError("synthetic_loss_weight must be in (0, 1]")
    weights = stage_sample_weights(metadata)
    confidence = []
    for item in metadata:
        if not item.get("Synthetic", False):
            confidence.append(1.0)
            continue
        quality = float(item.get("SyntheticQuality", 1.0))
        if not 0.0 <= quality <= 1.0:
            raise ValueError("SyntheticQuality must be in [0, 1]")
        confidence.append(synthetic_loss_weight * quality)
    confidence = np.asarray(confidence, dtype=np.float32)
    return weights * confidence


def scale_fmlp_split(x, y, metadata, scaler_x, scaler_y):
    x_scaled = scaler_x.transform(x.reshape(-1, x.shape[-1])).reshape(x.shape)
    persistence = np.asarray([item["Persistence"] for item in metadata])
    y_scaled = scaler_y.transform((y - persistence).reshape(-1, 1)).reshape(-1)
    return x_scaled.astype(np.float32), y_scaled.astype(np.float32), persistence


def train_and_evaluate_fmlp(
    x_train,
    y_train,
    train_metadata,
    x_validation,
    y_validation,
    validation_metadata,
    x_test,
    y_test,
    test_metadata,
    scaler_x,
    scaler_y,
    model_config,
    device,
    seed,
    max_epochs,
    synthetic_loss_weight,
):
    set_seed(seed)
    x_train_scaled, y_train_scaled, _ = scale_fmlp_split(
        x_train, y_train, train_metadata, scaler_x, scaler_y
    )
    x_validation_scaled, y_validation_scaled, validation_persistence = (
        scale_fmlp_split(
            x_validation,
            y_validation,
            validation_metadata,
            scaler_x,
            scaler_y,
        )
    )
    x_test_scaled, _, test_persistence = scale_fmlp_split(
        x_test, y_test, test_metadata, scaler_x, scaler_y
    )
    train_loader = DataLoader(
        TensorDataset(
            torch.from_numpy(x_train_scaled).float(),
            torch.from_numpy(y_train_scaled).float(),
            torch.from_numpy(routing_features(train_metadata)).float(),
            torch.from_numpy(
                downstream_sample_weights(
                    train_metadata, synthetic_loss_weight
                )
            ).float(),
        ),
        batch_size=32,
        shuffle=True,
    )
    validation_loader = DataLoader(
        TensorDataset(
            torch.from_numpy(x_validation_scaled).float(),
            torch.from_numpy(y_validation_scaled).float(),
            torch.from_numpy(routing_features(validation_metadata)).float(),
            torch.from_numpy(stage_sample_weights(validation_metadata)).float(),
        ),
        batch_size=64,
        shuffle=False,
    )
    model = HumidityStageFMLP(**model_config).to(device)
    model, best_loss, history = train_model(
        model,
        train_loader,
        validation_loader,
        device,
        max_epochs=max_epochs,
    )
    validation_quantiles, _, _ = predict_quantiles(
        model,
        x_validation_scaled,
        routing_features(validation_metadata),
        device,
        scaler_y,
        validation_persistence,
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
        routing_features(test_metadata),
        device,
        scaler_y,
        test_persistence,
    )
    test_quantiles = apply_calibration(test_quantiles, test_metadata, margins)
    metrics = {
        "train_sequences": int(len(x_train)),
        "best_validation_loss_scaled": float(best_loss),
        "conformal_margins": margins,
        "overall": {
            "fmlp": regression_metrics(y_test, test_quantiles[:, 1]),
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
    return {
        "model": model,
        "history": history,
        "metrics": metrics,
        "quantiles": test_quantiles,
        "gates": test_gates,
        "priors": test_priors,
    }


def comparison_metrics(baseline, augmented):
    result = {"overall": {}, "by_humidity_stage": {}}
    for metric in ("rmse", "mae", "r2"):
        base_value = baseline["overall"]["fmlp"][metric]
        augmented_value = augmented["overall"]["fmlp"][metric]
        result["overall"][metric] = {
            "baseline": base_value,
            "augmented": augmented_value,
            "delta": augmented_value - base_value,
        }
    stages = sorted(
        set(baseline["by_humidity_stage"])
        & set(augmented["by_humidity_stage"])
    )
    for stage in stages:
        result["by_humidity_stage"][stage] = {}
        for metric in ("rmse", "mae", "r2"):
            base_value = baseline["by_humidity_stage"][stage]["fmlp"][metric]
            augmented_value = augmented["by_humidity_stage"][stage]["fmlp"][metric]
            result["by_humidity_stage"][stage][metric] = {
                "baseline": base_value,
                "augmented": augmented_value,
                "delta": augmented_value - base_value,
            }
    return result


def save_synthetic_long_csv(
    path, windows, targets, metadata, columns, seq_len
):
    rows = []
    for sample_index, item in enumerate(metadata):
        synthetic_id = item["SyntheticID"]
        for step in range(seq_len):
            rows.append({
                "SyntheticID": synthetic_id,
                "Step": step,
                "IsTarget": False,
                "ConditionIMEI": item["IMEI"],
                "HumidityStage": stage_name(item["Hum"]),
                "Measure": float(windows[sample_index, step, columns.index("Measure")]),
                "Temp": float(windows[sample_index, step, columns.index("Temp")]),
                "Hum": float(windows[sample_index, step, columns.index("Hum")]),
                "Battery": float(windows[sample_index, step, columns.index("Battery")]),
                "DeltaHours": float(
                    windows[sample_index, step, columns.index("DeltaHours")]
                ),
            })
        rows.append({
            "SyntheticID": synthetic_id,
            "Step": seq_len,
            "IsTarget": True,
            "ConditionIMEI": item["IMEI"],
            "HumidityStage": stage_name(item["Hum"]),
            "Measure": float(targets[sample_index]),
            "Temp": float(item["Temp"]),
            "Hum": float(item["Hum"]),
            "Battery": float(
                windows[sample_index, -1, columns.index("Battery")]
            ),
            "DeltaHours": float(
                windows[sample_index, -1, columns.index("DeltaHours")]
            ),
        })
    pd.DataFrame(rows).to_csv(path, index=False)


def save_predictions(path, metadata, actual, run):
    with path.open("w", encoding="utf-8") as handle:
        for index, item in enumerate(metadata):
            row = dict(item)
            row.update({
                "HumidityStage": stage_name(item["Hum"]),
                "Actual": float(actual[index]),
                "P10": float(run["quantiles"][index, 0]),
                "P50": float(run["quantiles"][index, 1]),
                "P90": float(run["quantiles"][index, 2]),
                "GateWeights": {
                    stage: float(run["gates"][index, stage_index])
                    for stage_index, stage in enumerate(STAGE_NAMES)
                },
            })
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--training", default=str(DATA_DIR / "training.csv"))
    parser.add_argument("--test", default=str(DATA_DIR / "test.csv"))
    parser.add_argument(
        "--output-dir", default=str(SCRIPT_DIR / "diffusion_augmented_fmlp_evaluation")
    )
    parser.add_argument("--seq-len", type=int, default=24)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--synthetic-ratio", type=float, default=1.0)
    parser.add_argument("--synthetic-loss-weight", type=float, default=1.0)
    parser.add_argument("--moment-match-strength", type=float, default=0.0)
    parser.add_argument("--diffusion-epochs", type=int, default=120)
    parser.add_argument("--diffusion-steps", type=int, default=50)
    parser.add_argument("--diffusion-hidden-dim", type=int, default=192)
    parser.add_argument("--diffusion-depth", type=int, default=4)
    parser.add_argument("--fmlp-epochs", type=int, default=180)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    if args.synthetic_ratio <= 0:
        raise ValueError("synthetic-ratio must be positive")
    if not 0.0 < args.synthetic_loss_weight <= 1.0:
        raise ValueError("synthetic-loss-weight must be in (0, 1]")
    if not 0.0 <= args.moment_match_strength <= 1.0:
        raise ValueError("moment-match-strength must be in [0, 1]")

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
    device_to_index = {
        device_id: index for index, device_id in enumerate(device_ids)
    }

    training_raw["_split"] = "train"
    test_raw["_split"] = "test"
    training = prepare_rows(training_raw, device_ids)
    combined = prepare_rows(
        pd.concat([training_raw, test_raw], ignore_index=True), device_ids
    )
    columns = feature_columns(device_ids)
    all_x, all_y, all_metadata = build_all_windows(
        training, columns, args.seq_len
    )
    train_indices, validation_indices = metadata_split_indices(
        all_metadata, validation_ratio=0.15
    )
    x_train = all_x[train_indices]
    y_train = all_y[train_indices]
    train_metadata = [all_metadata[index] for index in train_indices]
    x_validation = all_x[validation_indices]
    y_validation = all_y[validation_indices]
    validation_metadata = [all_metadata[index] for index in validation_indices]
    x_test, y_test, test_metadata = build_test_windows(
        combined, columns, args.seq_len
    )
    if min(len(x_train), len(x_validation), len(x_test)) == 0:
        raise ValueError("Not enough rows to build all sequence splits")

    feature_count = len(columns)
    scaler_x, scaler_y = build_fmlp_scalers(
        x_train, y_train, train_metadata, feature_count
    )
    model_config = {
        "seq_len": args.seq_len,
        "pred_len": 1,
        "enc_in": feature_count,
        "hidden_dim": 64,
        "expert_hidden_dim": 32,
        "dropout": 0.15,
        "kernel_size": 7,
        "target_index": columns.index("Measure"),
        "humidity_index": columns.index("Hum"),
        "routing_temperature": 2.5,
        "learned_gate_scale": 0.25,
    }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        "Real sequences: train={} val={} test={} features={} device={}".format(
            len(x_train), len(x_validation), len(x_test), feature_count, device
        ),
        flush=True,
    )
    print("Training real-only FMLP baseline", flush=True)
    baseline_run = train_and_evaluate_fmlp(
        x_train,
        y_train,
        train_metadata,
        x_validation,
        y_validation,
        validation_metadata,
        x_test,
        y_test,
        test_metadata,
        scaler_x,
        scaler_y,
        model_config,
        device,
        args.seed,
        args.fmlp_epochs,
        1.0,
    )

    generation_vectors = pack_generation_vectors(
        x_train, y_train, train_metadata, columns
    )
    generation_scaler, generation_scaled, lower_bounds, upper_bounds = (
        fit_generation_scaler(generation_vectors)
    )
    train_stage_indices, train_device_indices = condition_indices(
        train_metadata, device_to_index
    )
    ddpm_config = {
        "input_dim": int(generation_scaled.shape[1]),
        "num_stages": len(STAGE_NAMES),
        "num_devices": len(device_ids),
        "hidden_dim": args.diffusion_hidden_dim,
        "depth": args.diffusion_depth,
        "dropout": 0.05,
        "diffusion_steps": args.diffusion_steps,
    }
    set_seed(args.seed + 101)
    ddpm = ConditionalDDPM(**ddpm_config).to(device)
    print("Training humidity-conditioned DDPM", flush=True)
    diffusion_history = train_diffusion(
        ddpm,
        generation_scaled,
        train_stage_indices,
        train_device_indices,
        device,
        max_epochs=args.diffusion_epochs,
        batch_size=args.batch_size,
    )

    very_humid_count = sum(
        stage_name(item["Hum"]) == "very_humid" for item in train_metadata
    )
    synthetic_count = max(1, int(round(very_humid_count * args.synthetic_ratio)))
    set_seed(args.seed + 202)
    (
        synthetic_x,
        synthetic_y,
        synthetic_metadata,
        synthetic_vectors,
        sampling_diagnostics,
    ) = generate_very_humid_windows(
        ddpm,
        generation_scaler,
        lower_bounds,
        upper_bounds,
        x_train,
        y_train,
        train_metadata,
        columns,
        device_to_index,
        synthetic_count,
        args.seq_len,
        args.seed + 202,
        args.moment_match_strength,
    )
    very_humid_mask = np.asarray([
        stage_name(item["Hum"]) == "very_humid" for item in train_metadata
    ])
    quality = generation_quality_metrics(
        generation_vectors[very_humid_mask],
        synthetic_vectors,
        y_train[very_humid_mask],
        synthetic_y,
        generation_scaler,
        args.seq_len,
    )
    quality["sampling"] = sampling_diagnostics
    augmented_x = np.concatenate([x_train, synthetic_x], axis=0)
    augmented_y = np.concatenate([y_train, synthetic_y], axis=0)
    augmented_metadata = train_metadata + synthetic_metadata
    print(
        "Training augmented FMLP with {} synthetic very-humid windows".format(
            synthetic_count
        ),
        flush=True,
    )
    augmented_run = train_and_evaluate_fmlp(
        augmented_x,
        augmented_y,
        augmented_metadata,
        x_validation,
        y_validation,
        validation_metadata,
        x_test,
        y_test,
        test_metadata,
        scaler_x,
        scaler_y,
        model_config,
        device,
        args.seed,
        args.fmlp_epochs,
        args.synthetic_loss_weight,
    )

    metrics = {
        "protocol": {
            "training_file": str(Path(args.training).resolve()),
            "test_file": str(Path(args.test).resolve()),
            "seq_len": args.seq_len,
            "real_train_sequences": int(len(x_train)),
            "validation_sequences": int(len(x_validation)),
            "test_sequences": int(len(x_test)),
            "real_very_humid_train_sequences": int(very_humid_count),
            "synthetic_very_humid_sequences": int(synthetic_count),
            "synthetic_ratio": args.synthetic_ratio,
            "synthetic_loss_weight": args.synthetic_loss_weight,
            "moment_match_strength": args.moment_match_strength,
            "generator_training_scope": "real downstream training split only",
            "validation_and_test": "real only",
            "device": str(device),
            "seed": args.seed,
        },
        "ddpm_config": ddpm_config,
        "generation_quality": quality,
        "baseline": baseline_run["metrics"],
        "augmented": augmented_run["metrics"],
        "comparison": comparison_metrics(
            baseline_run["metrics"], augmented_run["metrics"]
        ),
    }
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, ensure_ascii=False, indent=2)
    with (output_dir / "diffusion_history.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(diffusion_history, handle, ensure_ascii=False, indent=2)
    with (output_dir / "baseline_fmlp_history.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(baseline_run["history"], handle, ensure_ascii=False, indent=2)
    with (output_dir / "augmented_fmlp_history.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(augmented_run["history"], handle, ensure_ascii=False, indent=2)
    save_synthetic_long_csv(
        output_dir / "synthetic_training_windows.csv",
        synthetic_x,
        synthetic_y,
        synthetic_metadata,
        columns,
        args.seq_len,
    )
    np.savez_compressed(
        output_dir / "synthetic_training_windows.npz",
        windows=synthetic_x,
        targets=synthetic_y,
        routes=routing_features(synthetic_metadata),
        feature_columns=np.asarray(columns),
    )
    save_predictions(
        output_dir / "augmented_predictions.jsonl",
        test_metadata,
        y_test,
        augmented_run,
    )
    torch.save(
        {
            "state_dict": ddpm.state_dict(),
            "model_config": ddpm_config,
            "generation_columns": GENERATED_HISTORY_COLUMNS,
            "target_context_columns": TARGET_CONTEXT_COLUMNS,
            "generation_mean": generation_scaler.mean_,
            "generation_scale": generation_scaler.scale_,
            "clip_lower": lower_bounds,
            "clip_upper": upper_bounds,
            "stage_names": STAGE_NAMES,
            "device_ids": device_ids,
            "seq_len": args.seq_len,
            "seed": args.seed,
        },
        output_dir / "humidity_conditional_ddpm.pth",
    )
    for name, run in (("baseline", baseline_run), ("augmented", augmented_run)):
        torch.save(
            {
                "state_dict": run["model"].state_dict(),
                "model_config": model_config,
                "feature_columns": columns,
                "x_mean": scaler_x.mean_,
                "x_scale": scaler_x.scale_,
                "y_mean": scaler_y.mean_,
                "y_scale": scaler_y.scale_,
                "conformal_margins": run["metrics"]["conformal_margins"],
                "seed": args.seed,
            },
            output_dir / (name + "_humidity_stage_fmlp.pth"),
        )

    print(json.dumps(metrics["comparison"], ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()


