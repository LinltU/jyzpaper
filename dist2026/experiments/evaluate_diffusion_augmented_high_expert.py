"""Battery-free DDPM augmentation for the V2 dynamic high-humidity expert.

The DDPM sees only real downstream training windows. Synthetic very-humid
windows are added only to FMLP training; validation and test remain real.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from experiments.evaluate_diffusion_augmented_fmlp import (
    condition_indices,
    downstream_sample_weights,
    fit_generation_scaler,
    match_marginal_moments,
    train_diffusion,
)
from experiments.evaluate_fmlp_fixed_split import (
    build_all_windows,
    build_test_windows,
    regression_metrics,
    set_seed,
)
from experiments.evaluate_humidity_stage_fmlp import (
    STAGE_NAMES,
    apply_calibration,
    calibration_margins,
    interval_metrics,
    known_target_covariates,
    metadata_device_indices,
    metadata_split_indices,
    predict_quantiles,
    routing_features,
    stage_metrics,
    stage_name,
    stage_sample_weights,
    train_model,
)
from core.feature_engineering_v2 import prepare_rows_v2
from model.humidity_conditional_ddpm import ConditionalDDPM
from model.humidity_stage_fmlp import HumidityStageFMLP


GENERATED_HISTORY_COLUMNS = ("Measure", "Temp", "Hum", "DeltaHours")
STAGE_TO_INDEX = {name: index for index, name in enumerate(STAGE_NAMES)}


def generation_vector_layout(seq_len):
    history_size = seq_len * len(GENERATED_HISTORY_COLUMNS)
    return {
        "history": slice(0, history_size),
        "target_temp": history_size,
        "target_hum": history_size + 1,
        "target_delta": history_size + 2,
        "target_residual": history_size + 3,
        "size": history_size + 4,
    }


def pack_generation_vectors(windows, targets, metadata, columns):
    if not (len(windows) == len(targets) == len(metadata)):
        raise ValueError("windows, targets and metadata must have equal length")
    indices = [columns.index(column) for column in GENERATED_HISTORY_COLUMNS]
    history = windows[:, :, indices].reshape(len(windows), -1)
    target_context = np.asarray([
        [item["Temp"], item["Hum"], item["ForecastDeltaHours"]]
        for item in metadata
    ], dtype=np.float32)
    persistence = np.asarray(
        [item["Persistence"] for item in metadata], dtype=np.float32
    )
    residual = np.asarray(targets, dtype=np.float32) - persistence
    return np.concatenate(
        [history, target_context, residual[:, None]], axis=1
    ).astype(np.float32)


def rebuild_synthetic_windows(
    generated_vectors,
    anchor_metadata,
    location_map,
    columns,
    seq_len,
    target_bounds,
    humidity_fill_value,
    median_delta_hours,
):
    layout = generation_vector_layout(seq_len)
    history = generated_vectors[:, layout["history"]].reshape(
        len(generated_vectors), seq_len, len(GENERATED_HISTORY_COLUMNS)
    )
    synthetic_windows = []
    synthetic_targets = []
    synthetic_metadata = []

    for index, anchor in enumerate(anchor_metadata):
        imei = str(anchor["IMEI"])
        raw = history[index].copy()
        raw[:, GENERATED_HISTORY_COLUMNS.index("Hum")] = np.clip(
            raw[:, GENERATED_HISTORY_COLUMNS.index("Hum")], 0.1, 100.0
        )
        raw[:, GENERATED_HISTORY_COLUMNS.index("DeltaHours")] = np.clip(
            raw[:, GENERATED_HISTORY_COLUMNS.index("DeltaHours")], 1.0 / 60.0, 72.0
        )
        target_delta = float(np.clip(
            generated_vectors[index, layout["target_delta"]], 1.0 / 60.0, 72.0
        ))
        target_timestamp = pd.Timestamp(anchor["Timestamp"])
        timestamps = [None] * seq_len
        timestamps[-1] = target_timestamp - pd.Timedelta(hours=target_delta)
        delta_index = GENERATED_HISTORY_COLUMNS.index("DeltaHours")
        for step in range(seq_len - 1, 0, -1):
            timestamps[step - 1] = timestamps[step] - pd.Timedelta(
                hours=float(raw[step, delta_index])
            )

        frame = pd.DataFrame({
            "IMEI": imei,
            "Location": location_map[imei],
            "Timestamp": timestamps,
            "Measure": raw[:, GENERATED_HISTORY_COLUMNS.index("Measure")],
            "Temp": raw[:, GENERATED_HISTORY_COLUMNS.index("Temp")],
            "Hum": raw[:, GENERATED_HISTORY_COLUMNS.index("Hum")],
        })
        prepared = prepare_rows_v2(
            frame,
            humidity_fill_value=humidity_fill_value,
            median_delta_hours=median_delta_hours,
        )
        window = prepared[columns].to_numpy(dtype=np.float32)
        persistence = float(prepared.iloc[-1]["Measure"])
        residual = float(generated_vectors[index, layout["target_residual"]])
        target = float(np.clip(
            persistence + residual, target_bounds[0], target_bounds[1]
        ))
        target_hum = float(np.clip(
            generated_vectors[index, layout["target_hum"]], 80.0, 100.0
        ))
        item = dict(anchor)
        item.update({
            "Temp": float(generated_vectors[index, layout["target_temp"]]),
            "Hum": target_hum,
            "HumidityTrend": target_hum - float(prepared.iloc[-1]["Hum"]),
            "LastHum": float(prepared.iloc[-1]["Hum"]),
            "LastHumidityTrend": float(
                prepared.iloc[-1]["Hum"] - prepared.iloc[-2]["Hum"]
            ),
            "ForecastDeltaHours": target_delta,
            "Persistence": persistence,
            "Synthetic": True,
            "SyntheticID": "synthetic_high_{:05d}".format(index),
        })
        synthetic_windows.append(window)
        synthetic_targets.append(target)
        synthetic_metadata.append(item)

    return (
        np.asarray(synthetic_windows, dtype=np.float32),
        np.asarray(synthetic_targets, dtype=np.float32),
        synthetic_metadata,
    )


def generate_very_humid_windows(
    model,
    scaler,
    lower_bounds,
    upper_bounds,
    real_windows,
    real_targets,
    real_metadata,
    location_map,
    columns,
    device_to_index,
    count,
    seq_len,
    seed,
    moment_match_strength,
    humidity_fill_value,
    median_delta_hours,
):
    candidates = np.asarray([
        index for index, item in enumerate(real_metadata)
        if stage_name(item["Hum"]) == "very_humid"
    ])
    if len(candidates) == 0:
        raise ValueError("No real very-humid training windows")
    rng = np.random.RandomState(seed)
    accepted = []
    accepted_anchors = []
    layout = generation_vector_layout(seq_len)
    attempts = 0
    requested = 0
    while sum(len(batch) for batch in accepted) < count and attempts < 12:
        remaining = count - sum(len(batch) for batch in accepted)
        batch_size = max(remaining * 2, min(64, count))
        anchor_indices = rng.choice(candidates, size=batch_size, replace=True)
        stages = torch.full(
            (batch_size,), STAGE_TO_INDEX["very_humid"], dtype=torch.long
        )
        devices = torch.tensor([
            device_to_index[str(real_metadata[index]["IMEI"])]
            for index in anchor_indices
        ], dtype=torch.long)
        generated_scaled = model.sample(
            stages, devices, clip_denoised=4.5
        ).cpu().numpy()
        generated = scaler.inverse_transform(generated_scaled)
        generated = np.clip(generated, lower_bounds, upper_bounds)
        generated[:, layout["target_hum"]] = np.clip(
            generated[:, layout["target_hum"]], 0.0, 100.0
        )
        valid = generated[:, layout["target_hum"]] >= 80.0
        if valid.any():
            accepted.append(generated[valid][:remaining])
            accepted_anchors.append(anchor_indices[valid][:remaining])
        attempts += 1
        requested += batch_size
    if sum(len(batch) for batch in accepted) < count:
        raise RuntimeError("DDPM did not generate enough valid very-humid samples")

    generated = np.concatenate(accepted, axis=0)[:count]
    anchor_indices = np.concatenate(accepted_anchors, axis=0)[:count]
    real_high_vectors = pack_generation_vectors(
        real_windows[candidates],
        real_targets[candidates],
        [real_metadata[index] for index in candidates],
        columns,
    )
    generated = match_marginal_moments(
        generated, real_high_vectors, moment_match_strength
    )
    generated = np.clip(generated, lower_bounds, upper_bounds)
    generated[:, layout["target_hum"]] = np.clip(
        generated[:, layout["target_hum"]], 80.0, 100.0
    )
    generated[:, layout["target_delta"]] = np.clip(
        generated[:, layout["target_delta"]], 1.0 / 60.0, 72.0
    )
    high_targets = np.asarray([
        target for target, item in zip(real_targets, real_metadata)
        if stage_name(item["Hum"]) == "very_humid"
    ])
    target_bounds = (
        float(np.quantile(high_targets, 0.005)),
        float(np.quantile(high_targets, 0.995)),
    )
    synthetic_x, synthetic_y, synthetic_metadata = rebuild_synthetic_windows(
        generated,
        [real_metadata[index] for index in anchor_indices],
        location_map,
        columns,
        seq_len,
        target_bounds,
        humidity_fill_value,
        median_delta_hours,
    )
    return synthetic_x, synthetic_y, synthetic_metadata, generated.astype(np.float32), {
        "sampling_attempts": attempts,
        "raw_samples_requested": requested,
        "accepted_samples": int(count),
        "raw_stage_acceptance_rate": float(count / max(requested, 1)),
        "moment_match_strength": float(moment_match_strength),
    }


def generation_quality_metrics(real_vectors, synthetic_vectors, scaler, seq_len):
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
        measure = values[:, :, 0].reshape(-1)
        humidity = values[:, :, 2].reshape(-1)
        return float(np.corrcoef(measure, humidity)[0, 1])

    std_ratio = synthetic_scaled.std(axis=0) / np.maximum(
        real_scaled.std(axis=0), 1e-6
    )
    return {
        "real_count": int(len(real_vectors)),
        "synthetic_count": int(len(synthetic_vectors)),
        "mean_absolute_standardized_mean_shift": float(np.mean(np.abs(
            synthetic_scaled.mean(axis=0) - real_scaled.mean(axis=0)
        ))),
        "median_standardized_std_ratio": float(np.median(std_ratio)),
        "nearest_neighbor": {
            "synthetic_to_real_median": float(np.median(synthetic_to_real)),
            "real_to_real_median": float(np.median(real_to_real)),
            "distance_ratio": float(
                np.median(synthetic_to_real) / max(np.median(real_to_real), 1e-8)
            ),
            "synthetic_below_10pct_real_nn_rate": float(np.mean(
                synthetic_to_real < 0.1 * np.median(real_to_real)
            )),
        },
        "history_measure_humidity_correlation": {
            "real": correlation(real_history),
            "synthetic": correlation(synthetic_history),
        },
        "target_humidity": {
            "real_mean": float(np.mean(real_vectors[:, layout["target_hum"]])),
            "real_std": float(np.std(real_vectors[:, layout["target_hum"]])),
            "synthetic_mean": float(np.mean(synthetic_vectors[:, layout["target_hum"]])),
            "synthetic_std": float(np.std(synthetic_vectors[:, layout["target_hum"]])),
        },
        "target_residual": {
            "real_mean": float(np.mean(real_vectors[:, layout["target_residual"]])),
            "real_std": float(np.std(real_vectors[:, layout["target_residual"]])),
            "synthetic_mean": float(np.mean(synthetic_vectors[:, layout["target_residual"]])),
            "synthetic_std": float(np.std(synthetic_vectors[:, layout["target_residual"]])),
        },
    }


def _row_correlations(history):
    output = np.zeros(len(history), dtype=np.float32)
    for index, values in enumerate(history):
        measure = values[:, GENERATED_HISTORY_COLUMNS.index("Measure")]
        humidity = values[:, GENERATED_HISTORY_COLUMNS.index("Hum")]
        if np.std(measure) > 1e-6 and np.std(humidity) > 1e-6:
            output[index] = float(np.corrcoef(measure, humidity)[0, 1])
    return output


def _robust_plausibility(values, reference):
    values = np.asarray(values, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    median = float(np.median(reference))
    mad_scale = 1.4826 * float(np.median(np.abs(reference - median)))
    scale = max(mad_scale, float(np.std(reference)) * 0.25, 1e-6)
    z_score = np.abs(values - median) / scale
    return np.exp(-0.5 * np.minimum(z_score, 6.0) ** 2).astype(np.float32)


def select_quality_synthetic(
    real_vectors,
    candidate_x,
    candidate_y,
    candidate_metadata,
    candidate_vectors,
    scaler,
    seq_len,
    select_count,
    minimum_quality,
):
    """Select plausible, humidity-diverse candidates using real training only."""
    if not 0.0 < minimum_quality <= 1.0:
        raise ValueError("minimum_quality must be in (0, 1]")
    if select_count <= 0 or select_count > len(candidate_vectors):
        raise ValueError("select_count must be within the candidate pool")

    real_scaled = scaler.transform(real_vectors)
    candidate_scaled = scaler.transform(candidate_vectors)
    candidate_to_real_matrix = pairwise_distances(candidate_scaled, real_scaled)
    candidate_to_real = candidate_to_real_matrix.min(axis=1)
    nearest_real = candidate_to_real_matrix.argmin(axis=1)
    real_distances = pairwise_distances(real_scaled, real_scaled)
    np.fill_diagonal(real_distances, np.inf)
    real_to_real = real_distances.min(axis=1)
    real_distance_scale = max(float(np.median(real_to_real)), 1e-6)
    distance_ratio = candidate_to_real / real_distance_scale
    manifold_score = np.exp(-np.maximum(distance_ratio - 1.0, 0.0) / 3.0)
    copy_penalty = np.minimum(candidate_to_real / (0.1 * real_distance_scale), 1.0)
    manifold_score = (manifold_score * copy_penalty).astype(np.float32)

    layout = generation_vector_layout(seq_len)
    real_history = real_vectors[:, layout["history"]].reshape(
        len(real_vectors), seq_len, len(GENERATED_HISTORY_COLUMNS)
    )
    candidate_history = candidate_vectors[:, layout["history"]].reshape(
        len(candidate_vectors), seq_len, len(GENERATED_HISTORY_COLUMNS)
    )
    real_correlations = _row_correlations(real_history)
    candidate_correlations = _row_correlations(candidate_history)
    correlation_score = _robust_plausibility(
        candidate_correlations, real_correlations
    )
    measure_index = GENERATED_HISTORY_COLUMNS.index("Measure")
    real_roughness = np.std(np.diff(real_history[:, :, measure_index], axis=1), axis=1)
    candidate_roughness = np.std(
        np.diff(candidate_history[:, :, measure_index], axis=1), axis=1
    )
    roughness_score = _robust_plausibility(candidate_roughness, real_roughness)
    humidity_score = _robust_plausibility(
        candidate_vectors[:, layout["target_hum"]],
        real_vectors[:, layout["target_hum"]],
    )
    quality_score = (
        0.56 * manifold_score
        + 0.18 * correlation_score
        + 0.14 * roughness_score
        + 0.12 * humidity_score
    )

    # Preserve both humidity coverage and residual tails. A quality-only rank
    # collapses residual variance toward easy central samples.
    humidity_bins = np.asarray([80.0, 85.0, 90.0, 95.0, 100.0001])
    real_bin = np.clip(
        np.digitize(real_vectors[:, layout["target_hum"]], humidity_bins) - 1,
        0,
        len(humidity_bins) - 2,
    )
    candidate_bin = np.clip(
        np.digitize(
            candidate_vectors[:, layout["target_hum"]], humidity_bins
        ) - 1,
        0,
        len(humidity_bins) - 2,
    )
    real_residual = real_vectors[:, layout["target_residual"]]
    candidate_residual = candidate_vectors[:, layout["target_residual"]]
    residual_bins = np.quantile(real_residual, np.linspace(0.0, 1.0, 6))
    residual_bins[0] = -np.inf
    residual_bins[-1] = np.inf
    for index in range(1, len(residual_bins) - 1):
        if residual_bins[index] <= residual_bins[index - 1]:
            residual_bins[index] = residual_bins[index - 1] + 1e-6
    real_residual_bin = np.clip(
        np.digitize(real_residual, residual_bins) - 1, 0, 4
    )
    candidate_residual_bin = np.clip(
        np.digitize(candidate_residual, residual_bins) - 1, 0, 4
    )
    joint_bin_count = (len(humidity_bins) - 1) * 5
    real_joint_bin = real_bin * 5 + real_residual_bin
    candidate_joint_bin = candidate_bin * 5 + candidate_residual_bin
    real_counts = np.bincount(real_joint_bin, minlength=joint_bin_count)
    raw_quotas = real_counts / max(real_counts.sum(), 1) * select_count
    quotas = np.floor(raw_quotas).astype(int)
    for index in np.argsort(raw_quotas - quotas)[::-1][:select_count - quotas.sum()]:
        quotas[index] += 1

    selected = []
    selected_set = set()
    for bin_index, quota in enumerate(quotas):
        available = np.where(candidate_joint_bin == bin_index)[0]
        ranked = available[np.argsort(quality_score[available])[::-1]]
        for index in ranked[:quota]:
            selected.append(int(index))
            selected_set.add(int(index))
    if len(selected) < select_count:
        for index in np.argsort(quality_score)[::-1]:
            if int(index) not in selected_set:
                selected.append(int(index))
                selected_set.add(int(index))
            if len(selected) == select_count:
                break
    selected = np.asarray(selected[:select_count], dtype=np.int64)
    selected_scores = quality_score[selected]
    low = float(np.quantile(selected_scores, 0.10))
    high = float(np.quantile(selected_scores, 0.90))
    normalized = np.clip(
        (selected_scores - low) / max(high - low, 1e-6), 0.0, 1.0
    )
    quality_weights = minimum_quality + (1.0 - minimum_quality) * normalized

    selected_metadata = []
    for output_index, (source_index, quality, score) in enumerate(zip(
        selected, quality_weights, selected_scores
    )):
        item = dict(candidate_metadata[source_index])
        item.update({
            "SyntheticID": "synthetic_high_{:05d}".format(output_index),
            "SyntheticQuality": float(quality),
            "SyntheticQualityScore": float(score),
            "NearestRealIndex": int(nearest_real[source_index]),
        })
        selected_metadata.append(item)

    selected_bin = candidate_bin[selected]
    selected_joint_bin = candidate_joint_bin[selected]
    real_humidity_counts = np.bincount(
        real_bin, minlength=len(humidity_bins) - 1
    )
    diagnostics = {
        "candidate_count": int(len(candidate_vectors)),
        "selected_count": int(len(selected)),
        "minimum_quality_weight": float(minimum_quality),
        "quality_score": {
            "candidate_mean": float(np.mean(quality_score)),
            "selected_mean": float(np.mean(selected_scores)),
            "selected_min": float(np.min(selected_scores)),
            "selected_max": float(np.max(selected_scores)),
        },
        "training_quality_weight": {
            "mean": float(np.mean(quality_weights)),
            "min": float(np.min(quality_weights)),
            "max": float(np.max(quality_weights)),
        },
        "manifold_distance_ratio": {
            "candidate_median": float(np.median(distance_ratio)),
            "selected_median": float(np.median(distance_ratio[selected])),
        },
        "humidity_bin_edges": humidity_bins.tolist(),
        "real_bin_counts": real_humidity_counts.tolist(),
        "selected_bin_counts": np.bincount(
            selected_bin, minlength=len(humidity_bins) - 1
        ).tolist(),
        "residual_bin_edges": residual_bins.tolist(),
        "real_joint_bin_counts": real_counts.tolist(),
        "selected_joint_bin_counts": np.bincount(
            selected_joint_bin, minlength=joint_bin_count
        ).tolist(),
    }
    return (
        candidate_x[selected],
        candidate_y[selected],
        selected_metadata,
        candidate_vectors[selected],
        diagnostics,
    )


def scale_x(x, scaler):
    return scaler.transform(x.reshape(-1, x.shape[-1])).reshape(x.shape).astype(np.float32)


def tail_risk_multipliers(targets, metadata, alpha=0.9, strength=0.0):
    """Upweight large very-humid residuals using a real-training threshold."""
    if not 0.0 < alpha < 1.0:
        raise ValueError("tail risk alpha must be in (0, 1)")
    if strength < 0.0:
        raise ValueError("tail risk strength must be non-negative")
    if len(targets) != len(metadata):
        raise ValueError("targets and metadata must have equal length")
    residuals = np.abs(np.asarray(targets, dtype=np.float64) - np.asarray([
        item["Persistence"] for item in metadata
    ], dtype=np.float64))
    real_high_mask = np.asarray([
        not item.get("Synthetic", False)
        and stage_name(item["Hum"]) == "very_humid"
        for item in metadata
    ])
    if real_high_mask.sum() == 0:
        raise ValueError("No real very-humid samples for tail risk threshold")
    threshold = float(np.quantile(residuals[real_high_mask], alpha))
    eligible = np.asarray([
        stage_name(item["Hum"]) == "very_humid" for item in metadata
    ])
    tail_mask = eligible & (residuals >= threshold)
    multipliers = np.ones(len(metadata), dtype=np.float32)
    multipliers[tail_mask] += float(strength)
    synthetic_mask = np.asarray([
        item.get("Synthetic", False) for item in metadata
    ])
    diagnostics = {
        "alpha": float(alpha),
        "strength": float(strength),
        "multiplier_on_tail": float(1.0 + strength),
        "threshold_absolute_persistence_residual": threshold,
        "real_very_humid_count": int(real_high_mask.sum()),
        "real_tail_count": int((tail_mask & ~synthetic_mask).sum()),
        "synthetic_tail_count": int((tail_mask & synthetic_mask).sum()),
        "weighted_sample_count": int(tail_mask.sum()),
        "threshold_source": "real downstream training very-humid samples only",
    }
    return multipliers, diagnostics


def train_augmented_expert(
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
    device_to_index,
    baseline_margins,
    device,
    max_epochs,
    synthetic_loss_weight,
    seed,
    tail_risk_alpha=0.9,
    tail_risk_strength=0.0,
):
    train_persistence = np.asarray([item["Persistence"] for item in train_metadata])
    validation_persistence = np.asarray([
        item["Persistence"] for item in validation_metadata
    ])
    test_persistence = np.asarray([item["Persistence"] for item in test_metadata])
    train_y_scaled = scaler_y.transform(
        (y_train - train_persistence).reshape(-1, 1)
    ).reshape(-1).astype(np.float32)
    validation_y_scaled = scaler_y.transform(
        (y_validation - validation_persistence).reshape(-1, 1)
    ).reshape(-1).astype(np.float32)
    tail_multipliers, tail_risk = tail_risk_multipliers(
        y_train,
        train_metadata,
        alpha=tail_risk_alpha,
        strength=tail_risk_strength,
    )
    train_weights = downstream_sample_weights(
        train_metadata, synthetic_loss_weight
    ) * tail_multipliers
    train_tensors = [
        torch.from_numpy(scale_x(x_train, scaler_x)).float(),
        torch.from_numpy(train_y_scaled).float(),
        torch.from_numpy(routing_features(train_metadata)).float(),
        torch.from_numpy(train_weights).float(),
        torch.from_numpy(metadata_device_indices(
            train_metadata, device_to_index
        )).long(),
        torch.from_numpy(known_target_covariates(train_metadata)).float(),
    ]
    validation_tensors = [
        torch.from_numpy(scale_x(x_validation, scaler_x)).float(),
        torch.from_numpy(validation_y_scaled).float(),
        torch.from_numpy(routing_features(validation_metadata)).float(),
        torch.from_numpy(stage_sample_weights(validation_metadata)).float(),
        torch.from_numpy(metadata_device_indices(
            validation_metadata, device_to_index
        )).long(),
        torch.from_numpy(known_target_covariates(validation_metadata)).float(),
    ]
    set_seed(seed)
    model = HumidityStageFMLP(**model_config).to(device)
    model, best_loss, history = train_model(
        model,
        DataLoader(TensorDataset(*train_tensors), batch_size=32, shuffle=True),
        DataLoader(TensorDataset(*validation_tensors), batch_size=64, shuffle=False),
        device,
        max_epochs=max_epochs,
    )
    validation_quantiles, _, _ = predict_quantiles(
        model,
        scale_x(x_validation, scaler_x),
        routing_features(validation_metadata),
        device,
        scaler_y,
        validation_persistence,
        device_indices=metadata_device_indices(validation_metadata, device_to_index),
        known_covariates=known_target_covariates(validation_metadata),
    )
    validation_margins = calibration_margins(
        y_validation, validation_quantiles, validation_metadata, coverage=0.8
    )
    validation_metrics = {
        "fmlp": regression_metrics(y_validation, validation_quantiles[:, 1]),
        "interval": interval_metrics(y_validation, validation_quantiles),
        "very_humid": None,
    }
    validation_high_mask = np.asarray([
        stage_name(item["Hum"]) == "very_humid"
        for item in validation_metadata
    ])
    if validation_high_mask.any():
        validation_metrics["very_humid"] = {
            "fmlp": regression_metrics(
                y_validation[validation_high_mask],
                validation_quantiles[validation_high_mask, 1],
            ),
            "interval": interval_metrics(
                y_validation[validation_high_mask],
                validation_quantiles[validation_high_mask],
            ),
        }
    margins = {
        name: max(
            float(validation_margins[name]),
            float(baseline_margins.get(name, validation_margins[name])),
        )
        for name in validation_margins
    }
    test_quantiles, test_gates, test_priors = predict_quantiles(
        model,
        scale_x(x_test, scaler_x),
        routing_features(test_metadata),
        device,
        scaler_y,
        test_persistence,
        device_indices=metadata_device_indices(test_metadata, device_to_index),
        known_covariates=known_target_covariates(test_metadata),
    )
    test_quantiles = apply_calibration(test_quantiles, test_metadata, margins)
    metrics = {
        "train_sequences": int(len(x_train)),
        "best_validation_loss_scaled": float(best_loss),
        "validation_conformal_margins": validation_margins,
        "conformal_margins": margins,
        "calibration_policy": (
            "stagewise maximum of candidate validation margin and "
            "real-only baseline margin"
        ),
        "tail_risk_weighting": tail_risk,
        "real_validation_before_conformal_calibration": validation_metrics,
        "overall": {
            "fmlp": regression_metrics(y_test, test_quantiles[:, 1]),
            "persistence": regression_metrics(y_test, test_persistence),
            "interval": interval_metrics(y_test, test_quantiles),
        },
        "by_humidity_stage": stage_metrics(
            y_test, test_quantiles, test_persistence, test_gates, test_metadata
        ),
    }
    return {
        "model": model,
        "history": history,
        "metrics": metrics,
        "quantiles": test_quantiles,
        "gates": test_gates,
        "priors": test_priors,
        "margins": margins,
    }


def comparison_metrics(baseline, augmented):
    output = {"overall": {}, "by_humidity_stage": {}}
    for metric in ("rmse", "mae", "r2"):
        old = baseline["overall"]["fmlp"][metric]
        new = augmented["overall"]["fmlp"][metric]
        output["overall"][metric] = {
            "baseline": old, "augmented": new, "delta": new - old
        }
    for stage in STAGE_NAMES:
        output["by_humidity_stage"][stage] = {}
        for metric in ("rmse", "mae", "r2"):
            old = baseline["by_humidity_stage"][stage]["fmlp"][metric]
            new = augmented["by_humidity_stage"][stage]["fmlp"][metric]
            output["by_humidity_stage"][stage][metric] = {
                "baseline": old, "augmented": new, "delta": new - old
            }
    return output


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
                "Persistence": float(item["Persistence"]),
                "GateWeights": {
                    stage: float(run["gates"][index, stage_index])
                    for stage_index, stage in enumerate(STAGE_NAMES)
                },
                "StagePriors": {
                    stage: float(run["priors"][index, stage_index])
                    for stage_index, stage in enumerate(STAGE_NAMES)
                },
            })
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def save_synthetic_csv(path, windows, targets, metadata, columns):
    rows = []
    for sample_index, item in enumerate(metadata):
        for step in range(windows.shape[1]):
            rows.append({
                "SyntheticID": item["SyntheticID"],
                "Step": step,
                "IsTarget": False,
                "ConditionIMEI": item["IMEI"],
                "Measure": float(windows[sample_index, step, columns.index("Measure")]),
                "Temp": float(windows[sample_index, step, columns.index("Temp")]),
                "Hum": float(windows[sample_index, step, columns.index("Hum")]),
                "DeltaHours": float(windows[sample_index, step, columns.index("DeltaHours")]),
            })
        rows.append({
            "SyntheticID": item["SyntheticID"],
            "Step": windows.shape[1],
            "IsTarget": True,
            "ConditionIMEI": item["IMEI"],
            "Measure": float(targets[sample_index]),
            "Temp": float(item["Temp"]),
            "Hum": float(item["Hum"]),
            "DeltaHours": float(item["ForecastDeltaHours"]),
        })
    pd.DataFrame(rows).to_csv(path, index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--training", required=True)
    parser.add_argument("--test", required=True)
    parser.add_argument("--baseline-high-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--synthetic-ratio", type=float, default=0.25)
    parser.add_argument("--synthetic-loss-weight", type=float, default=0.5)
    parser.add_argument("--moment-match-strength", type=float, default=0.75)
    parser.add_argument("--quality-candidate-multiplier", type=float, default=1.0)
    parser.add_argument("--minimum-quality-weight", type=float, default=1.0)
    parser.add_argument("--tail-risk-alpha", type=float, default=0.9)
    parser.add_argument("--tail-risk-strength", type=float, default=0.0)
    parser.add_argument("--ddpm-checkpoint")
    parser.add_argument("--diffusion-epochs", type=int, default=100)
    parser.add_argument("--diffusion-steps", type=int, default=50)
    parser.add_argument("--diffusion-hidden-dim", type=int, default=192)
    parser.add_argument("--diffusion-depth", type=int, default=4)
    parser.add_argument("--fmlp-epochs", type=int, default=180)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.synthetic_ratio <= 0.0:
        raise ValueError("synthetic-ratio must be positive")
    if not 0.0 < args.synthetic_loss_weight <= 1.0:
        raise ValueError("synthetic-loss-weight must be in (0, 1]")
    if not 0.0 <= args.moment_match_strength <= 1.0:
        raise ValueError("moment-match-strength must be in [0, 1]")
    if args.quality_candidate_multiplier < 1.0:
        raise ValueError("quality-candidate-multiplier must be at least 1")
    if not 0.0 < args.minimum_quality_weight <= 1.0:
        raise ValueError("minimum-quality-weight must be in (0, 1]")
    if not 0.0 < args.tail_risk_alpha < 1.0:
        raise ValueError("tail-risk-alpha must be in (0, 1)")
    if args.tail_risk_strength < 0.0:
        raise ValueError("tail-risk-strength must be non-negative")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    baseline_dir = Path(args.baseline_high_dir)
    checkpoint = torch.load(
        str(baseline_dir / "humidity_stage_fmlp.pth"), map_location="cpu"
    )
    baseline_metrics = json.loads(
        (baseline_dir / "metrics.json").read_text(encoding="utf-8")
    )
    columns = list(checkpoint["feature_columns"])
    if "Battery" in columns:
        raise ValueError("Baseline high expert unexpectedly contains Battery")
    preprocessing = checkpoint.get("preprocessing", {})

    training_raw = pd.read_csv(args.training, dtype={"IMEI": str})
    test_raw = pd.read_csv(args.test, dtype={"IMEI": str})
    device_ids = sorted(training_raw["IMEI"].astype(str).unique().tolist())
    device_to_index = {device_id: index for index, device_id in enumerate(device_ids)}
    location_map = (
        pd.concat([training_raw, test_raw], ignore_index=True)
        .drop_duplicates("IMEI")
        .set_index("IMEI")["Location"]
        .astype(str)
        .to_dict()
    )
    training_raw["_split"] = "train"
    test_raw["_split"] = "test"
    training = prepare_rows_v2(
        training_raw,
        humidity_fill_value=preprocessing.get("humidity_fill_value"),
        median_delta_hours=preprocessing.get("median_delta_hours"),
        device_start_times=preprocessing.get("device_start_times") or None,
    )
    combined = prepare_rows_v2(
        pd.concat([training_raw, test_raw], ignore_index=True),
        humidity_fill_value=preprocessing.get("humidity_fill_value"),
        median_delta_hours=preprocessing.get("median_delta_hours"),
        device_start_times=preprocessing.get("device_start_times") or None,
    )
    seq_len = int(checkpoint["model_config"]["seq_len"])
    all_x, all_y, all_metadata = build_all_windows(training, columns, seq_len)
    train_indices, validation_indices = metadata_split_indices(all_metadata, 0.15)
    x_train, y_train = all_x[train_indices], all_y[train_indices]
    train_metadata = [all_metadata[index] for index in train_indices]
    x_validation, y_validation = all_x[validation_indices], all_y[validation_indices]
    validation_metadata = [all_metadata[index] for index in validation_indices]
    x_test, y_test, test_metadata = build_test_windows(combined, columns, seq_len)

    scaler_x = StandardScaler().fit(x_train.reshape(-1, len(columns)))
    train_persistence = np.asarray([item["Persistence"] for item in train_metadata])
    scaler_y = StandardScaler().fit((y_train - train_persistence).reshape(-1, 1))
    if not np.allclose(scaler_x.mean_, checkpoint["x_mean"], atol=1e-6):
        raise AssertionError("Reconstructed feature scaler differs from baseline")
    if not np.allclose(scaler_y.mean_, checkpoint["y_mean"], atol=1e-6):
        raise AssertionError("Reconstructed target scaler differs from baseline")

    generation_vectors = pack_generation_vectors(
        x_train, y_train, train_metadata, columns
    )
    generation_scaler, generation_scaled, lower, upper = fit_generation_scaler(
        generation_vectors
    )
    stage_indices, generation_device_indices = condition_indices(
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed + 101)
    ddpm = ConditionalDDPM(**ddpm_config).to(device)
    print(
        "Battery-free DDPM: train={} val={} test={} features={} device={}".format(
            len(x_train), len(x_validation), len(x_test), len(columns), device
        ),
        flush=True,
    )
    if args.ddpm_checkpoint:
        ddpm_checkpoint_path = Path(args.ddpm_checkpoint)
        saved_ddpm = torch.load(str(ddpm_checkpoint_path), map_location="cpu")
        saved_config = saved_ddpm["model_config"]
        if saved_config != ddpm_config:
            raise ValueError("Reused DDPM configuration differs from this run")
        if list(saved_ddpm.get("device_ids", [])) != device_ids:
            raise ValueError("Reused DDPM device mapping differs from this run")
        if tuple(saved_ddpm.get("generation_columns", ())) != tuple(
            GENERATED_HISTORY_COLUMNS
        ):
            raise ValueError("Reused DDPM generation columns differ from this run")
        if not np.allclose(saved_ddpm["generation_mean"], generation_scaler.mean_):
            raise ValueError("Reused DDPM generation mean differs from this run")
        if not np.allclose(saved_ddpm["generation_scale"], generation_scaler.scale_):
            raise ValueError("Reused DDPM generation scale differs from this run")
        ddpm.load_state_dict(saved_ddpm["state_dict"])
        diffusion_history = [{
            "reused_checkpoint": str(ddpm_checkpoint_path.resolve())
        }]
    else:
        diffusion_history = train_diffusion(
            ddpm,
            generation_scaled,
            stage_indices,
            generation_device_indices,
            device,
            max_epochs=args.diffusion_epochs,
            batch_size=args.batch_size,
        )
    high_mask = np.asarray([
        stage_name(item["Hum"]) == "very_humid" for item in train_metadata
    ])
    high_count = int(high_mask.sum())
    synthetic_count = max(1, int(round(high_count * args.synthetic_ratio)))
    candidate_count = max(
        synthetic_count,
        int(np.ceil(synthetic_count * args.quality_candidate_multiplier)),
    )
    set_seed(args.seed + 202)
    synthetic_x, synthetic_y, synthetic_metadata, synthetic_vectors, sampling = (
        generate_very_humid_windows(
            ddpm,
            generation_scaler,
            lower,
            upper,
            x_train,
            y_train,
            train_metadata,
            location_map,
            columns,
            device_to_index,
            candidate_count,
            seq_len,
            args.seed + 202,
            args.moment_match_strength,
            preprocessing.get("humidity_fill_value"),
            preprocessing.get("median_delta_hours"),
        )
    )
    real_high_vectors = generation_vectors[high_mask]
    candidate_quality = generation_quality_metrics(
        real_high_vectors, synthetic_vectors, generation_scaler, seq_len
    )
    selection = None
    if candidate_count > synthetic_count or args.minimum_quality_weight < 1.0:
        (
            synthetic_x,
            synthetic_y,
            synthetic_metadata,
            synthetic_vectors,
            selection,
        ) = select_quality_synthetic(
            real_high_vectors,
            synthetic_x,
            synthetic_y,
            synthetic_metadata,
            synthetic_vectors,
            generation_scaler,
            seq_len,
            synthetic_count,
            args.minimum_quality_weight,
        )
    quality = generation_quality_metrics(
        real_high_vectors, synthetic_vectors, generation_scaler, seq_len
    )
    quality["sampling"] = sampling
    quality["candidate_pool"] = candidate_quality
    quality["selection"] = selection

    augmented_x = np.concatenate([x_train, synthetic_x], axis=0)
    augmented_y = np.concatenate([y_train, synthetic_y], axis=0)
    augmented_metadata = train_metadata + synthetic_metadata
    print(
        "Training augmented high expert with {} synthetic windows".format(
            synthetic_count
        ),
        flush=True,
    )
    augmented_run = train_augmented_expert(
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
        checkpoint["model_config"],
        device_to_index,
        checkpoint.get("conformal_margins", {}),
        device,
        args.fmlp_epochs,
        args.synthetic_loss_weight,
        args.seed,
        args.tail_risk_alpha,
        args.tail_risk_strength,
    )
    metrics = {
        "protocol": {
            "training_file": str(Path(args.training).resolve()),
            "test_file": str(Path(args.test).resolve()),
            "baseline_high_dir": str(baseline_dir.resolve()),
            "seq_len": seq_len,
            "real_train_sequences": int(len(x_train)),
            "validation_sequences": int(len(x_validation)),
            "test_sequences": int(len(x_test)),
            "real_very_humid_train_sequences": high_count,
            "synthetic_very_humid_sequences": synthetic_count,
            "synthetic_candidate_sequences": candidate_count,
            "synthetic_ratio": args.synthetic_ratio,
            "synthetic_loss_weight": args.synthetic_loss_weight,
            "moment_match_strength": args.moment_match_strength,
            "quality_candidate_multiplier": args.quality_candidate_multiplier,
            "minimum_quality_weight": args.minimum_quality_weight,
            "tail_risk_alpha": args.tail_risk_alpha,
            "tail_risk_strength": args.tail_risk_strength,
            "ddpm_checkpoint_reused": (
                str(Path(args.ddpm_checkpoint).resolve())
                if args.ddpm_checkpoint else None
            ),
            "generated_history_columns": list(GENERATED_HISTORY_COLUMNS),
            "battery_in_generator": False,
            "battery_in_fmlp": False,
            "generator_training_scope": "real downstream training split only",
            "validation_and_test": "real only",
            "calibration_policy": (
                "stagewise maximum of candidate validation margin and "
                "real-only baseline margin"
            ),
            "seed": args.seed,
            "device": str(device),
        },
        "ddpm_config": ddpm_config,
        "generation_quality": quality,
        "baseline": {
            "overall": baseline_metrics["overall"],
            "by_humidity_stage": baseline_metrics["by_humidity_stage"],
        },
        "augmented": augmented_run["metrics"],
    }
    metrics["comparison"] = comparison_metrics(
        metrics["baseline"], metrics["augmented"]
    )
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "diffusion_history.json").write_text(
        json.dumps(diffusion_history, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "training_history.json").write_text(
        json.dumps(augmented_run["history"], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    save_predictions(
        output_dir / "predictions.jsonl", test_metadata, y_test, augmented_run
    )
    save_synthetic_csv(
        output_dir / "synthetic_training_windows.csv",
        synthetic_x,
        synthetic_y,
        synthetic_metadata,
        columns,
    )
    np.savez_compressed(
        output_dir / "synthetic_training_windows.npz",
        windows=synthetic_x,
        targets=synthetic_y,
        feature_columns=np.asarray(columns),
    )
    torch.save({
        "state_dict": ddpm.state_dict(),
        "model_config": ddpm_config,
        "generation_columns": GENERATED_HISTORY_COLUMNS,
        "generation_mean": generation_scaler.mean_,
        "generation_scale": generation_scaler.scale_,
        "clip_lower": lower,
        "clip_upper": upper,
        "stage_names": STAGE_NAMES,
        "device_ids": device_ids,
        "seq_len": seq_len,
        "seed": args.seed,
        "battery_in_generator": False,
    }, output_dir / "humidity_conditional_ddpm.pth")
    torch.save({
        "state_dict": augmented_run["model"].state_dict(),
        "model_config": checkpoint["model_config"],
        "feature_columns": columns,
        "x_mean": scaler_x.mean_,
        "x_scale": scaler_x.scale_,
        "y_mean": scaler_y.mean_,
        "y_scale": scaler_y.scale_,
        "conformal_margins": augmented_run["margins"],
        "stage_names": STAGE_NAMES,
        "quantiles": HumidityStageFMLP.QUANTILES,
        "feature_set": "v2",
        "v2_profile": "dynamic",
        "routing_mode": "target_humidity",
        "device_to_index": device_to_index,
        "preprocessing": preprocessing,
        "seed": args.seed,
        "battery_in_model": False,
        "diffusion_augmented": True,
        "quality_filtered_diffusion": bool(selection is not None),
        "tail_risk_weighting": augmented_run["metrics"][
            "tail_risk_weighting"
        ],
    }, output_dir / "humidity_stage_fmlp.pth")
    print(json.dumps(metrics["comparison"], ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()


