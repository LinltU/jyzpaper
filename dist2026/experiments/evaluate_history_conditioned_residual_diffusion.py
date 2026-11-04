"""Evaluate CSDI/TimeDiff-style history-conditioned residual augmentation."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from experiments.evaluate_diffusion_augmented_fmlp import condition_indices
from experiments.evaluate_diffusion_augmented_high_expert import (
    comparison_metrics,
    save_predictions,
    train_augmented_expert,
)
from experiments.evaluate_fmlp_fixed_split import (
    build_all_windows,
    build_test_windows,
    set_seed,
)
from experiments.evaluate_humidity_stage_fmlp import (
    STAGE_NAMES,
    known_target_covariates,
    metadata_split_indices,
    stage_name,
)
from core.feature_engineering_v2 import prepare_rows_v2
from model.history_conditioned_residual_ddpm import (
    HistoryConditionedResidualDDPM,
)
from model.humidity_stage_fmlp import HumidityStageFMLP
from core.project_paths import manifest_relative_path


def scaled_history_context(windows, scaler):
    scaled = scaler.transform(windows.reshape(-1, windows.shape[-1]))
    return scaled.reshape(len(windows), -1).astype(np.float32)


def trend_initialized_baseline(
    windows,
    metadata,
    columns,
    trend_strength,
    trend_cap,
):
    if not 0.0 <= trend_strength <= 1.0:
        raise ValueError("trend_strength must be in [0, 1]")
    persistence = np.asarray(
        [item["Persistence"] for item in metadata], dtype=np.float32
    )
    if "MeasureRate" in columns:
        last_rate = windows[:, -1, columns.index("MeasureRate")]
    else:
        delta = np.maximum(
            windows[:, -1, columns.index("DeltaHours")], 1.0 / 60.0
        )
        last_rate = (
            windows[:, -1, columns.index("Measure")]
            - windows[:, -2, columns.index("Measure")]
        ) / delta
    forecast_delta = np.asarray([
        item.get("ForecastDeltaHours", 0.0) for item in metadata
    ], dtype=np.float32)
    trend = np.clip(last_rate * forecast_delta, -trend_cap, trend_cap)
    return persistence + trend_strength * trend


def residual_target_context(metadata, trend_baseline, persistence, trend_cap):
    known = known_target_covariates(metadata)
    normalized_trend = (
        (trend_baseline - persistence) / max(float(trend_cap), 1e-6)
    )[:, None]
    return np.concatenate([known, normalized_trend], axis=1).astype(np.float32)


def train_residual_diffusion(
    model,
    innovations,
    histories,
    target_context,
    stages,
    devices,
    device,
    max_epochs,
    batch_size,
):
    counts = np.bincount(stages, minlength=len(STAGE_NAMES))
    weights = np.asarray([
        1.0 / max(counts[index], 1) for index in stages
    ], dtype=np.float64)
    loader = DataLoader(
        TensorDataset(
            torch.from_numpy(innovations).float(),
            torch.from_numpy(histories).float(),
            torch.from_numpy(target_context).float(),
            torch.from_numpy(stages).long(),
            torch.from_numpy(devices).long(),
        ),
        batch_size=batch_size,
        sampler=WeightedRandomSampler(
            torch.from_numpy(weights), len(weights), replacement=True
        ),
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=8e-4, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(max_epochs, 1), eta_min=1e-5
    )
    history = []
    for epoch in range(1, max_epochs + 1):
        model.train()
        total = 0.0
        for residual, history_x, target_x, stage, device_index in loader:
            optimizer.zero_grad()
            loss = model.loss(
                residual.to(device),
                history_x.to(device),
                target_x.to(device),
                stage.to(device),
                device_index.to(device),
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += loss.item()
        scheduler.step()
        epoch_loss = total / max(len(loader), 1)
        history.append({"epoch": epoch, "noise_mse": epoch_loss})
        if epoch == 1 or epoch % 10 == 0 or epoch == max_epochs:
            print(
                "Residual DDPM epoch {:3d}: noise_mse={:.6f}".format(
                    epoch, epoch_loss
                ),
                flush=True,
            )
    return history


def generate_conditioned_residual_samples(
    model,
    history_context,
    target_context,
    stages,
    devices,
    innovations,
    innovation_scaler,
    windows,
    metadata,
    trend_baseline,
    targets,
    count,
    seed,
    device,
    moment_match_strength,
):
    if not 0.0 <= moment_match_strength <= 1.0:
        raise ValueError("moment_match_strength must be in [0, 1]")
    candidates = np.asarray([
        index for index, item in enumerate(metadata)
        if stage_name(item["Hum"]) == "very_humid"
    ], dtype=np.int64)
    if len(candidates) == 0:
        raise ValueError("No real very-humid training windows")
    rng = np.random.RandomState(seed)
    anchors = rng.choice(
        candidates, size=count, replace=count > len(candidates)
    )
    batches = []
    model.eval()
    for start in range(0, count, 256):
        indices = anchors[start:start + 256]
        batch = model.sample(
            torch.from_numpy(history_context[indices]).float().to(device),
            torch.from_numpy(target_context[indices]).float().to(device),
            torch.from_numpy(stages[indices]).long().to(device),
            torch.from_numpy(devices[indices]).long().to(device),
            clip_denoised=5.0,
        )
        batches.append(batch.cpu().numpy())
    sampled_scaled = np.concatenate(batches, axis=0)
    raw_sampled_innovation = innovation_scaler.inverse_transform(
        sampled_scaled
    ).reshape(-1)
    high_innovation = innovations[candidates].reshape(-1)
    lower, upper = np.quantile(high_innovation, [0.005, 0.995])
    sampled_mean = float(np.mean(raw_sampled_innovation))
    sampled_std = float(np.std(raw_sampled_innovation))
    matched_innovation = (
        (raw_sampled_innovation - sampled_mean)
        / max(sampled_std, 1e-6)
        * float(np.std(high_innovation))
        + float(np.mean(high_innovation))
    )
    sampled_innovation = (
        (1.0 - moment_match_strength) * raw_sampled_innovation
        + moment_match_strength * matched_innovation
    )
    sampled_innovation = np.clip(sampled_innovation, lower, upper)
    high_targets = targets[candidates]
    target_lower, target_upper = np.quantile(high_targets, [0.005, 0.995])
    synthetic_targets = np.clip(
        trend_baseline[anchors] + sampled_innovation,
        target_lower,
        target_upper,
    ).astype(np.float32)
    synthetic_metadata = []
    for output_index, (anchor, innovation) in enumerate(zip(
        anchors, sampled_innovation
    )):
        item = dict(metadata[anchor])
        item.update({
            "Synthetic": True,
            "SyntheticID": "conditioned_residual_{:05d}".format(output_index),
            "SyntheticQuality": 1.0,
            "AnchorIndex": int(anchor),
            "TrendBaseline": float(trend_baseline[anchor]),
            "GeneratedInnovation": float(innovation),
        })
        synthetic_metadata.append(item)
    diagnostics = {
        "selected_anchor_count": int(count),
        "unique_anchor_count": int(len(np.unique(anchors))),
        "history_rows_generated": 0,
        "real_history_reused": True,
        "innovation_moment_match_strength": float(moment_match_strength),
        "real_high_innovation": {
            "mean": float(np.mean(high_innovation)),
            "std": float(np.std(high_innovation)),
            "q005": float(lower),
            "q995": float(upper),
        },
        "synthetic_innovation": {
            "mean": float(np.mean(sampled_innovation)),
            "std": float(np.std(sampled_innovation)),
            "min": float(np.min(sampled_innovation)),
            "max": float(np.max(sampled_innovation)),
        },
        "raw_synthetic_innovation": {
            "mean": float(np.mean(raw_sampled_innovation)),
            "std": float(np.std(raw_sampled_innovation)),
            "min": float(np.min(raw_sampled_innovation)),
            "max": float(np.max(raw_sampled_innovation)),
        },
        "synthetic_target": {
            "mean": float(np.mean(synthetic_targets)),
            "std": float(np.std(synthetic_targets)),
        },
    }
    return (
        windows[anchors].copy(),
        synthetic_targets,
        synthetic_metadata,
        anchors,
        diagnostics,
    )


def save_synthetic_targets(path, synthetic_y, metadata):
    rows = []
    for target, item in zip(synthetic_y, metadata):
        rows.append({
            "SyntheticID": item["SyntheticID"],
            "AnchorIndex": item["AnchorIndex"],
            "IMEI": item["IMEI"],
            "Timestamp": item["Timestamp"],
            "Temp": item["Temp"],
            "Hum": item["Hum"],
            "Persistence": item["Persistence"],
            "TrendBaseline": item["TrendBaseline"],
            "GeneratedInnovation": item["GeneratedInnovation"],
            "SyntheticTarget": float(target),
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
    parser.add_argument("--trend-strength", type=float, default=0.5)
    parser.add_argument(
        "--innovation-moment-match-strength", type=float, default=0.0
    )
    parser.add_argument("--ddpm-checkpoint")
    parser.add_argument("--tail-risk-alpha", type=float, default=0.9)
    parser.add_argument("--tail-risk-strength", type=float, default=0.0)
    parser.add_argument("--diffusion-epochs", type=int, default=80)
    parser.add_argument("--diffusion-steps", type=int, default=30)
    parser.add_argument("--diffusion-hidden-dim", type=int, default=128)
    parser.add_argument("--diffusion-depth", type=int, default=3)
    parser.add_argument("--fmlp-epochs", type=int, default=180)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.synthetic_ratio <= 0.0:
        raise ValueError("synthetic-ratio must be positive")
    if not 0.0 < args.synthetic_loss_weight <= 1.0:
        raise ValueError("synthetic-loss-weight must be in (0, 1]")
    if not 0.0 <= args.trend_strength <= 1.0:
        raise ValueError("trend-strength must be in [0, 1]")
    if not 0.0 <= args.innovation_moment_match_strength <= 1.0:
        raise ValueError("innovation-moment-match-strength must be in [0, 1]")
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
        raise ValueError("Battery must not enter residual diffusion or FMLP")
    preprocessing = checkpoint.get("preprocessing", {})

    training_raw = pd.read_csv(args.training, dtype={"IMEI": str})
    test_raw = pd.read_csv(args.test, dtype={"IMEI": str})
    device_ids = sorted(training_raw["IMEI"].astype(str).unique().tolist())
    device_to_index = {
        device_id: index for index, device_id in enumerate(device_ids)
    }
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
    train_indices, validation_indices = metadata_split_indices(
        all_metadata, 0.15
    )
    x_train, y_train = all_x[train_indices], all_y[train_indices]
    train_metadata = [all_metadata[index] for index in train_indices]
    x_validation, y_validation = (
        all_x[validation_indices], all_y[validation_indices]
    )
    validation_metadata = [
        all_metadata[index] for index in validation_indices
    ]
    x_test, y_test, test_metadata = build_test_windows(
        combined, columns, seq_len
    )

    scaler_x = StandardScaler().fit(x_train.reshape(-1, len(columns)))
    persistence = np.asarray([
        item["Persistence"] for item in train_metadata
    ], dtype=np.float32)
    scaler_y = StandardScaler().fit((y_train - persistence).reshape(-1, 1))
    if not np.allclose(scaler_x.mean_, checkpoint["x_mean"], atol=1e-6):
        raise AssertionError("Reconstructed feature scaler differs from baseline")
    if not np.allclose(scaler_y.mean_, checkpoint["y_mean"], atol=1e-6):
        raise AssertionError("Reconstructed target scaler differs from baseline")

    persistence_residual = y_train - persistence
    trend_cap = float(np.quantile(np.abs(persistence_residual), 0.995))
    trend_baseline = trend_initialized_baseline(
        x_train,
        train_metadata,
        columns,
        args.trend_strength,
        trend_cap,
    )
    innovations = (y_train - trend_baseline).astype(np.float32)
    innovation_scaler = StandardScaler().fit(innovations.reshape(-1, 1))
    innovations_scaled = innovation_scaler.transform(
        innovations.reshape(-1, 1)
    ).astype(np.float32)
    history_context = scaled_history_context(x_train, scaler_x)
    target_context = residual_target_context(
        train_metadata, trend_baseline, persistence, trend_cap
    )
    stages, devices = condition_indices(train_metadata, device_to_index)
    model_config = {
        "history_dim": int(history_context.shape[1]),
        "target_context_dim": int(target_context.shape[1]),
        "num_stages": len(STAGE_NAMES),
        "num_devices": len(device_ids),
        "hidden_dim": args.diffusion_hidden_dim,
        "depth": args.diffusion_depth,
        "dropout": 0.05,
        "diffusion_steps": args.diffusion_steps,
    }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed + 101)
    ddpm = HistoryConditionedResidualDDPM(**model_config).to(device)
    print(
        "History-conditioned residual DDPM: train={} history_dim={} device={}".format(
            len(x_train), history_context.shape[1], device
        ),
        flush=True,
    )
    if args.ddpm_checkpoint:
        saved_ddpm = torch.load(args.ddpm_checkpoint, map_location="cpu")
        if saved_ddpm["model_config"] != model_config:
            raise ValueError("Reused DDPM configuration differs from this run")
        if list(saved_ddpm["history_feature_columns"]) != columns:
            raise ValueError("Reused DDPM history columns differ from this run")
        if list(saved_ddpm["device_ids"]) != device_ids:
            raise ValueError("Reused DDPM device mapping differs from this run")
        if not np.allclose(saved_ddpm["history_mean"], scaler_x.mean_):
            raise ValueError("Reused DDPM history mean differs from this run")
        if not np.allclose(saved_ddpm["history_scale"], scaler_x.scale_):
            raise ValueError("Reused DDPM history scale differs from this run")
        if not np.allclose(
            saved_ddpm["innovation_mean"], innovation_scaler.mean_
        ):
            raise ValueError("Reused DDPM innovation mean differs from this run")
        if not np.allclose(
            saved_ddpm["innovation_scale"], innovation_scaler.scale_
        ):
            raise ValueError("Reused DDPM innovation scale differs from this run")
        ddpm.load_state_dict(saved_ddpm["state_dict"])
        ddpm.to(device)
        diffusion_history = [{
            "reused_checkpoint": str(Path(args.ddpm_checkpoint).resolve())
        }]
    else:
        diffusion_history = train_residual_diffusion(
            ddpm,
            innovations_scaled,
            history_context,
            target_context,
            stages,
            devices,
            device,
            args.diffusion_epochs,
            args.batch_size,
        )

    high_count = sum(
        stage_name(item["Hum"]) == "very_humid" for item in train_metadata
    )
    synthetic_count = max(1, int(round(high_count * args.synthetic_ratio)))
    set_seed(args.seed + 202)
    (
        synthetic_x,
        synthetic_y,
        synthetic_metadata,
        anchor_indices,
        generation_quality,
    ) = generate_conditioned_residual_samples(
        ddpm,
        history_context,
        target_context,
        stages,
        devices,
        innovations,
        innovation_scaler,
        x_train,
        train_metadata,
        trend_baseline,
        y_train,
        synthetic_count,
        args.seed + 202,
        device,
        args.innovation_moment_match_strength,
    )
    augmented_run = train_augmented_expert(
        np.concatenate([x_train, synthetic_x], axis=0),
        np.concatenate([y_train, synthetic_y], axis=0),
        train_metadata + synthetic_metadata,
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
            "real_train_sequences": int(len(x_train)),
            "validation_sequences": int(len(x_validation)),
            "test_sequences": int(len(x_test)),
            "real_very_humid_train_sequences": int(high_count),
            "synthetic_very_humid_sequences": int(synthetic_count),
            "synthetic_ratio": args.synthetic_ratio,
            "synthetic_loss_weight": args.synthetic_loss_weight,
            "trend_strength": args.trend_strength,
            "trend_cap": trend_cap,
            "innovation_moment_match_strength": (
                args.innovation_moment_match_strength
            ),
            "ddpm_checkpoint_reused": (
                str(Path(args.ddpm_checkpoint).resolve())
                if args.ddpm_checkpoint else None
            ),
            "tail_risk_alpha": args.tail_risk_alpha,
            "tail_risk_strength": args.tail_risk_strength,
            "history_generated": False,
            "residual_only_diffusion": True,
            "battery_in_generator": False,
            "battery_in_fmlp": False,
            "generator_training_scope": "real downstream training split only",
            "validation_and_test": "real only",
            "test_labels_used_for_generation": False,
            "seed": args.seed,
            "device": str(device),
        },
        "ddpm_config": model_config,
        "generation_quality": generation_quality,
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
        json.dumps(diffusion_history, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "training_history.json").write_text(
        json.dumps(augmented_run["history"], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    save_predictions(
        output_dir / "predictions.jsonl", test_metadata, y_test, augmented_run
    )
    save_synthetic_targets(
        output_dir / "synthetic_residual_targets.csv",
        synthetic_y,
        synthetic_metadata,
    )
    np.savez_compressed(
        output_dir / "synthetic_residual_targets.npz",
        windows=synthetic_x,
        targets=synthetic_y,
        anchor_indices=anchor_indices,
        feature_columns=np.asarray(columns),
    )
    torch.save({
        "state_dict": ddpm.state_dict(),
        "model_config": model_config,
        "history_feature_columns": columns,
        "history_mean": scaler_x.mean_,
        "history_scale": scaler_x.scale_,
        "innovation_mean": innovation_scaler.mean_,
        "innovation_scale": innovation_scaler.scale_,
        "trend_strength": args.trend_strength,
        "trend_cap": trend_cap,
        "device_ids": device_ids,
        "stage_names": STAGE_NAMES,
        "seed": args.seed,
        "battery_in_generator": False,
    }, output_dir / "history_conditioned_residual_ddpm.pth")
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
        "augmentation_type": "history_conditioned_residual_ddpm",
        "tail_risk_weighting": augmented_run["metrics"][
            "tail_risk_weighting"
        ],
    }, output_dir / "humidity_stage_fmlp.pth")

    base_checkpoint = (
        baseline_dir.parent
        / "clean_elapsed"
        / "humidity_stage_fmlp.pth"
    )
    if base_checkpoint.exists():
        manifest_path = output_dir / "ensemble_manifest.json"
        manifest = {
            "base_checkpoint": manifest_relative_path(
                base_checkpoint, manifest_path
            ),
            "high_humidity_checkpoint": manifest_relative_path(
                output_dir / "humidity_stage_fmlp.pth", manifest_path
            ),
            "humidity_threshold": 80.0,
        }
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    print(json.dumps(metrics["comparison"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()


