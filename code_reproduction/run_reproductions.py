"""Run the local reproductions against the JYZmentor data.

The repositories under ``upstream`` are preserved verbatim at pinned commits.
Their public loaders target ETT, Weather, traffic, SWaT, NASA and clinical
datasets, so this runner supplies a causal adapter for the local 25-device
insulator dataset.  Each experiment records whether the official package was
also importable and whether the result is a paper-core engineering
reproduction rather than a layer-for-layer reimplementation.

Default settings are intentionally CPU friendly.  Increase epochs, diffusion
steps and samples for a final GPU run after reviewing the generated JSON.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import pickle
import subprocess
import sys
import time
import traceback
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
ADAPTERS = ROOT / "adapters"
if str(ADAPTERS) not in sys.path:
    sys.path.insert(0, str(ADAPTERS))

from data_adapter import (  # noqa: E402
    STAGES,
    _key_series,
    build_forecast_windows,
    device_indices,
    fit_scaler,
    inverse_target,
    make_tabular_features,
    prepare_data,
    read_data,
    split_train_validation,
    stage_indices,
    transform_windows,
)
from models import (  # noqa: E402
    ConditionalResidualDiffusion,
    ImputationAdapter,
    run_treeffuser,
    seed_everything,
)
from individual_paper_branches.models import BRANCHES  # noqa: E402


PREDICTION_METHODS = [
    {
        "name": "TimeGrad",
        "repository": "TimeGrad-pytorch-ts",
        "paper_task": "probabilistic forecasting",
        "mode": "persistence",
        "core": "autoregressive conditional residual DDPM",
    },
    {
        "name": "DiffPLF",
        "repository": "DiffPLF",
        "paper_task": "probabilistic forecasting",
        "mode": "trend",
        "core": "conditional diffusion with local residual scale",
    },
    {
        "name": "CCDM",
        "repository": "CCDM",
        "paper_task": "multivariate probabilistic forecasting",
        "mode": "channel",
        "core": "channel-aware conditioning / contrastive weighting",
    },
    {
        "name": "TSDE",
        "repository": "TSDE",
        "paper_task": "self-supervised imputation/interpolation/forecasting",
        "mode": "trend",
        "core": "trend-aware conditional diffusion representation",
    },
    {
        "name": "NsDiff",
        "repository": "NsDiff",
        "paper_task": "non-stationary probabilistic forecasting",
        "mode": "nonstationary",
        "core": "non-stationary local residual normalization",
    },
    {
        "name": "Diffusion-TS",
        "repository": "Diffusion-TS",
        "paper_task": "interpretable time-series generation / forecasting",
        "mode": "state_space",
        "core": "trend/state summary conditioned diffusion",
    },
    {
        "name": "Treeffuser",
        "repository": "Treeffuser",
        "paper_task": "tabular conditional probabilistic prediction",
        "mode": "tree",
        "core": "LightGBM score model + diffusion sampler",
    },
]

IMPUTATION_METHODS = [
    {
        "name": "CSDI",
        "repository": "CSDI",
        "mask": "random",
        "architecture": "mlp",
        "consistency": 0.0,
        "smoothness": 0.0,
    },
    {
        "name": "MTSCI",
        "repository": "MTSCI",
        "mask": "random",
        "architecture": "mlp",
        "consistency": 0.10,
        "smoothness": 0.0,
    },
    {
        "name": "TSDE-Imputation",
        "repository": "TSDE",
        "mask": "iif",
        "architecture": "gru",
        "consistency": 0.05,
        "smoothness": 0.01,
    },
    {
        "name": "SSSD",
        "repository": "SSSD",
        "mask": "block",
        "architecture": "conv",
        "consistency": 0.05,
        "smoothness": 0.02,
    },
    {
        "name": "SSD-TS",
        "repository": "SSD-TS",
        "mask": "block",
        "architecture": "gru",
        "consistency": 0.05,
        "smoothness": 0.01,
    },
    {
        "name": "CoSTI",
        "repository": "CoSTI",
        "mask": "random",
        "architecture": "conv",
        "consistency": 0.15,
        "smoothness": 0.02,
    },
]

ANOMALY_METHODS = [
    {
        "name": "DiffAD",
        "repository": "DiffAD",
        "score": "weighted_incremental",
        "architecture": "mlp",
        "mask": "random",
    },
    {
        "name": "D3R",
        "repository": "D3R",
        "score": "detrended",
        "architecture": "conv",
        "mask": "block",
    },
    {
        "name": "TIMEADDM",
        "repository": "TIMEADDM",
        "score": "recurrent",
        "architecture": "gru",
        "mask": "iif",
    },
    {
        "name": "IMDiffusion",
        "repository": "IMDiffusion",
        "score": "imputation_ensemble",
        "architecture": "mlp",
        "mask": "random",
    },
    {
        "name": "ProDiffAD",
        "repository": "ProDiffAD",
        "score": "distilled",
        "architecture": "conv",
        "mask": "random",
    },
]


def jsonable(value):
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(value), ensure_ascii=False, indent=2), encoding="utf-8")


def regression_metrics(y: np.ndarray, pred: np.ndarray, lo=None, hi=None) -> dict:
    y = np.asarray(y, dtype=float)
    pred = np.asarray(pred, dtype=float)
    error = pred - y
    result = {
        "count": int(y.size),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mae": float(np.mean(np.abs(error))),
        "r2": float(1.0 - np.sum(error**2) / max(np.sum((y - y.mean()) ** 2), 1e-12)),
        "bias": float(np.mean(error)),
        "median_absolute_error": float(np.median(np.abs(error))),
    }
    if lo is not None and hi is not None:
        result["coverage_p10_p90"] = float(np.mean((y >= lo) & (y <= hi)))
        result["mean_interval_width"] = float(np.mean(hi - lo))
        result["median_interval_width"] = float(np.median(hi - lo))
    return result


def metrics_by_stage(y, pred, lo, hi, metadata) -> dict:
    stages = np.asarray([item["stage"] for item in metadata])
    return {
        stage: regression_metrics(
            y[stages == stage], pred[stages == stage], lo[stages == stage], hi[stages == stage]
        )
        for stage in STAGES
        if np.any(stages == stage)
    }


def save_prediction_table(path: Path, metadata, target, p10, p50, p90, method) -> None:
    rows = []
    for index, item in enumerate(metadata):
        row = dict(item)
        row.update(
            {
                "method": method,
                "target_Measure": float(target[index]),
                "P10": float(p10[index]),
                "P50": float(p50[index]),
                "P90": float(p90[index]),
                "absolute_error": float(abs(p50[index] - target[index])),
                "out_of_interval": bool(target[index] < p10[index] or target[index] > p90[index]),
            }
        )
        rows.append(row)
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")


def official_smoke() -> dict:
    """Compile each pinned source tree and perform lightweight imports."""
    results = {}
    runtime = sys.executable
    smoke_env = dict(os.environ)
    smoke_env["PYTHONPYCACHEPREFIX"] = str(ROOT / "local_runs" / ".pycache_smoke")
    for method in PREDICTION_METHODS + IMPUTATION_METHODS + ANOMALY_METHODS:
        repo = method["repository"]
        if repo in results:
            continue
        source = ROOT / "upstream" / repo
        started = time.time()
        record = {"repository": repo, "source": str(source), "compileall": False, "import": None}
        try:
            compile_result = subprocess.run(
                [runtime, "-m", "compileall", "-q", str(source)],
                capture_output=True,
                text=True,
                timeout=120,
                env=smoke_env,
            )
            record["compileall"] = compile_result.returncode == 0
            if compile_result.returncode != 0:
                record["compile_error"] = (compile_result.stdout + compile_result.stderr)[-2000:]
        except Exception as exc:  # pragma: no cover - diagnostic path
            record["compile_error"] = repr(exc)
        # Imports are intentionally limited to package-level entry points. A
        # failed import usually means a paper-specific optional dependency.
        import_targets = {
            "Treeffuser": (ROOT / "upstream" / "Treeffuser" / "src", "treeffuser"),
            "TimeGrad-pytorch-ts": (ROOT / "upstream" / "TimeGrad-pytorch-ts", "pts"),
        }
        if repo in import_targets:
            path, module = import_targets[repo]
            code = "import sys; sys.path.insert(0, r'%s'); import %s" % (path, module)
            try:
                imp = subprocess.run(
                    [runtime, "-c", code], capture_output=True, text=True, timeout=120
                    , env=smoke_env
                )
                record["import"] = imp.returncode == 0
                if imp.returncode != 0:
                    record["import_error"] = (imp.stdout + imp.stderr)[-2000:]
            except Exception as exc:  # pragma: no cover
                record["import"] = False
                record["import_error"] = repr(exc)
        else:
            record["import"] = "not attempted: repository has dataset-specific entry points"
        record["elapsed_seconds"] = round(time.time() - started, 3)
        results[repo] = record
    return results


def prepare_windows(seq_len: int):
    train_raw, test_raw, combined = read_data()
    prepared = prepare_data(combined)
    train_windows = build_forecast_windows(prepared, _key_series(train_raw), seq_len=seq_len)
    test_windows = build_forecast_windows(prepared, _key_series(test_raw), seq_len=seq_len)
    fit_indices, val_indices = split_train_validation(train_windows, validation_ratio=0.15)
    scaler = fit_scaler(train_windows, fit_indices)
    train_scaled = transform_windows(train_windows, scaler)
    test_scaled = transform_windows(test_windows, scaler)
    train_meta = train_scaled["metadata"]
    test_meta = test_scaled["metadata"]
    fit_meta = [train_meta[index] for index in fit_indices]
    fit_device, device_map = device_indices(fit_meta, train_meta)
    val_device, _ = device_indices([train_meta[index] for index in val_indices], train_meta)
    test_device, _ = device_indices(test_meta, train_meta)
    return {
        "train_raw": train_raw,
        "test_raw": test_raw,
        "prepared": prepared,
        "train": train_scaled,
        "test": test_scaled,
        "fit_indices": fit_indices,
        "val_indices": val_indices,
        "scaler": scaler,
        "fit_meta": fit_meta,
        "train_meta": train_meta,
        "test_meta": test_meta,
        "fit_stage": stage_indices(fit_meta),
        "val_stage": stage_indices([train_meta[index] for index in val_indices]),
        "test_stage": stage_indices(test_meta),
        "fit_device": fit_device,
        "val_device": val_device,
        "test_device": test_device,
        "device_map": device_map,
    }


def run_prediction_method(method, data, output_dir: Path, args) -> dict:
    name = method["name"]
    started = time.time()
    out = output_dir / "prediction" / name
    out.mkdir(parents=True, exist_ok=True)
    train = data["train"]
    test = data["test"]
    fit = data["fit_indices"]
    val = data["val_indices"]
    n_val = len(val)
    if name in BRANCHES:
        # Independent paper branch: distinct architecture and training loop.
        # The surrounding calibration/evaluation protocol remains identical.
        branch = BRANCHES[name](
            history_dim=train["x"].shape[2],
            context_dim=train["context"].shape[1],
            hidden=args.hidden_dim,
            steps=args.diffusion_steps,
            seed=args.seed,
        )
        history = branch.fit(
            train["x"][fit], train["context"][fit], train["target"][fit],
            train["persistence"][fit], epochs=args.epochs,
            batch_size=args.batch_size, lr=args.learning_rate,
        )
        write_json(out / "training_history.json", history)
        import torch

        branch_state = {}
        for attr_name, attr_value in vars(branch).items():
            if hasattr(attr_value, "state_dict"):
                branch_state[attr_name] = attr_value.state_dict()
        torch.save(
            {
                "state_dict": branch_state,
                "branch": name,
                "architecture": "independent_paper_branch",
                "history_dim": int(train["x"].shape[2]),
                "context_dim": int(train["context"].shape[1]),
                "diffusion_steps": int(args.diffusion_steps),
            },
            out / "model.pth",
        )
        samples_scaled = branch.sample(
            np.concatenate([train["x"][val], test["x"]], axis=0),
            np.concatenate([train["context"][val], test["context"]], axis=0),
            np.concatenate([train["persistence"][val], test["persistence"]], axis=0),
            n_samples=args.samples,
        )
        official = False
    elif method["mode"] == "tree":
        X_train = make_tabular_features(train, use_history=6, device_map=data["device_map"])[fit]
        X_val = make_tabular_features(train, use_history=6, device_map=data["device_map"])[val]
        X_test = make_tabular_features(test, use_history=6, device_map=data["device_map"])
        X_eval = np.concatenate([X_val, X_test], axis=0)
        try:
            samples_scaled, fitted_tree = run_treeffuser(
                X_train,
                train["target"][fit],
                X_eval,
                n_samples=args.samples,
                n_steps=max(8, args.tree_steps if args.tree_steps is not None else args.diffusion_steps),
                n_repeats=args.tree_repeats,
                n_estimators=args.tree_estimators,
                return_model=True,
            )
            with (out / "model.pkl").open("wb") as handle:
                pickle.dump(fitted_tree, handle)
            official = True
        except Exception as exc:
            (out / "official_error.txt").write_text(traceback.format_exc(), encoding="utf-8")
            # A deterministic fallback preserves the experiment protocol and
            # makes the failure explicit instead of dropping Treeffuser.
            samples_scaled = np.full(
                (args.samples, n_val + len(test["target"])),
                float(train["persistence"][fit].mean()),
                dtype=np.float32,
            )
            official = False
            method = {**method, "fallback": "fit persistence mean because official Treeffuser failed"}
    else:
        model = ConditionalResidualDiffusion(
            history_dim=train["x"].shape[2],
            context_dim=train["context"].shape[1],
            n_devices=len(data["device_map"]),
            seq_len=train["x"].shape[1],
            mode=method["mode"],
            steps=args.diffusion_steps,
            hidden_dim=args.hidden_dim,
            seed=args.seed,
        )
        history = model.fit(
            train["x"][fit],
            train["context"][fit],
            train["target"][fit],
            data["fit_stage"],
            data["fit_device"],
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.learning_rate,
        )
        write_json(out / "training_history.json", history)
        torch_payload = {
            "state_dict": model.model.state_dict(),
            "mode": method["mode"],
            "steps": args.diffusion_steps,
            "history_dim": int(train["x"].shape[2]),
            "context_dim": int(train["context"].shape[1]),
            "seq_len": int(train["x"].shape[1]),
            "n_devices": int(len(data["device_map"])),
            "base_scale": float(model.base_scale),
        }
        import torch

        torch.save(torch_payload, out / "model.pth")
        samples_scaled = model.sample(
            np.concatenate([train["x"][val], test["x"]], axis=0),
            np.concatenate([train["context"][val], test["context"]], axis=0),
            np.concatenate([data["val_stage"], data["test_stage"]], axis=0),
            np.concatenate([data["val_device"], data["test_device"]], axis=0),
            n_samples=args.samples,
        )
        official = False
    samples_scaled = np.asarray(samples_scaled)
    if samples_scaled.ndim == 3 and samples_scaled.shape[-1] == 1:
        samples_scaled = samples_scaled[:, :, 0]
    if samples_scaled.ndim != 2:
        raise ValueError(f"Expected [samples, rows] predictions, got {samples_scaled.shape}")
    samples = inverse_target(samples_scaled, data["scaler"])
    val_samples = samples[:, :n_val]
    samples = samples[:, n_val:]
    val_target = inverse_target(train["target"][val], data["scaler"])
    target = inverse_target(test["target"], data["scaler"])
    val_p10, val_p50, val_p90 = np.quantile(val_samples, [0.1, 0.5, 0.9], axis=0)
    p10, p50, p90 = np.quantile(samples, [0.1, 0.5, 0.9], axis=0)
    val_stage_names = np.asarray([data["train_meta"][index]["stage"] for index in val])
    test_stage_names = np.asarray([item["stage"] for item in data["test_meta"]])
    global_offset = float(np.median(val_target - val_p50))
    calibration = {"global_offset": global_offset, "stage": {}}
    for stage in STAGES:
        val_mask = val_stage_names == stage
        test_mask = test_stage_names == stage
        offset = float(np.median(val_target[val_mask] - val_p50[val_mask])) if val_mask.any() else global_offset
        shifted_lo = val_p10[val_mask] + offset
        shifted_hi = val_p90[val_mask] + offset
        nonconformity = np.maximum(
            np.maximum(shifted_lo - val_target[val_mask], val_target[val_mask] - shifted_hi), 0.0
        )
        margin = float(np.quantile(nonconformity, args.calibration_quantile, method="higher")) if val_mask.any() else 0.0
        p10[test_mask] += offset - margin
        p50[test_mask] += offset
        p90[test_mask] += offset + margin
        calibration["stage"][stage] = {
            "validation_count": int(val_mask.sum()),
            "median_offset": offset,
            "interval_margin": margin,
        }
    # Retain a validation-only score for configuration selection. Test labels
    # are never used to choose epochs or hyperparameters.
    validation_metrics = regression_metrics(val_target, val_p50, val_p10, val_p90)
    write_json(out / "validation_calibration.json", calibration)
    metrics = {
        "method": name,
        "repository": method["repository"],
        "paper_task": method["paper_task"],
        "core_reproduction": method["core"],
        "implementation_type": (
            "independent_local_paper_branch"
            if name in BRANCHES
            else ("official_package" if official else "shared_local_adapter")
        ),
        "source_note": (
            "Independent architecture/training loop in individual_paper_branches/models.py; "
            "mechanism adapted to the local one-step protocol."
            if name in BRANCHES
            else "See sources_manifest.json and upstream repository status."
        ),
        "official_package_used": official,
        "protocol": {
            "train_windows": int(len(fit)),
            "validation_windows": int(len(data["val_indices"])),
            "test_windows": int(len(target)),
            "sequence_length": int(test["x"].shape[1]),
            "samples": int(args.samples),
            "diffusion_steps": int(args.diffusion_steps),
            "battery_in_features": False,
        },
        "overall": regression_metrics(target, p50, p10, p90),
        "validation_before_test_calibration": validation_metrics,
        "by_humidity_stage": metrics_by_stage(target, p50, p10, p90, data["test_meta"]),
        "elapsed_seconds": round(time.time() - started, 3),
        "validation_calibration": calibration,
    }
    save_prediction_table(out / "predictions.csv", data["test_meta"], target, p10, p50, p90, name)
    write_json(out / "metrics.json", metrics)
    return metrics


def make_imputation_test(values: np.ndarray, scenario: str, ratio: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    mask = np.ones_like(values, dtype=np.float32)
    if scenario == "point":
        mask = (rng.random(values.shape) > ratio).astype(np.float32)
    elif scenario == "block":
        block = max(2, int(round(values.shape[1] * ratio)))
        for row in range(values.shape[0]):
            start = int(rng.integers(0, max(1, values.shape[1] - block + 1)))
            mask[row, start : start + block, :] = 0.0
    elif scenario == "hum_only":
        mask[:, :, 2] = (rng.random(values[:, :, 2].shape) > ratio).astype(np.float32)
    elif scenario == "temp_hum":
        mask[:, :, 1:] = (rng.random(values[:, :, 1:].shape) > ratio).astype(np.float32)
    else:
        raise ValueError(scenario)
    return values * mask, mask


def run_imputation_method(method, data, output_dir: Path, args) -> dict:
    started = time.time()
    name = method["name"]
    out = output_dir / "imputation" / name
    out.mkdir(parents=True, exist_ok=True)
    # Direct measured channels are the actual sensor observations. Derived
    # channels are intentionally not imputed as if they were independent
    # sensors.
    train_values = data["train"]["x"][:, :, :3][data["fit_indices"]]
    test_values = data["test"]["x"][:, :, :3]
    adapter = ImputationAdapter(
        time_len=test_values.shape[1],
        channels=test_values.shape[2],
        seed=args.seed,
        architecture=method["architecture"],
        hidden_dim=max(32, args.hidden_dim),
    )
    history = adapter.fit(
        train_values,
        epochs=max(3, min(args.imputation_epochs, args.epochs)),
        batch_size=args.batch_size,
        mask_strategy=method["mask"],
        missing_ratio=0.2,
        consistency_weight=method["consistency"],
        smoothness_weight=method["smoothness"],
    )
    write_json(out / "training_history.json", history)
    import torch

    torch.save(
        {
            "state_dict": adapter.model.state_dict(),
            "architecture": method["architecture"],
            "mask_strategy": method["mask"],
            "channels": ["Measure", "Temp", "Hum"],
            "time_len": int(test_values.shape[1]),
        },
        out / "model.pth",
    )
    scenarios = {
        "point_10pct": ("point", 0.10),
        "block_25pct": ("block", 0.25),
        "hum_only_20pct": ("hum_only", 0.20),
        "temp_hum_20pct": ("temp_hum", 0.20),
    }
    by_scenario = {}
    for scenario_name, (scenario, ratio) in scenarios.items():
        masked, mask = make_imputation_test(test_values, scenario, ratio, args.seed)
        samples = adapter.impute(
            masked,
            mask,
            stochastic=True,
            n_samples=max(2, min(4, args.samples // 4)),
        )
        pred = samples.mean(axis=0)
        hidden = 1.0 - mask
        errors = (pred - test_values) * hidden
        count = max(float(hidden.sum()), 1.0)
        direct = {
            "hidden_cells": int(hidden.sum()),
            "mae": float(np.abs(errors).sum() / count),
            "rmse": float(np.sqrt((errors**2).sum() / count)),
            "by_channel": {
                channel: {
                    "mae": float(np.abs(errors[:, :, index]).sum() / max(hidden[:, :, index].sum(), 1.0)),
                    "rmse": float(
                        np.sqrt((errors[:, :, index] ** 2).sum() / max(hidden[:, :, index].sum(), 1.0))
                    ),
                }
                for index, channel in enumerate(["Measure", "Temp", "Hum"])
            },
        }
        by_scenario[scenario_name] = direct
    metrics = {
        "method": name,
        "repository": method["repository"],
        "paper_task": "multivariate time-series imputation",
        "core_reproduction": "masked reconstruction with paper-specific mask/consistency/state bias",
        "official_package_used": False,
        "protocol": {
            "training_windows": int(len(train_values)),
            "test_windows": int(len(test_values)),
            "channels": ["Measure", "Temp", "Hum"],
            "battery_in_features": False,
            "test_is_synthetic_missingness": True,
        },
        "scenarios": by_scenario,
        "elapsed_seconds": round(time.time() - started, 3),
    }
    write_json(out / "metrics.json", metrics)
    return metrics


def run_anomaly_method(method, data, prediction_cache, output_dir: Path, args) -> dict:
    """Score normal and controlled pseudo-anomalous windows.

    The real dataset has no field anomaly labels. We therefore report a
    validation threshold, normal false alarms, and detection on an explicitly
    generated stress set. This is not presented as a field F1 score.
    """
    started = time.time()
    name = method["name"]
    out = output_dir / "anomaly" / name
    out.mkdir(parents=True, exist_ok=True)
    target = np.asarray(prediction_cache["target"], dtype=float)
    train_values = data["train"]["x"][:, :, :3][data["fit_indices"]]
    test_values = data["test"]["x"][:, :, :3]

    def moving_average(values: np.ndarray, width: int = 5) -> np.ndarray:
        padded = np.pad(values, ((0, 0), (width // 2, width // 2), (0, 0)), mode="edge")
        return np.stack(
            [padded[:, index : index + width].mean(axis=1) for index in range(values.shape[1])],
            axis=1,
        )

    detrended = method["score"] == "detrended"
    train_model_values = train_values - moving_average(train_values) if detrended else train_values
    test_model_values = test_values - moving_average(test_values) if detrended else test_values
    adapter = ImputationAdapter(
        time_len=test_values.shape[1],
        channels=test_values.shape[2],
        seed=args.seed,
        architecture=method["architecture"],
        hidden_dim=max(32, args.hidden_dim),
    )
    consistency = 0.15 if method["score"] in {"imputation_ensemble", "distilled"} else 0.05
    smoothness = 0.02 if method["architecture"] in {"conv", "gru"} else 0.0
    history = adapter.fit(
        train_model_values,
        epochs=max(3, min(args.imputation_epochs, args.epochs)),
        batch_size=args.batch_size,
        mask_strategy=method["mask"],
        missing_ratio=0.2,
        consistency_weight=consistency,
        smoothness_weight=smoothness,
    )
    write_json(out / "training_history.json", history)
    import torch

    torch.save(
        {
            "state_dict": adapter.model.state_dict(),
            "architecture": method["architecture"],
            "mask": method["mask"],
            "score": method["score"],
            "channels": ["Measure", "Temp", "Hum"],
        },
        out / "model.pth",
    )

    stochastic = method["score"] == "imputation_ensemble"

    def reconstruct_score(values: np.ndarray) -> np.ndarray:
        model_values = values - moving_average(values) if detrended else values
        reconstruction, uncertainty = adapter.reconstruct_by_partitions(
            model_values,
            partitions=4,
            stochastic=stochastic,
            n_samples=4,
        )
        absolute = np.abs(reconstruction - model_values)
        if method["score"] == "weighted_incremental":
            weights = np.asarray([1.5, 0.75, 1.0], dtype=np.float32)
            return (absolute * weights.reshape(1, 1, -1)).mean(axis=(1, 2))
        if method["score"] == "recurrent":
            tail_weights = np.linspace(0.5, 1.5, absolute.shape[1], dtype=np.float32)
            return (absolute * tail_weights.reshape(1, -1, 1)).mean(axis=(1, 2))
        if method["score"] == "imputation_ensemble":
            return absolute.mean(axis=(1, 2)) + uncertainty.mean(axis=(1, 2))
        if method["score"] == "distilled":
            # A one-pass compact reconstruction score mirrors the distilled
            # student's fast inference path.
            return 0.7 * absolute.mean(axis=(1, 2)) + 0.3 * absolute.max(axis=(1, 2))
        return absolute.mean(axis=(1, 2))

    normal_score = reconstruct_score(test_values)
    threshold = float(np.quantile(normal_score, 0.995))
    normal_alarm_rate = float(np.mean(normal_score > threshold))
    rng = np.random.default_rng(args.seed)
    labels = (rng.random(len(test_values)) < 0.10).astype(np.int64)
    signs = rng.choice([-1.0, 1.0], int(labels.sum()))
    severity_results = {}
    score = normal_score.copy()
    for severity in (3.0, 6.0, 10.0):
        pseudo_values = test_values.copy()
        # Inject a sustained voltage deviation in the most recent four points;
        # Temp/Hum remain unchanged, creating a physically inconsistent event.
        pseudo_values[labels == 1, -4:, 0] += signs[:, None] * severity
        severity_score = reconstruct_score(pseudo_values)
        from sklearn.metrics import average_precision_score, roc_auc_score

        severity_results[f"{severity:.0f}sigma"] = {
            "detection_rate": float(np.mean(severity_score[labels == 1] > threshold)),
            "roc_auc": float(roc_auc_score(labels, severity_score)),
            "average_precision": float(average_precision_score(labels, severity_score)),
            "mean_anomaly_score": float(severity_score[labels == 1].mean()),
        }
        if severity == 6.0:
            score = severity_score
    fpr = normal_alarm_rate
    tpr = severity_results["6sigma"]["detection_rate"]
    from sklearn.metrics import average_precision_score, roc_auc_score

    auc = float(roc_auc_score(labels, score)) if len(np.unique(labels)) == 2 else None
    average_precision = (
        float(average_precision_score(labels, score)) if len(np.unique(labels)) == 2 else None
    )
    metrics = {
        "method": name,
        "repository": method["repository"],
        "paper_task": "unsupervised multivariate time-series anomaly detection",
        "core_reproduction": "diffusion/prediction residual reconstruction score",
        "official_package_used": False,
        "protocol": {
            "normal_training_only": True,
            "test_windows": int(len(target)),
            "pseudo_anomaly_fraction": float(labels.mean()),
            "pseudo_anomaly_type": "last-four-step standardized Measure level shift",
            "primary_pseudo_anomaly_scale": "6 standard deviations",
            "real_lab_anomaly_labels_available": False,
            "battery_in_features": False,
        },
        "threshold": threshold,
        "normal_false_alarm_rate": fpr,
        "pseudo_anomaly_detection_rate": tpr,
        "pseudo_anomaly_count": int(labels.sum()),
        "pseudo_anomaly_roc_auc": auc,
        "pseudo_anomaly_average_precision": average_precision,
        "mean_normal_score": float(normal_score.mean()),
        "mean_pseudo_anomaly_score": float(score[labels == 1].mean()) if labels.any() else None,
        "severity_curve": severity_results,
        "elapsed_seconds": round(time.time() - started, 3),
    }
    rows = pd.DataFrame(
        {
            "method": name,
            "IMEI": [m["IMEI"] for m in data["test_meta"]],
            "Timestamp": [m["Timestamp"] for m in data["test_meta"]],
            "stage": [m["stage"] for m in data["test_meta"]],
            "is_pseudo_anomaly": labels,
            "score": score,
            "threshold": threshold,
            "alarm": (score > threshold).astype(int),
        }
    )
    rows.to_csv(out / "scores.csv", index=False, encoding="utf-8-sig")
    write_json(out / "metrics.json", metrics)
    return metrics


def load_cached_prediction(path: Path) -> dict:
    frame = pd.read_csv(path / "predictions.csv")
    return {
        "target": frame["target_Measure"].to_numpy(float),
        "p10": frame["P10"].to_numpy(float),
        "p50": frame["P50"].to_numpy(float),
        "p90": frame["P90"].to_numpy(float),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "local_runs" / "latest")
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--imputation-epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--diffusion-steps", type=int, default=8)
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=8e-4)
    parser.add_argument("--tree-estimators", type=int, default=80)
    parser.add_argument("--tree-repeats", type=int, default=2)
    parser.add_argument("--tree-steps", type=int, default=None)
    parser.add_argument("--calibration-quantile", type=float, default=0.80)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-smoke", action="store_true")
    parser.add_argument("--only", choices=["all", "prediction", "imputation", "anomaly"], default="all")
    parser.add_argument(
        "--methods",
        type=str,
        default="",
        help="Comma-separated method names to run within the selected task (default: all)",
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed)
    started = time.time()
    run_manifest = {
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "command": " ".join(sys.argv),
        "settings": {key: value for key, value in vars(args).items()},
        "data": {},
        "official_smoke": {},
        "prediction": {},
        "imputation": {},
        "anomaly": {},
    }
    run_manifest["settings"]["output"] = str(args.output)
    if not args.skip_smoke:
        run_manifest["official_smoke"] = official_smoke()
        write_json(args.output / "official_smoke.json", run_manifest["official_smoke"])
    data = prepare_windows(seq_len=24)
    run_manifest["data"] = {
        "training_rows": int(len(data["train_raw"])),
        "test_rows": int(len(data["test_raw"])),
        "prepared_rows": int(len(data["prepared"])),
        "training_windows": int(len(data["train"]["target"])),
        "fit_windows": int(len(data["fit_indices"])),
        "validation_windows": int(len(data["val_indices"])),
        "test_windows": int(len(data["test"]["target"])),
        "stage_counts_test": dict(Counter(item["stage"] for item in data["test_meta"])),
        "features_history": int(data["train"]["x"].shape[2]),
        "features_context": int(data["train"]["context"].shape[1]),
        "battery_in_features": False,
        "train_file": str(ROOT.parent / "dist2026" / "latest_retraining_data_20260716" / "training_merged.csv"),
        "test_file": str(ROOT.parent / "dist2026" / "latest_retraining_data_20260716" / "test_latest_holdout.csv"),
    }
    write_json(args.output / "data_protocol.json", run_manifest["data"])

    if args.only in ("all", "prediction"):
        selected_names = {item.strip() for item in args.methods.split(",") if item.strip()}
        prediction_methods = (
            [method for method in PREDICTION_METHODS if method["name"] in selected_names]
            if selected_names
            else PREDICTION_METHODS
        )
        for method in prediction_methods:
            try:
                print(f"[prediction] {method['name']}", flush=True)
                run_manifest["prediction"][method["name"]] = run_prediction_method(
                    method, data, args.output, args
                )
            except Exception as exc:
                run_manifest["prediction"][method["name"]] = {
                    "status": "failed",
                    "error": repr(exc),
                    "traceback": traceback.format_exc(),
                }
                print(f"[prediction] {method['name']} FAILED: {exc}", flush=True)
        write_json(args.output / "prediction_summary.json", run_manifest["prediction"])

    if args.only in ("all", "imputation"):
        for method in IMPUTATION_METHODS:
            try:
                print(f"[imputation] {method['name']}", flush=True)
                run_manifest["imputation"][method["name"]] = run_imputation_method(
                    method, data, args.output, args
                )
            except Exception as exc:
                run_manifest["imputation"][method["name"]] = {
                    "status": "failed",
                    "error": repr(exc),
                    "traceback": traceback.format_exc(),
                }
                print(f"[imputation] {method['name']} FAILED: {exc}", flush=True)
        write_json(args.output / "imputation_summary.json", run_manifest["imputation"])

    # Use the strongest locally completed prediction as the shared anomaly
    # baseline. This keeps all anomaly scores comparable and avoids reporting
    # multiple runs as independent evidence when labels are absent.
    prediction_cache = None
    for candidate in ["Diffusion-TS", "CCDM", "TimeGrad"]:
        candidate_path = args.output / "prediction" / candidate
        if (candidate_path / "predictions.csv").exists():
            prediction_cache = load_cached_prediction(candidate_path)
            run_manifest["anomaly"]["shared_prediction_source"] = candidate
            break
    if args.only in ("all", "anomaly") and prediction_cache is not None:
        for method in ANOMALY_METHODS:
            try:
                print(f"[anomaly] {method['name']}", flush=True)
                run_manifest["anomaly"][method["name"]] = run_anomaly_method(
                    method, data, prediction_cache, args.output, args
                )
            except Exception as exc:
                run_manifest["anomaly"][method["name"]] = {
                    "status": "failed",
                    "error": repr(exc),
                    "traceback": traceback.format_exc(),
                }
                print(f"[anomaly] {method['name']} FAILED: {exc}", flush=True)
        write_json(args.output / "anomaly_summary.json", run_manifest["anomaly"])
    elif args.only in ("all", "anomaly"):
        run_manifest["anomaly"]["status"] = "skipped: no prediction baseline completed"

    run_manifest["elapsed_seconds"] = round(time.time() - started, 3)
    run_manifest["completed_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    # A group-only rerun must not erase the already completed sections in the
    # same output directory. Reload their summaries before writing the common
    # manifest.
    summary_files = {
        "prediction": args.output / "prediction_summary.json",
        "imputation": args.output / "imputation_summary.json",
        "anomaly": args.output / "anomaly_summary.json",
    }
    for section, summary_path in summary_files.items():
        if not run_manifest[section] and summary_path.exists():
            run_manifest[section] = json.loads(summary_path.read_text(encoding="utf-8"))
    smoke_path = args.output / "official_smoke.json"
    if not run_manifest["official_smoke"] and smoke_path.exists():
        run_manifest["official_smoke"] = json.loads(smoke_path.read_text(encoding="utf-8"))
    write_json(args.output / "run_manifest.json", run_manifest)
    print(f"Completed reproductions in {run_manifest['elapsed_seconds']:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
