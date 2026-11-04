"""Train/evaluate the upstream Diffusion-TS Transformer on JYZmentor windows."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
ADAPTERS = ROOT / "adapters"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ADAPTERS) not in sys.path:
    sys.path.insert(0, str(ADAPTERS))
from data_adapter import STAGES  # noqa: E402
from run_reproductions import prepare_windows  # noqa: E402
from model import FullDiffusionTSPoint, TRANSFORMER_SOURCE  # noqa: E402


def seed_everything(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))


def make_full_sequence(x: np.ndarray, context: np.ndarray, target: np.ndarray, scaler: dict) -> tuple[np.ndarray, np.ndarray]:
    """Append a target position without using target Measure-derived features."""
    seq = np.zeros((len(x), x.shape[1] + 1, x.shape[2]), dtype=np.float32)
    seq[:, :-1] = x
    # Carry the latest causal state for target-derived channels.
    seq[:, -1] = x[:, -1]
    # The target Measure is the only unknown channel at the final position.
    seq[:, -1, 0] = target
    # Replace known target-time covariates with values transformed in the
    # history-feature scaler, not the separate context scaler.
    context_raw = context * scaler["context_std"] + scaler["context_mean"]
    history_index = {name: index for index, name in enumerate([
        "Measure", "Temp", "Hum", "AH", "DeltaHours", "MeasureRate", "TempRate", "HumRate",
        "MeasureEMA3", "HumEMA3", "MeasureMAD6", "HumMAD6", "HoursAbove65", "HoursAbove80",
        "HumRising", "HumFalling", "HumInvalid", "LineVoltageKV", "Loc_SideCode", "Loc_PhaseCode",
        "Loc_SizeCode", "Loc_StrNum", "Loc_PlateCount", "Loc_PlateStart", "Loc_PlateEnd",
        "Loc_PlatePositionRatio",
    ])}
    context_names = [
        "Temp", "Hum", "AH", "DeltaHours", "LineVoltageKV", "Loc_SideCode", "Loc_PhaseCode",
        "Loc_SizeCode", "Loc_StrNum", "Loc_PlateCount", "Loc_PlateStart", "Loc_PlateEnd",
        "Loc_PlatePositionRatio",
    ]
    for context_index, name in enumerate(context_names):
        h = history_index[name]
        raw = context_raw[:, context_index]
        seq[:, -1, h] = (raw - scaler["x_mean"][h]) / scaler["x_std"][h]
    observed = np.ones_like(seq, dtype=np.float32)
    observed[:, -1, 0] = 0.0
    return seq, observed


def metrics(target: np.ndarray, p10: np.ndarray, p50: np.ndarray, p90: np.ndarray) -> dict:
    error = p50 - target
    return {
        "count": int(len(target)),
        "rmse": float(np.sqrt(np.mean(error ** 2))),
        "mae": float(np.mean(np.abs(error))),
        "r2": float(1.0 - np.sum(error ** 2) / max(np.sum((target - target.mean()) ** 2), 1e-12)),
        "coverage_p10_p90": float(np.mean((target >= p10) & (target <= p90))),
        "mean_interval_width": float(np.mean(p90 - p10)),
    }


def stage_metrics(target, p10, p50, p90, metadata):
    output = {}
    stages = np.asarray([item["stage"] for item in metadata])
    for stage in STAGES:
        mask = stages == stage
        output[stage] = metrics(target[mask], p10[mask], p50[mask], p90[mask]) if mask.any() else {"count": 0}
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "local_runs" / "full_diffusion_ts_20260907")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--timesteps", type=int, default=24)
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use-fourier", action="store_true")
    parser.add_argument("--calibration-quantile", type=float, default=0.80)
    parser.add_argument("--target-weight", type=float, default=5.0)
    parser.add_argument("--fourier-weight", type=float, default=0.05)
    args = parser.parse_args()
    seed_everything(args.seed)
    started = time.time()
    data = prepare_windows(seq_len=24)
    fit = data["fit_indices"]
    val = data["val_indices"]
    train = data["train"]
    test = data["test"]
    train_seq, train_mask = make_full_sequence(train["x"][fit], train["context"][fit], train["target"][fit], data["scaler"])
    val_seq, val_mask = make_full_sequence(train["x"][val], train["context"][val], train["target"][val], data["scaler"])
    test_seq, test_mask = make_full_sequence(test["x"], test["context"], test["target"], data["scaler"])
    device = torch.device("cpu")
    model = FullDiffusionTSPoint(
        seq_length=25,
        feature_size=train_seq.shape[-1],
        n_layer_enc=args.layers,
        n_layer_dec=args.layers,
        d_model=args.d_model,
        n_heads=4,
        timesteps=args.timesteps,
        use_fourier=args.use_fourier,
        target_weight=args.target_weight,
        fourier_weight=args.fourier_weight,
    ).to(device)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(train_seq), torch.from_numpy(train_mask)),
        batch_size=args.batch_size,
        shuffle=True,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for batch_seq, batch_mask in loader:
            loss = model.loss(batch_seq.to(device), batch_mask.to(device))
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        history.append({"epoch": epoch, "loss": float(np.mean(losses))})
        print(f"epoch={epoch} loss={history[-1]['loss']:.6f}", flush=True)

    out = args.output / ("fourier_on" if args.use_fourier else "fourier_off")
    out.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "args": vars(args)}, out / "model.pth")
    (out / "training_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

    model.eval()
    with torch.no_grad():
        val_samples_scaled = model.sample_infill(torch.from_numpy(val_seq), torch.from_numpy(val_mask), n_samples=args.samples, seed=args.seed)
        test_samples_scaled = model.sample_infill(torch.from_numpy(test_seq), torch.from_numpy(test_mask), n_samples=args.samples, seed=args.seed + 1000)
    val_samples = val_samples_scaled.numpy() * float(data["scaler"]["target_std"]) + float(data["scaler"]["target_mean"])
    test_samples = test_samples_scaled.numpy() * float(data["scaler"]["target_std"]) + float(data["scaler"]["target_mean"])
    val_target = train["target"][val] * float(data["scaler"]["target_std"]) + float(data["scaler"]["target_mean"])
    target = test["target"] * float(data["scaler"]["target_std"]) + float(data["scaler"]["target_mean"])
    val_p10, val_p50, val_p90 = np.quantile(val_samples, [0.1, 0.5, 0.9], axis=0)
    p10, p50, p90 = np.quantile(test_samples, [0.1, 0.5, 0.9], axis=0)
    # Stagewise conformal calibration, using validation only.
    val_stages = np.asarray([data["train_meta"][i]["stage"] for i in val])
    test_stages = np.asarray([item["stage"] for item in data["test_meta"]])
    calibration = {}
    for stage in STAGES:
        vm = val_stages == stage
        tm = test_stages == stage
        if not vm.any():
            continue
        offset = float(np.median(val_target[vm] - val_p50[vm]))
        nonconformity = np.maximum(val_p10[vm] + offset - val_target[vm], val_target[vm] - (val_p90[vm] + offset))
        margin = float(np.quantile(nonconformity, args.calibration_quantile, method="higher"))
        p10[tm] += offset - margin
        p50[tm] += offset
        p90[tm] += offset + margin
        calibration[stage] = {"offset": offset, "margin": margin, "validation_count": int(vm.sum())}
    result = {
        "model": "full_upstream_Diffusion-TS_Transformer",
        "implementation_type": "independent_full_local_Diffusion-TS_branch",
        "transformer_source": TRANSFORMER_SOURCE,
        "use_fourier": bool(args.use_fourier),
        "protocol": {"train_windows": int(len(fit)), "validation_windows": int(len(val)), "test_windows": int(len(test["target"])), "sequence_length": 25, "history_length": 24, "feature_size": int(train_seq.shape[-1]), "samples": args.samples, "timesteps": args.timesteps, "epochs": args.epochs, "d_model": args.d_model, "layers": args.layers, "batch_size": args.batch_size, "learning_rate": args.lr, "target_weight": args.target_weight, "fourier_weight": args.fourier_weight, "battery_in_features": False},
        "validation_raw": metrics(val_target, val_p10, val_p50, val_p90),
        "overall": metrics(target, p10, p50, p90),
        "by_humidity_stage": stage_metrics(target, p10, p50, p90, data["test_meta"]),
        "calibration": calibration,
        "calibration_quantile": args.calibration_quantile,
        "elapsed_seconds": round(time.time() - started, 3),
    }
    (out / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    frame = pd.DataFrame({
        "IMEI": [m["IMEI"] for m in data["test_meta"]], "Timestamp": [m["Timestamp"] for m in data["test_meta"]],
        "Hum": [m["Hum"] for m in data["test_meta"]], "stage": test_stages,
        "target_Measure": target, "P10": p10, "P50": p50, "P90": p90,
    })
    frame.to_csv(out / "predictions.csv", index=False, encoding="utf-8-sig")
    print(json.dumps(result["overall"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
