"""Augmentation-first upstream anomaly detection benchmark for experiment.xlsx.

Pipeline requested: first train fixed-version upstream diffusion generators on
normal windows to augment the training set, then train the upstream anomaly
detection models themselves (D3R / IMDiffusion) on the augmented normal set and
score held-out test windows.

Synthetic normal windows are reused from runs/upstream_full_20260910 (per
method/seed/test_date class-conditional generators, relationship-projected).
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from upstream_large_augmentation import (
    FEATURES,
    LENGTH,
    ROOT,
    Upstream,
    scores,
    seed_all,
    threshold_from_validation,
)

HERE = Path(__file__).resolve().parent
SYNTH_ROOT = HERE / "runs" / "upstream_full_20260910"
AUG_METHODS = ["CSDI", "MTSCI", "IMDiffusion", "Diffusion-TS", "D3R"]
DETECTORS = ["D3R", "IMDiffusion"]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_split(out_dir, test_date):
    x = np.load(out_dir / "dataset.npz")["x"]
    meta = pd.read_json(out_dir / "metadata.json", dtype={"date": str})
    y = meta.label.to_numpy()
    fit = np.flatnonzero((meta.date != test_date) & (meta.role == "fit"))
    val = np.flatnonzero((meta.date != test_date) & (meta.role == "validation"))
    test = np.flatnonzero(meta.date == test_date)
    return x, y, fit, val, test


def load_synthetic_normal(aug, seed, test_date, volume, n_real_normals):
    if aug == "real":
        return None
    path = SYNTH_ROOT / aug / f"seed{seed}" / test_date / "synthetic.npz"
    data = np.load(path)
    proj, labels = data["projected"], data["label"]
    normal = proj[labels == 0]
    if volume == "10x":
        normal = normal[: 10 * n_real_normals]
    elif volume == "1x":
        normal = normal[: n_real_normals]
    return normal.astype(np.float32)


def scale_data(train, arrays):
    flat = train.reshape(-1, 5)
    lo, hi = flat.min(0), flat.max(0)
    scale = (hi - lo).clip(1e-6)
    offset = lo
    out = []
    for a in arrays:
        z = (a.reshape(-1, 5) - offset) / scale * 1.6 - 0.8
        out.append(z.reshape(a.shape).astype(np.float32))
    return out


@torch.no_grad()
def detector_score(up, z, batch_size=512):
    up.model.eval()
    outputs = []
    for start in range(0, len(z), batch_size):
        batch = torch.from_numpy(z[start:start + batch_size])
        if up.method == "D3R":
            tp = torch.linspace(0, 1, LENGTH).view(1, LENGTH, 1).repeat(len(batch), 1, 1)
            _, _, recon = up.model(batch, tp, p=0)
            err = ((batch - recon) ** 2).mean(dim=(1, 2))
        else:
            obs, mask, side, _ = up.mask_side(batch)
            kwargs = {"strategy_type": torch.zeros(len(batch), dtype=torch.long)}
            samples = up.model.impute(obs, mask, side, 1, **kwargs)[:, 0]
            masked = (samples - obs).abs() * (1 - mask)
            err = masked.sum(dim=(1, 2)) / (1 - mask).sum(dim=(1, 2)).clamp(min=1)
        outputs.append(err.numpy())
    return np.concatenate(outputs)


def train_detector(up, z_train, epochs, batch=64):
    optimizer = torch.optim.Adam(up.model.parameters(), lr=0.001)
    history = []
    inp = torch.from_numpy(z_train)
    for epoch in range(epochs):
        up.model.train()
        order = torch.randperm(len(inp))
        losses = []
        for start in range(0, len(inp), batch):
            selected = order[start:start + batch]
            loss = up.loss(inp[selected])
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(up.model.parameters(), 1)
            optimizer.step()
            losses.append(float(loss.detach()))
        history.append(float(np.mean(losses)))
        if epoch in (0, epochs - 1):
            print(f"  detector={up.method} aug epoch={epoch + 1}/{epochs} loss={history[-1]:.6f}", flush=True)
    return history


def run(args):
    out_dir = args.output
    run_dir = out_dir / args.detector / args.aug / args.volume / f"seed{args.seed}" / args.test_date
    if (run_dir / "result.json").exists():
        print(f"skip {run_dir}", flush=True)
        return
    run_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    seed_all(args.seed)
    x, y, fit, val, test = load_split(out_dir, args.test_date)
    real_normals = fit[y[fit] == 0]
    synthetic = load_synthetic_normal(args.aug, args.seed, args.test_date, args.volume, len(real_normals))
    train_ids = real_normals
    n_synthetic = 0
    if synthetic is not None:
        x_train = np.concatenate([x[real_normals], synthetic])
        n_synthetic = len(synthetic)
    else:
        x_train = x[real_normals]
    z_train, z_val, z_test = scale_data(x_train, [x_train, x[val], x[test]])
    up = Upstream(args.detector, args.steps, args.hidden)
    module_file = inspect.getfile(type(up.model))
    history = train_detector(up, z_train, args.epochs)
    score_val = detector_score(up, z_val)
    score_test = detector_score(up, z_test)
    threshold = threshold_from_validation(y[val], score_val)
    metrics = scores(y[test], score_test, threshold)
    val_metrics = scores(y[val], score_val, threshold)
    torch.save(up.model.state_dict(), run_dir / "detector.pt")
    pd.DataFrame({
        "test_id": test, "label": y[test], "score": score_test, "threshold": threshold,
    }).to_csv(run_dir / "test_predictions.csv", index=False)
    result = {
        "detector": args.detector,
        "augmentation": args.aug,
        "volume": args.volume,
        "seed": args.seed,
        "test_date": args.test_date,
        "status": "completed",
        "features": FEATURES,
        "window_length": LENGTH,
        "real_normal_train_windows": int(len(real_normals)),
        "synthetic_normal_windows": int(n_synthetic),
        "upstream_model_file": module_file,
        "upstream_model_sha256": sha(module_file),
        "training_loss_first": history[0],
        "training_loss_last": history[-1],
        "epochs": args.epochs,
        "diffusion_steps": args.steps,
        "threshold": float(threshold),
        "test_metrics": metrics,
        "validation_metrics": val_metrics,
        "elapsed_seconds": time.time() - started,
        "command": sys.argv,
    }
    (run_dir / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(f"COMPLETED det={args.detector} aug={args.aug} {args.volume} {args.test_date} seed={args.seed}: "
          f"{json.dumps(metrics)} {time.time() - started:.1f}s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--detector", choices=DETECTORS, required=True)
    ap.add_argument("--aug", choices=["real"] + AUG_METHODS, required=True)
    ap.add_argument("--volume", choices=["1x", "10x", "large"], required=True)
    ap.add_argument("--test-date", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--hidden", type=int, default=32)
    args = ap.parse_args()
    if args.aug == "real" and args.volume != "1x":
        args.volume = "1x"
    run(args)


if __name__ == "__main__":
    main()
