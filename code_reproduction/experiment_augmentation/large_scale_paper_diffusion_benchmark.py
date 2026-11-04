"""Large-scale paper-core diffusion augmentation benchmark for experiment.xlsx.

All methods use the same five dual-device channels and the same real holdout
folds. Upstream repositories are preserved under ../upstream; the model
branches below adapt their transferable mechanisms to this dataset.
"""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler


FEATURES = [
    "M4_residual",
    "M7_residual",
    "M4_M7_difference",
    "M4_M7_ratio",
    "log_M4_M7_ratio",
]
METHODS = ["TimeGrad", "NsDiff", "Diffusion-TS", "D3R", "CSDI", "MTSCI"]


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))


def read_pairs(path):
    sheets = {p.stem: pd.read_csv(p) for p in Path(path).glob("*.csv")}
    grouped = {}
    for name, frame in sheets.items():
        grouped.setdefault(name[:6], {})[int(name.split("(")[1][0])] = frame
    outputs = []
    audit = []
    for date, pair in sorted(grouped.items()):
        m4, m7 = pair[4].copy(), pair[7].copy()
        for frame in (m4, m7):
            frame["dt"] = pd.to_datetime(frame.time, format="%Y-%m-%d_%H:%M:%S")
            frame.sort_values("dt", inplace=True)
        aligned = pd.merge_asof(
            m4,
            m7,
            on="dt",
            direction="nearest",
            tolerance=pd.Timedelta("40s"),
            suffixes=("4", "7"),
        ).dropna(subset=["Measure7"])
        aligned["date"] = date
        aligned["label"] = ((aligned.error4 != 0) | (aligned.error7 != 0)).astype(int)
        aligned["stage"] = aligned.error4.astype(str) + ":" + aligned.error7.astype(str)
        kept = []
        for _, run in aligned.groupby((aligned.stage != aligned.stage.shift()).cumsum()):
            drop = min(len(run), max(5, int(math.ceil(0.20 * len(run)))))
            stable = run.iloc[drop:].copy()
            kept.append(stable)
            audit.append({
                "date": date,
                "stage": str(run.stage.iloc[0]),
                "paired_points": int(len(run)),
                "startup_points_removed": int(drop),
                "stable_points": int(len(stable)),
            })
        stable = pd.concat(kept, ignore_index=True)
        normal = stable[stable.label == 0]
        if len(normal) < 8:
            raise ValueError(f"{date} does not have enough stable normal calibration points")
        elapsed = (stable.dt - stable.dt.min()).dt.total_seconds().to_numpy() / 60.0
        design = np.column_stack([
            np.ones(len(stable)),
            stable.Temp4.to_numpy(),
            stable.Hum4.to_numpy(),
            elapsed,
        ])
        normal_mask = stable.label.to_numpy() == 0
        beta4 = np.linalg.lstsq(design[normal_mask], stable.Measure4.to_numpy()[normal_mask], rcond=None)[0]
        beta7 = np.linalg.lstsq(design[normal_mask], stable.Measure7.to_numpy()[normal_mask], rcond=None)[0]
        eps = 1e-4
        stable[FEATURES[0]] = stable.Measure4.to_numpy() - design @ beta4
        stable[FEATURES[1]] = stable.Measure7.to_numpy() - design @ beta7
        stable[FEATURES[2]] = stable.Measure4 - stable.Measure7
        stable[FEATURES[3]] = (stable.Measure4 + eps) / (stable.Measure7 + eps)
        stable[FEATURES[4]] = np.log((stable.Measure4 + eps) / (stable.Measure7 + eps))
        outputs.append(stable)
    return pd.concat(outputs, ignore_index=True), pd.DataFrame(audit)


def make_windows(points, length=6):
    windows, labels, dates, stages = [], [], [], []
    for (date, stage), run in points.groupby(["date", "stage"], sort=True):
        run = run.sort_values("dt")
        values = run[FEATURES].to_numpy(np.float32)
        label = int(run.label.iloc[0])
        for start in range(0, len(run) - length + 1):
            windows.append(values[start:start + length])
            labels.append(label)
            dates.append(date)
            stages.append(stage)
    return np.stack(windows), np.asarray(labels), np.asarray(dates), np.asarray(stages)


def time_embedding(t, dim=32):
    half = dim // 2
    freq = torch.exp(-math.log(10000.0) * torch.arange(half) / max(half - 1, 1))
    z = t.float()[:, None] * freq[None]
    return torch.cat([z.sin(), z.cos()], 1)


class Denoiser(nn.Module):
    def __init__(self, dim, cond_dim, hidden=192):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim * 2 + cond_dim + 32, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, noisy, t, cond, observed=None):
        if observed is None:
            observed = torch.zeros_like(noisy)
        return self.net(torch.cat([noisy, observed, cond, time_embedding(t)], 1))


def transform(method, windows):
    if method == "NsDiff":
        mu = windows.mean(1, keepdims=True)
        sd = windows.std(1, keepdims=True).clip(1e-3)
        return (windows - mu) / sd, np.concatenate([mu[:, 0], sd[:, 0]], 1)
    if method == "D3R":
        t = np.linspace(-1.0, 1.0, windows.shape[1], dtype=np.float32)
        slope = np.sum(windows * t[None, :, None], 1) / np.sum(t * t)
        intercept = windows.mean(1)
        trend = intercept[:, None] + slope[:, None] * t[None, :, None]
        return windows - trend, np.concatenate([intercept, slope], 1)
    if method == "Diffusion-TS":
        padded = np.pad(windows, ((0, 0), (1, 1), (0, 0)), mode="edge")
        trend = (padded[:, :-2] + padded[:, 1:-1] + padded[:, 2:]) / 3.0
        return np.concatenate([windows - trend, trend], 2), np.zeros((len(windows), 0), np.float32)
    return windows, np.zeros((len(windows), 0), np.float32)


def inverse_transform(method, generated, anchors):
    length = anchors.shape[1]
    if method == "NsDiff":
        mu = anchors.mean(1, keepdims=True)
        sd = anchors.std(1, keepdims=True).clip(1e-3)
        return generated * sd + mu
    if method == "D3R":
        t = np.linspace(-1.0, 1.0, length, dtype=np.float32)
        slope = np.sum(anchors * t[None, :, None], 1) / np.sum(t * t)
        intercept = anchors.mean(1)
        return generated + intercept[:, None] + slope[:, None] * t[None, :, None]
    if method == "Diffusion-TS":
        return generated[:, :, :5] + generated[:, :, 5:]
    return generated


def condition(labels, dates, all_dates, extra):
    label_onehot = np.eye(2, dtype=np.float32)[labels]
    date_onehot = np.column_stack([(dates == d).astype(np.float32) for d in all_dates])
    return np.concatenate([label_onehot, date_onehot, extra], 1).astype(np.float32)


def fit_ddpm(method, windows, labels, dates, all_dates, epochs, steps, seed):
    seed_all(seed)
    transformed = transform(method, windows)
    values, extra = transformed[0], transformed[1]
    flat = values.reshape(len(values), -1)
    scaler = StandardScaler().fit(flat)
    scaled = scaler.transform(flat).astype(np.float32)
    extra_scaler = StandardScaler().fit(extra) if extra.shape[1] else None
    extra_scaled = extra_scaler.transform(extra).astype(np.float32) if extra_scaler else extra
    cond = condition(labels, dates, all_dates, extra_scaled)
    model = Denoiser(scaled.shape[1], cond.shape[1])
    beta = torch.linspace(1e-4, 0.02, steps)
    alpha = 1.0 - beta
    abar = torch.cumprod(alpha, 0)
    counts = np.bincount(labels, minlength=2)
    weights = np.asarray([1.0 / max(counts[y], 1) for y in labels], np.float64)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(scaled), torch.from_numpy(cond)),
        batch_size=min(128, len(scaled)),
        sampler=WeightedRandomSampler(torch.from_numpy(weights), len(weights), replacement=True),
    )
    opt = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=1e-4)
    history = []
    for epoch in range(epochs):
        losses = []
        for clean, c in loader:
            t = torch.randint(0, steps, (len(clean),))
            noise = torch.randn_like(clean)
            noisy = abar[t, None].sqrt() * clean + (1.0 - abar[t, None]).sqrt() * noise
            observed = torch.zeros_like(clean)
            if method in ("CSDI", "MTSCI"):
                mask = (torch.rand_like(clean) > 0.25).float()
                observed = clean * mask
            pred = model(noisy, t, c, observed)
            loss = (pred - noise).pow(2).mean()
            if method == "MTSCI":
                mask2 = (torch.rand_like(clean) > 0.25).float()
                pred2 = model(noisy, t, c, clean * mask2)
                loss = loss + 0.10 * (pred - pred2).pow(2).mean()
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.detach()))
        history.append(float(np.mean(losses)))
    return {
        "model": model,
        "scaler": scaler,
        "extra_scaler": extra_scaler,
        "beta": beta,
        "alpha": alpha,
        "abar": abar,
        "steps": steps,
        "final_loss": history[-1],
    }


@torch.no_grad()
def generate(method, fitted, anchors, labels, dates, all_dates, seed, batch_size=1024):
    transformed = transform(method, anchors)
    extra = transformed[1]
    if fitted["extra_scaler"]:
        extra = fitted["extra_scaler"].transform(extra).astype(np.float32)
    cond_all = condition(labels, dates, all_dates, extra)
    outputs = []
    model = fitted["model"].eval()
    dim = fitted["scaler"].mean_.shape[0]
    for start in range(0, len(anchors), batch_size):
        c = torch.from_numpy(cond_all[start:start + batch_size])
        generator = torch.Generator().manual_seed(seed + start)
        z = torch.randn((len(c), dim), generator=generator)
        for step in range(fitted["steps"] - 1, -1, -1):
            t = torch.full((len(c),), step, dtype=torch.long)
            pred = model(z, t, c)
            alpha = fitted["alpha"][step]
            abar = fitted["abar"][step]
            z = (z - (1.0 - alpha) / (1.0 - abar).sqrt() * pred) / alpha.sqrt()
            if step > 0:
                z += fitted["beta"][step].sqrt() * torch.randn(z.shape, generator=generator)
        outputs.append(z.numpy())
    generated = fitted["scaler"].inverse_transform(np.concatenate(outputs)).reshape(
        len(anchors), anchors.shape[1], -1
    )
    generated = inverse_transform(method, generated, anchors)
    lower = np.quantile(anchors, 0.001, axis=(0, 1))
    upper = np.quantile(anchors, 0.999, axis=(0, 1))
    return np.clip(generated, lower[None, None], upper[None, None]).astype(np.float32)


def classifier_features(windows):
    return np.concatenate([
        windows.reshape(len(windows), -1),
        windows.mean(1),
        windows.std(1),
        windows[:, -1] - windows[:, 0],
    ], 1)


def metrics(y, probability, threshold):
    pred = (probability >= threshold).astype(int)
    return {
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, probability)),
        "average_precision": float(average_precision_score(y, probability)),
        "threshold": float(threshold),
    }


def fit_classifier(x, y, seed):
    model = RandomForestClassifier(
        n_estimators=600,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=seed,
        n_jobs=4,
        bootstrap=True,
        oob_score=True,
    )
    model.fit(classifier_features(x), y)
    oob = model.oob_decision_function_[:, 1]
    precision, recall, thresholds = precision_recall_curve(y, oob)
    f1 = 2.0 * precision * recall / np.maximum(precision + recall, 1e-12)
    best = int(np.nanargmax(f1))
    threshold = float(thresholds[min(best, len(thresholds) - 1)]) if len(thresholds) else 0.5
    return model, threshold


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--seeds", default="42,123,2024")
    parser.add_argument("--multipliers", default="1,5,20")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    seeds = [int(x) for x in args.seeds.split(",")]
    multipliers = [int(x) for x in args.multipliers.split(",")]
    points, startup_audit = read_pairs(args.input)
    windows, labels, dates, stages = make_windows(points)
    all_dates = sorted(np.unique(dates).tolist())
    records = []
    training_records = []
    for test_date in all_dates:
        train_mask = dates != test_date
        test_mask = dates == test_date
        x_train, y_train, d_train = windows[train_mask], labels[train_mask], dates[train_mask]
        x_test, y_test = windows[test_mask], labels[test_mask]
        for seed in seeds:
            baseline, base_threshold = fit_classifier(x_train, y_train, seed)
            base_prob = baseline.predict_proba(classifier_features(x_test))[:, 1]
            records.append({
                "method": "Real-only ExtraTrees baseline",
                "multiplier": 0,
                "seed": seed,
                "test_date": test_date,
                "synthetic_windows": 0,
                **metrics(y_test, base_prob, base_threshold),
            })
            rng = np.random.default_rng(seed)
            for method in METHODS:
                fitted = fit_ddpm(
                    method, x_train, y_train, d_train, all_dates,
                    args.epochs, args.steps, seed,
                )
                max_multiplier = max(multipliers)
                anchor_ids = np.concatenate([
                    rng.choice(np.where(y_train == cls)[0], size=max(1, int((y_train == cls).sum() * max_multiplier)), replace=True)
                    for cls in (0, 1)
                ])
                synthetic = generate(
                    method,
                    fitted,
                    x_train[anchor_ids],
                    y_train[anchor_ids],
                    d_train[anchor_ids],
                    all_dates,
                    seed + 10000,
                )
                synthetic_y = y_train[anchor_ids]
                training_records.append({
                    "method": method,
                    "seed": seed,
                    "test_date": test_date,
                    "real_train_windows": int(len(x_train)),
                    "real_normal": int((y_train == 0).sum()),
                    "real_abnormal": int((y_train == 1).sum()),
                    "max_synthetic_windows": int(len(synthetic)),
                    "epochs": args.epochs,
                    "diffusion_steps": args.steps,
                    "final_noise_loss": fitted["final_loss"],
                })
                for multiplier in multipliers:
                    selected = np.concatenate([
                        np.where(synthetic_y == cls)[0][:
                            int((y_train == cls).sum() * multiplier)
                        ] for cls in (0, 1)
                    ])
                    classifier, threshold = fit_classifier(
                        np.concatenate([x_train, synthetic[selected]], 0),
                        np.concatenate([y_train, synthetic_y[selected]], 0),
                        seed,
                    )
                    probability = classifier.predict_proba(classifier_features(x_test))[:, 1]
                    records.append({
                        "method": method,
                        "multiplier": multiplier,
                        "seed": seed,
                        "test_date": test_date,
                        "synthetic_windows": int(len(selected)),
                        **metrics(y_test, probability, threshold),
                    })
                print(f"completed date={test_date} seed={seed} method={method}", flush=True)
    raw = pd.DataFrame(records)
    group_cols = ["method", "multiplier"]
    summary = raw.groupby(group_cols).agg(
        runs=("f1", "size"),
        synthetic_windows_mean=("synthetic_windows", "mean"),
        precision_mean=("precision", "mean"),
        precision_std=("precision", "std"),
        recall_mean=("recall", "mean"),
        recall_std=("recall", "std"),
        f1_mean=("f1", "mean"),
        f1_std=("f1", "std"),
        roc_auc_mean=("roc_auc", "mean"),
        roc_auc_std=("roc_auc", "std"),
        average_precision_mean=("average_precision", "mean"),
        average_precision_std=("average_precision", "std"),
    ).reset_index().sort_values(["f1_mean", "roc_auc_mean"], ascending=False)
    raw.to_csv(args.output / "all_runs.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(args.output / "summary.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(training_records).to_csv(args.output / "training_audit.csv", index=False, encoding="utf-8-sig")
    startup_audit.to_csv(args.output / "startup_decay_audit.csv", index=False, encoding="utf-8-sig")
    protocol = {
        "source": str(args.input),
        "features": FEATURES,
        "window_length": 6,
        "startup_decay_exclusion": "remove first max(5 points, ceil(20%)) from every continuous error stage",
        "validation": "leave-one-experiment-date-out; test rows are always real",
        "methods": METHODS,
        "seeds": seeds,
        "generation_multipliers": multipliers,
        "epochs": args.epochs,
        "diffusion_steps": args.steps,
        "stable_points": int(len(points)),
        "windows": int(len(windows)),
        "normal_windows": int((labels == 0).sum()),
        "abnormal_windows": int((labels == 1).sum()),
    }
    (args.output / "protocol.json").write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(protocol, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
