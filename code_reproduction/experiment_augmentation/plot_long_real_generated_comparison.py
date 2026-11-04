"""Plot long real and generated Measure series with anomaly markers.

Generated windows do not contain physical timestamps or device IDs. They are
therefore plotted in source-window order on a synthetic observation index, and
the figure labels this limitation explicitly.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


RATIOS = {"2to8": 0.2, "3to7": 0.3, "4to6": 0.4, "5to5": 0.5}


def test_start_times(meta: pd.DataFrame, train_frac: float) -> dict[str, pd.Timestamp]:
    starts = {}
    for imei, group in meta.groupby("IMEI", sort=False):
        ids = group.index.to_numpy()
        cut = max(1, min(len(ids) - 1, int(len(ids) * train_frac)))
        starts[str(imei)] = pd.to_datetime(group.iloc[cut]["start_time"])
    return starts


def real_series(run: Path, train_frac: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    points = pd.read_csv(run / "annotated_device_points.csv", parse_dates=["Timestamp"])
    meta = pd.read_json(run / "window_metadata.json")
    starts = test_start_times(meta, train_frac)
    chunks, labels, separators = [], [], []
    total = 0
    for imei, group in points.groupby("IMEI", sort=False):
        g = group.sort_values("Timestamp")
        g = g[g["Timestamp"] >= starts[str(imei)]]
        if g.empty:
            continue
        values = g["Measure"].to_numpy(float)
        bad = g["pseudo_anomaly"].to_numpy(int)
        chunks.append(values)
        labels.append(bad)
        total += len(values)
        separators.append(total)
    return np.concatenate(chunks), np.concatenate(labels), np.asarray(separators[:-1])


def generated_series(run: Path, ratio: str, method: str, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    safe_method = method.replace("/", "_").replace("-", "_")
    path = run / f"raw_{ratio}_{safe_method}_seed{seed}.npz"
    with np.load(path) as data:
        windows = data["windows"]
        labels = data["labels"].astype(int)
        source_ids = data["source_ids"].astype(int)
    order = np.argsort(source_ids, kind="stable")
    values = windows[order, :, 0].reshape(-1)
    point_labels = np.repeat(labels[order], windows.shape[1])
    ends = np.arange(windows.shape[1], len(values), windows.shape[1])
    return values, point_labels, ends[:-1]


def draw_long(ax, values, labels, separators, title, ylabel=True):
    x = np.arange(len(values))
    ax.plot(x, values, color="#64748b", lw=0.55, alpha=0.72)
    normal = labels == 0
    anomaly = labels.astype(bool)
    ax.scatter(x[normal], values[normal], color="#2563eb", s=2.2, alpha=0.45, label="normal")
    ax.scatter(x[anomaly], values[anomaly], color="#dc2626", s=7, alpha=0.85, label="candidate anomaly")
    for pos in separators:
        ax.axvline(pos, color="#94a3b8", lw=0.45, alpha=0.35)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("Observation index")
    if ylabel:
        ax.set_ylabel("Measure")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    ax.text(0.01, 0.03, f"anomaly points: {int(anomaly.sum()):,}/{len(values):,} ({anomaly.mean():.2%})", transform=ax.transAxes, fontsize=8, bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none"})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run = args.run
    best = pd.read_csv(run / "formal_best_methods.csv")
    fig, axes = plt.subplots(len(RATIOS), 2, figsize=(18, 13), squeeze=False)

    for row, (ratio, train_frac) in enumerate(RATIOS.items()):
        method = str(best[best.ratio.eq(ratio)].iloc[0].method)
        rv, rl, rs = real_series(run, train_frac)
        gv, gl, gs = generated_series(run, ratio, method, args.seed)
        all_values = np.r_[rv, gv]
        axes[row, 0].set_ylim(np.nanmin(all_values), np.nanmax(all_values))
        axes[row, 1].set_ylim(np.nanmin(all_values), np.nanmax(all_values))
        draw_long(axes[row, 0], rv, rl, rs, f"{ratio} real test data", ylabel=True)
        draw_long(axes[row, 1], gv, gl, gs, f"{ratio} generated raw data ({method})", ylabel=False)
        axes[row, 0].text(-0.07, 1.08, f"{ratio} train:test", transform=axes[row, 0].transAxes, fontsize=11, weight="bold")

    fig.suptitle("Long-timeseries comparison: real pseudo-anomaly labels vs generated anomaly labels", fontsize=15, y=0.995)
    fig.text(0.5, 0.005, "Generated data has no physical timestamps/device IDs; its x-axis is the concatenated source-window order. Red points are candidate anomaly labels.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.02, 1, 0.985))
    out = run / "figures" / f"real_vs_generated_long_timeseries_seed{args.seed}.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    main()
