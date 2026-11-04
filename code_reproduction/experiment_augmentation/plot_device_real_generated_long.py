"""Plot per-device long real and generated series in the pseudo-label style."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def test_points(run: Path, train_frac: float) -> pd.DataFrame:
    points = pd.read_csv(run / "annotated_device_points.csv", parse_dates=["Timestamp"])
    out = []
    for imei, group in points.groupby("IMEI", sort=False):
        g = group.sort_values("Timestamp").reset_index(drop=True)
        cut = max(1, min(len(g) - 1, int(len(g) * train_frac)))
        out.append(g.iloc[cut:].copy())
    return pd.concat(out, ignore_index=True)


def generated_points(run: Path, ratio: str, method: str, seed: int) -> pd.DataFrame:
    safe = method.replace("/", "_").replace("-", "_")
    raw_path = run / f"raw_{ratio}_{safe}_seed{seed}.npz"
    metadata_path = run / f"generated_metadata_{ratio}_{safe}_seed{seed}.csv"
    with np.load(raw_path, allow_pickle=True) as data:
        windows = data["windows"]
        labels = data["labels"].astype(int)
    meta = pd.read_csv(metadata_path)
    if len(meta) != len(windows):
        raise ValueError(f"metadata/raw length mismatch: {len(meta)} != {len(windows)}")
    rows = []
    for i, row in meta.iterrows():
        start = pd.to_datetime(row["start_time"])
        end = pd.to_datetime(row["end_time"])
        times = pd.date_range(start, end, periods=windows.shape[1])
        for j, timestamp in enumerate(times):
            rows.append({
                "IMEI": str(row["IMEI"]),
                "Timestamp": timestamp,
                "Measure": float(windows[i, j, 0]),
                "label": int(labels[i]),
                "source_id": int(row["source_id"]),
            })
    return pd.DataFrame(rows)


def plot_one(run: Path, ratio: str, train_frac: float, method: str, seed: int) -> Path:
    real = test_points(run, train_frac)
    generated = generated_points(run, ratio, method, seed)
    devices = list(real["IMEI"].drop_duplicates())
    ncols = 4
    nrows = int(np.ceil(len(devices) / 2))
    fig, axes = plt.subplots(nrows, ncols, figsize=(22, 3.1 * nrows), squeeze=False)
    axes = axes.ravel()
    for device_index, imei in enumerate(devices):
        rg = real[real.IMEI.astype(str).eq(str(imei))]
        gg = generated[generated.IMEI.astype(str).eq(str(imei))]
        pair_row = device_index // 2
        pair_col = device_index % 2
        base = pair_row * ncols + pair_col * 2
        real_ax = axes[base]
        gen_ax = axes[base + 1]

        real_ax.plot(rg.Timestamp, rg.Measure, color="#64748b", lw=0.7, alpha=0.75)
        normal = rg.pseudo_anomaly.eq(0)
        real_ax.scatter(rg.loc[normal, "Timestamp"], rg.loc[normal, "Measure"], color="#2563eb", s=5, alpha=0.55, label="normal")
        bad = rg.pseudo_anomaly.eq(1)
        real_ax.scatter(rg.loc[bad, "Timestamp"], rg.loc[bad, "Measure"], color="#dc2626", s=12, alpha=0.95, label="candidate anomaly")
        real_ax.set_title(f"{imei} real | anomaly={int(bad.sum())}", fontsize=8)

        if not gg.empty:
            # Draw the raw line first, then draw every point on top. This keeps
            # zero and final samples visible instead of being hidden by a line.
            gen_ax.plot(gg.Timestamp, gg.Measure, color="#64748b", lw=0.7, alpha=0.75)
            gnormal = gg.label.eq(0)
            gbad = gg.label.eq(1)
            gen_ax.scatter(gg.loc[gnormal, "Timestamp"], gg.loc[gnormal, "Measure"], color="#2563eb", s=5, alpha=0.55, label="normal")
            gen_ax.scatter(gg.loc[gbad, "Timestamp"], gg.loc[gbad, "Measure"], color="#dc2626", s=12, alpha=0.95, label="candidate anomaly")
            gen_ax.set_title(f"{imei} generated raw | anomaly={int(gbad.sum())}", fontsize=8)
            values = gg.Measure.to_numpy(float)
            if np.nanmin(values) <= 0:
                gen_ax.set_ylim(min(np.nanmin(values), 0.0), np.nanmax(values) * 1.05)
        else:
            gen_ax.set_title(f"{imei} generated raw | no windows", fontsize=8)

        for ax in (real_ax, gen_ax):
            ax.grid(alpha=0.25)
            ax.tick_params(axis="x", labelrotation=30, labelsize=6)
            ax.tick_params(axis="y", labelsize=7)
            ax.set_ylabel("Measure")

    for ax in axes[len(devices) * 2:]:
        ax.axis("off")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.005), fontsize=9)
    fig.suptitle(f"Per-device long-timeseries comparison | {ratio} train:test | {method} | seed {seed}", fontsize=14, y=1.025)
    fig.text(0.5, 0.003, "Same background and styling as pseudo labels. Generated raw timestamps are inherited from source windows; all points are drawn above the line, including zero and final samples.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.02, 1, 0.98))
    out = run / "figures" / f"per_device_real_vs_generated_{ratio}_{method.replace('-', '_')}_seed{seed}.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--ratio", default="2to8")
    parser.add_argument("--train-fraction", type=float, required=True)
    parser.add_argument("--method", default="CCDM")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print("wrote", plot_one(args.run, args.ratio, args.train_fraction, args.method, args.seed))


if __name__ == "__main__":
    main()
