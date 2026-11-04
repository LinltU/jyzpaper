"""Diagnostic figure for device_data_110_220_final.xlsx characteristics.

Panels:
  1. Sampling interval vs record index (dense early -> sparse late).
  2. Measure vs humidity bins (mean and p95).
  3. Peer difference vs humidity (confounding check).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

sys.path.insert(0, str(Path(__file__).resolve().parent))
from device_data_paper_augmentation import DEVICE_LOCATIONS, location_group

WORKBOOK = Path("/home/test/linliting/jyzmentor/device_data_110_220_final.xlsx")


def load_points(workbook: Path) -> pd.DataFrame:
    rows = []
    for sheet, frame in pd.read_excel(workbook, sheet_name=None).items():
        q = frame.copy().dropna(subset=["Timestamp", "Measure"])
        q["Timestamp"] = pd.to_datetime(q["Timestamp"])
        for c in ["Measure", "Temp", "Hum"]:
            q[c] = pd.to_numeric(q[c], errors="coerce")
        q = q.dropna(subset=["Measure"]).sort_values("Timestamp")
        q["IMEI"] = str(sheet)
        q["Location"] = q.IMEI.map(DEVICE_LOCATIONS)
        q["group_id"] = q.Location.map(location_group)
        rows.append(q)
    return pd.concat(rows, ignore_index=True)


def main(args):
    args.output.mkdir(parents=True, exist_ok=True)
    p = load_points(args.workbook)

    fig, axes = plt.subplots(1, 3, figsize=(19, 5.2))

    # Panel 1: interval vs record index
    ax = axes[0]
    for imei, g in p.groupby("IMEI"):
        g = g.sort_values("Timestamp")
        dt = g.Timestamp.diff().dt.total_seconds().to_numpy(float)
        idx = np.arange(len(dt)) / max(1, len(dt) - 1)
        smooth = pd.Series(dt).rolling(30, min_periods=5).median().to_numpy()
        ax.plot(idx, smooth, lw=.9, alpha=.6)
    ax.set_yscale("log")
    ax.set_xlabel("record progress (0=deployment, 1=latest)")
    ax.set_ylabel("median sampling interval (s, log)")
    ax.set_title("Sampling interval grows over time")
    ax.grid(alpha=.25)

    # Panel 2: Measure vs humidity bins
    ax = axes[1]
    bins = [0, 30, 40, 50, 55, 60, 65, 70, 75, 80, 90, 100]
    p2 = p.dropna(subset=["Measure", "Hum"]).copy()
    p2["hum_bin"] = pd.cut(p2.Hum, bins=bins, right=False)
    agg = p2.groupby("hum_bin", observed=True).agg(
        center=("Hum", "mean"),
        mean=("Measure", "mean"),
        p95=("Measure", lambda s: np.percentile(s, 95)),
    ).reset_index()
    ax.plot(agg.center, agg["mean"], "o-", label="mean")
    ax.plot(agg.center, agg.p95, "s--", label="p95")
    ax.axvline(60, color="red", ls=":", lw=1.5, label="Hum=60")
    ax.set_xlabel("Humidity")
    ax.set_ylabel("Measure")
    ax.set_title("Measure rises with humidity (spike above 60)")
    ax.legend()
    ax.grid(alpha=.25)

    # Panel 3: peer difference vs humidity
    ax = axes[2]
    for gid, grp in p.groupby("group_id"):
        devs = sorted(grp.IMEI.unique())
        if len(devs) < 2:
            continue
        a = grp[grp.IMEI == devs[0]][["Timestamp", "Measure"]].sort_values("Timestamp")
        b = grp[grp.IMEI == devs[1]][["Timestamp", "Measure"]].sort_values("Timestamp")
        m = pd.merge_asof(a, b, on="Timestamp", direction="nearest", tolerance=pd.Timedelta("10min"), suffixes=("_a", "_b"))
        h = grp[grp.IMEI == devs[0]][["Timestamp", "Hum"]].sort_values("Timestamp")
        m = pd.merge_asof(m, h, on="Timestamp", direction="nearest")
        m = m.dropna()
        if len(m) < 50:
            continue
        ax.scatter(m.Hum, m.Measure_a - m.Measure_b, s=4, alpha=.25)
    ax.axvline(60, color="red", ls=":", lw=1.5)
    ax.set_xlabel("Humidity")
    ax.set_ylabel("peer difference (A - B)")
    ax.set_title("Peer difference still scales with humidity")
    ax.grid(alpha=.25)

    fig.suptitle("device_data_110_220_final.xlsx characteristics: time, humidity, peer", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, .96))
    fig.savefig(args.output / "device_data_characteristics.png", dpi=170, bbox_inches="tight")
    print("saved", args.output / "device_data_characteristics.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workbook", type=Path, default=WORKBOOK)
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args())
