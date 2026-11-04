"""Diagnostic: string1 (0686 / 9397 / 9449) sampling and spike synchrony.

Answers two questions:
  1. Is the visible difference caused by a different sampling interval?
     -> no: all three sample on the same ~12 h grid (11:00 / 23:00).
  2. 0686 has far more absolute spikes; are they device-specific?
     -> no: the spikes are synchronised with 9397 (corr of simultaneous
        hourly diffs = 0.89), so the peer difference cancels them. The larger
        absolute amplitude is mostly a level/scale difference.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.rcParams["font.sans-serif"] = ["Noto Sans CJK JP", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

RUN = Path("runs/device_data_110_220_nonoverlap_windows_v3_20260928_final")
STR1 = ["863644079730686", "863644079749397", "863644079729449"]


def robust_z(x: pd.Series) -> pd.Series:
    med = x.median()
    mad = (x - med).abs().median() * 1.4826 + 1e-9
    return (x - med) / mad


def main() -> None:
    p = pd.read_csv(RUN / "device_points_v3.csv", dtype={"IMEI": str}, parse_dates=["Timestamp"])
    series = {}
    interval = {}
    for imei in STR1:
        d = p[p.IMEI == imei].sort_values("Timestamp").copy()
        d["rounded"] = d.Timestamp.dt.round("h")
        d = d.drop_duplicates("rounded").set_index("rounded")
        series[imei[-4:]] = d
        dt = d.Timestamp.diff().dt.total_seconds() / 3600.0
        interval[imei[-4:]] = dt.dropna()

    fig, axes = plt.subplots(2, 2, figsize=(20, 11))

    ax = axes[0, 0]
    for k, d in series.items():
        ax.plot(d.index, d.Measure, lw=.9, alpha=.85, label=f"{k}  median={d.Measure.median():.3f}")
    ax.set_title("(a) 原始 Measure（同轴）：0686 的直流电平和摆幅都最大，所以看着明显不一样")
    ax.set_ylabel("Measure")
    ax.legend(fontsize=9)
    ax.grid(alpha=.25)

    ax = axes[0, 1]
    for k, d in series.items():
        ax.plot(d.index, robust_z(d.Measure), lw=.9, alpha=.85, label=f"{k}（去电平/尺度后）")
    ax.set_title("(b) 每台各自去中位数、按 MAD 归一后：三台几乎重合成一条 → 差异主要是电平/尺度")
    ax.set_ylabel("robust z of Measure")
    ax.legend(fontsize=9)
    ax.grid(alpha=.25)

    ax = axes[1, 0]
    j = pd.concat(
        [series["0686"].Measure.diff().rename("a"), series["9397"].Measure.diff().rename("b")], axis=1
    ).dropna()
    ax.scatter(j.a, j.b, s=12, alpha=.5, color="#2563eb")
    lim = np.nanmax(np.abs(j.to_numpy())) * 1.05
    ax.plot([-lim, lim], [-lim, lim], color="#94a3b8", lw=.8, ls="--")
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_title(f"(c) 同一时刻的两台一步差分 0686 vs 9397：r={j.a.corr(j.b):.2f} → 跳变高度同步")
    ax.set_xlabel("0686 hourly diff")
    ax.set_ylabel("9397 hourly diff")
    ax.grid(alpha=.25)

    ax = axes[1, 1]
    labels = ["|d|>0.3", "|d|>0.5", "|d|>0.8"]
    x = np.arange(len(labels))
    w = 0.26
    counts = {}
    for k, d in series.items():
        dif = d.Measure.diff().abs().dropna()
        counts[k] = [int((dif > t).sum()) for t in (0.3, 0.5, 0.8)]
    for i, (k, c) in enumerate(counts.items()):
        bars = ax.bar(x + (i - 1) * w, c, w, label=k)
        ax.bar_label(bars, fontsize=9)
    ax.set_xticks(x, labels)
    ax.set_title("(d) 绝对尖刺数：0686 约为同串另一台的 2~3 倍，但这些尖刺多数与 peer 同时发生")
    ax.set_ylabel("count")
    ax.legend(fontsize=9)
    ax.grid(alpha=.25, axis="y")

    iv = pd.DataFrame(interval)
    print("sampling interval (hours) per device:\n", iv.describe().loc[["mean", "50%", "min", "max"]])
    print("corr of simultaneous hourly diffs 0686-9397 =", round(j.a.corr(j.b), 3))

    fig.suptitle("串1 三台：采样节律相同、尖刺同步 —— 0686 的“波动大”主要是电平/尺度差异", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, .96))
    out = RUN / "diagnostic_0686_string1_sync.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print("saved", out)


if __name__ == "__main__":
    main()
