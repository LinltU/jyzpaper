"""Diagnostic for the corrected v3 peer-delta jump: 0694 vs 0611.

0611 (string1) and 0694 (string2) sit at the same tower position (第九~十片) on
the two parallel strings, so their raw Measure curves look nearly identical.
The old v3 compared the peer deltas of *different* devices at the same instant
(timestamps nearly coincide), so a between-device level difference was counted
as a "jump". The corrected rule differentiates each device's own peer delta over
time, only where the full peer panel is matched.

This figure shows (a) the raw Measure of both devices (almost identical, hence
the user's intuition), (b) the per-device peer delta with the corrected jump
points marked, and (c) the corrected jump z per point.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

matplotlib.rcParams["font.sans-serif"] = ["Noto Sans CJK JP", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

RUN = Path("runs/device_data_110_220_nonoverlap_windows_v3_20260928_final")
TARGETS = [("863644079730611", "0611 (串1)"), ("863644079730694", "0694 (串2)")]
THR = 8.0


def main() -> None:
    points = pd.read_csv(RUN / "device_points_v3.csv", dtype={"IMEI": str}, parse_dates=["Timestamp"])
    fig, axes = plt.subplots(3, 1, figsize=(19, 12), sharex=True)

    for imei, label in TARGETS:
        d = points[points.IMEI == imei].sort_values("Timestamp")
        axes[0].plot(d.Timestamp, d.Measure, lw=1.0, alpha=.85, label=label)
    axes[0].set_ylabel("Measure")
    axes[0].set_title("(a) 原始 Measure：0611 与 0694 曲线几乎重合（两台看起来一样）", fontsize=11)
    axes[0].legend(fontsize=9, ncol=2)
    axes[0].grid(alpha=.25)

    for imei, label in TARGETS:
        d = points[points.IMEI == imei].sort_values("Timestamp")
        axes[1].plot(d.Timestamp, d.peer_delta, lw=1.0, alpha=.8, label=f"{label} peer delta")
        hi = d[d.peer_delta_jump_z >= THR]
        axes[1].scatter(hi.Timestamp, hi.peer_delta, s=45, marker="o", facecolors="none",
                        edgecolors="#dc2626", lw=1.5, zorder=5)
    axes[1].axhline(0, color="#64748b", lw=.7)
    axes[1].set_ylabel("peer delta\n(自身残差 - 同串伙伴中位数)")
    axes[1].set_title(f"(b) 修正后的逐设备 peer delta；红圈 = 修正后判为跳变的点 (jump z >= {THR})", fontsize=11)
    axes[1].legend(fontsize=9, ncol=2)
    axes[1].grid(alpha=.25)

    for imei, label in TARGETS:
        d = points[points.IMEI == imei].sort_values("Timestamp")
        axes[2].plot(d.Timestamp, d.peer_delta_jump_z, lw=0.9, alpha=.8, label=f"{label} jump z")
    axes[2].axhline(THR, color="#dc2626", ls="--", lw=1.0, label=f"threshold = {THR}")
    axes[2].set_ylabel("peer delta jump z\n(以本设备 peer delta 的 MAD 为单位)")
    axes[2].set_xlabel("time")
    axes[2].set_title("(c) 修正后的跳变 z：两台在相近时刻都出现真实跳变（量级接近）", fontsize=11)
    axes[2].legend(fontsize=9, ncol=2)
    axes[2].grid(alpha=.25)

    fig.suptitle("修正 v3：0611 与 0694 的“跳变”现在按各自 peer delta 的时间差分计算，两台获得可比的处理", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, .97))
    out = RUN / "diagnostic_0611_0694_fixed.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print("saved", out)


if __name__ == "__main__":
    main()
