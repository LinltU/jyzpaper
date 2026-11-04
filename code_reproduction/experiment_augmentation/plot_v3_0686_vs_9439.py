"""Diagnostic: why 0686 (visually more volatile) is normal, but 9439 is flagged.

Two effects are shown:
  (a) string1 (3 devices): 0686 swings the most in raw Measure, but all three
      move together, so the per-device peer delta stays modest and no jump
      crosses the threshold.
  (b) string2 (2 devices): peer_delta(A) = -(peer_delta(B)), so any relative
      jump flags BOTH devices at the same timestamps with opposite signs.
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
STR1 = ["863644079730686", "863644079749397", "863644079729449"]
STR2 = ["863644079749439", "863644079749363"]
THR = 8.0


def main() -> None:
    p = pd.read_csv(RUN / "device_points_v3.csv", dtype={"IMEI": str}, parse_dates=["Timestamp"])
    fig, axes = plt.subplots(2, 2, figsize=(20, 11))

    # (a) raw Measure, string1
    ax = axes[0, 0]
    for imei in STR1:
        d = p[p.IMEI == imei].sort_values("Timestamp")
        ax.plot(d.Timestamp, d.Measure, lw=.9, alpha=.85, label=f"{imei[-4:]} std={d.Measure.std():.3f}")
    ax.set_title("(a) 串1（3台）原始 Measure：0686 波动最大，但三台同起同落")
    ax.set_ylabel("Measure")
    ax.legend(fontsize=9)
    ax.grid(alpha=.25)

    # (b) peer delta, string1 (should be small / no big jumps)
    ax = axes[0, 1]
    for imei in STR1:
        d = p[p.IMEI == imei].sort_values("Timestamp")
        ax.plot(d.Timestamp, d.peer_delta, lw=.9, alpha=.8, label=f"{imei[-4:]} peer delta")
    ax.axhline(0, color="#64748b", lw=.7)
    ax.set_title("(b) 串1 各设备 peer delta：0686 虽有波动，但没有越过 jump 阈值的大跳变")
    ax.set_ylabel("peer delta")
    ax.legend(fontsize=9)
    ax.grid(alpha=.25)

    # (c) raw Measure, string2
    ax = axes[1, 0]
    for imei in STR2:
        d = p[p.IMEI == imei].sort_values("Timestamp")
        ax.plot(d.Timestamp, d.Measure, lw=.9, alpha=.85, label=f"{imei[-4:]} std={d.Measure.std():.3f}")
    ax.set_title("(c) 串2（仅2台）原始 Measure")
    ax.set_ylabel("Measure")
    ax.legend(fontsize=9)
    ax.grid(alpha=.25)

    # (d) peer delta, string2: exact mirror
    ax = axes[1, 1]
    for imei in STR2:
        d = p[p.IMEI == imei].sort_values("Timestamp")
        ax.plot(d.Timestamp, d.peer_delta, lw=.9, alpha=.8, label=f"{imei[-4:]} peer delta")
        hi = d[d.peer_delta_jump_z >= THR]
        ax.scatter(hi.Timestamp, hi.peer_delta, s=50, marker="o", facecolors="none",
                   edgecolors="#dc2626", lw=1.6, zorder=5)
    ax.axhline(0, color="#64748b", lw=.7)
    ax.set_title(f"(d) 串2 peer delta：两台的曲线完全镜像（红圈=同一批 jump 点，两台被同时标上）")
    ax.set_ylabel("peer delta")
    ax.legend(fontsize=9)
    ax.grid(alpha=.25)

    fig.suptitle("为什么0686波动更大却没被标，而9439被标？—— 原始波动 vs 相对peer的跳变", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, .96))
    out = RUN / "diagnostic_0686_vs_9439.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print("saved", out)


if __name__ == "__main__":
    main()
