"""Trend comparison of normal vs abnormal: raw physical signal, and real vs augmented feature trends.

Design (control-variable friendly):
  Row 1 (physical signal): real measured voltage trend per test date, abnormal段标红 -> 直观看出异常存在。
  Row 2 (feature trend):   M4_residual 4-step trajectory, normal vs abnormal, for
                           (a) real windows and (b) CSDI-augmented synthetic windows.
                           Same feature / same axes / same style -> the only change is real vs augmented.
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

HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs"
FIG = RUNS / "figures"
FIG.mkdir(parents=True, exist_ok=True)
FIXED = RUNS / "upstream_gpu_full_20260910"
XLSX = Path("/home/test/linliting/jyzmentor/experiment.xlsx")

DATES = ["250605", "250626", "250718"]
SEEDS = (42, 123, 2024)
METHOD = "CSDI"
FEAT = "M4_residual"
J = 0  # feature index of M4_residual
NORMAL_C = "#1f77b4"
ABNORMAL_C = "#d62728"

DATE_DEVICE = {  # one device sheet per date (phase 4), for a clean physical-signal panel
    "250605": "250605_80kv_9p_0819(4)",
    "250626": "250626_80kv_9p_0603(4)",
    "250718": "250718_80kv_9p_0603(4)",
}


def load_real():
    x = np.load(FIXED / "dataset.npz")["x"]
    meta = pd.read_json(FIXED / "metadata.json", dtype={"date": str})
    return x, meta.label.to_numpy()


def load_synth_date(date):
    arrs, labs = [], []
    for s in SEEDS:
        p = FIXED / METHOD / f"seed{s}" / date / "synthetic.npz"
        if p.exists():
            d = np.load(p)
            arrs.append(d["raw"])
            labs.append(d["label"])
    return np.concatenate(arrs), np.concatenate(labs)


def _band(ax, X, y, color, label, alpha_band=0.20, n_faint=60):
    traj = X[:, :, J]
    t = np.arange(traj.shape[1])
    for row in traj[np.random.default_rng(0).choice(len(traj), min(n_faint, len(traj)), replace=False)]:
        ax.plot(t, row, color=color, lw=0.5, alpha=0.10)
    m, sd = traj.mean(0), traj.std(0)
    ax.plot(t, m, "-o", color=color, lw=2.6, ms=6, label=label, zorder=5)
    ax.fill_between(t, m - sd, m + sd, color=color, alpha=alpha_band, zorder=2)


def main():
    xr, yr = load_real()
    xl = pd.ExcelFile(XLSX)

    fig = plt.figure(figsize=(19, 11))
    gs = fig.add_gridspec(2, 6, height_ratios=[1.0, 1.15], hspace=0.32, wspace=0.5)

    # ---------------- Row 1: raw physical signal ----------------
    for k, date in enumerate(DATES):
        ax = fig.add_subplot(gs[0, 2 * k:2 * k + 2])
        df = xl.parse(DATE_DEVICE[date])
        idx = np.arange(len(df))
        meas = df["Measure"].to_numpy()
        err = df["error"].to_numpy() != 0
        ax.plot(idx, meas, color=NORMAL_C, lw=0.8, alpha=0.5)
        # highlight abnormal segments in red
        seg = np.where(err)[0]
        if len(seg):
            ax.plot(seg, meas[seg], color=ABNORMAL_C, lw=2.2)
            ax.scatter(seg, meas[seg], color=ABNORMAL_C, s=14, zorder=5,
                       label="abnormal (error≠0)")
        ax.scatter(idx[~err], meas[~err], color=NORMAL_C, s=8, alpha=0.55, label="normal")
        ax.set_title(f"{date}  ({DATE_DEVICE[date].split('_')[-1].strip('()')} device)")
        ax.set_xlabel("sample index")
        if k == 0:
            ax.set_ylabel("Measure (voltage)")
            ax.legend(fontsize=8, loc="upper right")
        ax.grid(alpha=0.25)

    # ---------------- Row 2: feature trend, real vs augmented ----------------
    syn_all, syn_all_lab = [], []
    for date in DATES:
        s, l = load_synth_date(date)
        syn_all.append(s); syn_all_lab.append(l)
    syn_all = np.concatenate(syn_all); syn_all_lab = np.concatenate(syn_all_lab)

    data = [
        ("(a) 原始真实数据: 正常 vs 异常", xr, yr),
        ("(b) 增强合成数据 (CSDI): 正常 vs 异常", syn_all, syn_all_lab),
    ]
    ymin = min(data[0][1][:, :, J].min(), np.percentile(data[1][1][:, :, J], 0.05))
    ymax = max(data[0][1][:, :, J].max(), np.percentile(data[1][1][:, :, J], 99.95))
    for k, (title, X, y) in enumerate(data):
        ax = fig.add_subplot(gs[1, 3 * k:3 * k + 3])
        _band(ax, X[y == 0], 0, NORMAL_C, f"normal (n={int((y==0).sum())})")
        _band(ax, X[y == 1], 1, ABNORMAL_C, f"abnormal (n={int((y==1).sum())})")
        ax.set_title(title)
        ax.set_xlabel("window step")
        ax.set_xticks(np.arange(4))
        ax.set_ylim(ymin, ymax)
        ax.set_ylabel(FEAT)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=9, loc="lower left")
        if k == 0:
            ax.text(0.02, 0.02, "overlap大 → 不易区分", transform=ax.transAxes,
                    fontsize=10, color="black", ha="left", va="bottom")
        else:
            ax.text(0.02, 0.02, "异常残差更负、分布更宽\n仍有重叠 → 有提升空间",
                    transform=ax.transAxes, fontsize=10, color="black", ha="left", va="bottom")

    fig.suptitle("Normal vs abnormal trends: raw voltage signal and real vs augmented feature trends",
                 y=0.98, fontsize=14)
    p = FIG / "trend_normal_vs_abnormal.png"
    fig.savefig(p, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("wrote", p)


if __name__ == "__main__":
    main()
