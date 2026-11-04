"""Dual-device (device 4 + device 7) voltage trend: normal vs abnormal, before/after augmentation.

Why dual device?
  The engineering task pairs insulator #4 and #7 measured on the SAME string at nearly the
  same time (40s nearest-timestamp merge_asof).  The 5 joint features are
  M4_residual, M7_residual, M4-M7, M4/M7, log(M4/M7) -- i.e. both devices and their
  RELATION.  So a trend figure must show device 4 AND device 7, plus their linkage,
  not device 4 alone.

Control variables (identical on both sides):
  * same physical quantity : experiment.xlsx "Measure" of both devices (volts), paired rows
  * same window length      : 4 steps (matches dataset.npz, shape (n, 4, 5))
  * same colours / style    : normal = blue, abnormal = red, mean +/- std band + faint lines
  * same sample size        : the CSDI-augmented side is subsampled per date to the exact
                              n_normal / n_abnormal of the real side (the raw synthetic pool is
                              50/50 balanced with 60k windows/date and would look unfairly clean)
  * residual zoom      : Rows 2-3 plot  Measure - daily_base.  Because base is one constant
                        per date, (abnormal_mean - normal_mean) is numerically IDENTICAL to
                        the raw-voltage gap; the residual axis merely zooms into the
                        fluctuation, a magnifier for the same gap (not a new quantity).
  * separate y-axis per row : synthetic residual scale differs from real by 5-8x on some dates

Layout (4 rows x 3 columns, column = held-out test date; wide rows span 2 sub-columns):
  Row 1 : raw paired Excel voltage of BOTH devices (anomalies directly visible)
  Row 2 : real 4-step residual trajectories, device 4 and device 7, normal vs abnormal
  Row 3 : CSDI-augmented 4-step residual trajectories (sample-count matched), both devices
  Row 4 : dual-device linkage (M4_residual vs M7_residual scatter) - does the relation break?
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

matplotlib.rcParams["font.sans-serif"] = ["Noto Sans CJK JP", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs"
FIG = RUNS / "figures"
FIG.mkdir(parents=True, exist_ok=True)
FIXED = RUNS / "upstream_gpu_full_20260910"

DATES = ["250605", "250626", "250718"]
SEEDS = (42, 123, 2024)
METHOD = "CSDI"

NORMAL_C = "#1f77b4"
ABNORMAL_C = "#d62728"
DEV4_C = "#1f77b4"
DEV7_C = "#ff7f0e"
DEV = {4: DEV4_C, 7: DEV7_C}
J4, J7 = 0, 1  # feature index of M4_residual / M7_residual


def load_dataset():
    x = np.load(FIXED / "dataset.npz")["x"]
    meta = pd.read_json(FIXED / "metadata.json", dtype={"date": str}).reset_index(drop=True)
    return x, meta


def load_pair_audit():
    df = pd.read_csv(FIXED / "paired_row_audit.csv", dtype={"date": str})
    df["dt"] = pd.to_datetime(df["dt"])
    return df


def load_synth(date, key="raw"):
    arrs, labs = [], []
    for s in SEEDS:
        p = FIXED / METHOD / f"seed{s}" / date / "synthetic.npz"
        if p.exists():
            d = np.load(p)
            arrs.append(d[key])
            labs.append(d["label"])
    return np.concatenate(arrs), np.concatenate(labs)


def band(ax, traj, color, label, ls="-", n_faint=40, seed=0):
    t = np.arange(traj.shape[1])
    pick = np.random.default_rng(seed).choice(len(traj), min(n_faint, len(traj)), replace=False)
    for row in traj[pick]:
        ax.plot(t, row, color=color, ls=ls, lw=0.5, alpha=0.10)
    m, sd = traj.mean(0), traj.std(0)
    ax.plot(t, m, ls, color=color, lw=2.4, marker="o", ms=5, label=label, zorder=5)
    ax.fill_between(t, m - sd, m + sd, color=color, alpha=0.18, zorder=2)


def ylim(a):
    lo_, hi_ = np.percentile(a, 0.5), np.percentile(a, 99.5)
    pad_ = 0.15 * (hi_ - lo_ + 1e-12)
    return lo_ - pad_, hi_ + pad_


def match(syn, syn_lab, n0, n1, rng):
    i0 = rng.choice(np.flatnonzero(syn_lab == 0), n0, replace=False)
    i1 = rng.choice(np.flatnonzero(syn_lab == 1), n1, replace=False)
    return syn[i0], syn[i1]


def main():
    x, meta = load_dataset()
    y = meta.label.to_numpy()
    audit = load_pair_audit()
    b4 = meta.groupby("date")["base4"].first().to_dict()
    b7 = meta.groupby("date")["base7"].first().to_dict()

    fig = plt.figure(figsize=(21, 17.5))
    gs = fig.add_gridspec(4, 6, height_ratios=[1.0, 1.0, 1.0, 1.05],
                          top=0.885, bottom=0.055, left=0.05, right=0.985,
                          hspace=0.62, wspace=0.34)

    recap = {}
    for k, date in enumerate(DATES):
        # ================= Row 1: raw paired dual-device voltage =================
        ax = fig.add_subplot(gs[0, 2 * k:2 * k + 2])
        g = audit[audit.date == date].sort_values("dt").reset_index(drop=True)
        idx = np.arange(len(g))
        m4 = g.Measure4.to_numpy()
        m7 = g.Measure7.to_numpy()
        bad = ((g.error4 != 0) | (g.error7 != 0)).to_numpy()
        ax.plot(idx, m4, color=DEV4_C, lw=1.0, alpha=0.75, label="device 4 的 Measure")
        ax.plot(idx, m7, color=DEV7_C, lw=1.0, alpha=0.75, label="device 7 的 Measure")
        if bad.any():
            for s in np.split(idx[bad], np.flatnonzero(np.diff(idx[bad]) > 1) + 1):
                ax.axvspan(s[0] - 0.5, s[-1] + 0.5, color=ABNORMAL_C, alpha=0.12, zorder=0)
            ax.scatter(idx[bad], m4[bad], color=ABNORMAL_C, s=14, zorder=5)
            ax.scatter(idx[bad], m7[bad], color=ABNORMAL_C, s=14, marker="s", zorder=5)
        ax.set_title(f"{date}  原始配对双设备电压 (device 4 / 7)", fontsize=11, pad=8)
        ax.set_xlabel("配对后采样点序号")
        ax.grid(alpha=0.25)
        if k == 0:
            ax.set_ylabel("Measure 电压 (V)")
            ax.legend(fontsize=8, loc="best")

        # ================= prepare real / augmented dual-device windows =================
        sel = (meta.date == date).to_numpy()
        r4, r7 = x[sel][:, :, J4], x[sel][:, :, J7]
        yr = y[sel]
        n0, n1 = int((yr == 0).sum()), int((yr == 1).sum())

        syn, syn_lab = load_synth(date)
        rng = np.random.default_rng(20240501 + k)
        a4n, a4a = match(syn[:, :, J4], syn_lab, n0, n1, rng)
        rng = np.random.default_rng(20240501 + 100 + k)
        a7n, a7a = match(syn[:, :, J7], syn_lab, n0, n1, rng)

        gap_real4 = abs(r4[yr == 1].mean(0) - r4[yr == 0].mean(0)).mean()
        gap_aug4 = abs(a4a.mean(0) - a4n.mean(0)).mean()
        gap_real7 = abs(r7[yr == 1].mean(0) - r7[yr == 0].mean(0)).mean()
        gap_aug7 = abs(a7a.mean(0) - a7n.mean(0)).mean()
        recap[date] = (n0, n1, gap_real4, gap_aug4, gap_real7, gap_aug7)

        # ================= Row 2/3: per-device residual trajectories =================
        panels = [
            (1, "真实残差 (增强前)", [[r4[yr == 0], r4[yr == 1]], [r7[yr == 0], r7[yr == 1]]]),
            (2, "CSDI 增强残差 (样本对齐)",
             [[a4n, a4a], [a7n, a7a]]),
        ]
        # Residual = voltage - base, a per-date constant, so the blue/red gap below is
        # numerically IDENTICAL to the raw-voltage gap; the residual axis just zooms in.
        for row, tag, dev_data in panels:
            for dcol, (dev_id, (dn, da)) in enumerate(zip((4, 7), dev_data)):
                ax = fig.add_subplot(gs[row, 2 * k + dcol])
                band(ax, dn, NORMAL_C, f"正常 (n={n0})", seed=10 + dcol)
                band(ax, da, ABNORMAL_C, f"异常 (n={n1})", seed=20 + dcol)
                ax.set_title(f"{date}  device {dev_id}\n{tag}", fontsize=9.5, pad=6)
                ax.set_xlabel("窗口内第几步"); ax.set_xticks(np.arange(4))
                ax.set_ylim(*ylim(np.concatenate([dn, da]))); ax.grid(alpha=0.25)
                if dcol == 0:
                    ax.set_ylabel("相对当日基线的偏移 (V)")
                ax.legend(fontsize=7, loc="best")
                gp = gap_aug4 if dev_id == 4 else gap_aug7
                gr = gap_real4 if dev_id == 4 else gap_real7
                val = gr if row == 1 else gp
                ax.text(0.03, 0.03, f"类间间隔 {val:.3g} V\n(纵轴=相对基线偏移, 缝值等于真实电压差)",
                        transform=ax.transAxes,
                        fontsize=8, ha="left", va="bottom",
                        bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.9))

        # ================= Row 4: dual-device linkage (M4 vs M7 residual) =================
        ax = fig.add_subplot(gs[3, 2 * k:2 * k + 2])
        rn = np.stack([r4[yr == 0], r7[yr == 0]], -1).reshape(-1, 2)
        ra = np.stack([r4[yr == 1], r7[yr == 1]], -1).reshape(-1, 2)
        an = np.stack([a4n, a7n], -1).reshape(-1, 2)
        aa = np.stack([a4a, a7a], -1).reshape(-1, 2)
        ax.scatter(rn[:, 0], rn[:, 1], s=18, color=NORMAL_C, alpha=0.45,
                   edgecolor="none", label="真实 正常")
        ax.scatter(ra[:, 0], ra[:, 1], s=26, color=ABNORMAL_C, alpha=0.75,
                   edgecolor="none", label="真实 异常")
        ax.scatter(an[:, 0], an[:, 1], s=16, color="none", edgecolor=NORMAL_C,
                   marker="^", alpha=0.7, label="增强 正常")
        ax.scatter(aa[:, 0], aa[:, 1], s=16, color="none", edgecolor=ABNORMAL_C,
                   marker="v", alpha=0.7, label="增强 异常")
        if len(rn) > 3:
            slope, intercept = np.polyfit(rn[:, 0], rn[:, 1], 1)
            xs = np.array([rn[:, 0].min(), rn[:, 0].max()])
            ax.plot(xs, slope * xs + intercept, color=NORMAL_C, ls="--", lw=1.6,
                    label="真实正常 拟合关系")
        c_rn = np.corrcoef(rn[:, 0], rn[:, 1])[0, 1] if len(rn) > 2 else np.nan
        c_ra = np.corrcoef(ra[:, 0], ra[:, 1])[0, 1] if len(ra) > 2 else np.nan
        c_an = np.corrcoef(an[:, 0], an[:, 1])[0, 1] if len(an) > 2 else np.nan
        ax.set_title(f"{date}  双设备联动 (M4残差 vs M7残差)", fontsize=11, pad=8)
        ax.set_xlabel("M4_residual (V)"); ax.set_ylabel("M7_residual (V)")
        ax.grid(alpha=0.25); ax.legend(fontsize=7, loc="best")
        ax.text(0.03, 0.97,
                f"真实正常 相关r={c_rn:.2f}\n真实异常 r={c_ra:.2f}\n增强正常 r={c_an:.2f}",
                transform=ax.transAxes, fontsize=8, ha="left", va="top",
                bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.9))

    fig.suptitle(
        "双设备 (device 4 + device 7) Measure 电压 / 残差 趋势: 正常 vs 异常, 增强前 vs 增强后\n"
        "同物理量 / 同窗口长 / 同样本数; 上排原始电压, 第2-3排为相对当日基线偏移(放大镜, 蓝红缝=真实电压差),\n"
        "末排检验两设备关系是否断裂",
        y=0.985, fontsize=13, linespacing=1.4)
    p = FIG / "trend_voltage_normal_vs_abnormal.png"
    fig.savefig(p, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("wrote", p)
    for d, (n0, n1, g4r, g4a, g7r, g7a) in recap.items():
        print(f"{d}: n=({n0},{n1}) device4 gap real={g4r:.4f} aug={g4a:.4f} | "
              f"device7 gap real={g7r:.4f} aug={g7a:.4f}")


if __name__ == "__main__":
    main()
