"""Plot the complete 260910 paired device-4/device-7 raw voltage trend."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.rcParams["font.sans-serif"] = ["Noto Sans CJK JP", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

HERE = Path(__file__).resolve().parent
WORKBOOK = HERE.parent.parent / "experiment.xlsx"
OUT = HERE / "runs" / "ratio_benchmark_20260918" / "final_test_train" / "figures"
DATE = "260910"
SHEET4 = "260910_80kv_9p_3373(4)"
SHEET7 = "260910_80kv_9p_0710(7)"
NORMAL_C = "#1f77b4"
ABNORMAL_C = "#d62728"
DEVICE4_C = "#1f77b4"
DEVICE7_C = "#ff7f0e"


def pair_data():
    sheets = pd.read_excel(WORKBOOK, sheet_name=None)
    a = sheets[SHEET4].copy()
    b = sheets[SHEET7].copy()
    a["row4"] = np.arange(len(a)) + 2
    b["row7"] = np.arange(len(b)) + 2
    a["dt"] = pd.to_datetime(a.time.astype(str).str.replace("_", " ", regex=False))
    b["dt"] = pd.to_datetime(b.time.astype(str).str.replace("_", " ", regex=False))
    a = a.sort_values("dt")
    b = b.sort_values("dt")
    left = a.rename(columns={c: f"{c}4" for c in a.columns if c not in {"dt", "row4"}})
    right = b.rename(columns={c: f"{c}7" for c in b.columns if c not in {"dt", "row7"}}).rename(columns={"dt": "dt7"})
    q = pd.merge_asof(
        left,
        right,
        left_on="dt", right_on="dt7", direction="nearest", tolerance=pd.Timedelta(seconds=40),
    ).dropna(subset=["Measure7"]).copy()
    q["lag_seconds"] = (q.dt - q.dt7).dt.total_seconds().abs()
    q = q.sort_values("lag_seconds").drop_duplicates("row7").sort_values("dt").reset_index(drop=True)
    q["abnormal"] = (q.error4 != 0) | (q.error7 != 0)
    q["pair_index"] = np.arange(len(q))
    return q


def add_spans(ax, abnormal):
    indices = np.flatnonzero(abnormal)
    if not len(indices):
        return
    for segment in np.split(indices, np.flatnonzero(np.diff(indices) > 1) + 1):
        ax.axvspan(segment[0] - 0.5, segment[-1] + 0.5, color=ABNORMAL_C, alpha=0.12, zorder=0)


def main():
    q = pair_data()
    OUT.mkdir(parents=True, exist_ok=True)
    q.to_csv(OUT / "260910_paired_raw_voltage.csv", index=False, encoding="utf-8-sig")

    x = q.pair_index.to_numpy()
    m4 = q.Measure4.to_numpy(float)
    m7 = q.Measure7.to_numpy(float)
    bad = q.abnormal.to_numpy(bool)

    fig, axes = plt.subplots(2, 1, figsize=(16, 9), sharex=True, gridspec_kw={"height_ratios": [1.8, 1]})
    ax = axes[0]
    add_spans(ax, bad)
    ax.plot(x, m4, color=DEVICE4_C, lw=1.3, alpha=0.80, label="device 4 Measure")
    ax.plot(x, m7, color=DEVICE7_C, lw=1.3, alpha=0.80, label="device 7 Measure")
    ax.scatter(x[bad], m4[bad], color=ABNORMAL_C, s=22, zorder=5, label="abnormal: either error != 0")
    ax.scatter(x[bad], m7[bad], color=ABNORMAL_C, s=22, marker="s", zorder=5)
    ax.set_ylabel("Measure voltage")
    ax.set_title("260910 original paired dual-device voltage", fontsize=14, pad=10)
    ax.grid(alpha=0.25)
    ax.legend(ncol=3, fontsize=9, loc="best", frameon=True)

    ax = axes[1]
    add_spans(ax, bad)
    ax.plot(x, q.error4.to_numpy(), color=DEVICE4_C, lw=1.2, drawstyle="steps-mid", label="device 4 error")
    ax.plot(x, q.error7.to_numpy(), color=DEVICE7_C, lw=1.2, drawstyle="steps-mid", label="device 7 error")
    ax.scatter(x[bad], q.error4.to_numpy()[bad], color=ABNORMAL_C, s=18, zorder=5)
    ax.set_xlabel("paired sample index (chronological)")
    ax.set_ylabel("error position")
    ax.set_yticks(sorted(set(q.error4.astype(int).tolist() + q.error7.astype(int).tolist())))
    ax.grid(alpha=0.25)
    ax.legend(ncol=2, fontsize=9, loc="best", frameon=True)

    n_bad = int(bad.sum())
    lag = q.lag_seconds.to_numpy(float)
    fig.suptitle(
        f"260910 paired raw voltage: device 4 (3373) + device 7 (0710)\n"
        f"paired rows={len(q)} | abnormal paired rows={n_bad} | max timestamp lag={lag.max():.1f}s",
        fontsize=15, y=0.98,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out = OUT / "260910_original_paired_dual_device_voltage.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)
    print("wrote", OUT / "260910_paired_raw_voltage.csv")


if __name__ == "__main__":
    main()
