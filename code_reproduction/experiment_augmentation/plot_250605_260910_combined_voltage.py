"""Combine 250605 and 260910 paired dual-device raw voltage in one figure."""
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
CONFIG = {
    "250605": ("250605_80kv_9p_0819(4)", "250605_80kv_9p_0785(7)"),
    "260910": ("260910_80kv_9p_3373(4)", "260910_80kv_9p_0710(7)"),
}
DEVICE4_C = "#1f77b4"
DEVICE7_C = "#ff7f0e"
ABNORMAL_C = "#d62728"


def pair_date(date, sheet4, sheet7):
    sheets = pd.read_excel(WORKBOOK, sheet_name=None)
    a = sheets[sheet4].copy()
    b = sheets[sheet7].copy()
    a["row4"] = np.arange(len(a)) + 2
    b["row7"] = np.arange(len(b)) + 2
    a["dt"] = pd.to_datetime(a.time.astype(str).str.replace("_", " ", regex=False))
    b["dt"] = pd.to_datetime(b.time.astype(str).str.replace("_", " ", regex=False))
    a = a.sort_values("dt")
    b = b.sort_values("dt")
    left = a.rename(columns={c: f"{c}4" for c in a.columns if c not in {"dt", "row4"}})
    right = b.rename(columns={c: f"{c}7" for c in b.columns if c not in {"dt", "row7"}}).rename(columns={"dt": "dt7"})
    q = pd.merge_asof(
        left, right, left_on="dt", right_on="dt7", direction="nearest",
        tolerance=pd.Timedelta(seconds=40),
    ).dropna(subset=["Measure7"]).copy()
    q["lag_seconds"] = (q.dt - q.dt7).dt.total_seconds().abs()
    q = q.sort_values("lag_seconds").drop_duplicates("row7").sort_values("dt").reset_index(drop=True)
    q["date"] = date
    q["abnormal"] = (q.error4 != 0) | (q.error7 != 0)
    # Remove power-on decay using the same conservative stable-normal rule as
    # the benchmark: terminal 10-point median, stable suffix within 15% over
    # the last 40 normal paired rows. Do not remove later abnormal stages.
    normal = q[q.error4 == 0].copy()
    tail = normal.tail(40)
    terminal = tail[["Measure4", "Measure7"]].tail(10).median().to_numpy()
    relative = np.abs(tail[["Measure4", "Measure7"]].to_numpy() / np.maximum(terminal, 1e-9) - 1).max(axis=1)
    bad = np.flatnonzero(relative > 0.15)
    stable = tail.iloc[(bad[-1] + 1 if len(bad) else 0):]
    stable_start = stable.dt.iloc[0] if len(stable) else q.dt.iloc[0]
    q["stable_start"] = stable_start
    q["keep_after_power_on_decay"] = q.dt >= stable_start
    q["pair_index"] = np.arange(len(q))
    return q


def add_spans(ax, abnormal):
    indices = np.flatnonzero(abnormal)
    for segment in np.split(indices, np.flatnonzero(np.diff(indices) > 1) + 1) if len(indices) else []:
        ax.axvspan(segment[0] - 0.5, segment[-1] + 0.5, color=ABNORMAL_C, alpha=0.12, zorder=0)


def main():
    frames = [pair_date(date, *sheet_names) for date, sheet_names in CONFIG.items()]
    combined_all = pd.concat(frames, ignore_index=True)
    combined = combined_all[combined_all.keep_after_power_on_decay].copy()
    for date in CONFIG:
        sub = combined[combined.date == date].sort_values("dt")
        combined.loc[sub.index, "pair_index_filtered"] = np.arange(len(sub))
    OUT.mkdir(parents=True, exist_ok=True)
    combined.to_csv(OUT / "250605_260910_paired_raw_voltage.csv", index=False, encoding="utf-8-sig")

    # Overlay both dates in one coordinate system. The x-axis is normalized
    # paired-sample progress because the two experiments have different lengths.
    fig, ax = plt.subplots(1, 1, figsize=(17, 7.5))
    date_styles = {"250605": "-", "260910": "--"}
    for q_all in frames:
        q = q_all[q_all.keep_after_power_on_decay].copy().reset_index(drop=True)
        x = np.arange(len(q))
        x = x / max(x.max(), 1) * 100.0
        m4 = q.Measure4.to_numpy(float)
        m7 = q.Measure7.to_numpy(float)
        bad = q.abnormal.to_numpy(bool)
        style = date_styles[q.date.iloc[0]]
        ax.plot(x, m4, color=DEVICE4_C, ls=style, lw=1.45, alpha=0.82, label=f"{q.date.iloc[0]} device 4")
        ax.plot(x, m7, color=DEVICE7_C, ls=style, lw=1.45, alpha=0.82, label=f"{q.date.iloc[0]} device 7")
        ax.scatter(x[bad], m4[bad], color=ABNORMAL_C, s=20, marker="o", zorder=5)
        ax.scatter(x[bad], m7[bad], color=ABNORMAL_C, s=20, marker="s", zorder=5)
    ax.set_xlabel("normalized paired-sample progress (%)")
    ax.set_ylabel("Measure voltage")
    ax.set_xlim(0, 100)
    ax.grid(alpha=0.25)
    ax.legend(ncol=2, fontsize=9, loc="best", frameon=True)
    fig.suptitle(
        "250605 and 260910 overlaid in one coordinate system: original paired dual-device voltage\n"
        "solid = 250605, dashed = 260910; blue = device 4, orange = device 7; red markers = abnormal pairs",
        fontsize=15, y=0.98,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    out = OUT / "250605_260910_overlaid_same_axes_dual_device_voltage.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)
    print("wrote", OUT / "250605_260910_paired_raw_voltage.csv")
    print(combined.groupby("date").agg(paired_rows=("pair_index", "size"), abnormal_pairs=("abnormal", "sum"), max_lag_seconds=("lag_seconds", "max")).to_string())
    print("stable starts:")
    print(combined_all.groupby("date").stable_start.first().to_string())
    print("removed rows:")
    print(combined_all.groupby("date").keep_after_power_on_decay.apply(lambda s: int((~s).sum())).to_string())


if __name__ == "__main__":
    main()
