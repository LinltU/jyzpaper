"""Extract stable paired raw voltage and make shuffled normal/abnormal plots.

The output keeps raw Measure values. It only removes the initial power-on decay
segment per date, then interleaves normal and abnormal paired rows for plotting.
"""
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
OUT = HERE / "runs" / "ratio_benchmark_20260918" / "final_test_train" / "stable_interleaved_voltage"
FIG = OUT / "figures"

DATE_SHEETS = {
    "250605": ("250605_80kv_9p_0819(4)", "250605_80kv_9p_0785(7)"),
    "250626": ("250626_80kv_9p_0603(4)", "250626_80kv_9p_0769(7)"),
    "250718": ("250718_80kv_9p_0603(4)", "250718_80kv_9p_0769(7)"),
    "260910": ("260910_80kv_9p_3373(4)", "260910_80kv_9p_0710(7)"),
}

NORMAL_C = "#2563eb"
ABNORMAL_C = "#dc2626"
M4_MARKER = "o"
M7_MARKER = "s"


def pair_date(sheets, date, sheet4, sheet7):
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
        left,
        right,
        left_on="dt",
        right_on="dt7",
        direction="nearest",
        tolerance=pd.Timedelta(seconds=40),
    ).dropna(subset=["Measure7"]).copy()
    q["lag_seconds"] = (q.dt - q.dt7).dt.total_seconds().abs()
    q = q.sort_values("lag_seconds").drop_duplicates("row7").sort_values("dt").reset_index(drop=True)
    q["date"] = date
    q["label"] = ((q.error4 != 0) | (q.error7 != 0)).astype(int)
    q["label_name"] = np.where(q.label == 0, "normal", "abnormal")
    q["original_pair_index"] = np.arange(len(q))

    normal = q[q.label == 0].copy()
    tail = normal.tail(40)
    terminal = tail[["Measure4", "Measure7"]].tail(10).median().to_numpy()
    relative = np.abs(tail[["Measure4", "Measure7"]].to_numpy() / np.maximum(terminal, 1e-9) - 1).max(axis=1)
    bad = np.flatnonzero(relative > 0.15)
    stable = tail.iloc[(bad[-1] + 1 if len(bad) else 0):]
    stable_start = stable.dt.iloc[0] if len(stable) else q.dt.iloc[0]
    filtered = q[q.dt >= stable_start].copy().reset_index(drop=True)
    filtered["stable_start"] = stable_start
    filtered["filtered_pair_index"] = np.arange(len(filtered))
    filtered["removed_power_on_rows"] = int((q.dt < stable_start).sum())
    return filtered


def interleave_shuffle(df, seed=20260920):
    rng = np.random.default_rng(seed)
    pieces = []
    for date, group in df.groupby("date", sort=True):
        normal = group[group.label == 0].sample(frac=1, random_state=int(rng.integers(1_000_000)))
        abnormal = group[group.label == 1].sample(frac=1, random_state=int(rng.integers(1_000_000)))
        rows = []
        max_len = max(len(normal), len(abnormal))
        normal_records = list(normal.to_dict("records"))
        abnormal_records = list(abnormal.to_dict("records"))
        for i in range(max_len):
            if i < len(normal_records):
                rows.append(normal_records[i])
            if i < len(abnormal_records):
                rows.append(abnormal_records[i])
        shuffled = pd.DataFrame(rows)
        shuffled["interleaved_index"] = np.arange(len(shuffled))
        pieces.append(shuffled)
    return pd.concat(pieces, ignore_index=True)


def plot_date(ax, group, date):
    normal = group[group.label == 0]
    abnormal = group[group.label == 1]
    for data, color, name in [(normal, NORMAL_C, "normal"), (abnormal, ABNORMAL_C, "abnormal")]:
        ax.scatter(data.interleaved_index, data.Measure4, color=color, marker=M4_MARKER, s=28, alpha=0.82, label=f"{name} M4")
        ax.scatter(data.interleaved_index, data.Measure7, color=color, marker=M7_MARKER, s=28, alpha=0.82, label=f"{name} M7")
    ax.set_title(f"{date}: shuffled/interleaved stable voltage", fontsize=11)
    ax.set_xlabel("interleaved shuffled index")
    ax.set_ylabel("Measure voltage")
    ax.grid(alpha=0.25)
    ax.text(
        0.02,
        0.98,
        f"normal={len(normal)}, abnormal={len(abnormal)}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8,
        bbox=dict(boxstyle="round", fc="white", ec="0.75", alpha=0.9),
    )


def make_plots(shuffled):
    FIG.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(17, 10), sharey=False)
    axes = axes.ravel()
    for ax, (date, group) in zip(axes, shuffled.groupby("date", sort=True)):
        plot_date(ax, group, date)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle(
        "Stable raw voltage by date: normal and abnormal rows shuffled/interleaved\n"
        "Power-on decay removed; raw Measure values are not clipped or projected",
        fontsize=15,
        y=1.06,
    )
    fig.tight_layout()
    fig.savefig(FIG / "stable_interleaved_voltage_by_date.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    for date, group in shuffled.groupby("date", sort=True):
        fig, ax = plt.subplots(figsize=(13, 5.8))
        plot_date(ax, group, date)
        ax.legend(ncol=4, fontsize=9, loc="best")
        fig.tight_layout()
        fig.savefig(FIG / f"stable_interleaved_voltage_{date}.png", dpi=180, bbox_inches="tight")
        plt.close(fig)


def make_chronological_plots(filtered):
    FIG.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(17, 10), sharey=False)
    axes = axes.ravel()
    for ax, (date, group) in zip(axes, filtered.groupby("date", sort=True)):
        group = group.sort_values("dt").reset_index(drop=True)
        x = np.arange(len(group))
        normal = group.label.to_numpy() == 0
        abnormal = ~normal
        ax.plot(x, group.Measure4, color="#64748b", lw=1.0, alpha=0.55, label="M4 chronological")
        ax.plot(x, group.Measure7, color="#94a3b8", lw=1.0, alpha=0.55, label="M7 chronological")
        ax.scatter(x[normal], group.Measure4[normal], color=NORMAL_C, marker=M4_MARKER, s=26, alpha=0.85, label="normal M4")
        ax.scatter(x[normal], group.Measure7[normal], color=NORMAL_C, marker=M7_MARKER, s=26, alpha=0.85, label="normal M7")
        ax.scatter(x[abnormal], group.Measure4[abnormal], color=ABNORMAL_C, marker=M4_MARKER, s=28, alpha=0.9, label="abnormal M4")
        ax.scatter(x[abnormal], group.Measure7[abnormal], color=ABNORMAL_C, marker=M7_MARKER, s=28, alpha=0.9, label="abnormal M7")
        if abnormal.any():
            for seg in np.split(np.flatnonzero(abnormal), np.flatnonzero(np.diff(np.flatnonzero(abnormal)) > 1) + 1):
                ax.axvspan(seg[0] - 0.5, seg[-1] + 0.5, color=ABNORMAL_C, alpha=0.10, zorder=0)
        ax.set_title(f"{date}: stable voltage in chronological order", fontsize=11)
        ax.set_xlabel("stable paired index (chronological)")
        ax.set_ylabel("Measure voltage")
        ax.grid(alpha=0.25)
        ax.text(
            0.02,
            0.98,
            f"normal={int(normal.sum())}, abnormal={int(abnormal.sum())}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8,
            bbox=dict(boxstyle="round", fc="white", ec="0.75", alpha=0.9),
        )
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=6, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle(
        "Stable raw voltage by date in chronological order\n"
        "Power-on decay removed; red shaded regions are abnormal phases",
        fontsize=15,
        y=1.06,
    )
    fig.tight_layout()
    fig.savefig(FIG / "stable_chronological_voltage_by_date.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    for date, group in filtered.groupby("date", sort=True):
        group = group.sort_values("dt").reset_index(drop=True)
        x = np.arange(len(group))
        normal = group.label.to_numpy() == 0
        abnormal = ~normal
        fig, ax = plt.subplots(figsize=(13, 5.8))
        ax.plot(x, group.Measure4, color="#64748b", lw=1.0, alpha=0.55, label="M4 chronological")
        ax.plot(x, group.Measure7, color="#94a3b8", lw=1.0, alpha=0.55, label="M7 chronological")
        ax.scatter(x[normal], group.Measure4[normal], color=NORMAL_C, marker=M4_MARKER, s=30, label="normal M4")
        ax.scatter(x[normal], group.Measure7[normal], color=NORMAL_C, marker=M7_MARKER, s=30, label="normal M7")
        ax.scatter(x[abnormal], group.Measure4[abnormal], color=ABNORMAL_C, marker=M4_MARKER, s=32, label="abnormal M4")
        ax.scatter(x[abnormal], group.Measure7[abnormal], color=ABNORMAL_C, marker=M7_MARKER, s=32, label="abnormal M7")
        if abnormal.any():
            bad_idx = np.flatnonzero(abnormal)
            for seg in np.split(bad_idx, np.flatnonzero(np.diff(bad_idx) > 1) + 1):
                ax.axvspan(seg[0] - 0.5, seg[-1] + 0.5, color=ABNORMAL_C, alpha=0.10, zorder=0)
        ax.set_title(f"{date}: stable voltage in chronological order")
        ax.set_xlabel("stable paired index (chronological)")
        ax.set_ylabel("Measure voltage")
        ax.grid(alpha=0.25)
        ax.legend(ncol=3, fontsize=8, loc="best")
        fig.tight_layout()
        fig.savefig(FIG / f"stable_chronological_voltage_{date}.png", dpi=180, bbox_inches="tight")
        plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    sheets = pd.read_excel(WORKBOOK, sheet_name=None)
    filtered = pd.concat(
        [pair_date(sheets, date, *sheet_pair) for date, sheet_pair in DATE_SHEETS.items()],
        ignore_index=True,
    )
    shuffled = interleave_shuffle(filtered)
    summary = filtered.groupby("date").agg(
        stable_rows=("date", "size"),
        normal_rows=("label", lambda s: int((s == 0).sum())),
        abnormal_rows=("label", lambda s: int((s == 1).sum())),
        stable_start=("stable_start", "first"),
        removed_power_on_rows=("removed_power_on_rows", "first"),
        max_lag_seconds=("lag_seconds", "max"),
    ).reset_index()

    filtered.to_csv(OUT / "stable_paired_voltage.csv", index=False, encoding="utf-8-sig")
    shuffled.to_csv(OUT / "stable_interleaved_voltage.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(OUT / "stable_interleaved_summary.csv", index=False, encoding="utf-8-sig")
    with pd.ExcelWriter(OUT / "stable_interleaved_voltage.xlsx", engine="openpyxl") as writer:
        summary.to_excel(writer, sheet_name="summary", index=False)
        filtered.to_excel(writer, sheet_name="stable_chronological", index=False)
        shuffled.to_excel(writer, sheet_name="stable_interleaved", index=False)
        for date, group in shuffled.groupby("date", sort=True):
            group.to_excel(writer, sheet_name=f"interleaved_{date}", index=False)

    make_plots(shuffled)
    make_chronological_plots(filtered)
    print("wrote", OUT / "stable_interleaved_voltage.xlsx")
    print("wrote", FIG / "stable_interleaved_voltage_by_date.png")
    print("wrote", FIG / "stable_chronological_voltage_by_date.png")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
