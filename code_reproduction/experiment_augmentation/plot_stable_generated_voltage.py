"""Make chronological-style plots for generated stable synthetic voltage."""
from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.rcParams["font.sans-serif"] = ["Noto Sans CJK JP", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

HERE = Path(__file__).resolve().parent
RUN = HERE / "runs" / "ratio_benchmark_20260918" / "sensitivity_stable"
OUT = RUN / "generated_voltage_plots"
FIG = OUT / "figures"
RATIO = "5to5"
METHOD = "NsDiff"
RAW = RUN / f"raw_{RATIO}_{METHOD}.npz"

NORMAL_C = "#2563eb"
ABNORMAL_C = "#dc2626"


def load_generated():
    meta = pd.DataFrame(json.loads((RUN / "metadata.json").read_text(encoding="utf-8")))
    payload = np.load(RAW)
    windows = payload["windows"].astype(float)
    labels = payload["labels"].astype(int)
    source_ids = payload["source_ids"].astype(int)
    source = meta.iloc[source_ids].reset_index(drop=True)
    rows = []
    for i in range(len(windows)):
        base4 = float(source.base4.iloc[i])
        base7 = float(source.base7.iloc[i])
        for step in range(windows.shape[1]):
            rows.append({
                "date": str(source.date.iloc[i]),
                "source_id": int(source_ids[i]),
                "generated_window_index": i,
                "step": step,
                "label": int(labels[i]),
                "label_name": "normal" if int(labels[i]) == 0 else "abnormal",
                "error_source": int(source.error.iloc[i]),
                "Measure4_generated": windows[i, step, 0] + base4,
                "Measure7_generated": windows[i, step, 1] + base7,
                "M4_residual_generated": windows[i, step, 0],
                "M7_residual_generated": windows[i, step, 1],
            })
    df = pd.DataFrame(rows).sort_values(["date", "source_id", "generated_window_index", "step"]).reset_index(drop=True)
    df["generated_sequence_index"] = df.groupby("date").cumcount()
    return df


def plot_date(ax, group, date):
    x = group.generated_sequence_index.to_numpy()
    normal = group.label.to_numpy() == 0
    abnormal = ~normal
    ax.plot(x, group.Measure4_generated, color="#64748b", lw=0.9, alpha=0.45, label="generated M4 trend")
    ax.plot(x, group.Measure7_generated, color="#94a3b8", lw=0.9, alpha=0.45, label="generated M7 trend")
    ax.scatter(x[normal], group.Measure4_generated[normal], color=NORMAL_C, marker="o", s=22, alpha=0.82, label="normal M4")
    ax.scatter(x[normal], group.Measure7_generated[normal], color=NORMAL_C, marker="s", s=22, alpha=0.82, label="normal M7")
    ax.scatter(x[abnormal], group.Measure4_generated[abnormal], color=ABNORMAL_C, marker="o", s=24, alpha=0.9, label="abnormal M4")
    ax.scatter(x[abnormal], group.Measure7_generated[abnormal], color=ABNORMAL_C, marker="s", s=24, alpha=0.9, label="abnormal M7")
    if abnormal.any():
        bad_idx = np.flatnonzero(abnormal)
        for seg in np.split(bad_idx, np.flatnonzero(np.diff(bad_idx) > 1) + 1):
            ax.axvspan(seg[0] - 0.5, seg[-1] + 0.5, color=ABNORMAL_C, alpha=0.08, zorder=0)
    ax.set_title(f"{date}: generated stable voltage ({METHOD}, {RATIO})", fontsize=11)
    ax.set_xlabel("generated point index (sorted by source window)")
    ax.set_ylabel("Generated Measure voltage")
    ax.grid(alpha=0.25)
    ax.text(
        0.02, 0.98,
        f"normal={int(normal.sum())}, abnormal={int(abnormal.sum())}",
        transform=ax.transAxes, ha="left", va="top", fontsize=8,
        bbox=dict(boxstyle="round", fc="white", ec="0.75", alpha=0.9),
    )


def make_plots(df):
    FIG.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(17, 10), sharey=False)
    axes = axes.ravel()
    for ax, (date, group) in zip(axes, df.groupby("date", sort=True)):
        plot_date(ax, group.reset_index(drop=True), date)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=6, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle(
        "Generated stable raw voltage by date, chronological-style source order\n"
        "Synthetic windows are sorted by their source real window; raw generated values are not clipped",
        fontsize=15, y=1.06,
    )
    fig.tight_layout()
    fig.savefig(FIG / f"generated_stable_chronological_voltage_by_date_{RATIO}_{METHOD}.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    for date, group in df.groupby("date", sort=True):
        fig, ax = plt.subplots(figsize=(13, 5.8))
        plot_date(ax, group.reset_index(drop=True), date)
        ax.legend(ncol=3, fontsize=8, loc="best")
        fig.tight_layout()
        fig.savefig(FIG / f"generated_stable_chronological_voltage_{date}_{RATIO}_{METHOD}.png", dpi=180, bbox_inches="tight")
        plt.close(fig)


def interleave_generated(df, seed=20260920):
    rng = np.random.default_rng(seed)
    pieces = []
    for date, group in df.groupby("date", sort=True):
        # Shuffle whole generated windows, not individual steps, then alternate
        # normal/abnormal windows. This keeps each 4-step synthetic trajectory intact.
        windows = group[["generated_window_index", "label"]].drop_duplicates()
        normal_ids = windows[windows.label == 0].generated_window_index.to_numpy()
        abnormal_ids = windows[windows.label == 1].generated_window_index.to_numpy()
        rng.shuffle(normal_ids)
        rng.shuffle(abnormal_ids)
        ordered_ids = []
        max_len = max(len(normal_ids), len(abnormal_ids))
        for i in range(max_len):
            if i < len(normal_ids):
                ordered_ids.append(normal_ids[i])
            if i < len(abnormal_ids):
                ordered_ids.append(abnormal_ids[i])
        ordered = []
        for new_win_idx, win_id in enumerate(ordered_ids):
            block = group[group.generated_window_index == win_id].sort_values("step").copy()
            block["interleaved_window_index"] = new_win_idx
            ordered.append(block)
        if ordered:
            date_df = pd.concat(ordered, ignore_index=True)
            date_df["interleaved_generated_index"] = np.arange(len(date_df))
            pieces.append(date_df)
    return pd.concat(pieces, ignore_index=True)


def make_interleaved_plots(interleaved):
    FIG.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(17, 10), sharey=False)
    axes = axes.ravel()
    for ax, (date, group) in zip(axes, interleaved.groupby("date", sort=True)):
        group = group.reset_index(drop=True)
        x = group.interleaved_generated_index.to_numpy()
        normal = group.label.to_numpy() == 0
        abnormal = ~normal
        ax.plot(x, group.Measure4_generated, color="#64748b", lw=0.9, alpha=0.45, label="generated M4 trend")
        ax.plot(x, group.Measure7_generated, color="#94a3b8", lw=0.9, alpha=0.45, label="generated M7 trend")
        ax.scatter(x[normal], group.Measure4_generated[normal], color=NORMAL_C, marker="o", s=22, alpha=0.82, label="normal M4")
        ax.scatter(x[normal], group.Measure7_generated[normal], color=NORMAL_C, marker="s", s=22, alpha=0.82, label="normal M7")
        ax.scatter(x[abnormal], group.Measure4_generated[abnormal], color=ABNORMAL_C, marker="o", s=24, alpha=0.9, label="abnormal M4")
        ax.scatter(x[abnormal], group.Measure7_generated[abnormal], color=ABNORMAL_C, marker="s", s=24, alpha=0.9, label="abnormal M7")
        if abnormal.any():
            bad_idx = np.flatnonzero(abnormal)
            for seg in np.split(bad_idx, np.flatnonzero(np.diff(bad_idx) > 1) + 1):
                ax.axvspan(seg[0] - 0.5, seg[-1] + 0.5, color=ABNORMAL_C, alpha=0.08, zorder=0)
        ax.set_title(f"{date}: generated stable voltage shuffled/interleaved", fontsize=11)
        ax.set_xlabel("interleaved generated point index")
        ax.set_ylabel("Generated Measure voltage")
        ax.grid(alpha=0.25)
        ax.text(
            0.02, 0.98,
            f"normal={int(normal.sum())}, abnormal={int(abnormal.sum())}",
            transform=ax.transAxes, ha="left", va="top", fontsize=8,
            bbox=dict(boxstyle="round", fc="white", ec="0.75", alpha=0.9),
        )
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=6, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle(
        "Generated stable raw voltage by date, normal/abnormal shuffled and interleaved\n"
        "Blue = normal, red = abnormal; raw generated values are not clipped",
        fontsize=15, y=1.06,
    )
    fig.tight_layout()
    fig.savefig(FIG / f"generated_stable_interleaved_voltage_by_date_{RATIO}_{METHOD}.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    for date, group in interleaved.groupby("date", sort=True):
        fig, ax = plt.subplots(figsize=(13, 5.8))
        group = group.reset_index(drop=True)
        x = group.interleaved_generated_index.to_numpy()
        normal = group.label.to_numpy() == 0
        abnormal = ~normal
        ax.plot(x, group.Measure4_generated, color="#64748b", lw=0.9, alpha=0.45, label="generated M4 trend")
        ax.plot(x, group.Measure7_generated, color="#94a3b8", lw=0.9, alpha=0.45, label="generated M7 trend")
        ax.scatter(x[normal], group.Measure4_generated[normal], color=NORMAL_C, marker="o", s=25, label="normal M4")
        ax.scatter(x[normal], group.Measure7_generated[normal], color=NORMAL_C, marker="s", s=25, label="normal M7")
        ax.scatter(x[abnormal], group.Measure4_generated[abnormal], color=ABNORMAL_C, marker="o", s=27, label="abnormal M4")
        ax.scatter(x[abnormal], group.Measure7_generated[abnormal], color=ABNORMAL_C, marker="s", s=27, label="abnormal M7")
        if abnormal.any():
            bad_idx = np.flatnonzero(abnormal)
            for seg in np.split(bad_idx, np.flatnonzero(np.diff(bad_idx) > 1) + 1):
                ax.axvspan(seg[0] - 0.5, seg[-1] + 0.5, color=ABNORMAL_C, alpha=0.08, zorder=0)
        ax.set_title(f"{date}: generated stable voltage shuffled/interleaved")
        ax.set_xlabel("interleaved generated point index")
        ax.set_ylabel("Generated Measure voltage")
        ax.grid(alpha=0.25)
        ax.legend(ncol=3, fontsize=8, loc="best")
        fig.tight_layout()
        fig.savefig(FIG / f"generated_stable_interleaved_voltage_{date}_{RATIO}_{METHOD}.png", dpi=180, bbox_inches="tight")
        plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = load_generated()
    summary = df.groupby("date").agg(
        generated_points=("date", "size"),
        normal_points=("label", lambda s: int((s == 0).sum())),
        abnormal_points=("label", lambda s: int((s == 1).sum())),
        m4_min=("Measure4_generated", "min"),
        m4_max=("Measure4_generated", "max"),
        m7_min=("Measure7_generated", "min"),
        m7_max=("Measure7_generated", "max"),
    ).reset_index()
    df.to_csv(OUT / f"generated_stable_voltage_{RATIO}_{METHOD}.csv", index=False, encoding="utf-8-sig")
    interleaved = interleave_generated(df)
    interleaved.to_csv(OUT / f"generated_stable_interleaved_voltage_{RATIO}_{METHOD}.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(OUT / f"generated_stable_voltage_summary_{RATIO}_{METHOD}.csv", index=False, encoding="utf-8-sig")
    with pd.ExcelWriter(OUT / f"generated_stable_voltage_{RATIO}_{METHOD}.xlsx", engine="openpyxl") as writer:
        summary.to_excel(writer, sheet_name="summary", index=False)
        df.to_excel(writer, sheet_name="generated_points", index=False)
        interleaved.to_excel(writer, sheet_name="generated_interleaved", index=False)
        for date, group in df.groupby("date", sort=True):
            group.to_excel(writer, sheet_name=f"generated_{date}", index=False)
    make_plots(df)
    make_interleaved_plots(interleaved)
    print("wrote", OUT / f"generated_stable_voltage_{RATIO}_{METHOD}.xlsx")
    print("wrote", FIG / f"generated_stable_chronological_voltage_by_date_{RATIO}_{METHOD}.png")
    print("wrote", FIG / f"generated_stable_interleaved_voltage_by_date_{RATIO}_{METHOD}.png")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
