"""Plot Measure over time for every device in device_data_110_220_final.xlsx."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.rcParams["font.sans-serif"] = ["Noto Sans CJK JP", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

ROOT = Path("/home/test/linliting/jyzmentor")
WORKBOOK = ROOT / "device_data_110_220_final.xlsx"
OUT = ROOT / "code_reproduction" / "experiment_augmentation" / "runs" / "device_data_110_220_plots"
FIG = OUT / "figures"


def load_all():
    sheets = pd.read_excel(WORKBOOK, sheet_name=None)
    frames = []
    summary = []
    for name, df in sheets.items():
        df = df.copy()
        df["sheet"] = name
        df["Timestamp"] = pd.to_datetime(df["Timestamp"])
        df = df.sort_values("Timestamp").reset_index(drop=True)
        df["time_index"] = np.arange(len(df))
        df["IMEI"] = df["IMEI"].astype(str)
        frames.append(df)
        summary.append({
            "sheet": name,
            "imei": str(df.IMEI.iloc[0]) if len(df) else name,
            "rows": len(df),
            "start": df.Timestamp.min(),
            "end": df.Timestamp.max(),
            "measure_min": df.Measure.min(),
            "measure_max": df.Measure.max(),
            "measure_mean": df.Measure.mean(),
            "measure_std": df.Measure.std(),
        })
    return pd.concat(frames, ignore_index=True), pd.DataFrame(summary)


def plot_all_overlay(data):
    fig, ax = plt.subplots(figsize=(18, 8))
    for imei, group in data.groupby("IMEI", sort=False):
        ax.plot(group.Timestamp, group.Measure, lw=0.75, alpha=0.55, label=imei)
    ax.set_title("All devices: Measure over time", fontsize=15, pad=12)
    ax.set_xlabel("Timestamp")
    ax.set_ylabel("Measure")
    ax.grid(alpha=0.25)
    ax.legend(ncol=5, fontsize=6.5, loc="upper center", bbox_to_anchor=(0.5, -0.13), frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "all_devices_measure_over_time.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_small_multiples(data):
    devices = list(data.IMEI.drop_duplicates())
    ncols = 5
    nrows = int(np.ceil(len(devices) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(22, 3.0 * nrows), sharex=False, sharey=False)
    axes = np.asarray(axes).ravel()
    for ax, imei in zip(axes, devices):
        group = data[data.IMEI == imei]
        ax.plot(group.Timestamp, group.Measure, color="#2563eb", lw=0.9)
        ax.set_title(imei, fontsize=9)
        ax.grid(alpha=0.25)
        ax.tick_params(axis="x", labelrotation=30, labelsize=6)
        ax.tick_params(axis="y", labelsize=7)
    for ax in axes[len(devices):]:
        ax.axis("off")
    fig.suptitle("Each device: Measure over time", fontsize=16, y=1.01)
    fig.tight_layout()
    fig.savefig(FIG / "each_device_measure_over_time_grid.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_grouped_by_start_period(data):
    early = data[data.Timestamp < pd.Timestamp("2025-08-01")]
    late = data[data.Timestamp >= pd.Timestamp("2025-08-01")]
    fig, axes = plt.subplots(2, 1, figsize=(18, 11), sharey=False)
    for ax, subset, title in [
        (axes[0], early, "April-start devices"),
        (axes[1], late, "November/December-start devices"),
    ]:
        for imei, group in subset.groupby("IMEI", sort=False):
            ax.plot(group.Timestamp, group.Measure, lw=0.8, alpha=0.65, label=imei)
        ax.set_title(title)
        ax.set_xlabel("Timestamp")
        ax.set_ylabel("Measure")
        ax.grid(alpha=0.25)
        ax.legend(ncol=5, fontsize=6.5, loc="upper center", bbox_to_anchor=(0.5, -0.14), frameon=False)
    fig.suptitle("Devices grouped by collection period", fontsize=15, y=0.99)
    fig.tight_layout()
    fig.savefig(FIG / "devices_measure_over_time_by_period.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    data, summary = load_all()
    data.to_csv(OUT / "device_data_110_220_long.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(OUT / "device_data_110_220_summary.csv", index=False, encoding="utf-8-sig")
    with pd.ExcelWriter(OUT / "device_data_110_220_plots_data.xlsx", engine="openpyxl") as writer:
        summary.to_excel(writer, sheet_name="summary", index=False)
        data.to_excel(writer, sheet_name="long_data", index=False)
    plot_all_overlay(data)
    plot_small_multiples(data)
    plot_grouped_by_start_period(data)
    print("wrote", FIG / "all_devices_measure_over_time.png")
    print("wrote", FIG / "each_device_measure_over_time_grid.png")
    print("wrote", FIG / "devices_measure_over_time_by_period.png")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
