"""Plot filtered 250605/260910 paired Measure against humidity on one axes."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.rcParams["font.sans-serif"] = ["Noto Sans CJK JP", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

HERE = Path(__file__).resolve().parent
FIG = HERE / "runs" / "ratio_benchmark_20260918" / "final_test_train" / "figures"
DATA = FIG / "250605_260910_paired_raw_voltage.csv"
OUT = FIG / "250605_260910_humidity_vs_measure_same_axes.png"

DATE_COLOR = {"250605": "#2563eb", "260910": "#ea580c"}
DEVICE = {
    4: {"humidity": "Hum4", "measure": "Measure4", "marker": "o"},
    7: {"humidity": "Hum7", "measure": "Measure7", "marker": "s"},
}
ABNORMAL_EDGE = "#b91c1c"


def main():
    q = pd.read_csv(DATA)
    q.date = q.date.astype(str)
    fig, ax = plt.subplots(figsize=(13, 8))
    for date in ["250605", "260910"]:
        sub = q[q.date == date].copy()
        date_color = DATE_COLOR[date]
        for device, spec in DEVICE.items():
            abnormal = sub.abnormal.to_numpy(bool)
            ax.scatter(
                sub[spec["humidity"]], sub[spec["measure"]],
                color=date_color, marker=spec["marker"], s=34,
                alpha=0.78, label=f"{date} device {device}",
            )
            ax.scatter(
                sub.loc[abnormal, spec["humidity"]],
                sub.loc[abnormal, spec["measure"]],
                facecolors="none", edgecolors=ABNORMAL_EDGE,
                marker=spec["marker"], s=70, linewidths=1.4, zorder=5,
            )
    ax.set_xlabel("Humidity (%)")
    ax.set_ylabel("Measure voltage")
    ax.set_title(
        "250605 and 260910: humidity vs Measure on the same axes\n"
        "power-on decay removed; date colors: blue = 250605, orange = 260910; red outlines = abnormal pairs",
        fontsize=14, pad=12,
    )
    ax.grid(alpha=0.25)
    ax.legend(ncol=2, fontsize=9, frameon=True)
    fig.tight_layout()
    fig.savefig(OUT, dpi=190, bbox_inches="tight")
    plt.close(fig)
    print("wrote", OUT)
    print(q.groupby("date").agg(rows=("date", "size"), abnormal_pairs=("abnormal", "sum"), hum4_min=("Hum4", "min"), hum4_max=("Hum4", "max"), hum7_min=("Hum7", "min"), hum7_max=("Hum7", "max")).to_string())


if __name__ == "__main__":
    main()
