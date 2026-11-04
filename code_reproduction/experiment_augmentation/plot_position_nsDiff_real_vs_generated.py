"""Plot original rows against raw position-conditioned NsDiff windows."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
RUN = HERE / "runs" / "position_conditioned_other_papers_20260917"
DATES = ("250605", "250626", "250718")
ERRORS = (0, 1, 2, 3, 5, 6, 8, 9)
COLORS = {
    0: "#4C78A8", 1: "#F58518", 2: "#E45756", 3: "#72B7B2",
    5: "#54A24B", 6: "#B279A2", 8: "#FF9DA6", 9: "#9D755D",
}


def main():
    meta = pd.read_json(RUN / "metadata.json", dtype={"date": str}).reset_index(drop=True)
    audit = pd.read_csv(RUN / "paired_row_audit.csv", dtype={"date": str})
    fig, axes = plt.subplots(3, 2, figsize=(15, 13), sharex=True)
    rng = np.random.default_rng(20260917)

    for row, date in enumerate(DATES):
        real = audit[audit.date == date].sort_values("dt").reset_index(drop=True)
        syn = np.load(RUN / "generators" / "NsDiff" / "seed42" / date / "synthetic.npz")
        labels = syn["label"].astype(int)
        anchors = syn["anchor_ids"].astype(int)
        raw = syn["raw"]
        bases = meta.loc[anchors, ["base4", "base7"]].to_numpy(float)
        m4 = raw[:, :, 0] + bases[:, None, 0]
        m7 = raw[:, :, 1] + bases[:, None, 1]

        for col, channel in enumerate(("Measure4", "Measure7")):
            ax = axes[row, col]
            real_col = real[channel].to_numpy(float)
            ax.plot(np.arange(len(real_col)), real_col, color="#222222", linewidth=1.4,
                    alpha=0.9, label="Original rows")
            # Show all real error rows as points, retaining their position label.
            for error in ERRORS[1:]:
                points = real[real.error4 == error]
                if not points.empty:
                    ids = points.index.to_numpy()
                    ax.scatter(ids, points[channel], s=22, color=COLORS[error], zorder=4)

            for error in ERRORS:
                ids = np.flatnonzero(labels == error)
                if not len(ids):
                    continue
                ids = rng.choice(ids, size=min(10, len(ids)), replace=False)
                values = m4[ids] if channel == "Measure4" else m7[ids]
                # Place generated windows after the original trace to make the
                # paper-code output visibly distinct without hiding raw outliers.
                start = len(real_col) + 3 + (error * 12)
                for offset, values_one in enumerate(values):
                    x0 = start + offset * 0.42
                    ax.plot(x0 + np.arange(4), values_one, color=COLORS[error], alpha=0.35, linewidth=1.0)

            ax.axvline(len(real_col) - 0.5, color="#777777", linestyle="--", linewidth=0.8)
            ax.text(0.01, 0.97, "original", transform=ax.transAxes, va="top", fontsize=9, color="#444444")
            ax.text(0.78, 0.97, "NsDiff raw", transform=ax.transAxes, va="top", fontsize=9, color="#444444")
            ax.set_title(f"{date} — {channel}", fontsize=12, weight="bold")
            ax.set_ylabel("Measure")
            ax.grid(True, alpha=0.22)
            ax.set_xlim(0, len(real_col) + 3 + 9 * 12 + 5)

    axes[-1, 0].set_xlabel("Original row index; generated windows shown after dashed line")
    axes[-1, 1].set_xlabel("Original row index; generated windows shown after dashed line")
    position_handles = [Line2D([0], [0], color=COLORS[e], linewidth=2, label=f"error={e}") for e in ERRORS]
    base_handles = [
        Line2D([0], [0], color="#222222", linewidth=1.6, label="Original trace"),
        Line2D([0], [0], color="#777777", linestyle="--", linewidth=1, label="Boundary"),
    ]
    fig.legend(handles=base_handles + position_handles, loc="upper center", ncol=6,
               bbox_to_anchor=(0.5, 0.995), frameon=False, fontsize=9)
    fig.suptitle("Original experiment.xlsx vs raw position-conditioned NsDiff generation",
                 fontsize=16, weight="bold", y=1.025)
    fig.text(0.5, 0.005,
             "NsDiff raw output only: no clipping, deletion, range filtering, or positivity correction. "
             "Generated lines are representative raw 4-step windows; colors indicate error position.",
             ha="center", fontsize=9, color="#444444")
    fig.tight_layout(rect=(0, 0.025, 1, 0.95), h_pad=2.0, w_pad=1.3)
    out = RUN / "original_vs_position_conditioned_NsDiff_raw.png"
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    main()
