"""Plot direct real-test versus generated-window anomaly comparisons."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


RATIOS = {"2to8": 0.2, "3to7": 0.3, "4to6": 0.4, "5to5": 0.5}
LABELS = {0: "normal", 1: "candidate anomaly"}


def split_test(meta: pd.DataFrame, train_frac: float) -> np.ndarray:
    test = []
    for _, group in meta.groupby("IMEI", sort=False):
        ids = group.index.to_numpy()
        cut = max(1, min(len(ids) - 1, int(len(ids) * train_frac)))
        test.extend(ids[cut:])
    return np.asarray(test, dtype=int)


def sample_indices(labels: np.ndarray, n_each: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    chosen = []
    for label in (0, 1):
        ids = np.flatnonzero(labels == label)
        if len(ids):
            chosen.extend(rng.choice(ids, min(n_each, len(ids)), replace=False).tolist())
    return np.asarray(chosen, dtype=int)


def draw_panel(ax, windows: np.ndarray, labels: np.ndarray, title: str, seed: int) -> None:
    ids = sample_indices(labels, 35, seed)
    x = np.arange(windows.shape[1])
    colors = {0: "#2563eb", 1: "#dc2626"}
    for idx in ids:
        label = int(labels[idx])
        ax.plot(x, windows[idx, :, 0], color=colors[label], alpha=0.22, lw=0.9)
        if label:
            ax.scatter(x, windows[idx, :, 0], color=colors[label], s=8, alpha=0.5)
    for label in (0, 1):
        count = int((labels[ids] == label).sum()) if len(ids) else 0
        ax.plot([], [], color=colors[label], lw=2, label=f"{LABELS[label]} (n={count})")
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("Window step")
    ax.set_ylabel("Measure")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8, loc="upper right")


def plot_comparison(run: Path, seed: int) -> None:
    data = np.load(run / "window_dataset.npz")
    x = data["x"]
    meta = pd.read_json(run / "window_metadata.json")
    best = pd.read_csv(run / "formal_best_methods.csv")
    fig, axes = plt.subplots(len(RATIOS), 2, figsize=(12, 4.0 * len(RATIOS)))
    axes = np.asarray(axes).reshape(len(RATIOS), 2)

    for row, (ratio, train_frac) in enumerate(RATIOS.items()):
        test_ids = split_test(meta, train_frac)
        real_x = x[test_ids]
        real_y = meta.iloc[test_ids].label.to_numpy(int)
        selected = best[best.ratio.eq(ratio)].iloc[0]
        method = str(selected.method)
        safe_method = method.replace("/", "_").replace("-", "_")
        raw_path = run / f"raw_{ratio}_{safe_method}_seed{seed}.npz"
        with np.load(raw_path) as generated:
            gen_x = generated["windows"]
            gen_y = generated["labels"].astype(int)

        draw_panel(axes[row, 0], real_x, real_y, f"{ratio} real test windows", seed + row)
        draw_panel(axes[row, 1], gen_x, gen_y, f"{ratio} generated: {method}", seed + row + 100)
        axes[row, 0].text(-0.08, 1.15, f"{ratio} train:test", transform=axes[row, 0].transAxes, fontsize=11, weight="bold")

    fig.suptitle("Direct real-versus-generated Measure windows with candidate anomaly labels", fontsize=14, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    out = run / "figures" / f"real_vs_generated_anomaly_windows_seed{seed}.png"
    fig.savefig(out, dpi=190, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    plot_comparison(args.run, args.seed)
    print("wrote real-versus-generated comparison to", args.run / "figures")


if __name__ == "__main__":
    main()
