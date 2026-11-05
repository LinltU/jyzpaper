"""Side-by-side comparison figure for the four v4.2 environment baselines."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

ORDER = ["huber", "archived", "nlinear", "last"]
TITLES = {
    "huber": "Huber (v4 reference)\n4 params, additive",
    "archived": "DecompositionNLinear (v4.2 official)\n10 features, non-linear MLP",
    "nlinear": "official LTSF NLinear\nseq_len=16, under-fitted",
    "last": "naive persistence\nprevious Measure",
}


def load_points(run: Path) -> pd.DataFrame:
    return pd.read_csv(run / "device_points_v42.csv")


def load_windows(run: Path) -> pd.DataFrame:
    return pd.read_csv(run / "device_nonoverlap_windows_v42.csv")


def main(args) -> None:
    args.output.mkdir(parents=True, exist_ok=True)
    runs = {k: args.root / f"v42b_{k}" for k in ORDER}
    W = {k: load_windows(v) for k, v in runs.items()}
    P = {k: load_points(v) for k, v in runs.items()}

    huber = set(W["huber"].loc[W["huber"].window_anomaly_v42 == 1, "window_id"])
    devices = sorted(W["archived"].IMEI.unique())

    fig = plt.figure(figsize=(26, 20))
    gs = fig.add_gridspec(3, 4, height_ratios=[1.0, 1.0, 1.1], hspace=0.42, wspace=0.22)

    for col, kind in enumerate(ORDER):
        w, p = W[kind], P[kind]
        ax = fig.add_subplot(gs[0, col])
        counts = w.groupby("IMEI").window_anomaly_v42.sum().reindex(devices, fill_value=0)
        totals = w.groupby("IMEI").size().reindex(devices, fill_value=0)
        color = "#cbd5e1" if kind == "huber" else ("#0f766e" if kind == "archived" else "#b45309")
        ax.bar(np.arange(len(devices)), counts.values, color=color)
        ax.set_xticks([])
        ax.set_title(f"{TITLES[kind]}\ncandidates = {int(w.window_anomaly_v42.sum())}/547 "
                     f"({100 * w.window_anomaly_v42.mean():.1f}%)", fontsize=11)
        ax.set_ylabel("candidate windows")
        ax.grid(axis="y", alpha=.2)

        ax = fig.add_subplot(gs[1, col])
        ok = p.env_baseline.notna() & np.isfinite(p.env_baseline)
        ax.scatter(p.loc[ok, "env_baseline"], (p.Measure - p.env_baseline)[ok],
                   s=1.2, alpha=.18, color="#334155", rasterized=True)
        ax.axhline(0, color="#dc2626", lw=.8)
        ax.set_xlabel("predicted Measure (baseline)")
        ax.set_ylabel("residual")
        err = (p.Measure - p.env_baseline)[ok]
        ss = ((p.Measure - p.Measure.mean()) ** 2)[ok].sum()
        ax.set_title(f"MAE={err.abs().mean():.4f}  R²={1 - (err ** 2).sum() / ss:.4f}\n"
                     f"residual std={p[ok].groupby('IMEI').env_residual.std().median():.4f}",
                     fontsize=10)
        ax.grid(alpha=.2)

        ax = fig.add_subplot(gs[2, col])
        reasons = w.anomaly_reason.value_counts()
        keys = ["peer-delta jump (per-device)", "isolated extreme peer jump",
                "environment residual and peer agree", "normal"]
        short = ["B: peer jump", "C: extreme jump", "A: residual+peer", "normal"]
        vals = [int(reasons.get(k, 0)) for k in keys]
        bars = ax.barh(short[::-1], vals[::-1],
                       color=["#cbd5e1", "#7c3aed", "#dc2626", "#f59e0b"][::-1])
        for b, v in zip(bars, vals[::-1]):
            ax.text(v + 3, b.get_y() + b.get_height() / 2, str(v), va="center", fontsize=9)
        ax.set_xlim(0, max(vals) * 1.18 if max(vals) else 1)
        ax.set_title("windows by triggering rule", fontsize=10)
        ax.grid(axis="x", alpha=.2)

    fig.suptitle("v4.2 environment-baseline comparison (same 32-point windows, same A/B/C/D thresholds)",
                 fontsize=17, y=0.975)
    fig.text(0.5, 0.945,
             "Only the baseline predicting Measure changes. Higher candidate counts for nlinear/last come from "
             "worse residuals, not from detecting more anomalies.",
             ha="center", fontsize=11)
    fig.text(0.5, 0.012,
             "archived = legacy_dlinear DecompositionNLinear with Measure_diff / Measure_lag1 / Measure_MA removed "
             "(those three reconstruct or contain the target). Its series decomposition degenerates at time length 1, "
             "so the effective model is a non-linear MLP.",
             ha="center", fontsize=9)
    out = args.output / "v42_baseline_comparison.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)

    rows = []
    for kind in ORDER:
        w, p = W[kind], P[kind]
        ok = p.env_baseline.notna() & np.isfinite(p.env_baseline)
        err = (p.Measure - p.env_baseline)[ok]
        ss = ((p.Measure - p.Measure.mean()) ** 2)[ok].sum()
        cand = set(w.loc[w.window_anomaly_v42 == 1, "window_id"])
        rows.append(dict(
            baseline=kind, candidates=int(w.window_anomaly_v42.sum()),
            candidate_pct=round(100 * w.window_anomaly_v42.mean(), 2),
            MAE=round(err.abs().mean(), 5), RMSE=round(np.sqrt((err ** 2).mean()), 5),
            R2=round(1 - (err ** 2).sum() / ss, 5),
            residual_std=round(p[ok].groupby("IMEI").env_residual.std().median(), 5),
            common_with_huber=len(cand & huber),
            jaccard_vs_huber=round(len(cand & huber) / len(cand | huber), 3),
            new_vs_huber=len(cand - huber), lost_vs_huber=len(huber - cand)))
    table = pd.DataFrame(rows)
    table.to_csv(args.output / "baseline_comparison.csv", index=False)
    print(table.to_string(index=False))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/tmp"))
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args())
