"""Plot comparisons between real windows and augmentation-generated synthetic windows."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs"
FIG = RUNS / "figures"
FIG.mkdir(parents=True, exist_ok=True)
FEATURES = ["M4_residual", "M7_residual", "M4-M7", "M4/M7", "log(M4/M7)"]
FIXED_ROOT = RUNS / "upstream_gpu_full_20260910"
EXT_ROOT = RUNS / "all_papers_extended_20260910" / "generators"
FIXED = {"CSDI", "MTSCI", "IMDiffusion", "Diffusion-TS", "D3R"}

COLORS = {"Real": "black", "CSDI": "#1f77b4", "NsDiff": "#d62728", "Diffusion-TS": "#2ca02c",
          "D3R": "#9467bd", "Treeffuser": "#ff7f0e", "TimeGrad": "#8c564b", "TabDDPM": "#17becf"}


def method_root(method):
    return (FIXED_ROOT if method in FIXED else EXT_ROOT) / method


def load_synth(method, key="raw"):
    root = method_root(method)
    arrs, labs = [], []
    for run in sorted(root.glob("seed*/2*")):
        p = run / "synthetic.npz"
        if p.exists():
            d = np.load(p)
            arrs.append(d[key])
            labs.append(d["label"])
    if not arrs:
        return None, None
    return np.concatenate(arrs), np.concatenate(labs)


def load_real():
    d = np.load(FIXED_ROOT / "dataset.npz")
    meta = pd.read_json(FIXED_ROOT / "metadata.json", dtype={"date": str})
    return d["x"], meta.label.to_numpy(), meta


def fig_distributions():
    xr, yr, _ = load_real()
    methods = ["CSDI", "NsDiff", "Diffusion-TS", "Treeffuser"]
    fig, axes = plt.subplots(1, 5, figsize=(22, 4.2))
    real_flat = xr.reshape(-1, 5)
    for j, ax in enumerate(axes):
        lo = np.percentile(real_flat[:, j], 0.5)
        hi = np.percentile(real_flat[:, j], 99.5)
        bins = np.linspace(lo, hi, 60)
        ax.hist(real_flat[:, j], bins=bins, density=True, histtype="step", lw=2.4,
                color=COLORS["Real"], label=f"Real (n={len(real_flat)})")
        for m in methods:
            syn, _ = load_synth(m, "raw")
            if syn is None:
                continue
            v = syn.reshape(-1, 5)[:, j]
            v = v[(v >= lo) & (v <= hi)]
            ax.hist(v, bins=bins, density=True, histtype="step", lw=1.6,
                    color=COLORS.get(m, None), label=m)
        ax.set_title(FEATURES[j])
        ax.set_yticks([])
    axes[0].set_ylabel("density")
    axes[-1].legend(fontsize=8, loc="upper right")
    fig.suptitle("Feature distributions: Real vs Synthetic (raw), aggregated over all runs", y=1.02)
    fig.tight_layout()
    p = FIG / "compare_distributions.png"
    fig.savefig(p, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("wrote", p)


def fig_per_method_std():
    xr, _, _ = load_real()
    real_std = xr.reshape(-1, 5).std(0)
    methods = ["CSDI", "MTSCI", "IMDiffusion", "Diffusion-TS", "D3R", "TimeGrad", "NsDiff",
               "FIDE", "PaD-TS", "WaveStitch", "TabDDPM", "CCDM", "DiffPLF", "TSDE", "SSSD",
               "CoSTI", "SSD-TS", "MCD-TSF", "Treeffuser"]
    data = {"Real": real_std}
    for m in methods:
        syn, _ = load_synth(m, "raw")
        if syn is not None:
            data[m] = syn.reshape(-1, 5).std(0)
    names = list(data)
    vals = np.array([data[n] for n in names])  # (methods, 5)
    x = np.arange(len(names))
    fig, axes = plt.subplots(1, 2, figsize=(20, 5.2), gridspec_kw={"width_ratios": [3, 1]})
    ax = axes[0]
    w = 0.15
    for j in range(5):
        ax.bar(x + (j - 2) * w, vals[:, j], width=w, label=FEATURES[j])
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=60, ha="right", fontsize=8)
    ax.set_ylabel("std (per feature, over windows)")
    ax.set_title("Per-feature standard deviation: Real vs each generator (raw synthetic)")
    ax.legend(fontsize=8)
    ax.axhline(0, color="k", lw=0.6)
    # normalized ratio
    ax2 = axes[1]
    ratio = vals[1:] / vals[0]
    im = ax2.imshow(ratio.T, aspect="auto", cmap="coolwarm", vmin=0, vmax=2)
    ax2.set_xticks(range(len(names) - 1))
    ax2.set_xticklabels(names[1:], rotation=90, fontsize=8)
    ax2.set_yticks(range(5))
    ax2.set_yticklabels(FEATURES, fontsize=8)
    ax2.set_title("std(generator) / std(Real)\n(1.0 = matches real spread)")
    fig.colorbar(im, ax=ax2, fraction=0.046)
    fig.tight_layout()
    p = FIG / "compare_per_method_std.png"
    fig.savefig(p, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("wrote", p)


def fig_scatter():
    xr, yr, _ = load_real()
    fig, axes = plt.subplots(1, 4, figsize=(19, 4.6))
    # real, colored by class
    ax = axes[0]
    for lab, c, nm in [(0, "#1f77b4", "normal"), (1, "#d62728", "abnormal")]:
        v = xr[yr == lab].reshape(-1, 5)
        ax.scatter(v[:, 0], v[:, 1], s=12, alpha=0.6, c=c, label=nm)
    ax.set_title(f"Real windows (n={len(xr)})")
    ax.set_xlabel(FEATURES[0]); ax.set_ylabel(FEATURES[1]); ax.legend(fontsize=8)
    for k, m in enumerate(["CSDI", "NsDiff", "Treeffuser"]):
        ax = axes[k + 1]
        syn, sy = load_synth(m, "raw")
        if syn is None:
            continue
        for lab, c, nm in [(0, "#1f77b4", "normal"), (1, "#d62728", "abnormal")]:
            v = syn[sy == lab].reshape(-1, 5)
            v = v[np.random.default_rng(0).choice(len(v), min(4000, len(v)), replace=False)]
            ax.scatter(v[:, 0], v[:, 1], s=5, alpha=0.35, c=c, label=nm)
        ax.set_title(f"{m} synthetic")
        ax.set_xlabel(FEATURES[0]); ax.set_ylabel(FEATURES[1])
    fig.suptitle("M4_residual vs M7_residual: real vs synthetic (normal/abnormal)", y=1.03)
    fig.tight_layout()
    p = FIG / "compare_scatter_M4_M7.png"
    fig.savefig(p, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("wrote", p)


def fig_windows():
    xr, yr, _ = load_real()
    fig, axes = plt.subplots(1, 5, figsize=(22, 4.2))
    methods = ["CSDI", "NsDiff", "Treeffuser"]
    t = np.arange(xr.shape[1])
    for j, ax in enumerate(axes):
        r = xr.reshape(-1, 5)[:, j].reshape(len(xr), -1)
        # reconstruct per-window matrix
        rw = xr[:, :, j]
        ax.plot(t, rw.mean(0), "-o", color="black", lw=2.4, label="Real mean")
        ax.fill_between(t, rw.mean(0) - rw.std(0), rw.mean(0) + rw.std(0), color="black", alpha=0.12)
        for m in methods:
            syn, _ = load_synth(m, "raw")
            if syn is None:
                continue
            sw = syn[:, :, j]
            ax.plot(t, sw.mean(0), "-", color=COLORS.get(m), lw=1.6, label=f"{m} mean")
            ax.fill_between(t, sw.mean(0) - sw.std(0), sw.mean(0) + sw.std(0), color=COLORS.get(m), alpha=0.08)
        ax.set_title(FEATURES[j]); ax.set_xticks(t); ax.set_xlabel("window step")
    axes[0].set_ylabel("value")
    axes[-1].legend(fontsize=7)
    fig.suptitle("Window shape: per-step mean ± std, Real vs Synthetic", y=1.02)
    fig.tight_layout()
    p = FIG / "compare_windows.png"
    fig.savefig(p, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("wrote", p)


def fig_raw_vs_projected():
    syn, sy = load_synth("NsDiff", "raw")
    proj, _ = load_synth("NsDiff", "projected")
    if syn is None:
        return
    # read the true physical-validity flag from the stored npz
    pos_flags = []
    for run in sorted(method_root("NsDiff").glob("seed*/2*")):
        p = run / "synthetic.npz"
        if p.exists():
            pos_flags.append(np.load(p)["positive"])
    pos = np.concatenate(pos_flags)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    ax = axes[0]
    ax.hist(syn[:, :, 0].reshape(-1), bins=80, alpha=0.6, label="raw M4_residual", color="#1f77b4")
    ax.hist(proj[:, :, 0].reshape(-1), bins=80, alpha=0.6, label="projected M4_residual", color="#d62728")
    ax.set_title("NsDiff: raw vs projected (M4_residual)"); ax.legend(fontsize=8); ax.set_yscale("log")
    ax = axes[1]
    ax.scatter(syn[:, :, 0].reshape(-1)[::50], proj[:, :, 0].reshape(-1)[::50], s=3, alpha=0.3)
    ax.set_xlabel("raw"); ax.set_ylabel("projected"); ax.set_title("M4_residual: raw vs projected (identity except clip)")
    ax = axes[2]
    fracs = [float(pos.mean()), 1.0]
    ax.bar(["raw", "projected"], fracs, color=["#1f77b4", "#d62728"])
    for i, v in enumerate(fracs):
        ax.text(i, v + 0.02, f"{v:.3f}", ha="center")
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("fraction with M4>0 and M7>0")
    ax.set_title("physical validity: raw vs projected")
    fig.suptitle("Effect of physical projection (NsDiff)", y=1.02)
    fig.tight_layout()
    p = FIG / "compare_raw_vs_projected.png"
    fig.savefig(p, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("wrote", p)


if __name__ == "__main__":
    fig_distributions()
    fig_per_method_std()
    fig_scatter()
    fig_windows()
    fig_raw_vs_projected()
    print("all figures in", FIG)
