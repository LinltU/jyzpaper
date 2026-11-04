"""Create reproducible summaries and figures for the formal device benchmark."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


RATIOS = ["2to8", "3to7", "4to6", "5to5"]
METRICS = ["Precision", "Recall", "F1", "ROC-AUC", "Average Precision"]


def load_results(run: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = pd.read_csv(run / "all_results.csv")
    real = raw[raw["method"].eq("Real-only")].drop_duplicates(
        ["ratio", "classifier", "seed"]
    )
    aug = raw[raw["method"].ne("Real-only")]
    detail = pd.concat([real, aug], ignore_index=True)
    return raw, detail


def summarize(detail: pd.DataFrame) -> pd.DataFrame:
    keys = ["ratio", "method", "mode", "classifier"]
    out = detail.groupby(keys).agg(
        runs=("seed", "nunique"),
        **{f"{m}_mean": (m, "mean") for m in METRICS},
        **{f"{m}_std": (m, "std") for m in METRICS},
    ).reset_index()
    return out.sort_values(["ratio", "classifier", "Average Precision_mean"], ascending=[True, True, False])


def annotation_tables(run: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    points = pd.read_csv(run / "annotated_device_points.csv")
    points["peer_matched"] = points["peer_match_seconds"].notna().astype(int)
    device = points.groupby("IMEI", as_index=False).agg(
        location=("Location", "first"),
        group_id=("group_id", "first"),
        rows=("IMEI", "size"),
        peer_matched=("peer_matched", "sum"),
        individual_anomalies=("pseudo_anomaly_individual", "sum"),
        peer_anomalies=("pseudo_anomaly_peer", "sum"),
        combined_anomalies=("pseudo_anomaly", "sum"),
    )
    device["peer_match_rate"] = device["peer_matched"] / device["rows"]
    device["combined_anomaly_rate"] = device["combined_anomalies"] / device["rows"]
    group = points.groupby("group_id", as_index=False).agg(
        devices=("IMEI", "nunique"),
        rows=("IMEI", "size"),
        peer_matched=("peer_matched", "sum"),
        individual_anomalies=("pseudo_anomaly_individual", "sum"),
        peer_anomalies=("pseudo_anomaly_peer", "sum"),
        combined_anomalies=("pseudo_anomaly", "sum"),
    )
    group["peer_match_rate"] = group["peer_matched"] / group["rows"]
    group["combined_anomaly_rate"] = group["combined_anomalies"] / group["rows"]
    return device.sort_values("group_id"), group.sort_values("group_id")


def raw_audit(run: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(run.glob("raw_*_seed*.npz")):
        body = path.stem[len("raw_"):]
        ratio, method_seed = body.split("_seed", 1)
        method = ratio.split("_", 1)[1]
        ratio = ratio.split("_", 1)[0]
        with np.load(path) as z:
            windows = z["windows"]
            labels = z["labels"]
        measure = windows[:, :, 0]
        rows.append({
            "file": path.name,
            "ratio": ratio,
            "method": method,
            "seed": int(method_seed),
            "windows": len(windows),
            "positive_windows": int(labels.sum()),
            "nan_values": int(np.isnan(windows).sum()),
            "nonpositive_measure_values": int((measure <= 0).sum()),
            "measure_min": float(np.nanmin(measure)),
            "measure_max": float(np.nanmax(measure)),
            "measure_mean": float(np.nanmean(measure)),
        })
    return pd.DataFrame(rows)


def figures(summary: pd.DataFrame, out: Path) -> None:
    fig_dir = out / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    hgb = summary[summary["classifier"].eq("HGB")].copy()
    aug = hgb[hgb["method"].ne("Real-only")]
    real = hgb[hgb["method"].eq("Real-only")]

    best = aug.loc[aug.groupby("ratio")["Average Precision_mean"].idxmax()].copy()
    best = best.sort_values("ratio", key=lambda s: s.map({r: i for i, r in enumerate(RATIOS)}))
    base = real.set_index("ratio").loc[best["ratio"]]
    x = np.arange(len(best))
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.errorbar(x - 0.12, base["Average Precision_mean"], yerr=base["Average Precision_std"], fmt="o", capsize=4, label="Real-only")
    ax.errorbar(x + 0.12, best["Average Precision_mean"], yerr=best["Average Precision_std"], fmt="o", capsize=4, label="Best augmentation")
    ax.set_xticks(x, [f"{r}\n{m}" for r, m in zip(best["ratio"], best["method"])])
    ax.set_ylabel("Average Precision (mean +/- std)")
    ax.set_ylim(0, 1.02)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    ax.set_title("Formal multi-seed benchmark: best augmentation vs Real-only")
    fig.tight_layout()
    fig.savefig(fig_dir / "formal_best_vs_real_mean_std.png", dpi=190)
    plt.close(fig)

    pivot = aug.pivot_table(index="method", columns="ratio", values="Average Precision_mean").reindex(columns=RATIOS)
    fig, ax = plt.subplots(figsize=(8.5, 7))
    im = ax.imshow(pivot.to_numpy(float), aspect="auto", cmap="YlGnBu", vmin=0, vmax=1)
    ax.set_xticks(np.arange(len(RATIOS)), RATIOS)
    ax.set_yticks(np.arange(len(pivot.index)), pivot.index)
    ax.set_title("HGB Average Precision mean by method and train:test ratio")
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            value = pivot.iloc[i, j]
            if pd.notna(value):
                ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(fig_dir / "formal_hgb_ap_heatmap_mean.png", dpi=190)
    plt.close(fig)


def write_report(run: Path, summary: pd.DataFrame, device: pd.DataFrame, groups: pd.DataFrame, audit: pd.DataFrame) -> None:
    hgb = summary[summary["classifier"].eq("HGB")]
    real = hgb[hgb["method"].eq("Real-only")].set_index("ratio")
    aug = hgb[hgb["method"].ne("Real-only")]
    best = aug.loc[aug.groupby("ratio")["Average Precision_mean"].idxmax()].copy()
    best = best.sort_values("ratio", key=lambda s: s.map({r: i for i, r in enumerate(RATIOS)}))
    best["baseline_ap"] = best["ratio"].map(real["Average Precision_mean"])
    best["delta_ap"] = best["Average Precision_mean"] - best["baseline_ap"]

    total_rows = int(device["rows"].sum())
    total_combined = int(device["combined_anomalies"].sum())
    total_individual = int(device["individual_anomalies"].sum())
    total_peer = int(device["peer_anomalies"].sum())
    matched = int(device["peer_matched"].sum())
    peer_rows = int(device.loc[device["peer_matched"].gt(0), "peer_matched"].sum())
    nonpositive = int(audit["nonpositive_measure_values"].sum())
    nan_values = int(audit["nan_values"].sum())
    generated_measure_values = int(audit["windows"].sum() * 4)

    lines = [
        "# Formal Device-data Benchmark",
        "",
        "This report summarizes the RTX 4090 experiment using the peer-aware pseudo-labels and three seeds (42, 123, 2024). The reported mean and standard deviation are computed from per-seed rows in `all_results.csv`; repeated Real-only rows are deduplicated by ratio, classifier, and seed.",
        "",
        "## Protocol",
        "",
        "- Source: `device_data_110_220_final.xlsx`; 25 devices and 16,985 valid points.",
        "- Ratios are train:test: 2:8, 3:7, 4:6, and 5:5.",
        "- Labels are candidate statistical anomalies, not confirmed maintenance faults.",
        "- Individual labels use Huber environment-adjusted residuals; corresponding serial-1/serial-2 positions additionally use nearest peer residuals within 30 minutes.",
        "- Models use five window features and length-four windows. Formal training uses 60 epochs, 20 diffusion steps, multiplier 1, CUDA, and seeds 42/123/2024.",
        "- The 18 generators are a mixture of fixed upstream adaptations and local mechanism implementations; results should not be described as verbatim official implementations for every paper.",
        "",
        "## Annotation Audit",
        "",
        f"- Candidate anomaly points: {total_combined:,}/{total_rows:,} ({total_combined / total_rows:.2%}).",
        f"- Individual-rule points: {total_individual:,}; peer-rule points: {total_peer:,}.",
        f"- Peer matches: {matched:,} device points; among peer-capable groups, {peer_rows:,} points have a valid match.",
        "- The 110 kV devices have different mapped positions and therefore are not forced into a same-position peer comparison.",
        "",
        "## HGB Results",
        "",
        "| Ratio | Best method by AP | AP mean +/- std | Real-only AP | Delta |",
        "|---|---|---:|---:|---:|",
    ]
    for _, row in best.iterrows():
        lines.append(f"| {row['ratio']} | {row['method']} | {row['Average Precision_mean']:.4f} +/- {row['Average Precision_std']:.4f} | {row['baseline_ap']:.4f} | {row['delta_ap']:+.4f} |")
    lines += [
        "",
        "The best method is selected independently for each ratio using mean HGB Average Precision, so this table is descriptive rather than a single globally tuned winner.",
        "",
        "## Raw Generation Audit",
        "",
        f"- Audited raw files: {len(audit)}.",
        f"- NaN values in generated windows: {nan_values}.",
        f"- Non-positive generated `Measure` values: {nonpositive:,}/{generated_measure_values:,} ({nonpositive / generated_measure_values:.2%}).",
        "- Generated values were audited as saved; no clipping, deletion, or projection was applied for presentation.",
        "",
        "## Files",
        "",
        "- `formal_summary.xlsx`: audit and mean/std tables.",
        "- `formal_summary_mean_std.csv`: per-seed mean/std summary.",
        "- `formal_best_methods.csv`: HGB best method and Real-only comparison by ratio.",
        "- `formal_raw_quality_audit.csv`: raw generated-window audit.",
        "- `figures/formal_best_vs_real_mean_std.png`: best-vs-baseline error bars.",
        "- `figures/formal_hgb_ap_heatmap_mean.png`: method/ratio heatmap.",
    ]
    (run / "DEVICE_DATA_GROUP_PAPER_FORMAL_RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    run = args.run
    raw, detail = load_results(run)
    summary = summarize(detail)
    device, groups = annotation_tables(run)
    audit = raw_audit(run)

    summary.to_csv(run / "formal_summary_mean_std.csv", index=False)
    device.to_csv(run / "formal_annotation_by_device.csv", index=False)
    groups.to_csv(run / "formal_annotation_by_group.csv", index=False)
    audit.to_csv(run / "formal_raw_quality_audit.csv", index=False)

    best = summary[(summary["classifier"].eq("HGB")) & (summary["method"].ne("Real-only"))].copy()
    real = summary[(summary["classifier"].eq("HGB")) & (summary["method"].eq("Real-only"))].set_index("ratio")
    best = best.loc[best.groupby("ratio")["Average Precision_mean"].idxmax()].copy()
    best["real_only_AP_mean"] = best["ratio"].map(real["Average Precision_mean"])
    best["real_only_AP_std"] = best["ratio"].map(real["Average Precision_std"])
    best["delta_AP_mean"] = best["Average Precision_mean"] - best["real_only_AP_mean"]
    best.to_csv(run / "formal_best_methods.csv", index=False)

    figures(summary, run)
    with pd.ExcelWriter(run / "formal_summary.xlsx", engine="openpyxl") as writer:
        summary.to_excel(writer, sheet_name="mean_std", index=False)
        best.to_excel(writer, sheet_name="best_HGB", index=False)
        device.to_excel(writer, sheet_name="annotation_device", index=False)
        groups.to_excel(writer, sheet_name="annotation_group", index=False)
        audit.to_excel(writer, sheet_name="raw_quality", index=False)
        detail.to_excel(writer, sheet_name="per_seed_results", index=False)
    write_report(run, summary, device, groups, audit)
    print("wrote summaries to", run)
    print(best[["ratio", "method", "Average Precision_mean", "delta_AP_mean"]].to_string(index=False))


if __name__ == "__main__":
    main()
