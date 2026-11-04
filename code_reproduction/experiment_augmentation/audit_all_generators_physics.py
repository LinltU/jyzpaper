"""Audit raw synthetic outputs from all available generator papers.

The audit does not correct generated values. It reconstructs Measure4/Measure7
from the saved raw feature windows and reports positivity/range failures.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs"
ROOTS = [
    RUNS / "upstream_gpu_full_20260910",
    RUNS / "all_papers_extended_20260910" / "generators",
]
METRICS = ["Precision", "Recall", "F1", "ROC-AUC", "Average Precision"]


def parse_synth_path(path: Path):
    parts = path.parts
    seed = next(p for p in parts if p.startswith("seed"))
    date = path.parent.name
    if "generators" in parts:
        idx = parts.index("generators")
        method = parts[idx + 1]
        root = Path(*parts[:idx])
        meta_root = root
        family = "mechanism-adapted"
    else:
        method = path.parent.parent.parent.name
        meta_root = path.parent.parent.parent.parent
        family = "fixed-upstream"
    return family, method, int(seed[4:]), date, meta_root


def load_meta(meta_root: Path):
    return pd.read_json(meta_root / "metadata.json", dtype={"date": str}).reset_index(drop=True)


def real_ranges(meta_root: Path):
    audit = pd.read_csv(meta_root / "paired_row_audit.csv", dtype={"date": str})
    out = {}
    for date, g in audit[audit.agree].groupby("date"):
        out[str(date)] = {
            "all": (g.Measure4.min(), g.Measure4.max(), g.Measure7.min(), g.Measure7.max()),
            "normal": (g[g.error4 == 0].Measure4.min(), g[g.error4 == 0].Measure4.max(),
                       g[g.error7 == 0].Measure7.min(), g[g.error7 == 0].Measure7.max()),
            "abnormal": (g[g.error4 != 0].Measure4.min(), g[g.error4 != 0].Measure4.max(),
                         g[g.error7 != 0].Measure7.min(), g[g.error7 != 0].Measure7.max()) if (g.error4 != 0).any() else None,
        }
    return out


def abs_measure(arr, labels, anchors, meta):
    base = meta.loc[anchors, ["base4", "base7"]].to_numpy(float)
    m4 = arr[:, :, 0] + base[:, None, 0]
    m7 = arr[:, :, 1] + base[:, None, 1]
    return m4, m7


def audit_synthetic():
    cache_meta, cache_ranges = {}, {}
    rows = []
    for root in ROOTS:
        if not root.exists():
            continue
        for path in root.glob("**/synthetic.npz"):
            family, method, seed, date, meta_root = parse_synth_path(path)
            key = str(meta_root)
            if key not in cache_meta:
                cache_meta[key] = load_meta(meta_root)
                cache_ranges[key] = real_ranges(meta_root)
            meta = cache_meta[key]
            ranges = cache_ranges[key][date]
            d = np.load(path)
            labels = d["label"].astype(int)
            anchors = d["anchor_ids"].astype(int)
            for mode in ("raw", "projected"):
                if mode not in d.files:
                    continue
                m4, m7 = abs_measure(d[mode], labels, anchors, meta)
                for label in (0, 1):
                    ids = labels == label
                    if not ids.any():
                        continue
                    x4, x7 = m4[ids], m7[ids]
                    source_dates = meta.loc[anchors[ids], "date"].astype(str).to_numpy()
                    outside_label = []
                    outside_all = []
                    for wi, src_date in enumerate(source_dates):
                        src_ranges = cache_ranges[key][src_date]
                        label_range = src_ranges["normal"] if label == 0 or src_ranges["abnormal"] is None else src_ranges["abnormal"]
                        lo4, hi4, lo7, hi7 = label_range
                        all_lo4, all_hi4, all_lo7, all_hi7 = src_ranges["all"]
                        outside_label.append(((x4[wi] < lo4) | (x4[wi] > hi4) | (x7[wi] < lo7) | (x7[wi] > hi7)).any())
                        outside_all.append(((x4[wi] < all_lo4) | (x4[wi] > all_hi4) | (x7[wi] < all_lo7) | (x7[wi] > all_hi7)).any())
                    rows.append({
                        "family": family, "method": method, "seed": seed, "date": date,
                        "mode": mode, "label": label, "n_windows": int(ids.sum()),
                        "M4_nonpositive_row_pct": float((x4 <= 0).mean() * 100),
                        "M7_nonpositive_row_pct": float((x7 <= 0).mean() * 100),
                        "nonpositive_window_pct": float(((x4 <= 0) | (x7 <= 0)).any(axis=1).mean() * 100),
                        "outside_label_range_window_pct": float(np.mean(outside_label) * 100),
                        "outside_all_range_window_pct": float(np.mean(outside_all) * 100),
                        "M4_p50": float(np.median(x4)), "M4_p95": float(np.quantile(x4, .95)), "M4_max": float(np.max(x4)),
                        "M7_p50": float(np.median(x7)), "M7_p95": float(np.quantile(x7, .95)), "M7_max": float(np.max(x7)),
                    })
    return pd.DataFrame(rows)


def collect_metrics():
    rows = []
    for root in ROOTS:
        if not root.exists():
            continue
        for rf in root.glob("**/result.json"):
            # only generator result layouts
            if "detectors" in rf.parts:
                continue
            try:
                family, method, seed, date, _ = parse_synth_path(rf.parent / "synthetic.npz")
            except Exception:
                continue
            if not (rf.parent / "synthetic.npz").exists():
                continue
            data = json.loads(rf.read_text(encoding="utf-8"))
            for r in data.get("results", []):
                rows.append({
                    "family": family, "method": method, "seed": seed, "date": date,
                    "classifier": r.get("classifier"), "regime": r.get("regime"),
                    "n_synthetic": r.get("n_synthetic"),
                    **{m: r.get("metrics", {}).get(m) for m in METRICS},
                })
    return pd.DataFrame(rows)


def fmt(x):
    return "-" if pd.isna(x) else f"{x:.4f}"


def md_table(df):
    if df.empty:
        return "(no rows)"
    lines = ["| " + " | ".join(df.columns) + " |", "|" + "|".join(["---"] * len(df.columns)) + "|"]
    for _, row in df.iterrows():
        vals = []
        for c in df.columns:
            v = row[c]
            vals.append(fmt(v) if isinstance(v, (float, np.floating)) else str(v))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def write_report(phys, metrics, out_md):
    raw = phys[phys["mode"] == "raw"].copy()
    raw_summary = raw.groupby(["family", "method"]).agg(
        runs=("date", "size"),
        nonpositive_window_pct=("nonpositive_window_pct", "mean"),
        outside_label_range_window_pct=("outside_label_range_window_pct", "mean"),
        outside_all_range_window_pct=("outside_all_range_window_pct", "mean"),
        M4_p95=("M4_p95", "mean"),
        M4_max=("M4_max", "max"),
        M7_p95=("M7_p95", "mean"),
        M7_max=("M7_max", "max"),
    ).reset_index().sort_values(["outside_all_range_window_pct", "nonpositive_window_pct"])

    m = metrics[(metrics.classifier == "RF") & (metrics.regime != "real")].copy()
    best_rows = []
    for (family, method), g in m.groupby(["family", "method"]):
        for regime, r in g.groupby("regime"):
            best_rows.append({"family": family, "method": method, "regime": regime, "runs": len(r),
                              **{k: r[k].mean() for k in METRICS}})
    best = pd.DataFrame(best_rows)
    if not best.empty:
        best["rank_score"] = best["ROC-AUC"] + best["Average Precision"]
        best = best.sort_values("rank_score", ascending=False)

    lines = ["# Existing paper generator audit for this project", ""]
    lines.append("## Main conclusion")
    lines.append("")
    lines.append("- The available paper generators can improve downstream RF ranking metrics, but raw samples commonly violate project physics.")
    lines.append("- This is an applicability gap: most paper code is an imputation/generation mechanism, not a calibrated physical simulator with support constraints.")
    lines.append("- `projected_*` results should be treated as post-processed classifier augmentation, not proof that the raw generator produces physically valid voltage sequences.")
    lines.append("")
    lines.append("## Best RF augmentation regimes by ROC-AUC + AP")
    lines.append("")
    if not best.empty:
        show = best.head(15)[["family", "method", "regime", "runs", "F1", "ROC-AUC", "Average Precision"]]
        lines.append(md_table(show))
    lines.append("")
    lines.append("## Raw physical audit by generator")
    lines.append("")
    show = raw_summary[["family", "method", "runs", "nonpositive_window_pct", "outside_label_range_window_pct", "outside_all_range_window_pct", "M4_max", "M7_max"]]
    lines.append(md_table(show))
    lines.append("")
    lines.append("## Interpretation for this engineering task")
    lines.append("")
    lines.append("1. Stable normal and abnormal real windows indeed occupy narrow ranges, but neural diffusion outputs are continuous samples and are not range-preserving by default.")
    lines.append("2. Leave-one-date-out creates date/environment shift; a model trained on two dates must extrapolate to the held-out date.")
    lines.append("3. Window length is only 4 and abnormal windows are extremely few, so class-conditional diffusion has high variance.")
    lines.append("4. None of the audited paper generators encodes the strong humidity/temperature response as a hard physical law.")
    lines.append("5. For deployment-quality data generation, this project needs a project-specific conditional physical model or constrained residual generator, not raw paper-code sampling alone.")
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    out_dir = RUNS / "paper_generator_physical_audit_20260917"
    out_dir.mkdir(parents=True, exist_ok=True)
    phys = audit_synthetic().round(6)
    metrics = collect_metrics().round(6)
    phys.to_csv(out_dir / "raw_physical_audit_by_run.csv", index=False, encoding="utf-8-sig")
    metrics.to_csv(out_dir / "generator_metrics_by_run.csv", index=False, encoding="utf-8-sig")
    with pd.ExcelWriter(out_dir / "paper_generator_audit.xlsx", engine="openpyxl") as xl:
        phys.to_excel(xl, sheet_name="physical_by_run", index=False)
        metrics.to_excel(xl, sheet_name="metrics_by_run", index=False)
        if not phys.empty:
            phys[phys["mode"] == "raw"].groupby(["family", "method"]).mean(numeric_only=True).reset_index().to_excel(xl, sheet_name="raw_summary", index=False)
    write_report(phys, metrics, out_dir / "PAPER_GENERATOR_EFFECT_AND_LIMITS.md")
    print("wrote", out_dir)


if __name__ == "__main__":
    main()
