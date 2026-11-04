"""Quantify raw CSDI outliers without changing any generated values."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RUN = HERE / "runs" / "target_date_log_residual_strict_20260916"
FIXED = HERE / "runs" / "upstream_gpu_full_20260910"
DATES = ("250605", "250626", "250718")


def main():
    meta = pd.read_json(RUN / "metadata_env_conditioned.json", dtype={"date": str}).reset_index(drop=True)
    audit = pd.read_csv(FIXED / "paired_row_audit.csv", dtype={"date": str})
    rows = []
    step_rows = []
    for date in DATES:
        syn = np.load(RUN / "CSDI" / "seed42" / date / "synthetic_env_conditioned.npz")
        raw = syn["raw_abs"]
        labels, anchors = syn["label"], syn["anchor_ids"]
        real = audit[audit.date == date]
        for label in (0, 1):
            x = raw[labels == label]
            if len(x) == 0:
                continue
            real_label = real[real.error4 == label]
            rows.append({
                "date": date, "label": label, "n_windows": len(x),
                "raw_M4_min": float(np.min(x[:, :, 0])), "raw_M4_p50": float(np.median(x[:, :, 0])),
                "raw_M4_p95": float(np.quantile(x[:, :, 0], .95)), "raw_M4_max": float(np.max(x[:, :, 0])),
                "raw_M7_min": float(np.min(x[:, :, 1])), "raw_M7_p50": float(np.median(x[:, :, 1])),
                "raw_M7_p95": float(np.quantile(x[:, :, 1], .95)), "raw_M7_max": float(np.max(x[:, :, 1])),
                "nonpositive_row_pct": float(((x[:, :, 0] <= 0) | (x[:, :, 1] <= 0)).mean() * 100),
                "above_real_label_max_window_pct": float(((x[:, :, 0].max(1) > real_label.Measure4.max()) | (x[:, :, 1].max(1) > real_label.Measure7.max())).mean() * 100) if len(real_label) else np.nan,
                "above_real_normal_max_window_pct": float(((x[:, :, 0].max(1) > real[real.error4 == 0].Measure4.max()) | (x[:, :, 1].max(1) > real[real.error7 == 0].Measure7.max())).mean() * 100),
            })
            for step in range(4):
                z = x[:, step]
                step_rows.append({
                    "date": date, "label": label, "step": step, "n": len(z),
                    "M4_p50": float(np.median(z[:, 0])), "M4_p95": float(np.quantile(z[:, 0], .95)), "M4_max": float(z[:, 0].max()),
                    "M7_p50": float(np.median(z[:, 1])), "M7_p95": float(np.quantile(z[:, 1], .95)), "M7_max": float(z[:, 1].max()),
                })
    summary = pd.DataFrame(rows).round(6)
    steps = pd.DataFrame(step_rows).round(6)
    summary.to_csv(RUN / "unconstrained_generation_audit.csv", index=False, encoding="utf-8-sig")
    with pd.ExcelWriter(RUN / "unconstrained_generation_audit.xlsx", engine="openpyxl") as xl:
        summary.to_excel(xl, sheet_name="summary", index=False)
        steps.to_excel(xl, sheet_name="by_step", index=False)
    print(summary.to_string(index=False))
    print("wrote", RUN / "unconstrained_generation_audit.xlsx")


if __name__ == "__main__":
    main()
