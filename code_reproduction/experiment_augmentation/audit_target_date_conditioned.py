"""Validate Measure generation under the held-out date's real environment path."""
from __future__ import annotations

import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DEFAULT_RUN = HERE / "runs" / "target_date_conditioned_log_20260916"
FIXED = HERE / "runs" / "upstream_gpu_full_20260910"
DATES = ("250605", "250626", "250718")


def corr(a, b):
    a, b = pd.Series(a, dtype=float), pd.Series(b, dtype=float)
    return float(a.corr(b)) if len(a) > 2 and a.nunique() > 1 and b.nunique() > 1 else np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, default=DEFAULT_RUN)
    args = ap.parse_args()
    run = args.run
    meta = pd.read_json(run / "metadata_env_conditioned.json", dtype={"date": str}).reset_index(drop=True)
    audit = pd.read_csv(FIXED / "paired_row_audit.csv", dtype={"date": str})
    rows = []
    for date in DATES:
        syn = np.load(run / "CSDI" / "seed42" / date / "synthetic_env_conditioned.npz")
        labels = syn["label"]
        anchors = syn["anchor_ids"]
        arr = syn["projected_abs"]
        real = audit[(audit.date == date) & audit.agree].copy()
        real_elapsed = (pd.to_datetime(real.dt) - pd.to_datetime(real.dt).min()).dt.total_seconds() / 60
        for label in (0, 1):
            ids = labels == label
            if not ids.any():
                continue
            cond_rows = []
            gen4, gen7 = [], []
            for i in np.flatnonzero(ids):
                a = meta.iloc[int(anchors[i])]
                # In target_date mode, all anchor rows should belong to test date.
                assert str(a.date) == date
                q = audit[(audit.date == date) & audit.row4.isin([int(x) for x in a.row4])].sort_values("dt")
                cond_rows.append(q)
                gen4.extend(arr[i, :, 0].tolist())
                gen7.extend(arr[i, :, 1].tolist())
            cond = pd.concat(cond_rows, ignore_index=True)
            env4 = cond.Hum4.to_numpy()
            env7 = cond.Hum7.to_numpy()
            temp4 = cond.Temp4.to_numpy()
            temp7 = cond.Temp7.to_numpy()
            # Compare generated distribution with the same-date real distribution.
            rows.append({
                "date": date, "label": label, "n_windows": int(ids.sum()),
                "anchor_date_check": "pass",
                "generated_M4_min": float(np.min(gen4)), "generated_M4_max": float(np.max(gen4)),
                "generated_M7_min": float(np.min(gen7)), "generated_M7_max": float(np.max(gen7)),
                "real_M4_min": float(real.Measure4.min()), "real_M4_max": float(real.Measure4.max()),
                "real_M7_min": float(real.Measure7.min()), "real_M7_max": float(real.Measure7.max()),
                "generated_M4_mean": float(np.mean(gen4)), "generated_M7_mean": float(np.mean(gen7)),
                "real_M4_mean": float(real.Measure4.mean()), "real_M7_mean": float(real.Measure7.mean()),
                "generated_M4_Hum4_corr": corr(gen4, env4),
                "generated_M7_Hum7_corr": corr(gen7, env7),
                "real_M4_Hum4_corr": corr(real.Measure4, real.Hum4),
                "real_M7_Hum7_corr": corr(real.Measure7, real.Hum7),
                "generated_M4_Temp4_corr": corr(gen4, temp4),
                "generated_M7_Temp7_corr": corr(gen7, temp7),
                "real_M4_Temp4_corr": corr(real.Measure4, real.Temp4),
                "real_M7_Temp7_corr": corr(real.Measure7, real.Temp7),
            })
    out = pd.DataFrame(rows).round(4)
    out.to_csv(run / "target_date_conditioned_physical_audit.csv", index=False, encoding="utf-8-sig")
    with pd.ExcelWriter(run / "target_date_conditioned_physical_audit.xlsx", engine="openpyxl") as xl:
        out.to_excel(xl, sheet_name="audit", index=False)
        pd.DataFrame({
            "item": ["condition source", "training source", "leakage note", "interpretation"],
            "value": [
                "Each generated window uses Temp/Hum/elapsed from a real window of its target date.",
                "CSDI weights are trained only on the other two dates.",
                "The first two Measure steps are real target-date anchors; this is physical continuation validation, not a leakage-free classifier benchmark.",
                "Use the generated-vs-real ranges, means, and environment correlation to judge physical plausibility.",
            ],
        }).to_excel(xl, sheet_name="notes", index=False)
    print(out.to_string(index=False))
    print("wrote", run / "target_date_conditioned_physical_audit.xlsx")


if __name__ == "__main__":
    main()
