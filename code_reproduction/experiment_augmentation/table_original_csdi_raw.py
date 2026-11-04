"""Unconstrained table from the original upstream CSDI feature run.

This uses `synthetic.npz[raw]` from upstream_gpu_full_20260910 exactly as saved:
no projection, clipping, deletion, or range filtering.
"""
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
FIXED = HERE / "runs" / "upstream_gpu_full_20260910"
OUT = FIXED / "original_csdi_raw_real_vs_generated.csv"
DATES = ("250605", "250626", "250718")
SEEDS = (42, 123, 2024)

def main():
    meta = pd.read_json(FIXED / "metadata.json", dtype={"date": str}).reset_index(drop=True)
    audit = pd.read_csv(FIXED / "paired_row_audit.csv", dtype={"date": str})
    rows = []
    for date in DATES:
        real = audit[audit.date == date].sort_values("dt")
        for r in real.itertuples(index=False):
            rows.append({"date": date, "source": "real", "time": r.time4, "Measure4": r.Measure4,
                         "Measure7": r.Measure7, "error": int(r.error4), "condition_source": "real"})
        for seed in SEEDS:
            d = np.load(FIXED / "CSDI" / f"seed{seed}" / date / "synthetic.npz")
            for i in range(min(200, len(d["label"]))):
                a = meta.iloc[int(d["anchor_ids"][i])]
                for step in range(4):
                    m4 = float(d["raw"][i, step, 0] + a.base4)
                    m7 = float(d["raw"][i, step, 1] + a.base7)
                    rows.append({"date": date, "source": "CSDI_raw", "time": "", "Measure4": m4,
                                 "Measure7": m7, "error": int(d["label"][i]),
                                 "condition_source": str(a.date)})
    out = pd.DataFrame(rows)
    out.to_csv(OUT, index=False, encoding="utf-8-sig")
    print("wrote", OUT)

if __name__ == "__main__":
    main()
