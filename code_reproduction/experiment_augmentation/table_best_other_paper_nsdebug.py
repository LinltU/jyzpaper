"""Direct table for the best-performing other generator: NsDiff raw output."""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
from openpyxl.styles import Alignment, Font, PatternFill

HERE = Path(__file__).resolve().parent
ROOT = HERE / "runs" / "all_papers_extended_20260910"
GEN = ROOT / "generators" / "NsDiff"
OUT = HERE / "runs" / "paper_generator_physical_audit_20260917"
DATES = ("250605", "250626", "250718")
SEEDS = (42, 123, 2024)
N_WINDOWS = 200


def main():
    meta = pd.read_json(ROOT / "metadata.json", dtype={"date": str}).reset_index(drop=True)
    audit = pd.read_csv(ROOT / "paired_row_audit.csv", dtype={"date": str})
    ranges = {}
    for date, g in audit[audit.agree].groupby("date"):
        ranges[str(date)] = (g.Measure4.min(), g.Measure4.max(), g.Measure7.min(), g.Measure7.max())
    parts = []
    with pd.ExcelWriter(OUT / "best_other_paper_NsDiff_raw_vs_real.xlsx", engine="openpyxl") as xl:
        for date in DATES:
            real_src = audit[audit.date == date].sort_values("dt")
            rows = []
            for r in real_src.itertuples(index=False):
                rows.append({"source": "real", "time": r.time4, "Measure4": r.Measure4, "Measure7": r.Measure7,
                             "error": int(r.error4), "source_date": date, "flag": "real"})
            pool = []
            for seed in SEEDS:
                d = np.load(GEN / f"seed{seed}" / date / "synthetic.npz")
                raw = d["raw"]
                label = d["label"]
                anchor = d["anchor_ids"]
                pick_seed = np.random.default_rng(20260917 + seed).choice(len(label), min(N_WINDOWS // len(SEEDS) + 1, len(label)), replace=False)
                for i in pick_seed:
                    pool.append((seed, int(i), int(label[i]), int(anchor[i]), raw[i].copy()))
            pick = np.arange(min(N_WINDOWS, len(pool)))
            for j in pick:
                seed, i, label, anchor_id, raw = pool[int(j)]
                a = meta.iloc[int(anchor_id)]
                lo4, hi4, lo7, hi7 = ranges[str(a.date)]
                for step in range(4):
                    m4 = float(raw[step, 0] + a.base4)
                    m7 = float(raw[step, 1] + a.base7)
                    if m4 <= 0 or m7 <= 0:
                        flag = "nonpositive"
                    elif m4 < lo4 or m4 > hi4 or m7 < lo7 or m7 > hi7:
                        flag = "outside_source_date_range"
                    else:
                        flag = "within_source_date_range"
                    rows.append({"source": "NsDiff_raw", "time": "", "Measure4": m4, "Measure7": m7,
                                 "error": int(label), "source_date": str(a.date), "flag": flag})
            df = pd.DataFrame(rows)
            df[["Measure4", "Measure7"]] = df[["Measure4", "Measure7"]].round(6)
            df.to_excel(xl, sheet_name=date, index=False)
            ws = xl.sheets[date]
            for cell in ws[1]:
                cell.font = Font(bold=True)
                cell.alignment = Alignment(horizontal="center")
            fill = PatternFill("solid", fgColor="E2F0D9")
            for row in range(2, ws.max_row + 1):
                is_gen = ws.cell(row, 1).value == "NsDiff_raw"
                for col in range(1, ws.max_column + 1):
                    ws.cell(row, col).alignment = Alignment(horizontal="center")
                    if is_gen:
                        ws.cell(row, col).fill = fill
            ws.freeze_panes = "A2"
            parts.append(df.assign(date=date))
        pd.DataFrame({
            "item": ["selected generator", "selection reason", "postprocessing"],
            "value": ["NsDiff raw", "Best RF ROC-AUC+AP among audited generator regimes", "None; table shows raw saved synthetic output"],
        }).to_excel(xl, sheet_name="notes", index=False)
    pd.concat(parts, ignore_index=True).to_csv(OUT / "best_other_paper_NsDiff_raw_vs_real.csv", index=False, encoding="utf-8-sig")
    print("wrote", OUT / "best_other_paper_NsDiff_raw_vs_real.xlsx")


if __name__ == "__main__":
    main()
