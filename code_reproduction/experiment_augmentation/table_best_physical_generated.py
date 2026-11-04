"""Best physical-view table for original rows vs constrained generated rows.

Uses the environment-conditioned CSDI run with log(Measure) targets and projected
physical bounds. Only the projected samples are shown; raw samples are kept in the
run audit files but excluded here because this table is for direct inspection.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from openpyxl.styles import Alignment, Font, PatternFill

HERE = Path(__file__).resolve().parent
RUN = HERE / "runs" / "env_conditioned_log_measure_projected_20260916"
FIXED = HERE / "runs" / "upstream_gpu_full_20260910"
DATES = ("250605", "250626", "250718")
N_WINDOWS = 200
ND = 4
COLS = ["source", "time", "Measure4", "Measure7", "Temp4", "Hum4", "Temp7", "Hum7", "error", "condition_source", "method"]


def main():
    meta = pd.read_json(RUN / "metadata_env_conditioned.json", dtype={"date": str}).reset_index(drop=True)
    audit = pd.read_csv(FIXED / "paired_row_audit.csv", dtype={"date": str})
    by_row4 = {(str(r.date), int(r.row4)): r for r in audit.itertuples(index=False)}
    combined = []
    out_xlsx = RUN / "best_real_vs_generated_projected.xlsx"
    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as xl:
        for date in DATES:
            real_src = audit[audit.date == date].sort_values("dt")
            real = pd.DataFrame({
                "source": "real",
                "time": real_src.time4,
                "Measure4": real_src.Measure4,
                "Measure7": real_src.Measure7,
                "Temp4": real_src.Temp4,
                "Hum4": real_src.Hum4,
                "Temp7": real_src.Temp7,
                "Hum7": real_src.Hum7,
                "error": real_src.error4.astype(int),
                "condition_source": "real",
                "method": "original",
            })

            syn = np.load(RUN / "CSDI" / "seed42" / date / "synthetic_env_conditioned.npz")
            labels = syn["label"]
            anchors = syn["anchor_ids"]
            arr = syn["projected_abs"]
            pick = np.random.default_rng(20260916).choice(len(labels), min(N_WINDOWS, len(labels)), replace=False)
            gen_rows = []
            for i in pick:
                anchor = meta.iloc[int(anchors[i])]
                for step, row4 in enumerate(anchor.row4):
                    env = by_row4[(str(anchor.date), int(row4))]
                    gen_rows.append({
                        "source": "generated",
                        "time": "",
                        "Measure4": arr[i, step, 0],
                        "Measure7": arr[i, step, 1],
                        "Temp4": env.Temp4,
                        "Hum4": env.Hum4,
                        "Temp7": env.Temp7,
                        "Hum7": env.Hum7,
                        "error": int(labels[i]),
                        "condition_source": str(anchor.date),
                        "method": "EnvCSDI_log_measure_projected",
                    })
            gen = pd.DataFrame(gen_rows)
            part = pd.concat([real, gen], ignore_index=True)[COLS]
            part[["Measure4", "Measure7", "Temp4", "Hum4", "Temp7", "Hum7"]] = part[["Measure4", "Measure7", "Temp4", "Hum4", "Temp7", "Hum7"]].round(ND)
            part.to_excel(xl, sheet_name=date, index=False)
            ws = xl.sheets[date]
            for cell in ws[1]:
                cell.font = Font(bold=True)
                cell.alignment = Alignment(horizontal="center")
            fill = PatternFill("solid", fgColor="E2F0D9")
            for row in range(2, ws.max_row + 1):
                is_gen = ws.cell(row, 1).value == "generated"
                for col in range(1, ws.max_column + 1):
                    ws.cell(row, col).alignment = Alignment(horizontal="center")
                    if is_gen:
                        ws.cell(row, col).fill = fill
                for col in range(3, 9):
                    ws.cell(row, col).number_format = "0.0000"
            ws.freeze_panes = "A2"
            widths = [11, 21, 12, 12, 10, 10, 10, 10, 7, 16, 30]
            for idx, width in enumerate(widths, 1):
                ws.column_dimensions[ws.cell(1, idx).column_letter].width = width
            combined.append(part.assign(date=date))
        pd.DataFrame({
            "item": ["generator", "physical constraint", "shown mode", "rows", "note"],
            "value": [
                "CSDI adapted from upstream/CSDI, fixed Temp/Hum/elapsed conditions, log(Measure) targets",
                "raw Measure is decoded with exp(); projected Measure is clipped to observed physical range by source date; derived detector features are recomputed",
                "projected only",
                f"each date: all real rows + {N_WINDOWS} generated windows x 4 rows",
                "Temp/Hum columns in generated rows are fixed conditioning context from the anchor window, not generated values",
            ],
        }).to_excel(xl, sheet_name="notes", index=False)
    out_csv = RUN / "best_real_vs_generated_projected.csv"
    pd.concat(combined, ignore_index=True)[["date"] + COLS].to_csv(out_csv, index=False, encoding="utf-8-sig")
    print("wrote", out_xlsx)
    print("wrote", out_csv)


if __name__ == "__main__":
    main()
