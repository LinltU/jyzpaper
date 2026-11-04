"""Direct comparison with unprojected CSDI output.

No clipping, deletion, positivity correction, or range filtering is applied.
The `flag` column is descriptive only and never removes rows.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from openpyxl.styles import Alignment, Font, PatternFill

HERE = Path(__file__).resolve().parent
RUN = HERE / "runs" / "target_date_log_residual_strict_20260916"
FIXED = HERE / "runs" / "upstream_gpu_full_20260910"
DATES = ("250605", "250626", "250718")
N_WINDOWS = 200
COLS = ["source", "time", "Measure4", "Measure7", "Temp4", "Hum4", "Temp7", "Hum7", "error", "condition_source", "flag"]


def main():
    meta = pd.read_json(RUN / "metadata_env_conditioned.json", dtype={"date": str}).reset_index(drop=True)
    audit = pd.read_csv(FIXED / "paired_row_audit.csv", dtype={"date": str})
    by_row4 = {(str(r.date), int(r.row4)): r for r in audit.itertuples(index=False)}
    parts = []
    with pd.ExcelWriter(RUN / "unconstrained_raw_real_vs_generated.xlsx", engine="openpyxl") as xl:
        for date in DATES:
            real_src = audit[audit.date == date].sort_values("dt")
            real = pd.DataFrame({
                "source": "real", "time": real_src.time4, "Measure4": real_src.Measure4,
                "Measure7": real_src.Measure7, "Temp4": real_src.Temp4, "Hum4": real_src.Hum4,
                "Temp7": real_src.Temp7, "Hum7": real_src.Hum7, "error": real_src.error4.astype(int),
                "condition_source": "real", "flag": "real_observation",
            })
            syn = np.load(RUN / "CSDI" / "seed42" / date / "synthetic_env_conditioned.npz")
            labels, anchors, raw = syn["label"], syn["anchor_ids"], syn["raw_abs"]
            pick = np.random.default_rng(20260917).choice(len(labels), min(N_WINDOWS, len(labels)), replace=False)
            generated = []
            observed = audit[audit.date == date]
            normal_max4 = observed[observed.error4 == 0].Measure4.max()
            normal_max7 = observed[observed.error7 == 0].Measure7.max()
            all_min4, all_min7 = observed.Measure4.min(), observed.Measure7.min()
            all_max4, all_max7 = observed.Measure4.max(), observed.Measure7.max()
            for i in pick:
                anchor = meta.iloc[int(anchors[i])]
                for step, row4 in enumerate(anchor.row4):
                    env = by_row4[(str(anchor.date), int(row4))]
                    m4, m7 = float(raw[i, step, 0]), float(raw[i, step, 1])
                    if m4 <= 0 or m7 <= 0:
                        flag = "nonpositive_raw"
                    elif m4 > normal_max4 or m7 > normal_max7:
                        flag = "above_observed_normal"
                    elif m4 < all_min4 or m7 < all_min7 or m4 > all_max4 or m7 > all_max7:
                        flag = "outside_observed_all"
                    else:
                        flag = "within_observed_range"
                    generated.append({
                        "source": "generated_raw", "time": "", "Measure4": m4, "Measure7": m7,
                        "Temp4": env.Temp4, "Hum4": env.Hum4, "Temp7": env.Temp7, "Hum7": env.Hum7,
                        "error": int(labels[i]), "condition_source": str(anchor.date), "flag": flag,
                    })
            part = pd.concat([real, pd.DataFrame(generated)], ignore_index=True)
            part["Measure4"] = part.Measure4.astype(float)
            part["Measure7"] = part.Measure7.astype(float)
            part.to_excel(xl, sheet_name=date, index=False)
            ws = xl.sheets[date]
            for cell in ws[1]:
                cell.font = Font(bold=True)
                cell.alignment = Alignment(horizontal="center")
            fill = PatternFill("solid", fgColor="FCE4D6")
            for row in range(2, ws.max_row + 1):
                is_gen = ws.cell(row, 1).value == "generated_raw"
                for col in range(1, ws.max_column + 1):
                    ws.cell(row, col).alignment = Alignment(horizontal="center")
                    if is_gen:
                        ws.cell(row, col).fill = fill
                for col in (3, 4, 5, 6, 7, 8):
                    ws.cell(row, col).number_format = "0.000000"
            ws.freeze_panes = "A2"
            parts.append(part.assign(date=date))
        pd.DataFrame({
            "item": ["generator", "postprocessing", "flag", "condition", "important"],
            "value": [
                "CSDI local adaptation using the upstream imputation mask; log-residual target is decoded with exp().",
                "None: raw values are shown exactly as decoded. No clipping, deletion, positivity correction, or range filtering.",
                "Descriptive only; flags do not remove or modify rows.",
                "Temp/Hum/elapsed are from the target date; first two Measure steps are observed anchors.",
                "Large values are evidence about model behavior, not corrected data.",
            ],
        }).to_excel(xl, sheet_name="notes", index=False)
    out = pd.concat(parts, ignore_index=True)
    out.to_csv(RUN / "unconstrained_raw_real_vs_generated.csv", index=False, encoding="utf-8-sig")
    print("wrote", RUN / "unconstrained_raw_real_vs_generated.xlsx")
    print("wrote", RUN / "unconstrained_raw_real_vs_generated.csv")


if __name__ == "__main__":
    main()
