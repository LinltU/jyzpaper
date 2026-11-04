"""Compare position-conditioned paper-mechanism samples with real windows.

The existing local experiment adapts a DDPM and an NsDiff-style DDPM, conditioning
on the actual error position (0, 1, 2, 3, 5, 6, 8, 9) and experiment date. This
script only reshapes and reports their saved raw output; it does not clip, delete,
or alter samples.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from openpyxl.styles import Alignment, Font, PatternFill

HERE = Path(__file__).resolve().parent
RUN = HERE / "runs" / "position_conditioned_paper_test_20260917"
INPUT = HERE.parent.parent / "experiment.xlsx"
FEATURES = ["M4", "M7", "Temp4", "Hum4", "Temp7", "Hum7", "Mdiff", "Mratio", "logratio", "Tdiff", "Hdiff", "elapsed_min"]
METHODS = ("DDPM_raw", "NsDiff_style_DDPM")
DATES = ("250605", "250626", "250718")
ERRORS = (0, 1, 2, 3, 5, 6, 8, 9)


def load_real():
    from run_experiment_augmentation import load_windows
    return load_windows(INPUT, 4, FEATURES)


def expected_position(error):
    if error == 4:
        return "M4 should be affected (not observed in data)"
    if error == 7:
        return "M7 should be affected (not observed in data)"
    return "position not represented by M4/M7 channels"


def main():
    real_x, real_meta = load_real()
    all_rows = []
    audit_rows = []
    with pd.ExcelWriter(RUN / "error_position_raw_real_vs_generated.xlsx", engine="openpyxl") as xl:
        for method in METHODS:
            flat = pd.read_csv(RUN / f"{method}_synthetic_windows.csv").to_numpy(float)
            syn = flat.reshape(-1, 4, len(FEATURES))
            sm = pd.read_csv(RUN / f"{method}_synthetic_metadata.csv", dtype={"date": str})
            for date in DATES:
                rows = []
                rid = np.flatnonzero(real_meta.date.astype(str).to_numpy() == date)
                for i in rid:
                    for step in range(4):
                        rows.append({"source": "real", "method": "experiment.xlsx", "date": date,
                                     "error_position": int(real_meta.iloc[i].error), "step": step,
                                     "Measure4": real_x[i, step, 0], "Measure7": real_x[i, step, 1],
                                     "expected_position_effect": expected_position(int(real_meta.iloc[i].error))})
                sid = np.flatnonzero(sm.date.astype(str).to_numpy() == date)
                for i in sid:
                    error = int(sm.iloc[i].error)
                    for step in range(4):
                        rows.append({"source": "generated_raw", "method": method, "date": date,
                                     "error_position": error, "step": step,
                                     "Measure4": syn[i, step, 0], "Measure7": syn[i, step, 1],
                                     "expected_position_effect": expected_position(error)})
                part = pd.DataFrame(rows)
                part[["Measure4", "Measure7"]] = part[["Measure4", "Measure7"]].round(6)
                sheet = f"{method[:8]}_{date}"
                part.to_excel(xl, sheet_name=sheet, index=False)
                ws = xl.sheets[sheet]
                for cell in ws[1]:
                    cell.font = Font(bold=True)
                    cell.alignment = Alignment(horizontal="center")
                fill = PatternFill("solid", fgColor="FCE4D6")
                for row in range(2, ws.max_row + 1):
                    generated = ws.cell(row, 1).value == "generated_raw"
                    for col in range(1, ws.max_column + 1):
                        ws.cell(row, col).alignment = Alignment(horizontal="center")
                        if generated:
                            ws.cell(row, col).fill = fill
                    for col in (7, 8):
                        ws.cell(row, col).number_format = "0.000000"
                ws.freeze_panes = "A2"
                all_rows.append(part)

                for error in ERRORS:
                    z = syn[sid][sm.iloc[sid].error.to_numpy() == error]
                    if len(z) == 0:
                        continue
                    real_z = real_x[rid][real_meta.iloc[rid].error.to_numpy() == error]
                    audit_rows.append({
                        "method": method, "date": date, "error_position": error,
                        "generated_windows": len(z), "real_windows": len(real_z),
                        "generated_M4_mean": float(z[:, :, 0].mean()), "generated_M4_min": float(z[:, :, 0].min()), "generated_M4_max": float(z[:, :, 0].max()),
                        "generated_M7_mean": float(z[:, :, 1].mean()), "generated_M7_min": float(z[:, :, 1].min()), "generated_M7_max": float(z[:, :, 1].max()),
                        "real_M4_mean": float(real_z[:, :, 0].mean()) if len(real_z) else np.nan,
                        "real_M7_mean": float(real_z[:, :, 1].mean()) if len(real_z) else np.nan,
                        "generated_M4_nonpositive_pct": float((z[:, :, 0] <= 0).mean() * 100),
                        "generated_M7_nonpositive_pct": float((z[:, :, 1] <= 0).mean() * 100),
                    })
        audit = pd.DataFrame(audit_rows).round(6)
        audit.to_excel(xl, sheet_name="position_audit", index=False)
        pd.DataFrame({
            "item": ["conditioning", "raw policy", "error meaning", "unobserved positions", "classifier caution"],
            "value": [
                "error position + experiment date; generated windows retain their error position",
                "Both paper mechanisms are reported as saved raw output; no clipping, deletion, or range restriction",
                "error is the zero-value insulator position among 9 insulators",
                "No real samples have error=4 or error=7, so direct M4/M7 zero-position learning for those positions cannot be validated",
                "This position-conditioned run uses error as an offline generation condition; it is not an online detector input",
            ],
        }).to_excel(xl, sheet_name="notes", index=False)
    out = pd.concat(all_rows, ignore_index=True)
    out.to_csv(RUN / "error_position_raw_real_vs_generated.csv", index=False, encoding="utf-8-sig")
    audit.to_csv(RUN / "error_position_raw_audit.csv", index=False, encoding="utf-8-sig")
    print("wrote", RUN / "error_position_raw_real_vs_generated.xlsx")
    print("wrote", RUN / "error_position_raw_real_vs_generated.csv")
    print(audit.to_string(index=False))


if __name__ == "__main__":
    main()
