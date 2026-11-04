"""Projected-only table for target-date environment conditioned validation."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from openpyxl.styles import Alignment, Font, PatternFill

HERE = Path(__file__).resolve().parent
RUN = HERE / "runs" / "target_date_log_residual_strict_20260916"
DATES = ("250605", "250626", "250718")
COLS = ["source", "time", "Measure4", "Measure7", "Temp4", "Hum4", "Temp7", "Hum7", "error", "condition_source"]


def main():
    df = pd.read_csv(RUN / "real_vs_env_conditioned_generated.csv", dtype={"date": str})
    df = df[(df.source == "real") | ((df.source == "EnvCSDI") & (df["mode"] == "projected"))].copy()
    df["source"] = df["source"].replace({"EnvCSDI": "generated"})
    df = df[COLS + ["date"]]
    nums = ["Measure4", "Measure7", "Temp4", "Hum4", "Temp7", "Hum7"]
    df[nums] = df[nums].round(4)
    out_xlsx = RUN / "best_target_date_real_vs_generated_projected.xlsx"
    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as xl:
        for date in DATES:
            part = df[df.date == date].drop(columns=["date"])
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
        pd.DataFrame({
            "item": ["mode", "condition", "normal projection", "target"],
            "value": [
                "projected only; raw diffusion samples are excluded from this inspection table",
                "generated rows use the same target date's Temp/Hum/elapsed trajectory",
                "normal generated rows are clipped to the target date's stable-normal range with 5% margin",
                "CSDI generates log(Measure / environment-regression-prediction), then reconstructs Measure",
            ],
        }).to_excel(xl, sheet_name="notes", index=False)
    out_csv = RUN / "best_target_date_real_vs_generated_projected.csv"
    df[["date"] + COLS].to_csv(out_csv, index=False, encoding="utf-8-sig")
    print("wrote", out_xlsx)
    print("wrote", out_csv)


if __name__ == "__main__":
    main()
