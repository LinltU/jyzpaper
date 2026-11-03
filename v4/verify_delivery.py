from pathlib import Path
import argparse
import hashlib
import json
import zipfile
import pandas as pd
import openpyxl

parser = argparse.ArgumentParser(description="Independent XLSX/CSV delivery verification for this dataset.")
parser.add_argument("--output", required=True, type=Path)
parser.add_argument("--source", required=True, type=Path)
args = parser.parse_args()
out = args.output
book = out / "device_nonoverlap_windows_v4.xlsx"
with zipfile.ZipFile(book) as archive:
    assert archive.testzip() is None
values = openpyxl.load_workbook(book, data_only=True)
formulas = openpyxl.load_workbook(book, data_only=False)
windows = pd.read_csv(out / "window_features_v4.csv", dtype={"device_id": str, "window_id": str})
summary = pd.read_csv(out / "device_summary_v4.csv", dtype={"device_id": str}).set_index("device_id")
points = pd.read_csv(out / "device_points_v4.csv", dtype={"device_id": str})
ws = values["窗口证据"]
rows = list(ws.values)
headers = rows[0]
assert len(rows) - 1 == len(windows) == 547
for row, (_, expected) in zip(rows[1:], windows.iterrows()):
    actual = dict(zip(headers, row))
    assert isinstance(actual["device_id"], str) and len(actual["device_id"]) == 15
    for key in ("device_id", "window_id", "n_points", "is_candidate"):
        assert actual[key] == expected[key], (key, actual[key], expected[key])
    for key in ("start", "end"):
        assert pd.Timestamp(actual[key]) == pd.Timestamp(expected[key])
assert sum(row[headers.index("is_candidate")] for row in rows[1:]) == 27
assert sum(row[headers.index("n_points")] for row in rows[1:]
           if row[headers.index("is_candidate")]) == 854
for rownum in range(12, 37):
    row = [values["设备汇总"].cell(rownum, col).value for col in range(1, 13)]
    assert isinstance(row[0], str) and len(row[0]) == 15
    expected = summary.loc[row[0]]
    for col, key in ((1, "total_points"), (2, "total_windows"), (3, "candidate_windows"),
                     (5, "candidate_points")):
        assert row[col] == expected[key]
    for col, key in ((8, "first_timestamp"), (9, "last_timestamp")):
        assert pd.Timestamp(row[col]) == pd.Timestamp(expected[key])
    assert formulas["设备汇总"].cell(rownum, 3).data_type == "f"
for cell, expected in {"B5":16985, "B6":547, "B7":27, "B8":854, "E8":2}.items():
    assert values["设备汇总"][cell].value == expected
errors = []
for sheet in values:
    for row in sheet:
        for cell in row:
            if cell.data_type == "e":
                errors.append((sheet.title, cell.coordinate, cell.value))
assert not errors
assert len(list((out / "device_plots").glob("*.png"))) == 25
source = args.source
source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
assert source_hash == "9760253f0a8be4c1fdc461a4ee4c1006e48e605bee8368af508970f55b1ae863"
result = dict(zip_integrity="passed", all_547_window_ids_labels_dates="passed",
              all_25_summary_counts_dates="passed", identifiers="15-digit strings",
              formula_caches="passed", formula_errors=errors, device_figures=25,
              source_sha256=source_hash)
(out / "delivery_check.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
print(json.dumps(result))
