"""Merge historical CSV data with the 2026-07-16 workbook extraction.

The raw inputs are never modified. Rows are de-duplicated by IMEI and timestamp,
with the latest workbook taking precedence, then split chronologically per device.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


LOCATION_MAP = {
    "863644079749421": "左边下相小号侧串1_220kV_18片_第三~四片",
    "863644079730603": "左边下相小号侧串1_220kV_18片_第九~十片",
    "863644079730652": "左边下相小号侧串1_220kV_18片_第十五~十六片",
    "863644079749355": "左边下相小号侧串2_220kV_18片_第三~四片",
    "863644079749231": "左边下相小号侧串2_220kV_18片_第九~十片",
    "863644079730728": "左边下相小号侧串2_220kV_18片_第十五~十六片",
    "863644079749413": "左边下相大号侧串1_220kV_18片_第三~四片",
    "863644079730611": "左边下相大号侧串1_220kV_18片_第九~十片",
    "863644079730645": "左边下相大号侧串1_220kV_18片_第十五~十六片",
    "863644079730637": "左边下相大号侧串2_220kV_18片_第三~四片",
    "863644079730694": "左边下相大号侧串2_220kV_18片_第九~十片",
    "863644079730660": "左边下相大号侧串2_220kV_18片_第十五~十六片",
    "863644079730629": "右边上相小号侧串1_220kV_18片_第三~四片",
    "863644079730744": "右边上相小号侧串1_220kV_18片_第九~十片",
    "863644079730736": "右边上相小号侧串1_220kV_18片_第十五~十六片",
    "863644079749454": "右边上相小号侧串2_220kV_18片_第三~四片",
    "863644079749249": "右边上相小号侧串2_220kV_18片_第九~十片",
    "863644079749405": "右边上相小号侧串2_220kV_18片_第十五~十六片",
    "863644079730686": "右边上相大号侧串1_220kV_18片_第三~四片",
    "863644079749397": "右边上相大号侧串1_220kV_18片_第九~十片",
    "863644079729449": "右边上相大号侧串1_220kV_18片_第十五~十六片",
    "863644079749439": "右边上相大号侧串2_220kV_18片_第九~十片",
    "863644079749363": "右边上相大号侧串2_220kV_18片_第十五~十六片",
    "863644079748878": "玉中线串1_110kV_9片_第三~四片",
    "863644079759867": "玉中线串1_110kV_9片_第六~七片",
}

OUTPUT_COLUMNS = [
    "IMEI",
    "Timestamp",
    "Battery",
    "Measure",
    "Temp",
    "Hum",
    "Version",
    "Location",
]


def load_csv(path, source, priority):
    frame = pd.read_csv(path, dtype={"IMEI": str})
    frame["Source"] = source
    frame["SourcePriority"] = priority
    return frame


def load_latest_json(path):
    sheets = json.loads(Path(path).read_text(encoding="utf-8"))
    frames = []
    for sheet in sheets:
        values = sheet["values"]
        if not values:
            continue
        frame = pd.DataFrame(values[1:], columns=values[0])
        frame["IMEI"] = frame["IMEI"].astype(str)
        if not frame["IMEI"].eq(str(sheet["name"])).all():
            raise ValueError("Sheet name and IMEI values disagree: {}".format(sheet["name"]))
        frames.append(frame)
    latest = pd.concat(frames, ignore_index=True)
    latest["Source"] = "latest_workbook"
    latest["SourcePriority"] = 3
    return latest


def normalize(frame):
    result = frame.copy()
    result["IMEI"] = result["IMEI"].astype(str).str.strip().str.replace(".0", "", regex=False)
    result["Timestamp"] = pd.to_datetime(
        result["Timestamp"].astype(str).str.replace("_", " ", regex=False),
        errors="coerce",
    )
    for column in ["Battery", "Measure", "Temp", "Hum"]:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    result["Version"] = result.get("Version", "").fillna("").astype(str)
    result["Location"] = result["IMEI"].map(LOCATION_MAP)
    return result


def humidity_stage(humidity):
    return pd.cut(
        humidity,
        bins=[-np.inf, 50.0, 65.0, 80.0, np.inf],
        labels=["dry", "transition", "humid", "very_humid"],
        right=False,
    )


def main():
    script_dir = Path(__file__).resolve().parent
    dist_dir = script_dir.parent
    data_dir = dist_dir / "data" / "raw"
    parser = argparse.ArgumentParser()
    parser.add_argument("--training", default=str(data_dir / "training.csv"))
    parser.add_argument("--test", default=str(data_dir / "test.csv"))
    parser.add_argument("--latest-json", default=str(data_dir / "latest_workbook_raw_20260716.json"))
    parser.add_argument("--output-dir", default=str(script_dir / "latest_retraining_data_20260716"))
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--min-test-rows", type=int, default=48)
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    sources = [
        load_csv(root / args.training, "historical_training", 1),
        load_csv(root / args.test, "historical_test", 2),
        load_latest_json(root / args.latest_json),
    ]
    normalized = [normalize(source) for source in sources]
    raw_counts = {source["Source"].iloc[0]: int(len(source)) for source in normalized}
    combined_raw = pd.concat(normalized, ignore_index=True)

    missing_required = combined_raw[
        ["IMEI", "Timestamp", "Measure", "Temp", "Hum", "Location"]
    ].isna().any(axis=1)
    invalid_range = (
        combined_raw["Hum"].lt(0)
        | combined_raw["Hum"].gt(100)
        | combined_raw["Temp"].lt(-50)
        | combined_raw["Temp"].gt(80)
        | combined_raw["Measure"].lt(0)
    )
    rejected = combined_raw[missing_required | invalid_range].copy()
    valid = combined_raw[~(missing_required | invalid_range)].copy()

    valid = valid.sort_values(
        ["IMEI", "Timestamp", "SourcePriority"],
        kind="mergesort",
    )
    duplicate_count = int(valid.duplicated(["IMEI", "Timestamp"], keep="last").sum())
    merged = valid.drop_duplicates(["IMEI", "Timestamp"], keep="last").copy()
    merged = merged.sort_values(["IMEI", "Timestamp"]).reset_index(drop=True)
    merged["HumidityStage"] = humidity_stage(merged["Hum"])

    split_frames = []
    split_audit = []
    for imei, group in merged.groupby("IMEI", sort=True):
        group = group.sort_values("Timestamp").copy()
        test_count = max(args.min_test_rows, int(np.ceil(len(group) * args.test_ratio)))
        test_count = min(test_count, max(1, len(group) - 25))
        group["Split"] = "training"
        group.iloc[-test_count:, group.columns.get_loc("Split")] = "test"
        training_group = group[group["Split"] == "training"]
        test_group = group[group["Split"] == "test"]
        split_audit.append({
            "IMEI": imei,
            "TotalRows": int(len(group)),
            "TrainingRows": int(len(training_group)),
            "TestRows": int(len(test_group)),
            "TrainingStart": training_group["Timestamp"].min().isoformat(),
            "TrainingEnd": training_group["Timestamp"].max().isoformat(),
            "TestStart": test_group["Timestamp"].min().isoformat(),
            "TestEnd": test_group["Timestamp"].max().isoformat(),
            "VeryHumidTraining": int((training_group["HumidityStage"] == "very_humid").sum()),
            "VeryHumidTest": int((test_group["HumidityStage"] == "very_humid").sum()),
        })
        split_frames.append(group)

    split_data = pd.concat(split_frames, ignore_index=True)
    training = split_data[split_data["Split"] == "training"].copy()
    test = split_data[split_data["Split"] == "test"].copy()

    if not (training.groupby("IMEI")["Timestamp"].max() < test.groupby("IMEI")["Timestamp"].min()).all():
        raise AssertionError("Temporal split leakage detected")

    training[OUTPUT_COLUMNS].to_csv(
        output_dir / "training_merged.csv",
        index=False,
        date_format="%Y-%m-%d %H:%M:%S",
    )
    test[OUTPUT_COLUMNS].to_csv(
        output_dir / "test_latest_holdout.csv",
        index=False,
        date_format="%Y-%m-%d %H:%M:%S",
    )
    split_data[OUTPUT_COLUMNS + ["Source", "HumidityStage", "Split"]].to_csv(
        output_dir / "combined_with_provenance.csv",
        index=False,
        date_format="%Y-%m-%d %H:%M:%S",
    )
    rejected.to_csv(output_dir / "rejected_rows.csv", index=False)
    pd.DataFrame(split_audit).to_csv(output_dir / "device_split_audit.csv", index=False)

    audit = {
        "input_rows": raw_counts,
        "input_total": int(len(combined_raw)),
        "rejected_rows": int(len(rejected)),
        "duplicate_imei_timestamp_rows_removed": duplicate_count,
        "merged_unique_rows": int(len(merged)),
        "training_rows": int(len(training)),
        "test_rows": int(len(test)),
        "devices": int(merged["IMEI"].nunique()),
        "time_range": {
            "start": merged["Timestamp"].min().isoformat(),
            "end": merged["Timestamp"].max().isoformat(),
        },
        "humidity_stage_counts": {
            split: {
                str(stage): int(count)
                for stage, count in frame["HumidityStage"].value_counts(sort=False).items()
            }
            for split, frame in [("training", training), ("test", test)]
        },
        "source_counts_after_dedup": {
            str(source): int(count)
            for source, count in merged["Source"].value_counts().items()
        },
        "split_protocol": {
            "method": "per-device chronological holdout",
            "test_ratio": args.test_ratio,
            "minimum_test_rows_per_device": args.min_test_rows,
            "deduplication_key": ["IMEI", "Timestamp"],
            "deduplication_precedence": [
                "latest_workbook",
                "historical_test",
                "historical_training",
            ],
        },
        "location_resolution": {
            "863644079749397": {
                "used": LOCATION_MAP["863644079749397"],
                "reason": "Existing config and historical CSV agree; supplied text duplicated a string-2 segment.",
            },
            "863644079729449": {
                "used": LOCATION_MAP["863644079729449"],
                "reason": "Existing config and historical CSV agree; supplied text duplicated a string-2 segment.",
            },
        },
    }
    (output_dir / "data_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

