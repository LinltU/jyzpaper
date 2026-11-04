"""Replay an online ensemble manifest over a chronological CSV holdout."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.evaluate_feature_stage_ensemble import metrics_from_rows
from production.humidity_stage_ensemble_predictor import HumidityStageEnsemblePredictor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--training", required=True)
    parser.add_argument("--test", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    predictor = HumidityStageEnsemblePredictor(args.manifest, device=args.device)

    training = pd.read_csv(args.training, dtype={"IMEI": str})
    test = pd.read_csv(args.test, dtype={"IMEI": str})
    training["_split"] = "training"
    test["_split"] = "test"
    combined = pd.concat([training, test], ignore_index=True)
    combined["Timestamp"] = pd.to_datetime(combined["Timestamp"])
    combined = combined.sort_values(["IMEI", "Timestamp"]).reset_index(drop=True)

    rows = []
    failures = []
    for imei, group in combined.groupby("IMEI", sort=True):
        group = group.sort_values("Timestamp").reset_index(drop=True)
        test_indices = group.index[group["_split"].eq("test")].tolist()
        for index in test_indices:
            target = group.iloc[index]
            history = group.iloc[
                max(0, index - predictor.required_history_records):index + 1
            ].sort_values("Timestamp", ascending=False)
            records = [
                (
                    row.Measure,
                    row.Temp,
                    row.Hum,
                    row.Timestamp,
                    row.Battery,
                    row.Version,
                )
                for row in history.itertuples()
            ]
            try:
                result = predictor.predict_from_records(
                    imei=imei,
                    location=target["Location"],
                    records=records,
                    target_temp=target["Temp"],
                    target_hum=target["Hum"],
                    target_timestamp=target["Timestamp"],
                    actual_measure=target["Measure"],
                )
                rows.append({
                    "IMEI": str(imei),
                    "Timestamp": pd.Timestamp(target["Timestamp"]).isoformat(),
                    "Temp": float(target["Temp"]),
                    "Hum": float(target["Hum"]),
                    "Actual": float(target["Measure"]),
                    "P10": float(result["P10"]),
                    "P50": float(result["P50"]),
                    "P90": float(result["P90"]),
                    "HumidityStage": result["HumidityStage"],
                    "SelectedModel": result["SelectedModel"],
                    "Persistence": float(result["Persistence"]),
                    "ForecastDeltaHours": float(result["ForecastDeltaHours"]),
                })
            except Exception as error:
                failures.append({
                    "IMEI": str(imei),
                    "Timestamp": pd.Timestamp(target["Timestamp"]).isoformat(),
                    "Error": str(error),
                })

    if not rows:
        raise ValueError("No holdout rows could be evaluated")
    metrics = metrics_from_rows(rows)
    actual = np.asarray([row["Actual"] for row in rows])
    persistence = np.asarray([row["Persistence"] for row in rows])
    metrics["persistence"] = {
        "rmse": float(np.sqrt(np.mean((actual - persistence) ** 2))),
        "mae": float(np.mean(np.abs(actual - persistence))),
    }
    metrics["protocol"] = {
        "manifest": str(Path(args.manifest).resolve()),
        "training": str(Path(args.training).resolve()),
        "test": str(Path(args.test).resolve()),
        "requested_test_rows": int(len(test)),
        "evaluated_rows": int(len(rows)),
        "failed_rows": int(len(failures)),
        "battery_in_model": False,
        "selected_counts": {
            name: int(sum(row["SelectedModel"] == name for row in rows))
            for name in ("clean_elapsed", "dynamic_high_expert")
        },
    }

    with (output_dir / "predictions.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    (output_dir / "failures.json").write_text(
        json.dumps(failures, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()


