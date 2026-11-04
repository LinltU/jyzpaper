"""Combine clean and dynamic FMLP predictions by target humidity stage."""

import argparse
import json
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
if __package__ in (None, ""):
    import sys
    sys.path.insert(0, str(PROJECT_DIR))

import numpy as np

from experiments.evaluate_fmlp_fixed_split import regression_metrics
from experiments.evaluate_humidity_stage_fmlp import STAGE_NAMES, interval_metrics, stage_name
from core.project_paths import DIST_DIR, manifest_relative_path


def read_predictions(path):
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def prediction_key(row):
    return str(row["IMEI"]), str(row["Timestamp"])


def combine_predictions(base_rows, high_rows, humidity_threshold=80.0):
    high_by_key = {prediction_key(row): row for row in high_rows}
    if len(high_by_key) != len(high_rows):
        raise ValueError("High-humidity predictions contain duplicate keys")

    combined = []
    for base_row in base_rows:
        key = prediction_key(base_row)
        if key not in high_by_key:
            raise ValueError("High-humidity predictions are missing key: " + str(key))
        high_row = high_by_key[key]
        if not np.isclose(base_row["Actual"], high_row["Actual"], atol=1e-7):
            raise ValueError("Actual values do not match for key: " + str(key))
        if not np.isclose(base_row["Hum"], high_row["Hum"], atol=1e-7):
            raise ValueError("Humidity values do not match for key: " + str(key))

        use_high_expert = float(base_row["Hum"]) >= humidity_threshold
        selected = high_row if use_high_expert else base_row
        output = dict(base_row)
        output.update({
            "P10": float(selected["P10"]),
            "P50": float(selected["P50"]),
            "P90": float(selected["P90"]),
            "SelectedModel": "dynamic_high_expert" if use_high_expert else "clean_elapsed",
        })
        combined.append(output)

    if len(combined) != len(high_rows):
        raise ValueError("Prediction sources have different row counts")
    return combined


def metrics_from_rows(rows):
    actual = np.asarray([row["Actual"] for row in rows], dtype=float)
    quantiles = np.asarray(
        [[row["P10"], row["P50"], row["P90"]] for row in rows],
        dtype=float,
    )
    output = {
        "fmlp": regression_metrics(actual, quantiles[:, 1]),
        "interval": interval_metrics(actual, quantiles),
    }
    by_stage = {}
    for name in STAGE_NAMES:
        mask = np.asarray([stage_name(row["Hum"]) == name for row in rows])
        if not mask.any():
            continue
        by_stage[name] = {
            "fmlp": regression_metrics(actual[mask], quantiles[mask, 1]),
            "interval": interval_metrics(actual[mask], quantiles[mask]),
        }
    output["by_humidity_stage"] = by_stage
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-dir",
        default="humidity_stage_fmlp_clean_elapsed_evaluation",
    )
    parser.add_argument(
        "--high-dir",
        default="humidity_stage_fmlp_v2_high_expert_evaluation",
    )
    parser.add_argument(
        "--reference-metrics",
        default="humidity_stage_fmlp_evaluation/metrics.json",
    )
    parser.add_argument(
        "--output-dir",
        default="humidity_stage_feature_ensemble_evaluation",
    )
    parser.add_argument("--humidity-threshold", type=float, default=80.0)
    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    high_dir = Path(args.high_dir)
    if not base_dir.is_absolute():
        base_dir = DIST_DIR / base_dir
    if not high_dir.is_absolute():
        high_dir = DIST_DIR / high_dir
    base_dir = base_dir.resolve()
    high_dir = high_dir.resolve()
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = DIST_DIR / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    combined = combine_predictions(
        read_predictions(base_dir / "predictions.jsonl"),
        read_predictions(high_dir / "predictions.jsonl"),
        humidity_threshold=args.humidity_threshold,
    )
    result_metrics = metrics_from_rows(combined)
    result_metrics["protocol"] = {
        "base_model_dir": str(base_dir),
        "high_humidity_model_dir": str(high_dir),
        "humidity_threshold": args.humidity_threshold,
        "selection_rule": "clean_elapsed below threshold; dynamic_high_expert at or above threshold",
        "battery_in_model": False,
        "test_sequences": len(combined),
        "selected_counts": {
            "clean_elapsed": sum(row["SelectedModel"] == "clean_elapsed" for row in combined),
            "dynamic_high_expert": sum(
                row["SelectedModel"] == "dynamic_high_expert" for row in combined
            ),
        },
    }

    reference_path = Path(args.reference_metrics)
    if not reference_path.is_absolute():
        reference_path = DIST_DIR / reference_path
    if reference_path.exists():
        with reference_path.open("r", encoding="utf-8") as handle:
            reference = json.load(handle)
        reference_fmlp = (
            reference["overall"]["fmlp"]
            if "overall" in reference
            else reference["fmlp"]
        )
        ensemble_fmlp = result_metrics["fmlp"]
        result_metrics["improvement_over_reference"] = {
            "reference_path": str(reference_path.resolve()),
            "reference_rmse": float(reference_fmlp["rmse"]),
            "ensemble_rmse": float(ensemble_fmlp["rmse"]),
            "rmse_percent": float(
                (reference_fmlp["rmse"] - ensemble_fmlp["rmse"])
                / reference_fmlp["rmse"]
                * 100.0
            ),
            "reference_mae": float(reference_fmlp["mae"]),
            "ensemble_mae": float(ensemble_fmlp["mae"]),
            "mae_percent": float(
                (reference_fmlp["mae"] - ensemble_fmlp["mae"])
                / reference_fmlp["mae"]
                * 100.0
            ),
        }

    with (output_dir / "predictions.jsonl").open("w", encoding="utf-8") as handle:
        for row in combined:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(result_metrics, handle, ensure_ascii=False, indent=2)
    manifest_path = output_dir / "ensemble_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "base_checkpoint": manifest_relative_path(
                    base_dir / "humidity_stage_fmlp.pth", manifest_path
                ),
                "high_humidity_checkpoint": manifest_relative_path(
                    high_dir / "humidity_stage_fmlp.pth", manifest_path
                ),
                "humidity_threshold": args.humidity_threshold,
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )

    print(json.dumps(result_metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()


