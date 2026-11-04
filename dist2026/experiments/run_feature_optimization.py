"""Train the battery-free feature models and build the stage ensemble."""

import argparse
import json
import subprocess
import sys
from pathlib import Path


def run(command, workdir):
    print("Running: " + " ".join(str(part) for part in command), flush=True)
    subprocess.run(command, cwd=workdir, check=True)


def main():
    script_dir = Path(__file__).resolve().parent
    dist_dir = script_dir.parent
    data_dir = dist_dir / "data" / "raw"
    parser = argparse.ArgumentParser()
    parser.add_argument("--training", default=str(data_dir / "training.csv"))
    parser.add_argument("--test", default=str(data_dir / "test.csv"))
    parser.add_argument("--output-root", default=str(script_dir / "feature_optimized_fmlp_run"))
    parser.add_argument("--reference-metrics", default=str(dist_dir / "humidity_stage_fmlp_evaluation" / "metrics.json"))
    parser.add_argument("--seq-len", type=int, default=24)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-epochs", type=int, default=180)
    args = parser.parse_args()

    workdir = Path(__file__).resolve().parent.parent
    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = workdir / output_root
    output_root.mkdir(parents=True, exist_ok=True)

    base_dir = output_root / "clean_elapsed"
    high_dir = output_root / "dynamic_high_expert"
    ensemble_dir = output_root / "ensemble"
    evaluator = [sys.executable, "-m", "experiments.evaluate_humidity_stage_fmlp"]
    ensemble = [sys.executable, "-m", "experiments.evaluate_feature_stage_ensemble"]
    training_path = Path(args.training)
    test_path = Path(args.test)
    reference_path = Path(args.reference_metrics)
    if not training_path.is_absolute():
        training_path = workdir / training_path
    if not test_path.is_absolute():
        test_path = workdir / test_path
    if not reference_path.is_absolute():
        reference_path = workdir / reference_path

    common = [
        *evaluator,
        "--training",
        str(training_path.resolve()),
        "--test",
        str(test_path.resolve()),
        "--seq-len",
        str(args.seq_len),
        "--seed",
        str(args.seed),
        "--max-epochs",
        str(args.max_epochs),
        "--battery-mode",
        "exclude",
        "--routing-mode",
        "target",
    ]

    run(
        common + [
            "--feature-set",
            "clean",
            "--output-dir",
            str(base_dir),
        ],
        workdir,
    )
    run(
        common + [
            "--feature-set",
            "v2",
            "--v2-profile",
            "dynamic",
            "--device-embedding-dim",
            "4",
            "--use-target-covariates",
            "--output-dir",
            str(high_dir),
        ],
        workdir,
    )
    run(
        [
            *ensemble,
            "--base-dir",
            str(base_dir),
            "--high-dir",
            str(high_dir),
            "--reference-metrics",
            str(reference_path.resolve()),
            "--output-dir",
            str(ensemble_dir),
        ],
        workdir,
    )

    with (ensemble_dir / "metrics.json").open("r", encoding="utf-8") as handle:
        metrics = json.load(handle)
    print(
        json.dumps(
            {
                "output_root": str(output_root),
                "ensemble_rmse": metrics["fmlp"]["rmse"],
                "ensemble_mae": metrics["fmlp"]["mae"],
                "very_humid_rmse": metrics["by_humidity_stage"]["very_humid"]["fmlp"]["rmse"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
