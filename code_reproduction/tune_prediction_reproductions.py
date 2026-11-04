"""Validation-driven tuning for all prediction paper adapters.

Each trial trains one paper mechanism on the fixed fitting split, evaluates
the untouched validation tail and writes a separate test result. The test
result is reported only after the best validation configuration is selected.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
RUNNER = ROOT / "run_reproductions.py"
PYTHON = ROOT / "runtime" / "python.exe"
METHODS = ["TimeGrad", "DiffPLF", "CCDM", "TSDE", "NsDiff", "Diffusion-TS", "Treeffuser"]

# CPU-friendly search space. The first row reproduces the previous official
# local protocol; the remaining rows vary capacity, diffusion steps and LR.
CONFIGS = [
    {"name": "baseline", "epochs": 40, "hidden_dim": 128, "diffusion_steps": 20, "samples": 32, "learning_rate": 8e-4, "tree_estimators": 80, "tree_repeats": 2},
    {"name": "low_lr_long", "epochs": 60, "hidden_dim": 128, "diffusion_steps": 30, "samples": 32, "learning_rate": 5e-4, "tree_estimators": 120, "tree_repeats": 3},
    {"name": "compact_fast", "epochs": 60, "hidden_dim": 96, "diffusion_steps": 16, "samples": 32, "learning_rate": 5e-4, "tree_estimators": 120, "tree_repeats": 3},
    {"name": "wide", "epochs": 60, "hidden_dim": 160, "diffusion_steps": 24, "samples": 32, "learning_rate": 5e-4, "tree_estimators": 120, "tree_repeats": 3},
]


def run_trial(output: Path, method: str, cfg: dict, seed: int) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(PYTHON if PYTHON.exists() else sys.executable),
        str(RUNNER),
        "--output", str(output),
        "--only", "prediction",
        "--methods", method,
        "--epochs", str(cfg["epochs"]),
        "--hidden-dim", str(cfg["hidden_dim"]),
        "--diffusion-steps", str(cfg["diffusion_steps"]),
        "--samples", str(cfg["samples"]),
        "--learning-rate", str(cfg["learning_rate"]),
        "--tree-estimators", str(cfg["tree_estimators"]),
        "--tree-repeats", str(cfg["tree_repeats"]),
        "--seed", str(seed),
        "--skip-smoke",
    ]
    completed = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True)
    log = output / "stdout_stderr.log"
    log.write_text(completed.stdout + "\n" + completed.stderr, encoding="utf-8")
    metrics_path = output / "prediction" / method / "metrics.json"
    if completed.returncode != 0 or not metrics_path.exists():
        return {"method": method, "config": cfg["name"], "status": "failed", "returncode": completed.returncode}
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    validation = metrics.get("validation_before_test_calibration", {})
    overall = metrics.get("overall", {})
    return {
        "method": method,
        "config": cfg["name"],
        "status": "ok",
        "validation_rmse": validation.get("rmse"),
        "validation_mae": validation.get("mae"),
        "validation_coverage_p10_p90": validation.get("coverage_p10_p90"),
        "test_rmse": overall.get("rmse"),
        "test_mae": overall.get("mae"),
        "test_r2": overall.get("r2"),
        "test_coverage_p10_p90": overall.get("coverage_p10_p90"),
        "test_mean_interval_width": overall.get("mean_interval_width"),
        "output": str(output),
        **cfg,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "local_runs" / "tuning_20260905")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--methods", default=", ".join(METHODS))
    args = parser.parse_args()
    methods = [item.strip() for item in args.methods.split(",") if item.strip()]
    rows = []
    for method in methods:
        for cfg in CONFIGS:
            trial = args.output / method / cfg["name"]
            print(f"[tune] {method} / {cfg['name']}", flush=True)
            rows.append(run_trial(trial, method, cfg, args.seed))
    frame = pd.DataFrame(rows)
    frame.to_csv(args.output / "all_trials.csv", index=False, encoding="utf-8-sig")
    valid = frame[frame["status"] == "ok"].copy()
    if not valid.empty:
        # Select on validation RMSE first, then validation MAE. No test metric
        # participates in model selection.
        valid = valid.sort_values(["method", "validation_rmse", "validation_mae"])
        best = valid.groupby("method", as_index=False).first()
        best.to_csv(args.output / "best_by_validation.csv", index=False, encoding="utf-8-sig")
        (args.output / "best_by_validation.json").write_text(
            json.dumps(best.to_dict(orient="records"), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    print(f"Completed {len(rows)} trials under {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
