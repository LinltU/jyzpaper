"""Create compact machine-readable and CSV summaries for the final run."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
RUN = ROOT / "local_runs" / "final"
OUT = ROOT / "results"


def load(name: str):
    return json.loads((RUN / name).read_text(encoding="utf-8"))


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    prediction = load("prediction_summary.json")
    imputation = load("imputation_summary.json")
    anomaly = load("anomaly_summary.json")
    smoke = load("official_smoke.json")
    protocol = load("data_protocol.json")
    manifest = load("run_manifest.json")
    manifest["prediction"] = prediction
    manifest["imputation"] = imputation
    manifest["anomaly"] = anomaly
    manifest["official_smoke"] = smoke
    write_json(RUN / "run_manifest.json", manifest)

    prediction_rows = []
    for method, result in prediction.items():
        overall = result.get("overall", {})
        prediction_rows.append(
            {
                "method": method,
                "repository": result.get("repository"),
                "rmse": overall.get("rmse"),
                "mae": overall.get("mae"),
                "r2": overall.get("r2"),
                "coverage_p10_p90": overall.get("coverage_p10_p90"),
                "mean_interval_width": overall.get("mean_interval_width"),
                "official_package_used": result.get("official_package_used"),
            }
        )
    pd.DataFrame(prediction_rows).sort_values("rmse").to_csv(
        OUT / "prediction_comparison.csv", index=False, encoding="utf-8-sig"
    )

    imputation_rows = []
    for method, result in imputation.items():
        scenarios = result.get("scenarios", {})
        imputation_rows.append(
            {
                "method": method,
                **{scenario: values.get("rmse") for scenario, values in scenarios.items()},
            }
        )
    pd.DataFrame(imputation_rows).to_csv(
        OUT / "imputation_comparison.csv", index=False, encoding="utf-8-sig"
    )

    anomaly_rows = []
    for method, result in anomaly.items():
        if not isinstance(result, dict) or "normal_false_alarm_rate" not in result:
            continue
        anomaly_rows.append(
            {
                "method": method,
                "normal_false_alarm_rate": result.get("normal_false_alarm_rate"),
                "detection_rate_3sigma": result.get("severity_curve", {}).get("3sigma", {}).get("detection_rate"),
                "detection_rate_6sigma": result.get("severity_curve", {}).get("6sigma", {}).get("detection_rate"),
                "detection_rate_10sigma": result.get("severity_curve", {}).get("10sigma", {}).get("detection_rate"),
                "roc_auc_6sigma": result.get("pseudo_anomaly_roc_auc"),
                "average_precision_6sigma": result.get("pseudo_anomaly_average_precision"),
            }
        )
    pd.DataFrame(anomaly_rows).to_csv(
        OUT / "anomaly_comparison.csv", index=False, encoding="utf-8-sig"
    )

    source_manifest = json.loads((ROOT / "sources_manifest.json").read_text(encoding="utf-8"))
    status_rows = []
    for source in source_manifest:
        repo = source["name"]
        local_groups = []
        if any(v.get("repository") == repo for v in prediction.values() if isinstance(v, dict)):
            local_groups.append("prediction")
        if any(v.get("repository") == repo for v in imputation.values() if isinstance(v, dict)):
            local_groups.append("imputation")
        if any(v.get("repository") == repo for v in anomaly.values() if isinstance(v, dict)):
            local_groups.append("anomaly")
        status_rows.append(
            {
                "repository": repo,
                "url": source["url"],
                "commit": source["commit"],
                "stars_snapshot": source.get("stars_snapshot"),
                "source_downloaded": source.get("status") == "downloaded",
                "source_compiles": smoke.get(repo, {}).get("compileall"),
                "official_import": smoke.get(repo, {}).get("import"),
                "local_reproduction_groups": "+".join(local_groups),
                "local_reproduction_complete": bool(local_groups),
            }
        )
    pd.DataFrame(status_rows).to_csv(
        OUT / "reproduction_status.csv", index=False, encoding="utf-8-sig"
    )
    write_json(
        OUT / "reproduction_results.json",
        {
            "data_protocol": protocol,
            "prediction": prediction,
            "imputation": imputation,
            "anomaly": anomaly,
            "official_smoke": smoke,
            "source_count": len(source_manifest),
        },
    )
    print(OUT)


if __name__ == "__main__":
    main()
