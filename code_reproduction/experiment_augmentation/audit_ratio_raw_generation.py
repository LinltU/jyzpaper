"""Audit raw samples saved by ratio_all_papers_benchmark.py.

The ratio benchmark generates residual/relationship features, not temperature or
humidity. This audit therefore checks measurement positivity, source-date ranges,
and consistency among the generated derived features without clipping values.
"""
from pathlib import Path
import argparse
import json
import re

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
RUN = HERE / "runs" / "ratio_benchmark_20260918" / "final_test_train"
FEATURES = ["M4_residual", "M7_residual", "M4-M7", "M4/M7", "log(M4/M7)"]
LENGTH = 4


def safe_float(value):
    return float(value) if np.isfinite(value) else np.nan


def corr(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 2 or np.std(a) == 0 or np.std(b) == 0:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def markdown_table(frame):
    columns = list(frame.columns)
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join(["---"] * len(columns)) + "|"]
    for _, row in frame.iterrows():
        values = []
        for column in columns:
            value = row[column]
            values.append(f"{value:.6g}" if isinstance(value, (float, np.floating)) else str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def main(run=RUN):
    global RUN
    RUN = Path(run)
    meta = pd.DataFrame(json.loads((RUN / "metadata.json").read_text(encoding="utf-8")))
    # Rebuild real windows using the benchmark's exact feature construction.
    import importlib.util
    spec = importlib.util.spec_from_file_location("ratio_benchmark", HERE / "ratio_all_papers_benchmark.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    real_x, real_meta, _ = module.build_windows()
    sheets = pd.read_excel(module.WORKBOOK, sheet_name=None)
    env_by_date_row = {}
    for date in sorted({s[:6] for s in sheets}):
        sheet4 = next(s for s in sheets if s[:6] == date and "(4)" in s)
        frame = sheets[sheet4].copy()
        frame["dt"] = pd.to_datetime(frame.time.astype(str).str.replace("_", " ", regex=False))
        frame = frame.sort_values("dt").reset_index(drop=True)
        env_by_date_row[date] = {
            int(i + 2): (float(row.Temp), float(row.Hum))
            for i, row in frame.iterrows()
        }
    real_temp = np.asarray([
        np.mean([env_by_date_row[row.date][int(r)][0] for r in row.rows4])
        for _, row in real_meta.iterrows()
    ])
    real_hum = np.asarray([
        np.mean([env_by_date_row[row.date][int(r)][1] for r in row.rows4])
        for _, row in real_meta.iterrows()
    ])
    real_m4_by_date = {}
    real_m7_by_date = {}
    for date, group in real_meta.groupby("date"):
        idx = group.index.to_numpy()
        real_m4_by_date[date] = (real_x[idx, :, 0] + group.base4.to_numpy(float)[:, None]).ravel()
        real_m7_by_date[date] = (real_x[idx, :, 1] + group.base7.to_numpy(float)[:, None]).ravel()

    audits = []
    previews = []
    files = sorted(RUN.glob("raw_*.npz"))
    for path in files:
        match = re.match(r"raw_(2to8|3to7|4to6|5to5)_(.+)\.npz$", path.name)
        if not match:
            continue
        ratio, method = match.groups()
        payload = np.load(path)
        windows = payload["windows"].astype(float)
        labels = payload["labels"].astype(int)
        source_ids = payload["source_ids"].astype(int)
        source = real_meta.iloc[source_ids].reset_index(drop=True)
        source_temp = real_temp[source_ids]
        source_hum = real_hum[source_ids]
        base4 = source.base4.to_numpy(float)[:, None]
        base7 = source.base7.to_numpy(float)[:, None]
        m4 = windows[:, :, 0] + base4
        m7 = windows[:, :, 1] + base7
        finite = np.isfinite(windows).all(axis=(1, 2))
        nonpositive_m4 = m4 <= 0
        nonpositive_m7 = m7 <= 0
        source_m4_min = []
        source_m4_max = []
        source_m7_min = []
        source_m7_max = []
        outside_m4 = np.zeros_like(m4, dtype=bool)
        outside_m7 = np.zeros_like(m7, dtype=bool)
        for i, date in enumerate(source.date):
            r4, r7 = real_m4_by_date[date], real_m7_by_date[date]
            lo4, hi4, lo7, hi7 = r4.min(), r4.max(), r7.min(), r7.max()
            source_m4_min.append(lo4); source_m4_max.append(hi4)
            source_m7_min.append(lo7); source_m7_max.append(hi7)
            outside_m4[i] = (m4[i] < lo4) | (m4[i] > hi4)
            outside_m7[i] = (m7[i] < lo7) | (m7[i] > hi7)
        expected_diff = m4 - m7
        expected_ratio = (m4 + 1e-4) / (m7 + 1e-4)
        expected_log = np.log(np.maximum(expected_ratio, 1e-12))
        relation_errors = {
            "diff_mae": np.nanmean(np.abs(windows[:, :, 2] - expected_diff)),
            "ratio_mae": np.nanmean(np.abs(windows[:, :, 3] - expected_ratio)),
            "logratio_mae": np.nanmean(np.abs(windows[:, :, 4] - expected_log)),
        }
        generated_mean_m4 = m4.mean(axis=1)
        source_real_mean_m4 = (real_x[source_ids, :, 0] + source.base4.to_numpy(float)[:, None]).mean(axis=1)
        audits.append({
            "ratio": ratio,
            "method": method,
            "n_windows": len(windows),
            "n_values": int(windows.shape[0] * windows.shape[1]),
            "nonfinite_window_rate": float((~finite).mean()),
            "nonpositive_m4_value_rate": float(nonpositive_m4.mean()),
            "nonpositive_m7_value_rate": float(nonpositive_m7.mean()),
            "window_any_nonpositive_rate": float((nonpositive_m4 | nonpositive_m7).any(axis=1).mean()),
            "outside_source_date_m4_value_rate": float(outside_m4.mean()),
            "outside_source_date_m7_value_rate": float(outside_m7.mean()),
            "outside_source_date_any_value_rate": float((outside_m4 | outside_m7).mean()),
            "relation_diff_mae": safe_float(relation_errors["diff_mae"]),
            "relation_ratio_mae": safe_float(relation_errors["ratio_mae"]),
            "relation_logratio_mae": safe_float(relation_errors["logratio_mae"]),
            "generated_m4_temp_corr": safe_float(corr(generated_mean_m4, source_temp)),
            "generated_m4_hum_corr": safe_float(corr(generated_mean_m4, source_hum)),
            "source_real_m4_temp_corr": safe_float(corr(source_real_mean_m4, source_temp)),
            "source_real_m4_hum_corr": safe_float(corr(source_real_mean_m4, source_hum)),
            "generated_environment_conditioned": False,
        })
        for sample_i in range(min(5, len(windows))):
            for step in range(LENGTH):
                real_source = real_x[source_ids[sample_i], step]
                previews.append({
                    "ratio": ratio,
                    "method": method,
                    "label": int(labels[sample_i]),
                    "source_id": int(source_ids[sample_i]),
                    "source_date": source.date.iloc[sample_i],
                    "step": step,
                    "M4_generated": m4[sample_i, step],
                    "M7_generated": m7[sample_i, step],
                    "M4_source_real": real_source[0] + source.base4.iloc[sample_i],
                    "M7_source_real": real_source[1] + source.base7.iloc[sample_i],
                    "M4_residual_generated": windows[sample_i, step, 0],
                    "M7_residual_generated": windows[sample_i, step, 1],
                    "M4_minus_M7_feature": windows[sample_i, step, 2],
                    "M4_over_M7_feature": windows[sample_i, step, 3],
                    "log_M4_over_M7_feature": windows[sample_i, step, 4],
                })

    audit_df = pd.DataFrame(audits).sort_values(["ratio", "method"])
    preview_df = pd.DataFrame(previews)
    audit_df.to_csv(RUN / "raw_generation_physics_audit.csv", index=False, encoding="utf-8-sig")
    preview_df.to_csv(RUN / "raw_generation_preview.csv", index=False, encoding="utf-8-sig")
    with pd.ExcelWriter(RUN / "raw_generation_audit.xlsx", engine="openpyxl") as writer:
        audit_df.to_excel(writer, sheet_name="physics_audit", index=False)
        preview_df.to_excel(writer, sheet_name="raw_preview", index=False)

    lines = [
        "# Ratio raw-generation audit",
        "",
        "- This audit uses the saved raw generated windows and applies no clipping or deletion.",
        "- M4/M7 are reconstructed as residual plus the source window's date baseline.",
        "- Range checks compare each generated sample with all real windows from its source date.",
        "- Temperature and humidity are attached from the source window for an association check; they are not generated or input as model conditions.",
        "",
        "## Audit table",
        "",
        markdown_table(audit_df),
        "",
        "## Interpretation",
        "",
        "- Non-positive rates describe the raw generator output; they are not repaired values.",
        "- Feature relation errors indicate whether generated M4/M7 agree with the separately generated difference/ratio channels.",
        "- Environment correlations compare generated M4 means against the source window's temperature/humidity, and are not evidence that the generator learned a causal environment condition.",
        "- Error labels are retained as class labels only. They do not prove a specific nine-insulator fault position was physically generated.",
    ]
    (RUN / "RAW_GENERATION_AUDIT.md").write_text("\n".join(lines), encoding="utf-8")
    print("wrote", RUN / "raw_generation_physics_audit.csv")
    print("wrote", RUN / "raw_generation_preview.csv")
    print("wrote", RUN / "raw_generation_audit.xlsx")
    print("wrote", RUN / "RAW_GENERATION_AUDIT.md")
    print(audit_df.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, default=RUN)
    main(parser.parse_args().run)
