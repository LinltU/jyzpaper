"""Exploratory analysis of device_data_110_220_final.xlsx data characteristics.

Focus:
  1. Sampling interval evolution over time (dense early, sparse later).
  2. Measure vs temperature/humidity, especially humidity > 60.
  3. Within-string peer correlation (serial-1 vs serial-2 at same position).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from device_data_paper_augmentation import DEVICE_LOCATIONS, location_group

WORKBOOK = Path("/home/test/linliting/jyzmentor/device_data_110_220_final.xlsx")


def load_points(workbook: Path) -> pd.DataFrame:
    rows = []
    for sheet, frame in pd.read_excel(workbook, sheet_name=None).items():
        q = frame.copy().dropna(subset=["Timestamp", "Measure"])
        q["Timestamp"] = pd.to_datetime(q["Timestamp"])
        for c in ["Measure", "Temp", "Hum"]:
            q[c] = pd.to_numeric(q[c], errors="coerce")
        q = q.dropna(subset=["Measure"]).sort_values("Timestamp")
        q["IMEI"] = str(sheet)
        q["Location"] = q.IMEI.map(DEVICE_LOCATIONS)
        q["group_id"] = q.Location.map(location_group)
        rows.append(q)
    return pd.concat(rows, ignore_index=True)


def interval_analysis(points: pd.DataFrame) -> pd.DataFrame:
    out = []
    for imei, g in points.groupby("IMEI", sort=False):
        g = g.sort_values("Timestamp")
        dt = g.Timestamp.diff().dt.total_seconds().dropna().to_numpy(float)
        span = (g.Timestamp.max() - g.Timestamp.min()).total_seconds()
        elapsed = (g.Timestamp - g.Timestamp.min()).dt.total_seconds().to_numpy(float)
        # split into 3 equal time thirds and report median interval in each
        third = elapsed / span if span > 0 else np.zeros_like(elapsed)
        meds = {}
        for i, name in enumerate(["early", "mid", "late"]):
            m = (third >= i / 3) & (third < (i + 1) / 3)
            vals = g.Timestamp[m].diff().dt.total_seconds().dropna().to_numpy(float)
            meds[name] = float(np.median(vals)) if len(vals) else np.nan
        out.append({
            "IMEI": imei,
            "Location": g.Location.iloc[0],
            "group_id": g.group_id.iloc[0],
            "n": len(g),
            "span_hours": span / 3600.0,
            "dt_median_s": float(np.median(dt)) if len(dt) else np.nan,
            "dt_p10_s": float(np.percentile(dt, 10)) if len(dt) else np.nan,
            "dt_p90_s": float(np.percentile(dt, 90)) if len(dt) else np.nan,
            "dt_max_s": float(np.max(dt)) if len(dt) else np.nan,
            "early_median_s": meds["early"],
            "mid_median_s": meds["mid"],
            "late_median_s": meds["late"],
        })
    return pd.DataFrame(out)


def humidity_analysis(points: pd.DataFrame) -> dict:
    p = points.dropna(subset=["Measure", "Hum", "Temp"]).copy()
    # global correlations
    corr_hum = float(np.corrcoef(p.Hum, p.Measure)[0, 1])
    corr_temp = float(np.corrcoef(p.Temp, p.Measure)[0, 1])
    # above/below humidity 60
    low = p[p.Hum <= 60].Measure
    high = p[p.Hum > 60].Measure
    # per-device correlation between Measure and Hum
    per_dev = []
    for imei, g in p.groupby("IMEI"):
        if g.Hum.nunique() < 5 or g.Measure.nunique() < 5:
            continue
        per_dev.append({
            "IMEI": imei,
            "n": len(g),
            "corr_measure_hum": float(np.corrcoef(g.Hum, g.Measure)[0, 1]),
            "corr_measure_temp": float(np.corrcoef(g.Temp, g.Measure)[0, 1]),
            "mean_hum": float(g.Hum.mean()),
            "frac_hum_gt60": float((g.Hum > 60).mean()),
        })
    # binned relationship
    bins = [0, 30, 40, 50, 55, 60, 65, 70, 75, 80, 90, 100]
    p["hum_bin"] = pd.cut(p.Hum, bins=bins, right=False)
    binned = p.groupby("hum_bin", observed=True).agg(
        n=("Measure", "size"),
        measure_mean=("Measure", "mean"),
        measure_p95=("Measure", lambda s: float(np.percentile(s, 95))),
        measure_max=("Measure", "max"),
    ).reset_index()
    binned["hum_bin"] = binned.hum_bin.astype(str)
    return {
        "corr_measure_hum_global": corr_hum,
        "corr_measure_temp_global": corr_temp,
        "measure_mean_hum_le60": float(low.mean()),
        "measure_p95_hum_le60": float(np.percentile(low, 95)),
        "measure_mean_hum_gt60": float(high.mean()),
        "measure_p95_hum_gt60": float(np.percentile(high, 95)),
        "n_hum_le60": int(len(low)),
        "n_hum_gt60": int(len(high)),
        "per_device": per_dev,
        "binned": binned.to_dict("records"),
    }


def peer_analysis(points: pd.DataFrame) -> pd.DataFrame:
    out = []
    for group_id, group in points.groupby("group_id", sort=False):
        devices = sorted(group.IMEI.unique())
        if len(devices) < 2:
            continue
        # nearest-timestamp merge between each pair in the group
        for i in range(len(devices)):
            for j in range(i + 1, len(devices)):
                a = group[group.IMEI == devices[i]][["Timestamp", "Measure", "Hum", "Temp"]].sort_values("Timestamp")
                b = group[group.IMEI == devices[j]][["Timestamp", "Measure", "Hum", "Temp"]].sort_values("Timestamp")
                merged = pd.merge_asof(
                    a.rename(columns={"Measure": "A"}),
                    b.rename(columns={"Measure": "B"}),
                    on="Timestamp", direction="nearest", tolerance=pd.Timedelta("10min"),
                ).dropna(subset=["A", "B"])
                if len(merged) < 10:
                    continue
                both = merged.dropna(subset=["Hum_x", "Hum_y"] if "Hum_x" in merged else [])
                out.append({
                    "group_id": group_id,
                    "dev_a": devices[i],
                    "dev_b": devices[j],
                    "n_matched": len(merged),
                    "corr_measure": float(np.corrcoef(merged.A, merged.B)[0, 1]),
                    "corr_hum": float(np.corrcoef(merged.Hum_x, merged.Hum_y)[0, 1]) if len(both) else np.nan,
                    "mean_abs_diff": float(np.mean(np.abs(merged.A - merged.B))),
                    "std_ratio": float(merged.B.std() / (merged.A.std() + 1e-9)),
                })
    return pd.DataFrame(out)


def main(args):
    args.output.mkdir(parents=True, exist_ok=True)
    points = load_points(args.workbook)
    intervals = interval_analysis(points)
    hum = humidity_analysis(points)
    peers = peer_analysis(points)

    intervals.to_csv(args.output / "analysis_sampling_interval.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(hum["per_device"]).to_csv(args.output / "analysis_measure_humidity_per_device.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(hum["binned"]).to_csv(args.output / "analysis_measure_humidity_binned.csv", index=False, encoding="utf-8-sig")
    peers.to_csv(args.output / "analysis_peer_correlation.csv", index=False, encoding="utf-8-sig")

    summary = {
        "n_devices": int(points.IMEI.nunique()),
        "n_points": int(len(points)),
        "interval": {
            "dt_median_s_overall": float(intervals.dt_median_s.median()),
            "early_median_s_overall": float(intervals.early_median_s.median()),
            "mid_median_s_overall": float(intervals.mid_median_s.median()),
            "late_median_s_overall": float(intervals.late_median_s.median()),
            "devices_with_growing_interval": int((intervals.late_median_s > 2 * intervals.early_median_s).sum()),
        },
        "humidity": {k: v for k, v in hum.items() if k not in ("per_device", "binned")},
        "peer": {
            "n_pairs": int(len(peers)),
            "corr_measure_median": float(peers.corr_measure.median()) if len(peers) else None,
            "corr_measure_min": float(peers.corr_measure.min()) if len(peers) else None,
            "corr_hum_median": float(peers.corr_hum.median()) if len(peers) else None,
        },
    }
    (args.output / "analysis_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workbook", type=Path, default=WORKBOOK)
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args())
