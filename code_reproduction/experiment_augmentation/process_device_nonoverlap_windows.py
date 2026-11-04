"""Extract large non-overlapping device windows and label anomalous windows."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import HuberRegressor
from sklearn.preprocessing import StandardScaler

from device_data_paper_augmentation import DEVICE_LOCATIONS, location_group


WORKBOOK = Path("/home/test/linliting/jyzmentor/device_data_110_220_final.xlsx")


def load_points(workbook: Path) -> pd.DataFrame:
    rows = []
    for sheet, frame in pd.read_excel(workbook, sheet_name=None).items():
        q = frame.copy().dropna(subset=["Timestamp", "Measure"])
        q["Timestamp"] = pd.to_datetime(q["Timestamp"])
        q["Measure"] = pd.to_numeric(q["Measure"], errors="coerce")
        q["Temp"] = pd.to_numeric(q["Temp"], errors="coerce")
        q["Hum"] = pd.to_numeric(q["Hum"], errors="coerce")
        q = q.dropna(subset=["Measure"]).sort_values("Timestamp")
        q["IMEI"] = str(sheet)
        q["Location"] = q.IMEI.map(DEVICE_LOCATIONS)
        q["group_id"] = q.Location.map(location_group)
        rows.append(q)
    return pd.concat(rows, ignore_index=True)


def add_point_residuals(points: pd.DataFrame) -> pd.DataFrame:
    out = []
    for imei, g in points.groupby("IMEI", sort=False):
        g = g.sort_values("Timestamp").copy()
        elapsed = (g.Timestamp - g.Timestamp.min()).dt.total_seconds().to_numpy(float) / 86400
        cov = SimpleImputer(strategy="median").fit_transform(np.c_[g.Temp, g.Hum, elapsed])
        y = g.Measure.to_numpy(float)
        scaler = StandardScaler().fit(cov)
        model = HuberRegressor(epsilon=1.5, alpha=1e-4, max_iter=300).fit(scaler.transform(cov), y)
        residual = y - model.predict(scaler.transform(cov))
        mad = np.median(np.abs(residual - np.median(residual))) * 1.4826 + 1e-9
        diff = np.r_[0, np.diff(y)]
        dmad = np.median(np.abs(diff - np.median(diff))) * 1.4826 + 1e-9
        g["env_residual"] = residual
        g["residual_z"] = (residual - np.median(residual)) / mad
        g["delta_measure"] = diff
        g["jump_z"] = np.abs(diff - np.median(diff)) / dmad
        out.append(g)
    return pd.concat(out, ignore_index=True)


def robust_z(values: pd.Series) -> pd.Series:
    x = values.to_numpy(float)
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med)) * 1.4826 + 1e-9
    return pd.Series((x - med) / mad, index=values.index)


def make_windows(points: pd.DataFrame, size: int, feature_z: float = 5.0,
                 point_fraction: float = 0.15, peer_z: float = 3.5) -> pd.DataFrame:
    rows = []
    for imei, g in points.groupby("IMEI", sort=False):
        g = g.sort_values("Timestamp").reset_index(drop=True)
        n_windows = len(g) // size
        for number in range(n_windows):
            start = number * size
            w = g.iloc[start:start + size]
            measure = w.Measure.to_numpy(float)
            temp = w.Temp.to_numpy(float)
            hum = w.Hum.to_numpy(float)
            residual = w.env_residual.to_numpy(float)
            rows.append({
                "IMEI": imei,
                "Location": w.Location.iloc[0],
                "group_id": w.group_id.iloc[0],
                "window_id": f"{imei}_{number:04d}",
                "window_number": number,
                "start_row": start,
                "end_row": start + size - 1,
                "start_time": w.Timestamp.iloc[0],
                "end_time": w.Timestamp.iloc[-1],
                "n_points": len(w),
                "measure_mean": np.mean(measure),
                "measure_median": np.median(measure),
                "measure_min": np.min(measure),
                "measure_max": np.max(measure),
                "measure_std": np.std(measure),
                "measure_range": np.ptp(measure),
                "measure_slope": np.polyfit(np.arange(size), measure, 1)[0],
                "temp_mean": np.nanmean(temp),
                "temp_std": np.nanstd(temp),
                "temp_range": np.nanmax(temp) - np.nanmin(temp),
                "hum_mean": np.nanmean(hum),
                "hum_std": np.nanstd(hum),
                "hum_range": np.nanmax(hum) - np.nanmin(hum),
                "residual_mean": np.mean(residual),
                "residual_std": np.std(residual),
                "residual_abs_max": np.max(np.abs(residual)),
                "residual_abs_q95": np.quantile(np.abs(residual), 0.95),
                "residual_z_abs_max": np.max(np.abs(w.residual_z)),
                "point_high_residual_fraction": np.mean(np.abs(w.residual_z) >= 3.5),
                "jump_z_max": np.max(w.jump_z),
                "measure_nonpositive": int((measure <= 0).sum()),
            })
    windows = pd.DataFrame(rows)
    # Score each window relative to that device's own window distribution.
    # Fluctuation features describe shape/volatility rather than absolute level,
    # so a device with a persistently high or low Measure is not flagged just
    # for its level. Every device is nevertheless guaranteed at least one
    # candidate stage (see guaranteed_min_window below).
    fluctuation_features = [
        "measure_std", "measure_range", "measure_slope",
        "residual_std", "residual_abs_q95",
        "point_high_residual_fraction", "jump_z_max",
    ]
    context_features = ["temp_std", "hum_std", "temp_range", "hum_range"]
    for feature in fluctuation_features + context_features:
        windows[f"{feature}_z"] = windows.groupby("IMEI", group_keys=False)[feature].transform(robust_z).abs()
    windows["fluctuation_score"] = windows[[f"{f}_z" for f in fluctuation_features]].max(axis=1)
    windows["context_score"] = windows[[f"{f}_z" for f in context_features]].max(axis=1)
    windows["feature_anomaly_score"] = np.maximum(windows["fluctuation_score"], windows["context_score"])
    windows["window_anomaly_individual"] = (
        (windows["point_high_residual_fraction"] >= point_fraction)
        | (windows["fluctuation_score"] >= feature_z)
    ).astype(int)
    windows["peer_delta_mean"] = np.nan
    windows["peer_delta_z"] = np.nan
    for group_id, group in windows.groupby("group_id", sort=False):
        if group.IMEI.nunique() < 2:
            continue
        for imei in group.IMEI.unique():
            target = group[group.IMEI == imei].sort_values("start_time")
            peer = group[group.IMEI != imei].sort_values("start_time")
            match = pd.merge_asof(target, peer[["start_time", "residual_mean"]].rename(columns={"residual_mean": "peer_residual_mean"}), on="start_time", direction="nearest", tolerance=pd.Timedelta("30min"))
            valid = match.peer_residual_mean.notna()
            delta = match.residual_mean - match.peer_residual_mean
            windows.loc[target.index[valid], "peer_delta_mean"] = delta[valid].to_numpy()
        valid = windows.loc[group.index, "peer_delta_mean"].notna()
        if valid.sum() >= 4:
            z = robust_z(windows.loc[group.index[valid], "peer_delta_mean"])
            windows.loc[group.index[valid], "peer_delta_z"] = z.to_numpy()
    windows["window_anomaly_peer"] = windows.peer_delta_z.abs().ge(peer_z).fillna(False).astype(int)
    windows["window_anomaly"] = ((windows.window_anomaly_individual == 1) | (windows.window_anomaly_peer == 1)).astype(int)
    # Per-device guarantee and cap. Each device keeps at least one candidate
    # stage (its largest-fluctuation window) so no unit is silently all-normal;
    # the cap prevents a single device from being almost entirely flagged.
    windows["guaranteed_min_window"] = 0
    windows["capped_window"] = 0
    for imei, group in windows.groupby("IMEI", sort=False):
        if group.window_anomaly.sum() == 0:
            best = group.fluctuation_score.idxmax()
            windows.loc[best, "window_anomaly"] = 1
            windows.loc[best, "guaranteed_min_window"] = 1
        cap = max(1, int(np.ceil(len(group) * 0.5)))
        flagged = group.index[windows.loc[group.index, "window_anomaly"] == 1]
        if len(flagged) > cap:
            order = windows.loc[flagged, "fluctuation_score"].sort_values(ascending=False).index
            drop = order[cap:]
            windows.loc[drop, "window_anomaly"] = 0
            windows.loc[drop, "capped_window"] = 1
    windows["anomaly_reason"] = np.select(
        [
            windows.window_anomaly.eq(0),
            windows.guaranteed_min_window.eq(1),
        ],
        ["normal", "largest within-device fluctuation stage"],
        default="measure/environment fluctuation or corresponding-position peer deviation",
    )
    return windows


def plot_windows(points: pd.DataFrame, windows: pd.DataFrame, out: Path, size: int) -> None:
    devices = list(windows.IMEI.drop_duplicates())
    ncols = 5
    nrows = int(np.ceil(len(devices) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(22, 3.1 * nrows), squeeze=False)
    axes = axes.ravel()
    for ax, imei in zip(axes, devices):
        p = points[points.IMEI == imei].sort_values("Timestamp")
        g = windows[windows.IMEI == imei].sort_values("window_number")
        point_window = np.full(len(p), -1, dtype=int)
        for _, row in g.iterrows():
            point_window[int(row.start_row):int(row.end_row) + 1] = int(row.window_number)
        anomaly_windows = set(g.loc[g.window_anomaly.eq(1), "window_number"].astype(int))
        in_anomaly = np.isin(point_window, list(anomaly_windows))
        ax.plot(p.Timestamp, p.Measure, color="#64748b", lw=.7, alpha=.7)
        ax.scatter(p.loc[~in_anomaly, "Timestamp"], p.loc[~in_anomaly, "Measure"], color="#2563eb", s=5, alpha=.55, label="normal window")
        ax.scatter(p.loc[in_anomaly, "Timestamp"], p.loc[in_anomaly, "Measure"], color="#dc2626", s=12, alpha=.9, label="candidate anomaly window")
        # Light spans make the non-overlapping window segments visible without
        # hiding the raw Measure line.
        for _, row in g.iterrows():
            if int(row.window_anomaly) == 1:
                ax.axvspan(row.start_time, row.end_time, color="#fecaca", alpha=.35, lw=0)
            ax.axvline(row.start_time, color="#cbd5e1", lw=.4, alpha=.5)
        ax.set_title(f"{imei}\nanomaly windows={int(g.window_anomaly.sum())}/{len(g)}", fontsize=8)
        ax.grid(alpha=.25)
        ax.set_ylabel("Measure", fontsize=7)
        ax.tick_params(axis="x", labelrotation=30, labelsize=6)
        ax.tick_params(axis="y", labelsize=7)
    for ax in axes[len(devices):]: ax.axis("off")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(.5, 1.005))
    fig.suptitle(f"Pseudo anomaly window labels by device ({size}-point non-overlapping windows)", fontsize=15, y=1.035)
    fig.text(.5, .005, "Windows are consecutive non-overlapping blocks. A window is a candidate anomaly if its within-device fluctuation score (Measure std/range/slope, residual std/Q95, high-residual fraction, jump) is large, or it departs from the corresponding serial-1/serial-2 peer. Every device is guaranteed at least its largest-fluctuation stage, capped at half its windows.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .02, 1, .98))
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main(args) -> None:
    args.output.mkdir(parents=True, exist_ok=True)
    points = add_point_residuals(load_points(args.workbook))
    windows = make_windows(points, args.window_size, args.feature_z, args.point_fraction, args.peer_z)
    points.to_csv(args.output / "device_points_with_residuals.csv", index=False, encoding="utf-8-sig")
    windows.to_csv(args.output / "device_nonoverlap_windows.csv", index=False, encoding="utf-8-sig")
    with pd.ExcelWriter(args.output / "device_nonoverlap_windows.xlsx", engine="openpyxl") as writer:
        windows.to_excel(writer, sheet_name="window_features", index=False)
        windows.groupby(["IMEI", "Location", "group_id"], as_index=False).agg(
            windows=("window_id", "size"), anomalies=("window_anomaly", "sum"), anomaly_rate=("window_anomaly", "mean"), points=("n_points", "sum")
        ).to_excel(writer, sheet_name="summary", index=False)
    plot_windows(points, windows, args.output / "nonoverlap_window_anomaly_labels_by_device.png", args.window_size)
    protocol = {"source": str(args.workbook), "devices": int(windows.IMEI.nunique()), "window_size": args.window_size, "overlap": 0, "window_rule": "floor(len(device)/window_size), chronological contiguous blocks", "label_features": ["Measure std/range/slope", "environment-adjusted residual std/Q95", "high-residual point fraction", "jump z", "Temp/Hum volatility", "corresponding-position peer residual"], "candidate_label": f"within-device fluctuation robust z >= {args.feature_z} OR high-residual point fraction >= {args.point_fraction} OR peer window residual robust z >= {args.peer_z}; plus every device is guaranteed its single largest-fluctuation window and capped at half its windows", "total_points": int(len(points)), "total_windows": int(len(windows)), "devices_with_anomaly": int((windows.groupby("IMEI").window_anomaly.sum() > 0).sum()), "candidate_anomaly_windows": int(windows.window_anomaly.sum())}
    (args.output / "protocol.json").write_text(json.dumps(protocol, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(protocol, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workbook", type=Path, default=WORKBOOK)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--window-size", type=int, default=64)
    parser.add_argument("--feature-z", type=float, default=5.0, help="within-device window feature robust-z threshold")
    parser.add_argument("--point-fraction", type=float, default=0.15, help="fraction of high-residual points that flags a window")
    parser.add_argument("--peer-z", type=float, default=3.5, help="corresponding-position peer window robust-z threshold")
    main(parser.parse_args())
