"""v4.1: v4 peer logic with main-branch per-point local time tolerance.

Only one protocol item changes relative to v4 (process_device_nonoverlap_windows_v4.py):

    tolerance: one scalar per device (0.5 * global median interval)
            -> per point (0.5 * median of the centered 9 nearest intervals)

Everything else is kept byte-for-byte equivalent to v4 so that any label change
can be attributed to the tolerance rule alone.
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.impute import SimpleImputer
from sklearn.linear_model import HuberRegressor
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
for candidate in (HERE, HERE.parents[1] / "code_reproduction" / "experiment_augmentation"):
    if (candidate / "device_data_paper_augmentation.py").is_file() and str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from device_data_paper_augmentation import (  # noqa: E402
    DEVICE_LOCATIONS,
    location_group,
    location_group_intra_string,
)

WORKBOOK = Path("/home/test/linliting/jyzmentor/dataset/device_data_110_220_final.xlsx")
HUM_BINS = [0, 30, 40, 50, 55, 60, 65, 70, 75, 80, 90, 100, 101]
HUM_EDGE = 60.0
VERSION = "v41"
PEER_JUMP_Z_THR = 8.0
PEER_SINGLE_JUMP_Z_THR = 12.0
MIN_TAIL_POINTS = 2
LOCAL_INTERVAL_POINTS = 9
TOL_FLOOR_S = 1800.0
TOL_CEIL_S = 43200.0

GROUP_FUNCTIONS = {
    "cross-string": location_group,
    "intra-string": location_group_intra_string,
}


def load_points(workbook: Path, group_mode: str = "intra-string") -> pd.DataFrame:
    group_fn = GROUP_FUNCTIONS[group_mode]
    rows = []
    for sheet, frame in pd.read_excel(workbook, sheet_name=None).items():
        q = frame.copy().dropna(subset=["Timestamp", "Measure"])
        q["Timestamp"] = pd.to_datetime(q["Timestamp"])
        for c in ["Measure", "Temp", "Hum"]:
            q[c] = pd.to_numeric(q[c], errors="coerce")
        q = q.dropna(subset=["Measure"]).sort_values("Timestamp")
        q["IMEI"] = str(sheet)
        q["Location"] = q.IMEI.map(DEVICE_LOCATIONS)
        q["group_id"] = q.Location.map(group_fn)
        rows.append(q)
    return pd.concat(rows, ignore_index=True)


def _robust_scale(x: np.ndarray) -> tuple[float, float]:
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med)) * 1.4826 + 1e-9
    return med, mad


def _rank_score(s: pd.Series) -> pd.Series:
    r = s.rank(method="average", pct=True).to_numpy(float)
    r = np.clip(r, 1e-4, 1 - 1e-4)
    return pd.Series(norm.ppf(r), index=s.index)


def _local_tolerance_s(dt: pd.Series) -> pd.Series:
    """v4.1 change: centered 9-point local median interval, clipped 30min..12h.

    dt is the per-device sequence of consecutive timestamp gaps (seconds).
    The window is centered, so a point's tolerance can depend on observations
    that occur after the target timestamp. This makes v4.1 a retrospective
    analysis, exactly like the main-branch reference, not an online detector.
    """
    local = dt.rolling(LOCAL_INTERVAL_POINTS, center=True, min_periods=1).median()
    tol = (0.5 * local).clip(TOL_FLOOR_S, TOL_CEIL_S)
    fallback = 0.5 * np.nanmedian(dt.to_numpy(float)) if np.isfinite(dt.to_numpy(float)).any() else np.nan
    if not np.isfinite(fallback):
        fallback = TOL_FLOOR_S
    return tol.fillna(float(np.clip(fallback, TOL_FLOOR_S, TOL_CEIL_S)))


def add_point_features(points: pd.DataFrame) -> pd.DataFrame:
    out = []
    for imei, g in points.groupby("IMEI", sort=False):
        g = g.sort_values("Timestamp").copy()
        elapsed = (g.Timestamp - g.Timestamp.min()).dt.total_seconds().to_numpy(float) / 86400
        cov = SimpleImputer(strategy="median").fit_transform(np.c_[g.Temp, g.Hum, elapsed])
        y = g.Measure.to_numpy(float)
        scaler = StandardScaler().fit(cov)
        model = HuberRegressor(epsilon=1.5, alpha=1e-4, max_iter=300).fit(scaler.transform(cov), y)
        g["env_residual"] = y - model.predict(scaler.transform(cov))
        dt = g.Timestamp.diff().dt.total_seconds()
        g["interval_s"] = dt
        g["median_interval_s"] = dt.median()
        g["local_median_interval_s"] = dt.rolling(
            LOCAL_INTERVAL_POINTS, center=True, min_periods=1).median()
        g["peer_tolerance_s"] = _local_tolerance_s(dt)
        g["hum_gt60"] = (g.Hum > HUM_EDGE).astype(float)
        out.append(g)
    points = pd.concat(out, ignore_index=True)

    points["env_residual_z_device"] = np.nan
    for imei, idx in points.groupby("IMEI", sort=False).groups.items():
        sub = points.loc[idx]
        m, s = _robust_scale(sub.env_residual.to_numpy(float))
        points.loc[idx, "env_residual_z_device"] = (sub.env_residual.to_numpy(float) - m) / s

    points["hum_bin"] = pd.cut(points.Hum, bins=HUM_BINS, right=False)
    points["env_residual_z"] = np.nan
    for b, sub in points.groupby("hum_bin", observed=True):
        if len(sub) >= 50:
            points.loc[sub.index, "env_residual_z"] = _rank_score(sub.env_residual_z_device).to_numpy()
    mask = points.env_residual_z.isna()
    points.loc[mask, "env_residual_z"] = _rank_score(points.loc[mask, "env_residual_z_device"]).to_numpy()

    jumps = np.full(len(points), np.nan)
    for imei, idx in points.groupby("IMEI", sort=False).groups.items():
        sub = points.loc[idx].sort_values("Timestamp")
        diff = np.r_[np.nan, np.diff(sub.env_residual_z.to_numpy(float))]
        jumps[sub.index.to_numpy()] = np.abs(diff) / np.sqrt(2.0)
    points["residual_jump_z"] = jumps

    points["peer_delta"] = np.nan
    points["peer_hum"] = np.nan
    points["n_peers_matched"] = 0
    points["peer_offset_max_s"] = np.nan
    for group_id, group in points.groupby("group_id", sort=False):
        devices = sorted(group.IMEI.unique())
        if len(devices) < 2:
            continue
        for imei in devices:
            target = group[group.IMEI == imei].sort_values("Timestamp")
            own = target.Timestamp.to_numpy(dtype="datetime64[ns]").astype("int64")
            tol_ns = target.peer_tolerance_s.to_numpy(float) * 1e9
            peer_res, peer_hum, peer_off = [], [], []
            for other in devices:
                if other == imei:
                    continue
                peer = group[group.IMEI == other][["Timestamp", "env_residual", "Hum"]].sort_values("Timestamp")
                pt = peer.Timestamp.to_numpy(dtype="datetime64[ns]").astype("int64")
                insert = np.searchsorted(pt, own)
                left = np.clip(insert - 1, 0, len(pt) - 1)
                right = np.clip(insert, 0, len(pt) - 1)
                li, ri = np.abs(pt[left] - own), np.abs(pt[right] - own)
                nearest = np.where(li <= ri, left, right)
                off = pt[nearest] - own
                accepted = np.abs(off) <= tol_ns
                peer_res.append(np.where(accepted, peer.env_residual.to_numpy(float)[nearest], np.nan))
                peer_hum.append(np.where(accepted, peer.Hum.to_numpy(float)[nearest], np.nan))
                peer_off.append(np.where(accepted, np.abs(off / 1e9), np.nan))
            res_mat = np.vstack(peer_res)
            hum_mat = np.vstack(peer_hum)
            off_mat = np.vstack(peer_off)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                peer_res_med = np.nanmedian(res_mat, axis=0)
                peer_hum_med = np.nanmedian(hum_mat, axis=0)
            n_matched = np.sum(np.isfinite(res_mat), axis=0)
            ok = np.isfinite(peer_res_med)
            points.loc[target.index[ok], "peer_delta"] = (
                target.env_residual.to_numpy(float)[ok] - peer_res_med[ok]
            )
            points.loc[target.index[ok], "peer_hum"] = peer_hum_med[ok]
            points.loc[target.index[ok], "n_peers_matched"] = n_matched[ok]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                points.loc[target.index[ok], "peer_offset_max_s"] = np.nanmax(off_mat[:, ok], axis=0)

    points["n_expected_peers"] = points.groupby("group_id")["IMEI"].transform("nunique") - 1
    points["peer_complete"] = (
        points.peer_delta.notna()
        & (points.n_expected_peers >= 1)
        & (points.n_peers_matched >= points.n_expected_peers)
    )
    points["peer_delta_z"] = np.nan
    points["peer_hum_bin"] = pd.cut(points.peer_hum, bins=HUM_BINS, right=False)
    for (group_id, b), sub in points.dropna(subset=["peer_delta"]).groupby(["group_id", "peer_hum_bin"], observed=True):
        if len(sub) >= 20:
            points.loc[sub.index, "peer_delta_z"] = _rank_score(sub.peer_delta).to_numpy()
    mask = points.peer_delta_z.isna() & points.peer_delta.notna()
    points.loc[mask, "peer_delta_z"] = _rank_score(points.loc[mask, "peer_delta"]).to_numpy()

    raw_jump = np.full(len(points), np.nan)
    for imei, idx in points.groupby("IMEI", sort=False).groups.items():
        sub = points.loc[idx].dropna(subset=["peer_delta"]).sort_values("Timestamp")
        if len(sub) < 2:
            continue
        diff = np.r_[np.nan, np.diff(sub.peer_delta.to_numpy(float))]
        complete = sub.peer_complete.to_numpy(bool)
        valid = complete & np.r_[False, complete[:-1]]
        raw_jump[sub.index.to_numpy()] = np.where(valid, np.abs(diff) / np.sqrt(2.0), np.nan)
    points["peer_delta_jump_z"] = np.nan
    points["peer_jump_magnitude"] = raw_jump
    scale = {}
    for imei, grp in points.dropna(subset=["peer_delta"]).groupby("IMEI", sort=False):
        _, s = _robust_scale(grp.peer_delta.to_numpy(float))
        scale[imei] = s
    denom = points.IMEI.map(scale).to_numpy(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        points["peer_delta_jump_z"] = points.peer_jump_magnitude.to_numpy(float) / denom
    return points


def _max_run(flags: np.ndarray) -> int:
    best = run = 0
    for f in flags:
        run = run + 1 if f else 0
        best = max(best, run)
    return best


def make_windows(points: pd.DataFrame, size: int) -> pd.DataFrame:
    resid_z_thr, peer_z_thr = 3.0, 3.0
    persist_fraction, jump_z_thr = 0.15, 2.5
    peer_jump_z_thr = PEER_JUMP_Z_THR
    single_peer_jump_z_thr = PEER_SINGLE_JUMP_Z_THR
    label_col = f"window_anomaly_{VERSION}"
    rows = []

    def build_row(imei: str, w: pd.DataFrame, start: int, end: int,
                  number: int, window_type: str) -> dict:
        n = len(w)
        measure = w.Measure.to_numpy(float)
        rz = w.env_residual_z.to_numpy(float)
        pz = w.peer_delta_z.to_numpy(float)
        rjump = w.residual_jump_z.to_numpy(float)
        pjump = w.peer_delta_jump_z.to_numpy(float)
        has_peer = np.isfinite(pz).any()
        pz_valid = pz[np.isfinite(pz)]
        duration_h = (w.Timestamp.iloc[-1] - w.Timestamp.iloc[0]).total_seconds() / 3600.0
        slope = (np.polyfit(np.arange(n), measure, 1)[0] / max(duration_h, 1e-9)) if n >= 2 else np.nan
        return {
            "IMEI": imei,
            "Location": w.Location.iloc[0],
            "group_id": w.group_id.iloc[0],
            "window_id": f"{imei}_{number:04d}",
            "window_number": number,
            "window_type": window_type,
            "start_row": start,
            "end_row": end,
            "start_time": w.Timestamp.iloc[0],
            "end_time": w.Timestamp.iloc[-1],
            "n_points": n,
            "duration_hours": duration_h,
            "median_interval_hours": float(np.nanmedian(w.interval_s)) / 3600.0,
            "tolerance_hours_median": float(np.nanmedian(w.peer_tolerance_s)) / 3600.0,
            "tolerance_hours_max": float(np.nanmax(w.peer_tolerance_s)) / 3600.0,
            "measure_mean": np.mean(measure),
            "measure_std": np.std(measure),
            "measure_slope_per_hour": slope,
            "residual_z_mean": float(np.mean(rz)),
            "residual_z_abs_mean": float(np.mean(np.abs(rz))),
            "residual_z_abs_q90": float(np.quantile(np.abs(rz), 0.90)),
            "high_residual_fraction": float(np.mean(np.abs(rz) >= 3.0)),
            "abs2_residual_fraction": float(np.mean(np.abs(rz) >= 2.0)),
            "max_consecutive_high": int(_max_run(np.abs(rz) >= 2.0)),
            "residual_jump_z_max": float(np.nanmax(rjump)) if np.isfinite(rjump).any() else np.nan,
            "n_large_residual_jumps": int(np.nansum(rjump >= jump_z_thr)),
            "peer_z_mean": float(np.mean(pz_valid)) if len(pz_valid) else np.nan,
            "peer_z_abs_mean": float(np.mean(np.abs(pz_valid))) if len(pz_valid) else np.nan,
            "peer_z_frac_valid": float(len(pz_valid)) / n,
            "peer_delta_jump_z_max": float(np.nanmax(pjump)) if has_peer else np.nan,
            "n_large_peer_jumps": int(np.nansum(pjump >= peer_jump_z_thr)) if has_peer else 0,
            "has_peer": int(has_peer),
            "hum_mean": float(np.nanmean(w.Hum.to_numpy(float))),
            "hum_gt60_fraction": float(np.nanmean(w.hum_gt60.to_numpy(float))),
        }

    for imei, g in points.groupby("IMEI", sort=False):
        g = g.sort_values("Timestamp").reset_index(drop=True)
        n_windows = len(g) // size
        for number in range(n_windows):
            start = number * size
            rows.append(build_row(imei, g.iloc[start:start + size], start,
                                  start + size - 1, number, "full"))
        tail_start = n_windows * size
        if len(g) - tail_start >= MIN_TAIL_POINTS:
            rows.append(build_row(imei, g.iloc[tail_start:], tail_start, len(g) - 1,
                                  n_windows, "tail"))
    windows = pd.DataFrame(rows)

    for col, name in [("residual_z_mean", "residual_z_mean_z"), ("peer_z_mean", "peer_z_mean_z")]:
        m, s = _robust_scale(np.abs(windows[col].to_numpy(float)))
        windows[name] = (windows[col].abs() - m) / s
    same_direction = np.sign(windows.residual_z_mean) == np.sign(windows.peer_z_mean)
    resid_flag = (windows.residual_z_mean_z >= resid_z_thr) & (
        windows.abs2_residual_fraction >= persist_fraction)
    peer_flag = windows.peer_z_mean_z >= peer_z_thr
    jump_flag = windows.n_large_peer_jumps >= 2
    strong_jump_flag = windows.peer_delta_jump_z_max.ge(single_peer_jump_z_thr)
    no_peer_flag = (windows.has_peer == 0) & (windows.abs2_residual_fraction >= persist_fraction) & (
        windows.residual_z_abs_q90 >= 2.0) & (windows.n_large_residual_jumps >= 2)
    consistent_flag = resid_flag & peer_flag & same_direction
    windows[label_col] = (consistent_flag | jump_flag | strong_jump_flag | no_peer_flag).astype(int)
    windows["anomaly_reason"] = np.select(
        [windows[label_col].eq(0), consistent_flag, jump_flag, strong_jump_flag, no_peer_flag],
        ["normal", "environment residual and peer agree", "peer-delta jump (per-device)",
         "isolated extreme peer jump", "no-peer strong residual (solo device)"],
        default="normal",
    )
    windows["candidate_score"] = np.nanmax(np.column_stack([
        windows.residual_z_mean_z,
        windows.peer_z_mean_z.fillna(0.0),
        windows.n_large_peer_jumps / 2.0,
        windows.peer_delta_jump_z_max.fillna(0.0) / single_peer_jump_z_thr,
        windows.n_large_residual_jumps / 2.0,
    ]), axis=1)
    windows["candidate_rank"] = windows.candidate_score.rank(ascending=False, method="first")
    return windows


def peer_humidity_audit(points: pd.DataFrame) -> dict:
    q = points.dropna(subset=["peer_delta"])
    base = float((q.Hum > 60).mean())
    out = {"baseline_hum_gt60": base, "n_peer_points": int(len(q))}
    for col, key in [("peer_delta", "raw_peer_delta"), ("peer_delta_z", "stratified_peer_z")]:
        thr = q[col].abs().quantile(0.95)
        top = q[q[col].abs() >= thr]
        out[f"{key}_top5_hum_gt60"] = float((top.Hum > 60).mean())
    return out


def plot_labels(points: pd.DataFrame, windows: pd.DataFrame, out: Path, mode: str) -> None:
    devices = list(windows.IMEI.drop_duplicates())
    label_col = f"window_anomaly_{VERSION}"
    ncols = 5
    nrows = int(np.ceil(len(devices) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(23, 3.3 * nrows), squeeze=False)
    axes = axes.ravel()
    for ax, imei in zip(axes, devices):
        p = points[points.IMEI == imei].sort_values("Timestamp")
        g = windows[windows.IMEI == imei].sort_values("window_number")
        flagged = set(g.loc[g[label_col].eq(1), "window_number"].astype(int))
        pw = np.full(len(p), -1, dtype=int)
        for _, row in g.iterrows():
            pw[int(row.start_row):int(row.end_row) + 1] = int(row.window_number)
        inv = np.isin(pw, list(flagged))
        ax.plot(p.Timestamp, p.Measure, color="#94a3b8", lw=.6, alpha=.6)
        ax.scatter(p.loc[~inv, "Timestamp"], p.loc[~inv, "Measure"], color="#93c5fd", s=4, alpha=.5, label="normal")
        ax.scatter(p.loc[inv, "Timestamp"], p.loc[inv, "Measure"], color="#dc2626", s=13, alpha=.9, label=f"{VERSION} candidate")
        for _, row in g.iterrows():
            if int(row[label_col]) == 1:
                ax.axvspan(row.start_time, row.end_time, color="#fecaca", alpha=.35, lw=0)
        ax.set_title(f"{imei}\n{VERSION} candidates={int(g[label_col].sum())}/{len(g)}", fontsize=8)
        ax.grid(alpha=.25)
        ax.set_ylabel("Measure", fontsize=7)
        ax.tick_params(axis="x", labelrotation=30, labelsize=6)
        ax.tick_params(axis="y", labelsize=7)
    for ax in axes[len(devices):]:
        ax.axis("off")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(.5, 1.005))
    fig.suptitle(f"{VERSION} candidate anomaly windows by device [{mode} peer group]", fontsize=15, y=1.035)
    fig.text(.5, .01, f"{VERSION} = v4 rules with a per-point centered {LOCAL_INTERVAL_POINTS}-interval "
                      "local median tolerance. The centered window uses later observations, so this is "
                      "retrospective, not an online detector.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .03, 1, .98))
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main(args) -> None:
    label_col = f"window_anomaly_{VERSION}"
    args.output.mkdir(parents=True, exist_ok=True)
    points = add_point_features(load_points(args.workbook, args.peer_group_mode))
    windows = make_windows(points, args.window_size)
    points.to_csv(args.output / f"device_points_{VERSION}.csv", index=False, encoding="utf-8-sig")
    windows.to_csv(args.output / f"device_nonoverlap_windows_{VERSION}.csv", index=False, encoding="utf-8-sig")
    plot_labels(points, windows, args.output / f"{VERSION}_anomaly_labels_by_device.png", args.peer_group_mode)

    with pd.ExcelWriter(args.output / f"device_nonoverlap_windows_{VERSION}.xlsx", engine="openpyxl") as writer:
        windows.to_excel(writer, sheet_name="window_features", index=False)
        windows.groupby(["IMEI", "Location", "group_id"], as_index=False).agg(
            windows=("window_id", "size"),
            anomalies=(label_col, "sum"),
            anomaly_rate=(label_col, "mean"),
        ).to_excel(writer, sheet_name="summary", index=False)

    protocol = {
        "version": VERSION,
        "change_vs_v4": "peer tolerance: per-device scalar -> per-point centered local median interval",
        "base": "v4 (process_device_nonoverlap_windows_v4.py) unchanged otherwise",
        "source": str(args.workbook),
        "devices": int(windows.IMEI.nunique()),
        "peer_group_mode": args.peer_group_mode,
        "n_peer_groups": int(windows.group_id.nunique()),
        "window_size": args.window_size,
        "windows": int(len(windows)),
        "windows_by_type": windows.window_type.value_counts().to_dict(),
        f"{VERSION}_candidate_windows": int(windows[label_col].sum()),
        f"{VERSION}_candidate_rate": float(windows[label_col].mean()),
        "tolerance": {
            "formula": "clip(0.5 * median(dt[k-4..k+4]), 1800s, 43200s)",
            "rolling_points": LOCAL_INTERVAL_POINTS,
            "centered": True,
            "floor_s": TOL_FLOOR_S,
            "ceil_s": TOL_CEIL_S,
            "uses_future_observations": True,
            "online_causal": False,
        },
        "thresholds": {"window_evidence_robust_z": 3.0, "persist_fraction_abs2": 0.15,
                       "min_large_peer_jumps": 2, "residual_jump_point_z": 2.5,
                       "peer_jump_point_z": PEER_JUMP_Z_THR,
                       "isolated_peer_jump_point_z": PEER_SINGLE_JUMP_Z_THR,
                       "min_tail_points": MIN_TAIL_POINTS},
        "humidity_audit": peer_humidity_audit(points),
    }
    (args.output / f"protocol_{VERSION}.json").write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(protocol, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workbook", type=Path, default=WORKBOOK)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--window-size", type=int, default=32)
    parser.add_argument("--peer-group-mode", choices=sorted(GROUP_FUNCTIONS), default="intra-string")
    main(parser.parse_args())
