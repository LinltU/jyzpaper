"""Time/humidity-conditioned device window anomaly labeling (v3).

Final version of the corrected principle in
runs/device_data_110_220_characteristics_20260924/DEVICE_DATA_ANOMALY_WINDOW_PRINCIPLE.md.
Unlike v2 this script only produces v3 labels (no v1/v2 comparison columns) and
only plots v3 candidates.

  1. Keep the 64-point non-overlapping windows for comparability with v1, but
     record each window's real duration (duration_hours, n_points, median
     interval) and normalise rate features by real time.
  2. Per-device environment baseline Measure ~ f(Hum, Temp, elapsed) via Huber,
     then humidity-stratified (rank -> normal score) standardisation of the
     residual.
  3. Peer comparison on the environment-corrected residual, with adaptive time
     tolerance and within-(group, humidity-bin) normal-score standardisation.
     `--peer-group-mode intra-string` (default) treats the several devices
     mounted on the same physical string (same 串号, any 片位) as peers and uses
     the per-timestamp median across the matched peers as the reference.
  4. A window is a candidate only if the environment residual and the peer
     deviation agree in direction (and persist), or if there is an unsynchronised
     peer jump (>=2 large jumps, or a single isolated extreme jump), or (solo
     device) a strong residual with jumps. Any leftover tail points are kept in
     a final partial window so no point is dropped.
  5. No per-device guarantee / cap: output a confidence ranking instead.
"""
from __future__ import annotations

import argparse
import json
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

from device_data_paper_augmentation import (
    DEVICE_LOCATIONS,
    location_group,
    location_group_intra_string,
)

WORKBOOK = Path("/home/test/linliting/jyzmentor/device_data_110_220_final.xlsx")
HUM_BINS = [0, 30, 40, 50, 55, 60, 65, 70, 75, 80, 90, 100, 101]
HUM_EDGE = 60.0
VERSION = "v3"
# the peer-delta jump z is now in units of the unit's own robust peer-delta
# scale, so the threshold is expressed in robust standard deviations. 8.0 is
# calibrated on this dataset to flag ~0.4% of jump points (for a Gaussian
# reference this is ~5 robust sigma), which keeps the window-level rule (>=2
# such points) selective while still catching the ~0.2-0.4 visible jumps.
PEER_JUMP_Z_THR = 8.0
# an isolated endpoint spike (e.g. a unit that drops out of the fleet and its
# last sample jumps) can never reach ">=2 large jumps within one window". A
# single, far larger unsynchronised jump is therefore flagged on its own.
PEER_SINGLE_JUMP_Z_THR = 12.0
# 64-point non-overlapping windows leave a tail that enters no window; the tail
# is now emitted as a final (partial) window so no point is dropped.
MIN_TAIL_POINTS = 2

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
    """Within-bin rank mapped to a standard-normal score.

    A single median/MAD standardisation equalises the scale of a distribution
    but not the shape of its tail, so extreme values still cluster in the
    high-humidity bins (which have heavier tails). Mapping the within-bin rank
    to a normal score removes the location, the scale and the tail-shape
    dependence on humidity, which is what we need for the confounding audit.
    """
    r = s.rank(method="average", pct=True).to_numpy(float)
    r = np.clip(r, 1e-4, 1 - 1e-4)
    return pd.Series(norm.ppf(r), index=s.index)


def add_point_features(points: pd.DataFrame) -> pd.DataFrame:
    """Environment baseline, humidity-stratified residual z, adaptive peer z."""
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
        # adaptive peer tolerance: half the local median interval, clipped 30min..12h
        tol = float(np.clip(0.5 * (dt.median() if np.isfinite(dt.median()) else 1800.0), 1800.0, 43200.0))
        g["peer_tolerance_s"] = tol
        g["hum_gt60"] = (g.Hum > HUM_EDGE).astype(float)
        out.append(g)
    points = pd.concat(out, ignore_index=True)

    # environment baseline residual, then per-device robust standardisation so
    # that residuals are on a comparable scale across units.
    points["env_residual_z_device"] = np.nan
    for imei, idx in points.groupby("IMEI", sort=False).groups.items():
        sub = points.loc[idx]
        m, s = _robust_scale(sub.env_residual.to_numpy(float))
        points.loc[idx, "env_residual_z_device"] = (sub.env_residual.to_numpy(float) - m) / s

    # humidity-stratified normal-score of the device-standardised residual.
    points["hum_bin"] = pd.cut(points.Hum, bins=HUM_BINS, right=False)
    points["env_residual_z"] = np.nan
    for b, sub in points.groupby("hum_bin", observed=True):
        if len(sub) >= 50:
            points.loc[sub.index, "env_residual_z"] = _rank_score(sub.env_residual_z_device).to_numpy()
    mask = points.env_residual_z.isna()
    points.loc[mask, "env_residual_z"] = _rank_score(points.loc[mask, "env_residual_z_device"]).to_numpy()

    # jump on the standardised residual: step size relative to unit noise.
    # For independent standardised values a first difference has std sqrt(2),
    # so dividing by sqrt(2) keeps the jump z on a common, transferable scale.
    jumps = np.full(len(points), np.nan)
    for imei, idx in points.groupby("IMEI", sort=False).groups.items():
        sub = points.loc[idx].sort_values("Timestamp")
        diff = np.r_[np.nan, np.diff(sub.env_residual_z.to_numpy(float))]
        jumps[sub.index.to_numpy()] = np.abs(diff) / np.sqrt(2.0)
    points["residual_jump_z"] = jumps

    # peer comparison on the environment residual, humidity-stratified.
    # Every other device in the same group is matched by nearest time; the peer
    # reference is then the per-timestamp median across those matched peers.
    points["peer_delta"] = np.nan
    points["peer_hum"] = np.nan
    points["n_peers_matched"] = 0
    for group_id, group in points.groupby("group_id", sort=False):
        devices = sorted(group.IMEI.unique())
        if len(devices) < 2:
            continue
        for imei in devices:
            target = group[group.IMEI == imei].sort_values("Timestamp")
            tol = pd.Timedelta(seconds=float(target.peer_tolerance_s.iloc[0]))
            peer_res, peer_hum = [], []
            for other in devices:
                if other == imei:
                    continue
                peer = group[group.IMEI == other][["Timestamp", "env_residual", "Hum"]].sort_values("Timestamp")
                match = pd.merge_asof(
                    target[["Timestamp"]],
                    peer.rename(columns={"env_residual": "peer_residual", "Hum": "peer_hum"}),
                    on="Timestamp", direction="nearest", tolerance=tol,
                )
                peer_res.append(match.peer_residual.to_numpy(float))
                peer_hum.append(match.peer_hum.to_numpy(float))
            res_mat = np.vstack(peer_res)
            hum_mat = np.vstack(peer_hum)
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
    # a step in the peer delta is only trustworthy when the *full* peer panel was
    # matched at that sample; otherwise a change in which peers were matched
    # would itself look like a jump.
    points["n_expected_peers"] = points.groupby("group_id")["IMEI"].transform("nunique") - 1
    points["peer_complete"] = (
        points.peer_delta.notna()
        & (points.n_expected_peers >= 1)
        & (points.n_peers_matched >= points.n_expected_peers)
    )
    # standardise peer delta within (group, humidity bin) with the normal-score
    # transform, so the peer ranking no longer depends on humidity.
    points["peer_delta_z"] = np.nan
    points["peer_hum_bin"] = pd.cut(points.peer_hum, bins=HUM_BINS, right=False)
    for (group_id, b), sub in points.dropna(subset=["peer_delta"]).groupby(["group_id", "peer_hum_bin"], observed=True):
        if len(sub) >= 20:
            points.loc[sub.index, "peer_delta_z"] = _rank_score(sub.peer_delta).to_numpy()
    mask = points.peer_delta_z.isna() & points.peer_delta.notna()
    points.loc[mask, "peer_delta_z"] = _rank_score(points.loc[mask, "peer_delta"]).to_numpy()
    # unsynchronised jump: the *per-device* time jump of the peer delta, i.e.
    # how much this unit's deviation from its peers changes from one sample of
    # that unit to the next. A per-device first difference is required: pooling
    # all units of a group and sorting by time would mostly compare two
    # different units sampled at the same instant (their timestamps nearly
    # coincide), turning a between-unit level difference into a spurious
    # "jump". The step is kept only when the full peer panel was matched at
    # both endpoints, so a change in the matched peer set cannot masquerade as
    # a jump either.
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
    # express the jump in units of each unit's own robustly-estimated peer-delta
    # scale (median/MAD of peer_delta). Using the scale of the level instead of
    # the (heavily zero-inflated) scale of the differences keeps a fixed
    # threshold meaningful: a pooled rank score made arbitrarily small
    # differences look extreme whenever a (group, humidity) bin was very flat.
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
            "measure_mean": np.mean(measure),
            "measure_std": np.std(measure),
            "measure_slope_per_hour": slope,
            # environment-corrected evidence
            "residual_z_mean": float(np.mean(rz)),
            "residual_z_abs_mean": float(np.mean(np.abs(rz))),
            "residual_z_abs_q90": float(np.quantile(np.abs(rz), 0.90)),
            "high_residual_fraction": float(np.mean(np.abs(rz) >= 3.0)),
            "abs2_residual_fraction": float(np.mean(np.abs(rz) >= 2.0)),
            "max_consecutive_high": int(_max_run(np.abs(rz) >= 2.0)),
            "residual_jump_z_max": float(np.nanmax(rjump)) if np.isfinite(rjump).any() else np.nan,
            "n_large_residual_jumps": int(np.nansum(rjump >= jump_z_thr)),
            # peer evidence (humidity stratified, on residual)
            "peer_z_mean": float(np.mean(pz_valid)) if len(pz_valid) else np.nan,
            "peer_z_abs_mean": float(np.mean(np.abs(pz_valid))) if len(pz_valid) else np.nan,
            "peer_z_frac_valid": float(len(pz_valid)) / n,
            "peer_delta_jump_z_max": float(np.nanmax(pjump)) if has_peer else np.nan,
            "n_large_peer_jumps": int(np.nansum(pjump >= peer_jump_z_thr)) if has_peer else 0,
            "has_peer": int(has_peer),
            # humidity context (for the confounding audit, not for the label)
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

    # window-level evidence is scored across the pooled window population, so a
    # candidate means "unusual relative to the whole dataset", not "unusual for
    # this unit" (the v1 per-device guarantee is deliberately dropped).
    for col, name in [("residual_z_mean", "residual_z_mean_z"), ("peer_z_mean", "peer_z_mean_z")]:
        m, s = _robust_scale(np.abs(windows[col].to_numpy(float)))
        windows[name] = (windows[col].abs() - m) / s
    resid_z_thr, peer_z_thr = 3.0, 3.0
    persist_fraction, min_peer_jumps = 0.15, 2
    same_direction = np.sign(windows.residual_z_mean) == np.sign(windows.peer_z_mean)
    resid_flag = (windows.residual_z_mean_z >= resid_z_thr) & (
        windows.abs2_residual_fraction >= persist_fraction)
    peer_flag = windows.peer_z_mean_z >= peer_z_thr
    jump_flag = windows.n_large_peer_jumps >= min_peer_jumps
    # one single but far larger unsynchronised jump (e.g. an endpoint spike that
    # no window otherwise contains) is a candidate on its own.
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
    # confidence ranking (no guarantee, no cap)
    windows["candidate_score"] = np.nanmax(np.column_stack([
        windows.residual_z_mean_z,
        windows.peer_z_mean_z.fillna(0.0),
        windows.n_large_peer_jumps / min_peer_jumps,
        windows.peer_delta_jump_z_max.fillna(0.0) / single_peer_jump_z_thr,
        windows.n_large_residual_jumps / 2.0,
    ]), axis=1)
    windows["candidate_rank"] = windows.candidate_score.rank(ascending=False, method="first")
    return windows


def _window_point_mask(points: pd.DataFrame, windows: pd.DataFrame, imei: str, flag: bool):
    p = points[points.IMEI == imei].sort_values("Timestamp")
    g = windows[windows.IMEI == imei].sort_values("window_number")
    pw = np.full(len(p), -1, dtype=int)
    for _, row in g.iterrows():
        pw[int(row.start_row):int(row.end_row) + 1] = int(row.window_number)
    label_col = f"window_anomaly_{VERSION}"
    flagged = set(g.loc[g[label_col].eq(1), "window_number"].astype(int))
    return p, g, np.isin(pw, list(flagged)) if flag else None, flagged


def plot_v3_labels(points: pd.DataFrame, windows: pd.DataFrame, out: Path, mode: str) -> None:
    """Pure v3 label plot: raw Measure with the v3 candidate windows marked."""
    devices = list(windows.IMEI.drop_duplicates())
    label_col = f"window_anomaly_{VERSION}"
    ncols = 5
    nrows = int(np.ceil(len(devices) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(23, 3.3 * nrows), squeeze=False)
    axes = axes.ravel()
    for ax, imei in zip(axes, devices):
        p, g, inv3, _ = _window_point_mask(points, windows, imei, True)
        ax.plot(p.Timestamp, p.Measure, color="#94a3b8", lw=.6, alpha=.6)
        ax.scatter(p.loc[~inv3, "Timestamp"], p.loc[~inv3, "Measure"], color="#93c5fd", s=4, alpha=.5, label="normal")
        ax.scatter(p.loc[inv3, "Timestamp"], p.loc[inv3, "Measure"], color="#dc2626", s=13, alpha=.9, label="v3 candidate")
        for _, row in g.iterrows():
            if int(row[label_col]) == 1:
                ax.axvspan(row.start_time, row.end_time, color="#fecaca", alpha=.35, lw=0)
        n_cand = int(g[label_col].sum())
        ax.set_title(f"{imei}\nv3 candidates={n_cand}/{len(g)}", fontsize=8)
        ax.grid(alpha=.25)
        ax.set_ylabel("Measure", fontsize=7)
        ax.tick_params(axis="x", labelrotation=30, labelsize=6)
        ax.tick_params(axis="y", labelsize=7)
    for ax in axes[len(devices):]:
        ax.axis("off")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(.5, 1.005))
    fig.suptitle(f"{VERSION} candidate anomaly windows by device [{mode} peer group]", fontsize=15, y=1.035)
    fig.text(.5, .01, f"{VERSION} keeps the 64-point non-overlapping windows (a trailing partial window carries any leftover tail points) and adds a tail window. A window is a candidate when the environment residual and the humidity-stratified peer deviation agree in direction with persistence, or when there are >=2 unsynchronised peer jumps, or a single isolated extreme peer jump, or a strong solo-device residual. No per-device guarantee/cap.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .03, 1, .98))
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)


def peer_humidity_audit(points: pd.DataFrame) -> dict:
    """Point-level check that humidity stratification removes the confounding.

    Rank points by the raw peer difference (v1 style) and by the humidity
    stratified peer z (v3); if the confounding is gone, the top-5% points under
    the v3 ranking should have the same Hum>60 share as the baseline.
    """
    q = points.dropna(subset=["peer_delta"])
    base = float((q.Hum > 60).mean())
    out = {"baseline_hum_gt60": base, "n_peer_points": int(len(q))}
    for col, key in [("peer_delta", "raw_peer_delta"), ("peer_delta_z", "stratified_peer_z")]:
        thr = q[col].abs().quantile(0.95)
        top = q[q[col].abs() >= thr]
        out[f"{key}_top5_hum_gt60"] = float((top.Hum > 60).mean())
    return out


def plot_v3_audit(windows: pd.DataFrame, point_audit: dict, out: Path) -> dict:
    label_col = f"window_anomaly_{VERSION}"
    v3 = windows[label_col].eq(1)
    base_hum = float(windows.hum_gt60_fraction.mean())
    v3_hum = float(windows.loc[v3, "hum_gt60_fraction"].mean()) if v3.any() else np.nan

    fig, axes = plt.subplots(1, 3, figsize=(19, 5))

    reasons = windows.anomaly_reason.value_counts()
    order = ["normal"] + [r for r in reasons.index if r != "normal"]
    vals = [int(reasons.get(r, 0)) for r in order]
    colors = ["#cbd5e1", "#f87171", "#dc2626", "#b45309", "#7c3aed"][:len(order)]
    axes[0].bar(range(len(order)), vals, color=colors)
    axes[0].set_xticks(range(len(order)))
    axes[0].set_xticklabels([r if len(r) <= 22 else r[:20] + ".." for r in order], rotation=20, ha="right", fontsize=8)
    for i, v in enumerate(vals):
        axes[0].text(i, v + 2, str(v), ha="center", fontsize=9)
    axes[0].set_title(f"{VERSION} window labels by reason")
    axes[0].set_ylabel("windows")

    bars = {
        "baseline\n(all peer points)": 100 * point_audit["baseline_hum_gt60"],
        "raw peer\ndifference": 100 * point_audit["raw_peer_delta_top5_hum_gt60"],
        f"{VERSION} stratified\npeer z": 100 * point_audit["stratified_peer_z_top5_hum_gt60"],
    }
    axes[1].bar(list(bars), list(bars.values()), color=["#94a3b8", "#7c3aed", "#dc2626"])
    for i, b in enumerate(bars.values()):
        axes[1].text(i, b + 1, f"{b:.1f}%", ha="center", fontsize=10)
    axes[1].set_title("Hum>60 share of the top-5% peer-difference points\n(lower / closer to baseline is better)")
    axes[1].set_ylabel("% of points with Hum>60")
    axes[1].set_ylim(0, 105)
    axes[1].tick_params(axis="x", labelsize=8)

    axes[2].scatter(windows.loc[~v3, "hum_gt60_fraction"], windows.loc[~v3, "candidate_score"],
                    s=12, color="#cbd5e1", label="normal")
    axes[2].scatter(windows.loc[v3, "hum_gt60_fraction"], windows.loc[v3, "candidate_score"],
                    s=24, color="#dc2626", label=f"{VERSION} candidate")
    axes[2].axhline(1.0, color="#0f172a", ls="--", lw=.8)
    axes[2].set_xlabel("window Hum>60 fraction")
    axes[2].set_ylabel("candidate score")
    axes[2].set_title(f"{VERSION} candidate score vs humidity context")
    axes[2].legend(fontsize=8)
    axes[2].grid(alpha=.25)

    fig.suptitle(f"{VERSION} anomaly-label audit", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, .95))
    fig.savefig(out, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return {
        "window_base_hum_gt60": base_hum,
        "window_v3_hum_gt60": v3_hum,
        "labels_by_reason": {r: int(windows.anomaly_reason.eq(r).sum()) for r in order},
        "point_audit": point_audit,
    }


def main(args) -> None:
    label_col = f"window_anomaly_{VERSION}"
    args.output.mkdir(parents=True, exist_ok=True)
    points = add_point_features(load_points(args.workbook, args.peer_group_mode))
    windows = make_windows(points, args.window_size)
    points.to_csv(args.output / f"device_points_{VERSION}.csv", index=False, encoding="utf-8-sig")
    windows.to_csv(args.output / f"device_nonoverlap_windows_{VERSION}.csv", index=False, encoding="utf-8-sig")

    plot_v3_labels(points, windows, args.output / f"{VERSION}_anomaly_labels_by_device.png",
                   args.peer_group_mode)
    point_audit = peer_humidity_audit(points)
    audit = plot_v3_audit(windows, point_audit, args.output / f"{VERSION}_anomaly_audit.png")

    with pd.ExcelWriter(args.output / f"device_nonoverlap_windows_{VERSION}.xlsx", engine="openpyxl") as writer:
        windows.to_excel(writer, sheet_name="window_features", index=False)
        windows.groupby(["IMEI", "Location", "group_id"], as_index=False).agg(
            windows=("window_id", "size"),
            anomalies_v3=(label_col, "sum"),
            anomaly_rate_v3=(label_col, "mean"),
        ).to_excel(writer, sheet_name="summary", index=False)

    protocol = {
        "source": str(args.workbook), "devices": int(windows.IMEI.nunique()),
        "peer_group_mode": args.peer_group_mode,
        "n_peer_groups": int(windows.group_id.nunique()),
        "window_size": args.window_size, "windows": int(len(windows)),
        "windows_by_type": windows.window_type.value_counts().to_dict(),
        "v3_candidate_windows": int(windows[label_col].sum()),
        "v3_candidate_rate": float(windows[label_col].mean()),
        "rule_v3": "env residual z and humidity-stratified peer z agree in sign with persistence, OR >=2 per-device peer-delta jumps (only where the full peer panel is matched), OR a single isolated extreme peer jump, OR strong solo-device residual; trailing tail points are kept in a final partial window; no per-device guarantee/cap",
        "thresholds": {"window_evidence_robust_z": 3.0, "persist_fraction_abs2": 0.15,
                       "min_large_peer_jumps": 2, "residual_jump_point_z": 2.5,
                       "peer_jump_point_z": PEER_JUMP_Z_THR,
                       "isolated_peer_jump_point_z": PEER_SINGLE_JUMP_Z_THR,
                       "min_tail_points": MIN_TAIL_POINTS},
        "humidity_audit": audit,
    }
    (args.output / f"protocol_{VERSION}.json").write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(protocol, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workbook", type=Path, default=WORKBOOK)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--window-size", type=int, default=64)
    parser.add_argument("--peer-group-mode", choices=sorted(GROUP_FUNCTIONS), default="intra-string",
                        help="intra-string (default) keeps the serial number so devices on the same string "
                             "are peers; cross-string matches 串1/串2 at the same 片位")
    main(parser.parse_args())
