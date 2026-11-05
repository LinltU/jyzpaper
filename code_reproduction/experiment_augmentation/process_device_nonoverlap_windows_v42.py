"""v4.2: v4 labeling with an LTSF NLinear environment baseline.

Only the environment baseline changes relative to v4:

    v4   : HuberRegressor(Measure ~ Temp + Hum + elapsed), fitted on full history
    v4.2 : LTSF NLinear over a causal history window + current Temp/Hum conditions,
           also fitted on full history (same retrospective protocol as v4)

Everything else (humidity bins, rank-normal scoring, peer matching with the
v4.1 per-point local tolerance, A/B/C/D thresholds, 32-point windows) is
unchanged so any label change is attributable to the baseline alone.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import norm

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for candidate in (HERE, ROOT / "code_reproduction" / "experiment_augmentation", ROOT / "legacy_dlinear" / "models"):
    if candidate.is_dir() and str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from device_data_paper_augmentation import (  # noqa: E402
    DEVICE_LOCATIONS,
    location_group,
    location_group_intra_string,
)

_spec = importlib.util.spec_from_file_location(
    "upstream_nlinear", ROOT / "legacy_dlinear" / "models" / "NLinear.py")
if _spec is None or not (ROOT / "legacy_dlinear" / "models" / "NLinear.py").is_file():
    _spec = importlib.util.spec_from_file_location(
        "upstream_nlinear", Path("/home/test/linliting/jyzmentor/dist2026/LTSF-Linear/models/NLinear.py"))
upstream_nlinear = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(upstream_nlinear)

_aspec = importlib.util.spec_from_file_location(
    "archived_linear", ROOT / "legacy_dlinear" / "model" / "ltsf_nlinear.py")
archived_linear = importlib.util.module_from_spec(_aspec)
_aspec.loader.exec_module(archived_linear)

WORKBOOK = ROOT / "dataset" / "device_data_110_220_final.xlsx"
HUM_BINS = [0, 30, 40, 50, 55, 60, 65, 70, 75, 80, 90, 100, 101]
HUM_EDGE = 60.0
VERSION = "v42"
PEER_JUMP_Z_THR = 8.0
PEER_SINGLE_JUMP_Z_THR = 12.0
MIN_TAIL_POINTS = 2
LOCAL_INTERVAL_POINTS = 9
TOL_FLOOR_S = 1800.0
TOL_CEIL_S = 43200.0
SEQ_LEN = 16
MIN_HISTORY = 4
MAX_EPOCHS = 120
PATIENCE = 15
BATCH_SIZE = 128
LEARNING_RATE = 1e-3
SEED = 42
# Fixed CPU thread count so baseline fits are bit-reproducible regardless of
# what the caller does with torch.set_num_threads().
DETERMINISTIC_THREADS = 2

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
    local = dt.rolling(LOCAL_INTERVAL_POINTS, center=True, min_periods=1).median()
    tol = (0.5 * local).clip(TOL_FLOOR_S, TOL_CEIL_S)
    cand = dt.to_numpy(float)
    fallback = 0.5 * np.nanmedian(cand) if np.isfinite(cand).any() else TOL_FLOOR_S
    return tol.fillna(float(np.clip(fallback, TOL_FLOOR_S, TOL_CEIL_S)))


class NLinearBaseline(nn.Module):
    """Upstream LTSF NLinear (measure history) + a linear current-condition head.

    yhat = NLinear(Measure history)[..., 0, 0] + Linear(Temp, Hum, Temp*Hum, elapsed)
    """

    def __init__(self, seq_len: int, n_conditions: int):
        super().__init__()
        configs = SimpleNamespace(seq_len=seq_len, pred_len=1, enc_in=1, individual=False)
        self.history = upstream_nlinear.Model(configs)
        self.condition = nn.Linear(n_conditions, 1)

    def forward(self, x, c):
        return self.history(x)[:, 0, 0] + self.condition(c).squeeze(-1)


ARCHIVED_FEATURES = ["Temp", "Hum", "AH", "Temp_diff", "Hum_diff",
                     "Temp_Hum_interact", "Temp_AH_interact",
                     "Hum_low", "Hum_mid", "Hum_high"]


def build_archived_features(g: pd.DataFrame, seq_len: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Leak-free port of legacy_dlinear/train.py feature set.

    Dropped on purpose (they reconstruct the target directly):
      Measure_diff + Measure_lag1  ->  Measure_t exactly
      Measure_MA (rolling(3))      ->  contains Measure_t
    Temp_diff / Hum_diff are shifted to the previous row because the archived
    predict() path cannot supply them at inference time and passes 0.
    """
    measure = g.Measure.to_numpy(float)
    n = len(measure)
    idx = np.flatnonzero(np.arange(n) >= max(seq_len, MIN_HISTORY))
    if not len(idx):
        return np.empty((0, len(ARCHIVED_FEATURES))), np.empty(0), np.empty(0, dtype=int)

    temp = g.Temp.to_numpy(float)
    hum = g.Hum.to_numpy(float)
    ah = 216.7 * (hum / 100.0 * 6.112 * np.exp(17.67 * temp / (temp + 243.5))) / (temp + 273.15)
    temp_diff = np.r_[0.0, np.diff(temp)]
    hum_diff = np.r_[0.0, np.diff(hum)]

    feats = np.c_[
        temp, hum, ah,
        temp_diff, hum_diff,
        temp * hum, temp * ah,
        (hum < 40).astype(float),
        ((hum >= 40) & (hum < 55)).astype(float),
        (hum >= 55).astype(float),
    ]
    last_measure = np.array([measure[i - 1] for i in idx])
    return feats[idx], last_measure, idx


class ArchivedNLinearBaseline(nn.Module):
    """legacy_dlinear DecompositionNLinear, used as archived, plus a condition head."""

    def __init__(self, input_dim: int, n_conditions: int):
        super().__init__()
        self.body = archived_linear.DecompositionNLinear(
            input_dim=input_dim, hidden_dim=128, dropout=0.2, kernel_size=25)
        self.condition = nn.Linear(n_conditions, 1)

    def forward(self, f, c):
        return self.body(f) + self.condition(c).squeeze(-1)



def build_sequences(g: pd.DataFrame, seq_len: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Causal history windows: row k uses rows k-seq_len..k-1, never row k."""
    measure = g.Measure.to_numpy(float)
    n = len(measure)
    keep = np.arange(n) >= max(seq_len, MIN_HISTORY)
    idx = np.flatnonzero(keep)
    if not len(idx):
        return np.empty((0, seq_len, 3)), np.empty((0, 4)), np.empty(0, dtype=int)
    win = np.stack([measure[i - seq_len:i] for i in idx])[:, :, None]
    elapsed = (g.Timestamp - g.Timestamp.iloc[0]).dt.total_seconds().to_numpy(float)[idx] / 86400.0
    gap_h = g.interval_s.to_numpy(float)[idx] / 3600.0
    gap_h = np.where(np.isfinite(gap_h), gap_h, 0.0)
    cond = np.c_[g.Temp.to_numpy(float)[idx], g.Hum.to_numpy(float)[idx],
                 g.Temp.to_numpy(float)[idx] * g.Hum.to_numpy(float)[idx], elapsed]
    return win, cond, idx


def _fit_huber(g: pd.DataFrame, idx: np.ndarray) -> tuple[np.ndarray, dict]:
    """Reference baseline: v4's per-device Huber on Temp, Hum, elapsed."""
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import HuberRegressor
    from sklearn.preprocessing import StandardScaler

    elapsed = (g.Timestamp - g.Timestamp.iloc[0]).dt.total_seconds().to_numpy(float) / 86400.0
    cov = SimpleImputer(strategy="median").fit_transform(
        np.c_[g.Temp.to_numpy(float), g.Hum.to_numpy(float), elapsed])
    scaler = StandardScaler().fit(cov)
    model = HuberRegressor(epsilon=1.5, alpha=1e-4, max_iter=2000).fit(
        scaler.transform(cov), g.Measure.to_numpy(float))
    pred = model.predict(scaler.transform(cov))
    return pred, {"status": "ok", "baseline_kind": "huber", "n_train": int(len(g)),
                  "protocol": "full_history_in_sample"}


def _fit_last(g: pd.DataFrame, seq_len: int) -> tuple[np.ndarray, dict]:
    """naive: predict Measure_t by the previous observed Measure_{t-1}."""
    measure = g.Measure.to_numpy(float)
    pred = np.full(len(g), np.nan)
    if len(measure) >= 2:
        pred[1:] = measure[:-1]
    return pred, {"status": "ok", "baseline_kind": "last", "n_train": int(len(g) - 1),
                  "protocol": "persistence_previous_value"}


def fit_device_baseline(g: pd.DataFrame, seq_len: int, kind: str = "archived") -> tuple[np.ndarray, dict]:
    if kind == "huber":
        return _fit_huber(g, np.empty(0, dtype=int))
    if kind == "last":
        return _fit_last(g, seq_len)

    # Labels must not depend on the caller's thread count or global RNG state.
    # CPU reductions (and therefore the fitted baseline) change with
    # torch.set_num_threads, which silently moved this dataset between 40, 41
    # and 43 candidate windows. Pin the thread count and restore it afterwards.
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(DETERMINISTIC_THREADS)
    try:
        return _fit_device_baseline(g, seq_len, kind)
    finally:
        torch.set_num_threads(previous_threads)


def _fit_device_baseline(g: pd.DataFrame, seq_len: int, kind: str) -> tuple[np.ndarray, dict]:
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    if kind == "archived":
        feats, last_measure, idx = build_archived_features(g, seq_len)
        if len(feats) < 16:
            return np.full(len(g), np.nan), {"status": "insufficient_history", "n_train": int(len(feats))}
        elapsed = (g.Timestamp - g.Timestamp.iloc[0]).dt.total_seconds().to_numpy(float)[idx] / 86400.0
        cond = np.c_[g.Temp.to_numpy(float)[idx], g.Hum.to_numpy(float)[idx],
                     g.Temp.to_numpy(float)[idx] * g.Hum.to_numpy(float)[idx], elapsed]
        xt = torch.tensor(feats, dtype=torch.float32)
    else:
        win, cond, idx = build_sequences(g, seq_len)
        if len(win) < 16:
            return np.full(len(g), np.nan), {"status": "insufficient_history", "n_train": int(len(win))}
        xt = torch.tensor(win, dtype=torch.float32)
    y = g.Measure.to_numpy(float)[idx]

    ct = torch.tensor(cond, dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.float32)
    x_mean, x_std = xt.mean(), xt.std().clamp_min(1e-6)
    c_mean, c_std = ct.mean(dim=0), ct.std(dim=0).clamp_min(1e-6)
    y_mean, y_std = yt.mean(), yt.std().clamp_min(1e-6)
    xt_n = (xt - x_mean) / x_std
    ct_n = (ct - c_mean) / c_std
    yt_n = (yt - y_mean) / y_std

    order = rng.permutation(len(y))
    split = max(int(0.8 * len(y)), 8)
    tr, va = order[:split], order[split:]
    if not len(va):
        tr, va = order[:-1], order[-1:]

    if kind == "archived":
        model = ArchivedNLinearBaseline(xt.shape[1], cond.shape[1])
    else:
        model = NLinearBaseline(seq_len, cond.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.SmoothL1Loss(beta=0.1)
    best, best_state, wait, best_epoch = np.inf, None, 0, 0
    epoch = 0
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        perm = rng.permutation(len(tr))
        for s in range(0, len(perm), BATCH_SIZE):
            b = tr[perm[s:s + BATCH_SIZE]]
            opt.zero_grad()
            loss = loss_fn(model(xt_n[b], ct_n[b]), yt_n[b])
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            vloss = float(loss_fn(model(xt_n[va], ct_n[va]), yt_n[va]))
        if vloss < best - 1e-5:
            best, wait, best_epoch = vloss, 0, epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= PATIENCE:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pred_n = model(xt_n, ct_n)
    pred = pred_n.numpy() * float(y_std) + float(y_mean)
    out = np.full(len(g), np.nan)
    out[idx] = pred
    return out, {
        "status": "ok", "baseline_kind": kind, "n_train": int(len(y)), "seq_len": seq_len,
        "stopped_epoch": int(epoch), "best_epoch": int(best_epoch),
        "best_val_loss": float(best) if np.isfinite(best) else None,
        "protocol": "full_history_in_sample",
    }


def add_point_features(points: pd.DataFrame, seq_len: int, kind: str = "archived") -> tuple[pd.DataFrame, list[dict]]:
    out, models = [], []
    for imei, g in points.groupby("IMEI", sort=False):
        g = g.sort_values("Timestamp").copy().reset_index(drop=True)
        dt = g.Timestamp.diff().dt.total_seconds()
        g["interval_s"] = dt
        g["median_interval_s"] = dt.median()
        g["local_median_interval_s"] = dt.rolling(
            LOCAL_INTERVAL_POINTS, center=True, min_periods=1).median()
        g["peer_tolerance_s"] = _local_tolerance_s(dt)
        g["hum_gt60"] = (g.Hum > HUM_EDGE).astype(float)
        baseline, info = fit_device_baseline(g, seq_len, kind)
        info["device_id"] = imei
        models.append(info)
        g["env_baseline"] = baseline
        g["env_residual"] = g.Measure.to_numpy(float) - baseline
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
    mask = points.env_residual_z.isna() & points.env_residual_z_device.notna()
    if mask.any():
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
    for group_id, group in points.groupby("group_id", sort=False):
        devices = sorted(group.IMEI.unique())
        if len(devices) < 2:
            continue
        for imei in devices:
            target = group[group.IMEI == imei].sort_values("Timestamp")
            own = target.Timestamp.to_numpy(dtype="datetime64[ns]").astype("int64")
            tol_ns = target.peer_tolerance_s.to_numpy(float) * 1e9
            peer_res, peer_hum = [], []
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
            res_mat = np.vstack(peer_res)
            hum_mat = np.vstack(peer_hum)
            import warnings
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
    if mask.any():
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
    return points, models


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
        pj_max = float(np.nanmax(pjump)) if np.isfinite(pjump).any() else np.nan
        return {
            "IMEI": imei, "Location": w.Location.iloc[0], "group_id": w.group_id.iloc[0],
            "window_id": f"{imei}_{number:04d}", "window_number": number,
            "window_type": window_type, "start_row": start, "end_row": end,
            "start_time": w.Timestamp.iloc[0], "end_time": w.Timestamp.iloc[-1],
            "n_points": n, "n_scored": int(np.isfinite(rz).sum()),
            "duration_hours": duration_h,
            "median_interval_hours": float(np.nanmedian(w.interval_s)) / 3600.0,
            "tolerance_hours_median": float(np.nanmedian(w.peer_tolerance_s)) / 3600.0,
            "measure_mean": np.mean(measure), "measure_std": np.std(measure),
            "measure_slope_per_hour": slope,
            "residual_z_mean": float(np.nanmean(rz)),
            "residual_z_abs_q90": float(np.nanquantile(np.abs(rz), 0.90)) if np.isfinite(rz).any() else np.nan,
            "abs2_residual_fraction": float(np.nanmean(np.abs(rz) >= 2.0)) if np.isfinite(rz).any() else np.nan,
            "max_consecutive_high": int(_max_run(np.abs(rz) >= 2.0)),
            "residual_jump_z_max": float(np.nanmax(rjump)) if np.isfinite(rjump).any() else np.nan,
            "n_large_residual_jumps": int(np.nansum(rjump >= jump_z_thr)),
            "peer_z_mean": float(np.mean(pz_valid)) if len(pz_valid) else np.nan,
            "peer_z_frac_valid": float(len(pz_valid)) / n,
            "peer_delta_jump_z_max": pj_max,
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


BASELINE_LABELS = {
    "archived": "DecompositionNLinear (legacy_dlinear, leakage removed)",
    "nlinear": "official LTSF NLinear",
    "huber": "Huber (v4 reference)",
    "last": "naive persistence (previous Measure)",
}


def plot_labels(points: pd.DataFrame, windows: pd.DataFrame, out: Path, mode: str,
                kind: str = "archived") -> None:
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
        ax.plot(p.Timestamp, p.Measure, color="#94a3b8", lw=.6, alpha=.6, label="Measure")
        ax.plot(p.Timestamp, p.env_baseline, color="#0f766e", lw=.7, alpha=.8,
                label=f"baseline ({kind})")
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
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(.5, 1.005))
    fig.suptitle(f"{VERSION} candidate windows: {BASELINE_LABELS.get(kind, kind)} [{mode}]",
                 fontsize=15, y=1.035)
    note = (f"{VERSION} environment baseline = {BASELINE_LABELS.get(kind, kind)}. "
            "32-point windows, humidity bins, peer matching and A/B/C/D thresholds are unchanged from v4.")
    if kind == "archived":
        note += (" Target-leaking features Measure_diff / Measure_lag1 / Measure_MA were removed; "
                 "its internal sequence decomposition degenerates at time length 1, so the effective "
                 "model is a non-linear MLP, not a true decomposition DLinear.")
    fig.text(.5, .01, note, ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .03, 1, .98))
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main(args) -> None:
    label_col = f"window_anomaly_{VERSION}"
    args.output.mkdir(parents=True, exist_ok=True)
    points, models = add_point_features(load_points(args.workbook, args.peer_group_mode), args.seq_len, args.baseline)
    windows = make_windows(points, args.window_size)
    points.to_csv(args.output / f"device_points_{VERSION}.csv", index=False, encoding="utf-8-sig")
    windows.to_csv(args.output / f"device_nonoverlap_windows_{VERSION}.csv", index=False, encoding="utf-8-sig")
    plot_labels(points, windows, args.output / f"{VERSION}_anomaly_labels_by_device.png",
                args.peer_group_mode, args.baseline)

    with pd.ExcelWriter(args.output / f"device_nonoverlap_windows_{VERSION}.xlsx", engine="openpyxl") as writer:
        windows.to_excel(writer, sheet_name="window_features", index=False)
        windows.groupby(["IMEI", "Location", "group_id"], as_index=False).agg(
            windows=("window_id", "size"),
            anomalies=(label_col, "sum"),
            anomaly_rate=(label_col, "mean"),
        ).to_excel(writer, sheet_name="summary", index=False)

    ok = [m for m in models if m.get("status") == "ok"]
    protocol = {
        "version": VERSION,
        "change_vs_v4": "environment baseline: HuberRegressor -> LTSF NLinear (causal history window)",
        "base": "v4 rules unchanged; peer tolerance = v4.1 per-point local median",
        "source": str(args.workbook),
        "devices": int(windows.IMEI.nunique()),
        "peer_group_mode": args.peer_group_mode,
        "n_peer_groups": int(windows.group_id.nunique()),
        "window_size": args.window_size,
        "windows": int(len(windows)),
        "windows_by_type": windows.window_type.value_counts().to_dict(),
        f"{VERSION}_candidate_windows": int(windows[label_col].sum()),
        f"{VERSION}_candidate_rate": float(windows[label_col].mean()),
        "baseline": {
            "model": "LTSF-Linear models/NLinear.py (linear, non-individual)",
            "source_path": "legacy_dlinear/models/NLinear.py",
            "seq_len": args.seq_len,
            "min_history": MIN_HISTORY,
            "optimizer": "Adam", "lr": LEARNING_RATE, "batch_size": BATCH_SIZE,
            "max_epochs": MAX_EPOCHS, "patience": PATIENCE, "seed": SEED,
            "loss": "SmoothL1(beta=0.1)",
            "conditions": ["Temp", "Hum", "Temp*Hum", "elapsed_days"],
            "protocol": "full_history_in_sample (aligned with v4, retrospective)",
            "causal_history": True,
            "target_leakage_guard": "row k uses only rows k-seq_len..k-1",
            "devices_fitted": len(ok),
            "devices_failed": [m["device_id"] for m in models if m.get("status") != "ok"],
        },
        "tolerance": {
            "formula": "clip(0.5 * median(dt[k-4..k+4]), 1800s, 43200s)",
            "rolling_points": LOCAL_INTERVAL_POINTS, "centered": True,
            "floor_s": TOL_FLOOR_S, "ceil_s": TOL_CEIL_S,
            "uses_future_observations": True, "online_causal": False,
        },
        "thresholds": {"window_evidence_robust_z": 3.0, "persist_fraction_abs2": 0.15,
                       "min_large_peer_jumps": 2, "residual_jump_point_z": 2.5,
                       "peer_jump_point_z": PEER_JUMP_Z_THR,
                       "isolated_peer_jump_point_z": PEER_SINGLE_JUMP_Z_THR,
                       "min_tail_points": MIN_TAIL_POINTS},
        "device_models": models,
    }
    (args.output / f"protocol_{VERSION}.json").write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in protocol.items() if k != "device_models"},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workbook", type=Path, default=WORKBOOK)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--window-size", type=int, default=32)
    parser.add_argument("--seq-len", type=int, default=SEQ_LEN)
    parser.add_argument("--baseline", choices=["archived", "nlinear", "huber", "last"], default="archived",
                        help="archived (v4.2 default) = legacy_dlinear DecompositionNLinear, leakage removed; "
                             "nlinear = official LTSF NLinear; huber = v4 baseline; last = persistence")
    parser.add_argument("--peer-group-mode", choices=sorted(GROUP_FUNCTIONS), default="intra-string")
    main(parser.parse_args())
