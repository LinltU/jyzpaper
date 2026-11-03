# -*- coding: utf-8 -*-
"""Report-based v4 reconstruction. No target-count calibration or device quota."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
from scipy.special import ndtri
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import HuberRegressor
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

OUT = Path("outputs/v4")
SOURCE: Path
REPORT: Path | None = None
LOCATIONS: Path
SKIP_PLOTS = False
EDGES = [0, 30, 40, 50, 55, 60, 65, 70, 75, 80, 90, 100, 101]
N_WINDOW = 32
LOCAL_INTERVAL_POINTS = 9
REASONS = {
    "A": "environment residual and peer agree",
    "B": "peer-delta jump (per-device)",
    "C": "isolated extreme peer jump",
    "D": "no-peer strong residual (solo device)",
}
REASONS_CN = {
    "A": "环境残差与同串偏离同向持续",
    "B": "窗口内至少两次同串不同步大跳变",
    "C": "至少一次同串不同步极端跳变",
    "D": "无同串设备时的强残差",
}
ASSUMPTIONS = [
    ["原始代码", "本地未找到报告所列原始 v4 脚本及 protocol；本版是依据报告明确规则重建，不能称为原脚本逐行复现。"],
    ["Huber 输入", "以原始 Measure 为因变量；Temp、Hum、距设备首点的 elapsed_days 三个自变量先做均值/标准差缩放；使用截距。"],
    ["Huber 拟合", "逐设备全历史离线拟合；epsilon=1.5、alpha=0.0001、max_iter=2000、tol=1e-8；不是面向未来的在线报警评估。"],
    ["稳健尺度", "1.4826×MAD；若退化则 IQR/1.349，再退化用标准差，最后用 1e-12；逐设备记录实际采用的尺度。"],
    ["残差湿度分层", "先逐设备残差中位数/MAD 标准化，再在全设备合并的各湿度箱内做正态分位变换。报告未指定第二步是否继续按设备分组。"],
    ["分位变换", "左闭右开湿度箱；平均秩处理并列值；z=Phi^-1((rank-0.5)/n)；单点箱 z=0；缺失维持缺失。"],
    ["同串定义", "电压、左右侧、上下相、大小号侧、串号完全一致；110kV 玉中线两台为一组；排除自身。"],
    ["peer 比较量", "peer_delta=目标原始 Measure－已匹配同串其他设备原始 Measure 的中位数；原始差之后按 group×目标湿度箱分位标准化。报告未明示 peer 输入量。"],
    ["局部间隔", "目标设备真实相邻时间差的居中 9 点滚动中位数，min_periods=1；容差=0.5×此间隔，裁剪至 0.5～12 小时。报告未给局部滚动跨度。"],
    ["最近匹配", "不插值；等距时选较早时间；每台 peer 各选最近一点；允许同一 peer 点被相邻目标点使用，完整保留匹配时间差。"],
    ["时间单位修正", "2026-11-02 复核修正：时间搜索前显式转换为 datetime64[ns]，避免微秒整数按纳秒换算使时间差缩小1000倍。匹配偏差另由原始时间相减独立校验。不得再使用时间单位错误版标签。"],
    ["peer 缺测", "至少匹配一台时计算 peer_delta；peer_complete 仅当该点全部同串其他设备均匹配时为真；缺测不补零参与标注。"],
    ["has_peer", "表示物理分组存在其他设备，不等同于某时刻匹配成功；本数据 25 台均有物理 peer，所以 D 不适用。"],
    ["跳变", "peer_delta_jump_z=abs(相邻观测 peer_delta 差)/sqrt(2)/逐设备 peer_delta 水平量稳健尺度；首点或相邻任一点缺测时不计算，不跨缺测桥接。"],
    ["残差跳变", "D 所需残差跳变采用 abs(相邻 env_residual_z 差)/sqrt(2)/该设备 env_residual_z 水平量稳健尺度；阈值 2.5；本数据 D 不触发。"],
    ["窗口证据", "先计算带符号的 residual_z_mean 与 peer_z_mean；其绝对值分别在全部窗口池上做中位数/MAD 稳健 z，得到两个 mean_z 证据；同号用原带符号均值判断。报告未给窗口证据精确公式。"],
    ["窗口边界跳变", "先逐设备计算差分，再分窗；跨窗口边界的跳变计入后一个窗口，并保留前一点时间和间隔；不另设报告外的时间间隔剔除阈值。"],
    ["尾窗", "每台剩余 2～31 点作为 tail；若仅剩 1 点，保留该点但不给窗口判断，明确列入未评估点。本次 0629、0694 各有 1 个点；报告未说明这类边界如何实现零丢点标注。"],
    ["原因重叠", "保存 A/B/C/D 全部布尔标志；主原因优先顺序 B、C、A、D；主原因仅用于统计，不改变标签。报告未说明原因覆盖顺序。"],
    ["排序分数", "仅候选窗口排序：取各已触发规则的最小条件/阈值比中的最大值；B=count8/2，C=maxjump/12，A=min(rmean_z/3,占比/0.15,pmean_z/3)。分数不是概率，不参与标签阈值。"],
    ["标签含义", "候选窗口内全部观测点标 1，其他已评估窗口标 0；1 是统计候选异常，不是检修确认故障；0 表示本规则未触发。"],
]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def robust_stats(values):
    a = np.asarray(values, float)
    a = a[np.isfinite(a)]
    if not len(a):
        return np.nan, np.nan, "no_finite_data"
    center = float(np.median(a))
    scale = float(1.4826 * np.median(np.abs(a - center)))
    method = "1.4826*MAD"
    if scale <= 1e-12:
        scale = float((np.quantile(a, .75) - np.quantile(a, .25)) / 1.349)
        method = "IQR/1.349"
    if scale <= 1e-12:
        scale = float(np.std(a))
        method = "std"
    if scale <= 1e-12:
        scale, method = 1e-12, "epsilon_floor"
    return center, scale, method


def normal_scores(values):
    s = pd.Series(values, index=values.index, dtype=float)
    good = s.notna()
    out = pd.Series(np.nan, index=s.index, dtype=float)
    n = int(good.sum())
    if n:
        out.loc[good] = ndtri((s.loc[good].rank(method="average") - .5) / n)
    return out


def load():
    locations = json.loads(LOCATIONS.read_text(encoding="utf-8-sig"))
    if not isinstance(locations, dict) or not all(
            isinstance(k, str) and re.fullmatch(r"\d{15}", k) and isinstance(v, str)
            for k, v in locations.items()):
        raise ValueError("Locations must be a JSON object of 15-digit device IDs to location strings.")
    frames, cleanlog = [], []
    book = pd.ExcelFile(SOURCE, engine="openpyxl")
    for device in book.sheet_names:
        f = pd.read_excel(book, sheet_name=device)
        f["source_excel_row"] = np.arange(len(f)) + 2
        n_raw = len(f)
        valid_id = f.IMEI.astype(str).str.fullmatch(r"\d{15}").fillna(False)
        invalid_id = int((~valid_id).sum())
        f = f.loc[valid_id].copy()
        assert (f.IMEI.astype(str) == str(device)).all()
        f["Timestamp"] = pd.to_datetime(f.Timestamp, errors="coerce")
        for c in ["Measure", "Temp", "Hum"]:
            f[c] = pd.to_numeric(f[c], errors="coerce")
        invalid_values = ~f[["Measure", "Temp", "Hum"]].apply(np.isfinite).all(axis=1) | f.Timestamp.isna()
        invalid_n = int(invalid_values.sum())
        f = f.loc[~invalid_values].sort_values(["Timestamp", "source_excel_row"])
        duplicates = int(f.Timestamp.duplicated(keep="last").sum())
        f = f.drop_duplicates("Timestamp", keep="last").copy()
        loc = locations[str(device)]
        kv = 220 if "220kV" in loc else 110
        side = "left" if "左边" in loc else ("right" if "右边" in loc else "single")
        phase = "down" if "下相" in loc else ("up" if "上相" in loc else "single")
        size = "small" if "小号侧" in loc else ("big" if "大号侧" in loc else "single")
        string = 2 if "串2" in loc else 1
        f["device_id"], f["IMEI"], f["location"] = str(device), str(device), loc
        f["peer_group"] = f"{kv}_{side}_{phase}_{size}_string{string}"
        f["voltage_kv"], f["string_no"] = kv, string
        f["elapsed_days"] = (f.Timestamp - f.Timestamp.iloc[0]).dt.total_seconds() / 86400
        f["interval_hours"] = f.Timestamp.diff().dt.total_seconds() / 3600
        f["local_median_interval_hours"] = f.interval_hours.rolling(
            LOCAL_INTERVAL_POINTS, center=True, min_periods=1).median()
        f["peer_tolerance_hours"] = (.5 * f.local_median_interval_hours).clip(.5, 12)
        f["point_index_device"] = np.arange(len(f))
        frames.append(f)
        cleanlog.append(dict(device_id=str(device), raw_rows=n_raw, invalid_id_rows=invalid_id,
                             invalid_value_rows=invalid_n, duplicate_time_rows=duplicates,
                             retained_points=len(f)))
    data = pd.concat(frames, ignore_index=True).sort_values(["device_id", "Timestamp"]).reset_index(drop=True)
    data["humidity_bin"] = pd.cut(data.Hum, EDGES, right=False, labels=False)
    assert data.humidity_bin.notna().all(), "Humidity outside report bins; requires explicit treatment."
    data.humidity_bin = data.humidity_bin.astype(int)
    return data, pd.DataFrame(cleanlog)


def environment(data):
    models = []
    for device, f in data.groupby("device_id", sort=False):
        x = f[["Temp", "Hum", "elapsed_days"]].to_numpy(float)
        scaler = StandardScaler()
        xs = scaler.fit_transform(x)
        model = HuberRegressor(epsilon=1.5, alpha=1e-4, max_iter=2000, tol=1e-8)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            model.fit(xs, f.Measure.to_numpy(float))
        if caught:
            raise RuntimeError(f"Huber warning on {device}: {caught}")
        predicted = model.predict(xs)
        residual = f.Measure.to_numpy(float) - predicted
        center, scale, method = robust_stats(residual)
        data.loc[f.index, "env_baseline"] = predicted
        data.loc[f.index, "env_residual"] = residual
        data.loc[f.index, "env_residual_device_z"] = (residual - center) / scale
        models.append(dict(device_id=device, residual_center=center, residual_scale=scale,
                           residual_scale_method=method, huber_iterations=int(model.n_iter_),
                           huber_intercept=float(model.intercept_), huber_scale=float(model.scale_),
                           predictor_center=scaler.mean_.tolist(), predictor_scale=scaler.scale_.tolist(),
                           coefficients_scaled=model.coef_.tolist()))
    data["env_residual_z"] = data.groupby("humidity_bin", group_keys=False).env_residual_device_z.transform(normal_scores)
    return data, models


def peers(data):
    data["n_expected_peers"] = data.groupby("peer_group").device_id.transform("nunique") - 1
    data["has_peer"] = (data.n_expected_peers > 0).astype(int)
    matchrows = []
    for device, f in data.groupby("device_id", sort=False):
        f = f.sort_values("Timestamp")
        own_times = f.Timestamp.to_numpy(dtype="datetime64[ns]").astype("int64")
        tolerance_ns = f.peer_tolerance_hours.to_numpy() * 3600e9
        other_ids = sorted(data.loc[(data.peer_group == f.peer_group.iloc[0]) & (data.device_id != device),
                                   "device_id"].unique())
        panel = []
        offsets = []
        for other in other_ids:
            p = data.loc[data.device_id == other].sort_values("Timestamp")
            ptimes = p.Timestamp.to_numpy(dtype="datetime64[ns]").astype("int64")
            insert = np.searchsorted(ptimes, own_times)
            left = np.clip(insert - 1, 0, len(p) - 1)
            right = np.clip(insert, 0, len(p) - 1)
            li, ri = np.abs(ptimes[left] - own_times), np.abs(ptimes[right] - own_times)
            nearest = np.where(li <= ri, left, right)
            off_ns = ptimes[nearest] - own_times
            accepted = np.abs(off_ns) <= tolerance_ns
            values = p.Measure.to_numpy()[nearest].astype(float)
            panel.append(np.where(accepted, values, np.nan))
            offsets.append(np.where(accepted, np.abs(off_ns / 60e9), np.nan))
            matchrows.append(pd.DataFrame({
                "device_id": device, "Timestamp": f.Timestamp.to_numpy(), "peer_group": f.peer_group.iloc[0],
                "peer_device_id": other, "nearest_peer_timestamp": p.Timestamp.to_numpy()[nearest],
                "offset_minutes": off_ns / 60e9, "tolerance_minutes": tolerance_ns / 60e9,
                "accepted": accepted.astype(int),
                "peer_measure": np.where(accepted, values, np.nan),
                "target_measure": f.Measure.to_numpy(),
            }))
        if panel:
            panel = np.asarray(panel).T
            offsets = np.asarray(offsets).T
            count = np.isfinite(panel).sum(axis=1)
            med = np.full(len(f), np.nan)
            maxoff = np.full(len(f), np.nan)
            good = count > 0
            med[good] = np.nanmedian(panel[good], axis=1)
            maxoff[good] = np.nanmax(offsets[good], axis=1)
        else:
            count, med, maxoff = np.zeros(len(f)), np.full(len(f), np.nan), np.full(len(f), np.nan)
        data.loc[f.index, "n_matched_peers"] = count
        data.loc[f.index, "peer_measure_median"] = med
        data.loc[f.index, "peer_offset_max_minutes"] = maxoff
        data.loc[f.index, "peer_delta"] = f.Measure.to_numpy() - med
    data["peer_complete"] = ((data.n_matched_peers == data.n_expected_peers) & (data.n_expected_peers > 0)).astype(int)
    data["peer_delta_z"] = data.groupby(["peer_group", "humidity_bin"], group_keys=False).peer_delta.transform(normal_scores)
    jump_scales = []
    for device, f in data.groupby("device_id", sort=False):
        _, ps, method = robust_stats(f.peer_delta)
        _, es, emethod = robust_stats(f.env_residual_z)
        data.loc[f.index, "peer_delta_jump_signed_z"] = f.peer_delta.diff() / np.sqrt(2) / ps
        data.loc[f.index, "peer_delta_jump_z"] = f.peer_delta.diff().abs() / np.sqrt(2) / ps
        data.loc[f.index, "residual_jump_z"] = f.env_residual_z.diff().abs() / np.sqrt(2) / es
        data.loc[f.index, "previous_timestamp"] = f.Timestamp.shift()
        data.loc[f.index, "peer_level_scale"] = ps
        jump_scales.append(dict(device_id=device, peer_level_scale=ps, peer_scale_method=method,
                                residual_z_level_scale=es, residual_jump_scale_method=emethod))
    return data, pd.concat(matchrows, ignore_index=True), jump_scales


def slope_per_hour(f, column):
    hours = (f.Timestamp - f.Timestamp.iloc[0]).dt.total_seconds().to_numpy() / 3600
    v = f[column].to_numpy(float)
    good = np.isfinite(v)
    if good.sum() < 2 or np.ptp(hours[good]) <= 0:
        return np.nan
    return float(np.polyfit(hours[good], v[good], 1)[0])


def windows(data):
    rows = []
    data["window_id"] = ""
    data["is_candidate"] = -1
    for device, f in data.groupby("device_id", sort=False):
        for start in range(0, len(f), N_WINDOW):
            w = f.iloc[start:start + N_WINDOW]
            if len(w) < 2:
                continue
            wid = f"{device}_W{start // N_WINDOW + 1:03d}"
            data.loc[w.index, "window_id"] = wid
            rows.append(dict(
                device_id=device, window_id=wid, peer_group=w.peer_group.iloc[0],
                location=w.location.iloc[0], start=w.Timestamp.iloc[0], end=w.Timestamp.iloc[-1],
                window_type="full" if len(w) == N_WINDOW else "tail", n_points=len(w),
                duration_hours=(w.Timestamp.iloc[-1] - w.Timestamp.iloc[0]).total_seconds() / 3600,
                median_interval_hours=w.Timestamp.diff().dt.total_seconds().median() / 3600,
                has_peer=int(w.has_peer.iloc[0]), n_expected_peers=int(w.n_expected_peers.iloc[0]),
                peer_matched_fraction=float(w.peer_delta.notna().mean()),
                peer_complete_fraction=float(w.peer_complete.mean()),
                max_accepted_peer_offset_minutes=float(w.peer_offset_max_minutes.max()),
                hum_mean=float(w.Hum.mean()), high_humidity_fraction=float((w.Hum > 60).mean()),
                measure_mean=float(w.Measure.mean()), residual_z_mean=float(w.env_residual_z.mean()),
                residual_z_abs_q90=float(w.env_residual_z.abs().quantile(.9)),
                abs2_residual_fraction=float((w.env_residual_z.abs() >= 2).mean()),
                peer_z_mean=float(w.peer_delta_z.mean()), peer_delta_mean=float(w.peer_delta.mean()),
                peer_jump_count8=int((w.peer_delta_jump_z >= 8).sum()),
                peer_jump_count12=int((w.peer_delta_jump_z >= 12).sum()),
                peer_jump_max=float(w.peer_delta_jump_z.max()),
                residual_jump_count2_5=int((w.residual_jump_z >= 2.5).sum()),
                measure_slope_per_hour=slope_per_hour(w, "Measure"),
                residual_z_slope_per_hour=slope_per_hour(w, "env_residual_z"),
                peer_z_slope_per_hour=slope_per_hour(w, "peer_delta_z"),
            ))
    w = pd.DataFrame(rows)
    poolstats = {}
    for source, target in [("residual_z_mean", "residual_z_mean_z"), ("peer_z_mean", "peer_z_mean_z")]:
        center, scale, method = robust_stats(w[source].abs())
        w[target] = (w[source].abs() - center) / scale
        poolstats[target] = dict(center=center, scale=scale, method=method, transformation="abs(window_signed_mean)")
    w["same_direction"] = (w.residual_z_mean * w.peer_z_mean > 0).astype(int)
    w["rule_A"] = ((w.residual_z_mean_z >= 3) & (w.abs2_residual_fraction >= .15) &
                   (w.peer_z_mean_z >= 3) & (w.same_direction == 1)).astype(int)
    w["rule_B"] = (w.peer_jump_count8 >= 2).astype(int)
    w["rule_C"] = (w.peer_jump_max >= 12).astype(int)
    w["rule_D"] = ((w.has_peer == 0) & (w.abs2_residual_fraction >= .15) &
                   (w.residual_z_abs_q90 >= 2) & (w.residual_jump_count2_5 >= 2)).astype(int)
    w["is_candidate"] = w[["rule_A", "rule_B", "rule_C", "rule_D"]].max(axis=1)
    w["triggered_rules"] = w.apply(lambda r: ",".join(k for k in "ABCD" if r[f"rule_{k}"]), axis=1)
    w["primary_rule"] = w.apply(lambda r: next((k for k in "BCAD" if r[f"rule_{k}"]), ""), axis=1)
    w["anomaly_reason"] = w.primary_rule.map(REASONS).fillna("no rule triggered")
    w["reason_cn"] = w.primary_rule.map(REASONS_CN).fillna("未触发报告规则")
    def score(r):
        margins = []
        if r.rule_A:
            margins.append(min(r.residual_z_mean_z / 3, r.abs2_residual_fraction / .15, r.peer_z_mean_z / 3))
        if r.rule_B:
            margins.append(r.peer_jump_count8 / 2)
        if r.rule_C:
            margins.append(r.peer_jump_max / 12)
        if r.rule_D:
            margins.append(min(r.abs2_residual_fraction / .15, r.residual_z_abs_q90 / 2, r.residual_jump_count2_5 / 2))
        return max(margins) if margins else np.nan
    w["candidate_score"] = w.apply(score, axis=1)
    w["candidate_rank"] = w.candidate_score.rank(method="min", ascending=False).astype("Int64")
    mapping = w.set_index("window_id")
    for c in ["is_candidate", "primary_rule", "triggered_rules", "reason_cn"]:
        data[c] = data.window_id.map(mapping[c])
    data["is_candidate"] = data.is_candidate.fillna(-1).astype(int)
    return data, w, poolstats


def frame_matrix(frame):
    f = frame.copy()
    datecols = []
    for c in f:
        if pd.api.types.is_datetime64_any_dtype(f[c]):
            datecols.append(c)
            f[c] = f[c].dt.strftime("%Y-%m-%dT%H:%M:%S")
    values = json.loads(f.to_json(orient="values", force_ascii=False))
    return dict(columns=f.columns.tolist(), rows=values, date_columns=datecols)


def plot_panel(ax, d, w, detailed=False):
    ax.set_facecolor("#fcfcfd")
    ax.plot(d.Timestamp, d.Measure, color="#8b96a7", lw=.65 if not detailed else .9, zorder=1)
    ax.scatter(d.Timestamp, d.Measure, s=3 if not detailed else 7, color="#547eae", alpha=.72, zorder=2)
    for r in w.loc[w.is_candidate == 1].itertuples():
        ax.axvspan(r.start, r.end, color="#d65b62", alpha=.09, lw=0)
        sel = d.loc[d.window_id == r.window_id]
        ax.plot(sel.Timestamp, sel.Measure, color="#ca4c55", lw=.9, zorder=3)
        ax.scatter(sel.Timestamp, sel.Measure, s=6 if not detailed else 12, color="#ca4c55", zorder=4)
    jumps = d.loc[d.peer_delta_jump_z >= 8]
    ax.scatter(jumps.Timestamp, jumps.Measure, s=22 if not detailed else 36,
               marker="x", linewidths=.9, color="#80333a", zorder=5)
    unassessed = d.loc[d.is_candidate == -1]
    ax.scatter(unassessed.Timestamp, unassessed.Measure, s=20 if not detailed else 36,
               marker="D", facecolors="white", edgecolors="#545b68", linewidths=.9, zorder=6)
    ax.grid(axis="y", color="#dfe3e8", lw=.45)
    for sp in ax.spines.values():
        sp.set_color("#cbd1d8")
        sp.set_linewidth(.6)
    ax.tick_params(colors="#555d68", labelsize=7 if not detailed else 9, length=2)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2 if not detailed else 1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_minor_locator(mdates.MonthLocator())
    ax.set_ylabel("Measure（原始单位）", fontsize=8 if not detailed else 10)


def figures(data, w, summary, audit):
    plt.rcParams.update({"font.family": "Microsoft YaHei", "font.size": 10,
                         "axes.unicode_minus": False, "savefig.facecolor": "white"})
    # Preserve the workbook's physical order, rather than sorting unrelated IMEI values.
    order = pd.ExcelFile(SOURCE, engine="openpyxl").sheet_names
    fig, axes = plt.subplots(5, 5, figsize=(24, 17.7), dpi=180)
    for ax, device in zip(axes.flat, order):
        d, ww = data.loc[data.device_id == device], w.loc[w.device_id == device]
        plot_panel(ax, d, ww)
        ax.set_xlim(d.Timestamp.min() - pd.Timedelta(days=2), d.Timestamp.max() + pd.Timedelta(days=3))
        ax.set_title(f"{device}\n候选 {ww.is_candidate.sum()}/{len(ww)} 窗口 · {d.is_candidate.eq(1).sum()}/{len(d)} 点",
                     fontsize=9, loc="left", color="#303946", pad=6)
    fig.suptitle("真实挂线数据 v4：报告规则重建标注", fontsize=19, y=.987, color="#263649")
    fig.text(.5, .961, "32 点无重叠窗口 · 温湿度与时间基线 · 湿度分层 · 同串时间匹配 · 保留全部观测点",
             ha="center", fontsize=11, color="#566170")
    legend = [Line2D([0], [0], color="#8b96a7", marker=".", markersize=5, lw=.7, label="全部原始观测（实际时间连接）"),
              Line2D([0], [0], color="#ca4c55", marker=".", markersize=5, lw=1, label="候选窗口内观测"),
              Patch(facecolor="#d65b62", alpha=.13, label="候选窗口起止时间"),
              Line2D([0], [0], color="#80333a", marker="x", linestyle="none", label="大跳变点（z ≥ 8）"),
              Line2D([0], [0], color="#545b68", marker="D", markerfacecolor="white", linestyle="none", label="单点尾部：未评估")]
    fig.legend(handles=legend, loc="upper center", bbox_to_anchor=(.5, .951), ncol=5, frameon=False, fontsize=10)
    fig.text(.5, .014, f"{len(data):,} 点 · {len(w)} 窗口 · {int(w.is_candidate.sum())} 个候选窗口；红色为统计候选，不等于确认故障。"
             "各图坐标按实际观测范围独立；长间隔连线仅为视觉连接。", ha="center", fontsize=10, color="#59616c")
    fig.subplots_adjust(left=.045, right=.985, bottom=.045, top=.91, hspace=.60, wspace=.31)
    fig.savefig(OUT / "v4_all_devices.png")
    plt.close(fig)
    detaildir = OUT / "device_plots"
    detaildir.mkdir(exist_ok=True)
    for device in order:
        d, ww = data.loc[data.device_id == device], w.loc[w.device_id == device]
        fig, ax = plt.subplots(figsize=(15, 4.4), dpi=160)
        plot_panel(ax, d, ww, True)
        ax.set_title(f"{device}    候选窗口 {int(ww.is_candidate.sum())}/{len(ww)}", loc="left", fontsize=14, pad=15)
        ax.set_xlabel("实际时间", fontsize=10)
        fig.legend(handles=legend, loc="lower center", ncol=5, frameon=False, fontsize=9)
        fig.subplots_adjust(left=.07, right=.985, bottom=.23, top=.85)
        fig.savefig(detaildir / f"{device}_v4.png")
        plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), dpi=160)
    s = summary.set_index("device_id").loc[order]
    axes[0, 0].bar(np.arange(25), s.candidate_windows, color="#c95b63")
    axes[0, 0].set_xticks(np.arange(25), [x[-4:] for x in order], rotation=60)
    axes[0, 0].set_title("各设备候选窗口数（末四位设备号）")
    axes[0, 0].set_ylabel("窗口数")
    reason_counts = w.loc[w.is_candidate == 1].primary_rule.value_counts().reindex(list("ABCD"), fill_value=0)
    axes[0, 1].bar(list("ABCD"), reason_counts.values, color="#547eae")
    axes[0, 1].set_title("主触发规则（重叠时 B、C、A、D 顺序）")
    axes[0, 1].set_ylabel("窗口数")
    vals = [audit["high_humidity_fraction_matched_points"], audit["top5_raw_peer_high_humidity_fraction"],
            audit["top5_stratified_peer_high_humidity_fraction"]]
    axes[1, 0].bar(["匹配点总体", "|原始 peer 差| top 5%", "|分层 peer z| top 5%"],
                    np.asarray(vals) * 100, color=["#9aa4b3", "#d19164", "#547eae"])
    axes[1, 0].set_ylim(0, 105)
    axes[1, 0].set_ylabel("Hum > 60 点占比（%）")
    axes[1, 0].set_title("高湿混淆审计（同一有效 peer 点总体）")
    for i, v in enumerate(vals):
        axes[1, 0].text(i, v * 100 + 2, f"{v:.1%}", ha="center")
    axes[1, 1].hist(w.duration_hours / 24, bins=24, color="#859ebd", edgecolor="white")
    axes[1, 1].set_title("32 点窗口仍覆盖不同的真实时间长度")
    axes[1, 1].set_xlabel("窗口实际持续天数")
    axes[1, 1].set_ylabel("窗口数")
    for ax in axes.flat:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=.15)
    fig.suptitle("v4 标注审计", fontsize=17)
    fig.tight_layout(rect=[0, .02, 1, .95], h_pad=3)
    fig.savefig(OUT / "v4_audit.png")
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    original_hash = digest(SOURCE)
    data, cleanlog = load()
    data, models = environment(data)
    data, matches, scales = peers(data)
    data, w, poolstats = windows(data)
    summaries = []
    for device, d in data.groupby("device_id"):
        ww = w.loc[w.device_id == device]
        summaries.append(dict(device_id=device, peer_group=d.peer_group.iloc[0], location=d.location.iloc[0],
                              total_points=len(d), total_windows=len(ww),
                              full_windows=int(ww.window_type.eq("full").sum()), tail_windows=int(ww.window_type.eq("tail").sum()),
                              candidate_windows=int(ww.is_candidate.sum()), candidate_window_fraction=float(ww.is_candidate.mean()),
                              candidate_points=int(d.is_candidate.eq(1).sum()), candidate_point_fraction=float(d.is_candidate.eq(1).mean()),
                              first_timestamp=d.Timestamp.min(), last_timestamp=d.Timestamp.max(),
                              peer_complete_fraction=float(d.peer_complete.mean()),
                              max_peer_offset_minutes=float(d.peer_offset_max_minutes.max())))
    summary = pd.DataFrame(summaries)
    matched = data.loc[data.peer_delta.notna() & data.peer_delta_z.notna()].copy()
    topn = max(1, int(np.ceil(len(matched) * .05)))
    rawtop = matched.loc[matched.peer_delta.abs().sort_values(ascending=False, kind="stable").index[:topn]]
    ztop = matched.loc[matched.peer_delta_z.abs().sort_values(ascending=False, kind="stable").index[:topn]]
    audit = dict(
        data_points=len(data), devices=int(data.device_id.nunique()),
        groups=int(data.peer_group.nunique()), windows=len(w),
        full_windows=int(w.window_type.eq("full").sum()), tail_windows=int(w.window_type.eq("tail").sum()),
        candidate_windows=int(w.is_candidate.sum()), candidate_window_fraction=float(w.is_candidate.mean()),
        candidate_points=int(data.is_candidate.eq(1).sum()), candidate_point_fraction=float(data.is_candidate.eq(1).mean()),
        unassessed_points=int(data.is_candidate.eq(-1).sum()),
        unassessed_observations=data.loc[data.is_candidate == -1, ["device_id", "Timestamp", "Measure"]].assign(
            Timestamp=lambda x: x.Timestamp.dt.strftime("%Y-%m-%d %H:%M:%S")).to_dict("records"),
        primary_rule_counts=w.loc[w.is_candidate == 1].primary_rule.value_counts().to_dict(),
        all_rule_counts={k: int(w[f"rule_{k}"].sum()) for k in "ABCD"},
        matched_point_fraction=float(data.peer_delta.notna().mean()),
        complete_peer_panel_fraction=float(data.peer_complete.mean()),
        accepted_peer_matches=int(matches.accepted.sum()), attempted_peer_matches=len(matches),
        rejected_peer_matches=int(matches.accepted.eq(0).sum()),
        max_accepted_offset_minutes=float(matches.loc[matches.accepted == 1, "offset_minutes"].abs().max()),
        high_humidity_fraction_matched_points=float((matched.Hum > 60).mean()),
        top5_count=topn, top5_denominator=len(matched),
        top5_raw_peer_high_humidity_fraction=float((rawtop.Hum > 60).mean()),
        top5_stratified_peer_high_humidity_fraction=float((ztop.Hum > 60).mean()),
        all_window_high_humidity_point_fraction=float(w.high_humidity_fraction.mean()),
        candidate_window_high_humidity_point_fraction=float(w.loc[w.is_candidate == 1, "high_humidity_fraction"].mean()),
        source_sha256=original_hash,
    )
    # Independent structural and label checks. Never assert a target candidate count.
    assert digest(SOURCE) == original_hash
    assert not data.duplicated(["device_id", "Timestamp"]).any()
    assert int(w.n_points.sum()) + audit["unassessed_points"] == len(data)
    assert w.window_id.is_unique
    assert (w.loc[w.window_type == "full", "n_points"] == 32).all()
    assert w.loc[w.window_type == "tail", "n_points"].between(2, 31).all()
    for device, f in data.groupby("device_id"):
        for wid, wf in f.loc[f.window_id != ""].groupby("window_id", sort=False):
            assert np.all(np.diff(wf.point_index_device) == 1)
        for c in ["env_residual_z", "peer_delta_z"]:
            assert not np.isinf(f[c]).any()
    assert data.device_id.nunique() == 25 and data.peer_group.nunique() == 9
    assert (data.n_expected_peers > 0).all()
    assert (matches.device_id != matches.peer_device_id).all()
    accepted = matches.loc[matches.accepted == 1]
    assert (accepted.offset_minutes.abs() <= accepted.tolerance_minutes + 1e-8).all()
    # Recompute from timestamp arithmetic, independent of the integer search representation.
    actual_offset = (matches.nearest_peer_timestamp - matches.Timestamp).dt.total_seconds() / 60
    assert np.allclose(matches.offset_minutes, actual_offset, rtol=0, atol=1e-9)
    actual_accept = (actual_offset.abs() <= matches.tolerance_minutes).astype(int)
    assert actual_accept.equals(matches.accepted)
    observed_times = data.set_index(["device_id", "Timestamp"]).Measure
    matched_values = observed_times.reindex(pd.MultiIndex.from_arrays(
        [accepted.peer_device_id, accepted.nearest_peer_timestamp])).to_numpy()
    assert np.allclose(matched_values, accepted.peer_measure, rtol=0, atol=1e-12)
    expected = ((w.rule_A == 1) | (w.rule_B == 1) | (w.rule_C == 1) | (w.rule_D == 1)).astype(int)
    assert expected.equals(w.is_candidate)
    assert (data.groupby("window_id").is_candidate.nunique() == 1).all()
    assert int(w.loc[w.is_candidate == 1, "n_points"].sum()) == audit["candidate_points"]
    protocol = dict(
        version="v4_report_reconstruction_20261102_time_unit_fixed", exact_original_code_reproduction=False,
        source=str(SOURCE), source_sha256=original_hash,
        report=str(REPORT) if REPORT else None, report_sha256=digest(REPORT) if REPORT else None,
        metadata_source=str(LOCATIONS), metadata_section="explicit location mapping only",
        window_size=32, tail_min_points=2, huber_epsilon=1.5, huber_alpha=1e-4,
        humidity_edges=EDGES, peer_group_mode="intra-string", local_interval_rolling_points=LOCAL_INTERVAL_POINTS,
        thresholds=dict(window_evidence_z=3, residual_abs_z=2, residual_fraction=.15,
                        peer_jump_z=8, peer_jump_min_count=2, extreme_peer_jump_z=12, residual_jump_z=2.5),
        implementation_decisions=ASSUMPTIONS, pooled_window_normalization=poolstats,
        device_environment_models=models, device_jump_scales=scales,
        python=sys.version, versions=dict(numpy=np.__version__, pandas=pd.__version__,
                                         scipy=scipy.__version__, sklearn=sklearn.__version__),
        report_reference_counts=dict(points_not_stated=True, windows=547, full=524, tail=23, candidate=38),
        validation="All structural, source-hash, timestamp-tolerance, no-self-peer and label-mapping assertions passed.",
        audit=audit,
    )
    for name, frame in [
        ("device_points_v4.csv", data), ("window_features_v4.csv", w),
        ("candidate_windows_v4.csv", w.loc[w.is_candidate == 1].sort_values("candidate_rank")),
        ("device_summary_v4.csv", summary), ("peer_matches_v4.csv", matches), ("source_cleaning_audit.csv", cleanlog),
        ("peer_jump_events_v4.csv", data.loc[data.peer_delta_jump_z >= 8]),
    ]:
        frame.to_csv(OUT / name, index=False, encoding="utf-8-sig", date_format="%Y-%m-%d %H:%M:%S")
    (OUT / "protocol_v4.json").write_text(json.dumps(protocol, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    (OUT / "audit_v4.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    # Original report workbook contains windows and summary; full point evidence is in CSV.
    preferred = ["device_id", "window_id", "start", "end", "n_points", "window_type",
                 "is_candidate", "triggered_rules", "reason_cn", "candidate_rank", "candidate_score",
                 "duration_hours", "median_interval_hours", "hum_mean", "high_humidity_fraction",
                 "residual_z_mean", "residual_z_mean_z", "abs2_residual_fraction",
                 "peer_z_mean", "peer_z_mean_z", "same_direction", "peer_jump_count8",
                 "peer_jump_count12", "peer_jump_max", "rule_A", "rule_B", "rule_C", "rule_D",
                 "peer_matched_fraction", "peer_complete_fraction", "max_accepted_peer_offset_minutes",
                 "measure_mean", "residual_z_abs_q90", "residual_jump_count2_5", "has_peer",
                 "measure_slope_per_hour", "residual_z_slope_per_hour", "peer_z_slope_per_hour",
                 "peer_group", "location"]
    payload = dict(audit=audit, summary=frame_matrix(summary), windows=frame_matrix(w[preferred]),
                   assumptions=ASSUMPTIONS)
    (OUT / "workbook_data.json").write_text(json.dumps(payload, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    if not SKIP_PLOTS:
        figures(data, w, summary, audit)
    decision_lines = "\n".join(f"- **{k}**：{v}" for k, v in ASSUMPTIONS)
    readme = f"""# 真实挂线数据 v4 标注（报告规则重建）

## 结果

- 时间匹配修正版：上一版将微秒整数当作纳秒换算，导致偏差小估1000倍；本版显式统一为纳秒，并用原始时间相减独立验证。请停止使用时间单位错误版标签与匹配统计。
- 原始有效观测：{len(data):,} 点，25 台设备，9 个同串组。
- 窗口：{len(w)} 个，完整窗口 {audit['full_windows']}，尾窗口 {audit['tail_windows']}；未评估 {audit['unassessed_points']} 点。
- 候选：{audit['candidate_windows']} 个窗口（{audit['candidate_window_fraction']:.2%}），映射为 {audit['candidate_points']:,} 个候选点（{audit['candidate_point_fraction']:.2%}）。
- 主原因：{audit['primary_rule_counts']}。所有重叠触发标志均保留，不靠原因覆盖来决定标签。
- 报告给出的对照为 547 个窗口、38 个候选。本次候选数与其差值为 {audit['candidate_windows'] - 38:+d}。没有为匹配该数字而调节任何阈值。

**本地未找到报告的原始 v4 脚本和 protocol。这里实现了报告全部明确的流程与阈值，但其未明示的实现细节必须补充约定。因此本版不能称为原始脚本的完全一致复现。**

## 看图

`v4_all_devices.png`：25 台完整时间序列，实际时间作横轴，所有观测点连接。蓝点为原始观测；红点/红线为候选窗口内原始观测；浅红底为候选窗口起止范围；深红叉表示该点 peer 不同步跳变 z≥8。红线不是阈值线，原始 Measure 未修改。

候选窗口内所有点都标 1，即使只有其中几个点触发跳变；这正是报告的窗口转点级标签规则。0 仅表示未触发本规则，不是确定正常。-1 表示单点尾部未评估，图中用灰色空心菱形表示。分窗按 32 个观测点，窗口真实时长可不同。各图时间和纵轴按实际观测范围独立。长间隔连线只是连接观测，不表示中间有观测或插值。

两条单点尾部为设备 863644079730629 的 2026-09-20 11:00:28，以及设备 863644079730694 的 2026-09-20 11:00:29；全部原始有效点均保留，16983 点进入报告规定窗口、2 点未评估。报告声称逐点覆盖且零丢点，但未说明“仅剩1点”的处理，不能把这两点擅自当正常。

`device_plots` 内有 25 张单设备高清图。`v4_audit.png` 展示设备数量、原因及高湿混淆审计。

## 报告明确规则

1. 32 点、无重叠、按设备时间排序；剩余≥2点补 tail。保留真实时间戳，窗口斜率以小时归一化。
2. 原始 Measure 对 Temp、Hum、elapsed 拟合逐设备 Huber 环境基线，epsilon=1.5，alpha=1e-4。
3. 先逐设备中位数/MAD，再湿度分箱正态分位标准化；边界 {EDGES}。
4. 同串多设备；逐时间最近匹配；容差=0.5×局部中位间隔，裁剪30分钟～12小时；多个 peer 取中位数；group×湿度箱标准化。
5. 不同步跳变采用 peer_delta 一阶差分绝对值 / sqrt(2) / 本设备 peer_delta 水平量稳健尺度。
6. 窗口 A：residual_z_mean_z≥3，abs2_residual_fraction≥0.15，peer_z_mean_z≥3，两个原始均值同号。
7. 窗口 B：peer_delta_jump_z≥8 的点数≥2。
8. 窗口 C：任何一点 peer_delta_jump_z≥12。
9. 窗口 D：无物理 peer，abs2_residual_fraction≥0.15，residual_z_abs_q90≥2，残差跳变≥2.5的点数≥2。
10. A/B/C/D 任一满足即候选；无每设备最低数量或最多一半等配额。

## 真实时间戳如何参与

环境基线采用设备首点起的 elapsed_days；每个窗口记录实际 duration_hours 和采样间隔；斜率除以真实小时。每条 peer 匹配都记录目标时间、peer 时间、允许容差和实际偏差，超过容差的最近点被拒绝，不插值。该自适应容差是报告规定，最宽可到12小时，并不意味着两点严格同时。差分记录 previous_timestamp 和 interval_hours，可复核跨长间隔跳变。

本版共尝试 {audit['attempted_peer_matches']:,} 条设备对匹配，接受 {audit['accepted_peer_matches']:,} 条，拒绝 {audit['rejected_peer_matches']:,} 条。没有可用 peer 的时刻保留缺失；不补零，也不跨缺失计算跳变。物理同串仍存在，故不因某时刻时间匹配失败就转用 D 规则。

## 文件

- `device_nonoverlap_windows_v4.xlsx`：设备汇总、全部窗口及其证据、方法边界。
- `device_points_v4.csv`：全部原始观测、源工作表行号、真实时间戳、点级证据、窗口 ID 和最终标签。
- `candidate_windows_v4.csv`：仅候选窗口，按描述性分数排序，含全部触发标志。
- `peer_matches_v4.csv`：逐设备逐 peer 最近匹配及拒绝记录。
- `peer_jump_events_v4.csv`：直接触发 z≥8 的点，便于区分触发点与整窗标签。
- `protocol_v4.json`：参数、补充约定、Huber模型、尺度、依赖版本和输入 SHA256。
- `audit_v4.json` / `source_cleaning_audit.csv`：结果及源数据清洗核对。

## 报告未写明处的补充约定

{decision_lines}

## 验证与用途

已验证源文件哈希未变、每个保留观测唯一、窗口不重叠、窗口与点级标签一致、25台均有物理peer、9组、无自身比较、全部接受匹配满足逐点容差、缺测没有补零参与标注。分位标化和基线用全历史拟合，标签仅用于离线探索/伪标签研究；若进行时间外推评估，需在训练期拟合转换并应用到测试期，避免信息泄漏。

高湿混淆审计：有效 peer 点总体 Hum>60 占比 {audit['high_humidity_fraction_matched_points']:.2%}；|原始peer差| top5% 中为 {audit['top5_raw_peer_high_humidity_fraction']:.2%}；|分层peer z| top5% 中为 {audit['top5_stratified_peer_high_humidity_fraction']:.2%}。top5% 用固定数量 {topn}/{len(matched)}，并列时按稳定原始顺序选取，未为复现报告审计数字而调整。
"""
    (OUT / "说明_v4.md").write_text(readme, encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path, help="Private field-data XLSX")
    parser.add_argument("--locations", required=True, type=Path, help="Private device-location JSON")
    parser.add_argument("--report", type=Path, help="Optional private report PDF, hashed for provenance")
    parser.add_argument("--output", type=Path, default=OUT)
    parser.add_argument("--skip-plots", action="store_true")
    args = parser.parse_args()
    SOURCE, LOCATIONS, REPORT, OUT = args.source, args.locations, args.report, args.output
    SKIP_PLOTS = args.skip_plots
    for path in (SOURCE, LOCATIONS, REPORT):
        if path is not None and not path.is_file():
            parser.error(f"Input file not found: {path}")
    main()
