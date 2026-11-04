"""Plot independent generated v4 windows at their SOURCE timestamps, not a synthetic timeline.

No training imports or pickle loading are needed. Numeric source_ids are resolved
against window_metadata_v4.json and device-local source_rows. Fail fast when the
real dataset, timestamps, labels, or generated provenance are inconsistent.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DEFAULT_RUN = HERE / "runs/device_data_110_220_paper_aug_v4evi_20261009"
BLUE, RED, GRAY, PINK = "#2563eb", "#dc2626", "#64748b", "#fecaca"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def integer_array(values, name: str) -> np.ndarray:
    raw = np.asarray(values)
    require(np.issubdtype(raw.dtype, np.integer), f"{name} must contain integers")
    return raw.astype(np.int64)


@dataclass
class Context:
    protocol: dict
    meta: pd.DataFrame
    points: dict[str, pd.DataFrame]
    candidates: dict[str, pd.DataFrame]
    times: np.ndarray
    real: np.ndarray


def load_context(run: Path) -> Context:
    protocol = json.loads((run / "protocol.json").read_text(encoding="utf-8"))
    require(protocol["window_length"] == 4, "Expected length-4 v4 generation")
    require(protocol["features"][0] == "Measure", "Channel 0 must be Measure")
    # json.loads preserves string IMEIs; read_json may infer integers and lose leading zeros.
    meta = pd.DataFrame(json.loads((run / "window_metadata_v4.json").read_text(encoding="utf-8")))
    require(len(meta) > 0, "Real window metadata is empty")
    meta["IMEI"] = meta["IMEI"].astype(str)
    for col in ("start_time", "end_time"):
        meta[col] = pd.to_datetime(meta[col], errors="raise")
    points = pd.read_csv(run / "annotated_device_points_v4.csv", dtype={"IMEI": str})
    points["Timestamp"] = pd.to_datetime(points["Timestamp"], errors="raise")
    require(points.Timestamp.notna().all(), "Missing point timestamp")
    require(np.isfinite(points.Measure).all(), "Nonfinite real Measure")
    require(points.v4_point_label.isin([0, 1]).all(), "Invalid real point labels")
    grouped = {str(k): g.sort_values("Timestamp").reset_index(drop=True)
               for k, g in points.groupby("IMEI", sort=True)}
    candidates = pd.read_csv(run / "v4_window_labels.csv", dtype={"IMEI": str})
    for col in ("start_time", "end_time"):
        candidates[col] = pd.to_datetime(candidates[col], errors="raise")
    require(candidates.window_anomaly_v4.isin([0, 1]).all(), "Invalid candidate labels")
    require(set(candidates.IMEI) <= set(grouped), "Candidate table contains unknown devices")
    for imei, p in grouped.items():
        inherited = np.zeros(len(p), dtype=int)
        for row in candidates[candidates.IMEI == imei].itertuples():
            require(isinstance(row.start_row, (int, np.integer)) and isinstance(row.end_row, (int, np.integer)),
                    "Candidate row indices must be integers")
            require(0 <= row.start_row <= row.end_row < len(p), "Candidate row indices out of bounds")
            require(row.start_time == p.Timestamp.iloc[row.start_row] and row.end_time == p.Timestamp.iloc[row.end_row],
                    "Candidate interval timestamps do not match source points")
            if row.window_anomaly_v4:
                inherited[row.start_row:row.end_row + 1] = 1
        require(np.array_equal(inherited, p.v4_point_label.to_numpy()), "Candidate/point labels mismatch")
    bad = candidates[candidates.window_anomaly_v4 == 1]
    candidate_groups = {str(k): g for k, g in bad.groupby("IMEI", sort=False)}
    with np.load(run / "window_dataset_v4.npz", allow_pickle=False) as z:
        real, labels = z["x"], z["y"]
    expected_shape = (len(meta), 4, len(protocol["features"]))
    require(real.shape == expected_shape, f"Real dataset shape must be {expected_shape}")
    require(np.array_equal(labels, meta.label.to_numpy()), "Real window labels do not match metadata")
    require(meta.label.isin([0, 1]).all(), "Invalid metadata labels")
    times = np.empty((len(meta), 4), dtype="datetime64[ns]")
    for imei, subset in meta.groupby("IMEI", sort=False):
        require(imei in grouped, f"Unknown source device {imei}")
        rows = integer_array(subset.source_rows.tolist(), "source_rows")
        p = grouped[imei]
        require(rows.shape == (len(subset), 4), "source_rows must have exactly four indices")
        require(((rows >= 0) & (rows < len(p))).all(), "source_rows out of bounds")
        require((np.diff(rows, axis=1) == 1).all(), "Source windows must be consecutive rows")
        ids = subset.index.to_numpy()
        times[ids] = p.Timestamp.to_numpy(dtype="datetime64[ns]")[rows]
        require(np.allclose(real[ids, :, 0], p.Measure.to_numpy()[rows], rtol=1e-6, atol=1e-8),
                f"Real Measure/source_rows mismatch for {imei}")
        inherited = p.v4_point_label.to_numpy()[rows].max(axis=1)
        require(np.array_equal(inherited, subset.label.to_numpy()), f"Point/window labels mismatch for {imei}")
    require(np.array_equal(times[:, 0], meta.start_time.to_numpy(dtype="datetime64[ns]")),
            "Source start timestamps do not match source_rows")
    require(np.array_equal(times[:, -1], meta.end_time.to_numpy(dtype="datetime64[ns]")),
            "Source end timestamps do not match source_rows")
    return Context(protocol, meta, grouped, candidate_groups, times, real)


def fit_ids(meta: pd.DataFrame, fraction: float) -> np.ndarray:
    """Exact chronological fit portion used by paper.split_ratio, without training imports."""
    result = []
    for _, group in meta.groupby("IMEI", sort=False):
        ids = group.index.to_numpy()
        cut = max(1, min(len(ids) - 1, int(len(ids) * fraction))) if len(ids) > 1 else 1
        train = ids[:cut]
        result.extend(train[:max(1, int(len(train) * .8))] if len(train) > 4 else train)
    return np.asarray(result, dtype=np.int64)


def load_generated(path: Path, ctx: Context, ratio: str):
    # Object source_* arrays are deliberately not accessed: no pickle is needed.
    with np.load(path, allow_pickle=False) as z:
        values = z["windows"]
        labels = integer_array(z["labels"], "generated labels")
        sources = integer_array(z["source_ids"], "source_ids")
    require(values.ndim == 3 and values.shape[1:] == ctx.real.shape[1:], "Invalid generated window shape")
    n = len(values)
    require(labels.shape == sources.shape == (n,), "Generated array lengths differ")
    require(((sources >= 0) & (sources < len(ctx.meta))).all(), "source_ids out of bounds")
    require(np.isin(labels, [0, 1]).all(), "Invalid generated labels")
    require(np.isfinite(values).all(), "Generated data contains nonfinite values; refusing to hide them")
    require(np.array_equal(labels, ctx.meta.label.to_numpy()[sources]), "Generated/source label mismatch")
    require(np.isin(sources, fit_ids(ctx.meta, ctx.protocol["ratios"][ratio])).all(),
            "Generated source outside the fit/training split")
    return values[:, :, 0], labels, sources


def select_windows(ids, labels, sources, cap: int) -> np.ndarray:
    """Deterministic time-spread selection per device AND class; 0 means all."""
    require(cap >= 0, "Window cap must be >=0")
    selected = []
    for label in (0, 1):
        pool = np.asarray(ids, dtype=int)[labels[ids] == label]
        pool = pool[np.argsort(sources[pool], kind="stable")]
        if cap and len(pool) > cap:
            pool = pool[np.linspace(0, len(pool) - 1, cap, dtype=int)]
        selected.extend(pool.tolist())
    return np.asarray(selected, dtype=int)


def merge_intervals(intervals) -> list[tuple[float, float]]:
    """Union overlapping source intervals; duplicates never deepen the shading."""
    merged = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
        else:
            merged.append((start, end))
    return merged


def add_windows(ax, times, values, color: str) -> LineCollection:
    segments = np.stack([times, values], axis=-1)
    collection = LineCollection(segments, colors=color, linewidths=.65, alpha=.6, zorder=3)
    ax.add_collection(collection)
    ax.scatter(times.ravel(), values.ravel(), color=color, s=3, alpha=.55, zorder=3)
    return collection


def style_axis(ax, local=False) -> None:
    locator = mdates.AutoDateLocator(minticks=2, maxticks=8, interval_multiples=False)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator) if local else mdates.DateFormatter("%Y-%m-%d"))
    ax.xaxis.get_offset_text().set_fontsize(6)
    ax.tick_params(axis="x", labelrotation=30, labelsize=6)
    ax.tick_params(axis="y", labelsize=7)
    ax.set_ylabel("Measure", fontsize=7)
    ax.grid(alpha=.2)


def set_limits(ax, times, values, bottom_room=False) -> None:
    # Explicit limits include ALL plotted negatives/extremes, including collections.
    lo, hi = float(np.min(values)), float(np.max(values))
    margin = max((hi - lo) * .08, .01)
    ax.set_ylim(lo - margin * (2 if bottom_room else 1), hi + margin)
    start, end = float(np.min(times)), float(np.max(times))
    padding = max((end - start) * .025, 1 / 24)
    ax.set_xlim(start - padding, end + padding)
    if lo <= 0 <= hi:
        ax.axhline(0, color=GRAY, linestyle="--", linewidth=.5, alpha=.5)


def finish_grid(fig, axes, n, title, handles, footer, path, dpi):
    for ax in axes[n:]:
        ax.axis("off")
    fig.suptitle(title, fontsize=15, y=.995)
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, .973),
               ncol=len(handles), frameon=False, fontsize=9)
    fig.text(.5, .015, footer, ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .045, 1, .94))
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_grids(ctx, values, labels, sources, title, output, cap, dpi, details):
    devices = list(ctx.points)
    ncols, nrows = 5, int(np.ceil(len(devices) / 5))
    generated_devices = ctx.meta.IMEI.to_numpy()[sources]
    dates = mdates.date2num(ctx.times[sources])
    rows = []
    selections = {}
    for imei in devices:
        ids = np.flatnonzero(generated_devices == imei)
        selected = select_windows(ids, labels, sources, cap)
        selections[imei] = (ids, selected)
        unique_sources = np.unique(sources[ids])
        covered = (np.unique(np.concatenate(ctx.meta.iloc[unique_sources].source_rows.to_list()))
                   if len(unique_sources) else np.array([], dtype=int))
        measure = values[ids]
        rows.append({
            "IMEI": imei, "generated_windows": len(ids), "positive_windows": int(labels[ids].sum()),
            "unique_sources": len(unique_sources), "plotted_overlay_windows": len(selected),
            "plotted_overlay_positive": int(labels[selected].sum()),
            "covered_real_points": len(covered), "real_points": len(ctx.points[imei]),
            "point_coverage_fraction": len(covered) / len(ctx.points[imei]),
            "source_start_time": str(ctx.times[sources[ids]].min()) if len(ids) else "",
            "source_end_time": str(ctx.times[sources[ids]].max()) if len(ids) else "",
            "nonpositive_measure_points": int((measure <= 0).sum()), "measure_points": measure.size,
            "nonpositive_fraction": float((measure <= 0).mean()) if measure.size else 0.,
        })
    for kind in ("real_vs_generated", "anomaly_localization"):
        fig, axes = plt.subplots(nrows, ncols, figsize=(23, 3.5 * nrows), squeeze=False)
        axes = axes.ravel()
        for ax, imei, stats in zip(axes, devices, rows):
            p = ctx.points[imei]
            real_times = mdates.date2num(p.Timestamp.to_numpy())
            ax.plot(real_times, p.Measure, color=GRAY, linewidth=.7, alpha=.7, zorder=2)
            for candidate in ctx.candidates.get(imei, pd.DataFrame()).itertuples():
                ax.axvspan(mdates.date2num(candidate.start_time), mdates.date2num(candidate.end_time),
                           color=PINK, alpha=.35, linewidth=0, zorder=0)
            ids, selected = selections[imei]
            if kind == "real_vs_generated":
                for label, color in ((0, BLUE), (1, RED)):
                    sel = selected[labels[selected] == label]
                    if len(sel):
                        add_windows(ax, dates[sel], values[sel], color)
                displayed = selected
                note = f"gen={len(ids)}, += {stats['positive_windows']}; shown={len(selected)}"
            else:
                # All positive generated curves, even when overlay sampling is requested.
                positive = ids[labels[ids] == 1]
                if len(positive):
                    add_windows(ax, dates[positive], values[positive], RED)
                for label, color, y in ((0, BLUE, .025), (1, RED, .075)):
                    label_ids = ids[labels[ids] == label]
                    intervals = merge_intervals([(dates[j, 0], dates[j, -1]) for j in label_ids])
                    if intervals:
                        rug = LineCollection([[(a, y), (b, y)] for a, b in intervals],
                                             colors=color, linewidths=3,
                                             transform=ax.get_xaxis_transform(), zorder=4)
                        ax.add_collection(rug, autolim=False)
                        ax.scatter(dates[label_ids, 0], np.full(len(label_ids), y),
                                   transform=ax.get_xaxis_transform(), color=color, s=3, zorder=4)
                displayed = positive
                note = f"gen+= {len(positive)}; unique anchors={stats['unique_sources']}"
            if not len(ids):
                ax.text(.5, .5, "No generated windows", transform=ax.transAxes, ha="center", fontsize=8)
            elif kind == "anomaly_localization" and not len(displayed):
                ax.text(.5, .9, "No positive generated windows", transform=ax.transAxes, ha="center", fontsize=7)
            all_values = np.r_[p.Measure.to_numpy(), values[displayed].ravel()]
            set_limits(ax, real_times, all_values, bottom_room=kind == "anomaly_localization")
            ax.set_title(f"{imei}\n{note}", fontsize=8)
            style_axis(ax)
        handles = [Line2D([], [], color=GRAY, label="Real Measure"),
                   Line2D([], [], color=BLUE, label="Generated label=0" if kind == "real_vs_generated" else "Normal anchor rug (all)"),
                   Line2D([], [], color=RED, label="Generated label=1 (inherited)"),
                   Patch(facecolor=PINK, label="Real v4 candidate interval")]
        footer = ("SOURCE timestamps only; independent 4-point windows, never joined across samples. "
                  "Labels are inherited, not re-detected. Negatives are retained.\n" +
                  ("Overlay: all windows" if cap == 0 else f"Overlay cap: {cap} per device per class; deterministic time-spread sampling")
                  if kind == "real_vs_generated" else
                  "SOURCE timestamps only; ALL positive generated windows shown. Bottom rugs locate all normal/positive anchors.\n"
                  "No synthetic continuous timeline; no anchor coverage outside the fit split; labels are inherited, not re-detected.")
        finish_grid(fig, axes, len(devices), f"{title} | {kind.replace('_', ' ')}", handles, footer,
                    output / f"{kind}_by_device.png", dpi)
    if details:
        fig, axes = plt.subplots(nrows, ncols, figsize=(23, 3.5 * nrows), squeeze=False)
        axes = axes.ravel()
        for ax, imei in zip(axes, devices):
            ids, _ = selections[imei]
            if not len(ids):
                ax.text(.5, .5, "No generated windows", transform=ax.transAxes, ha="center")
                ax.set_title(imei, fontsize=8)
                continue
            positive = ids[labels[ids] == 1]
            pool = positive if len(positive) else ids
            pool = pool[np.argsort(sources[pool], kind="stable")]
            j = pool[len(pool) // 2]
            sid, label = sources[j], labels[j]
            ax.plot(dates[j], ctx.real[sid, :, 0], "o-", color=GRAY, markersize=4, label="Real source window")
            ax.plot(dates[j], values[j], "o-", color=RED if label else BLUE, markersize=4)
            set_limits(ax, dates[j], np.r_[ctx.real[sid, :, 0], values[j]])
            ax.set_title(f"{imei}\nsample={j}, source={sid}, inherited label={label}", fontsize=8)
            style_axis(ax, local=True)
        finish_grid(fig, axes, len(devices), f"{title} | local 4-point source comparison",
                    [Line2D([], [], color=GRAY, marker="o", label="Real source window"),
                     Line2D([], [], color=BLUE, marker="o", label="Generated label=0"),
                     Line2D([], [], color=RED, marker="o", label="Generated label=1")],
                    "One source-time-median sample per device; prefer positive class when available. Not a quality ranking.\n"
                    "Exact four source timestamps; generators may retain observed context points; labels are inherited.",
                    output / "local_window_comparison_by_device.png", dpi)
    return rows


def write_report(output, summaries, audit, args, protocol):
    pd.DataFrame(summaries).to_csv(output / "configuration_summary.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(audit).to_csv(output / "device_generation_audit.csv", index=False, encoding="utf-8-sig")
    (output / "plot_protocol.json").write_text(json.dumps({
        "run": str(args.run.resolve()), "feature_set": protocol.get("feature_set", "base"),
        "configurations": len(summaries), "ratios": args.ratios, "methods": args.methods, "seeds": args.seeds,
        "dpi": args.dpi, "max_windows_per_device_class": args.max_windows_per_class,
        "details": args.details, "timestamp_semantics": "source/anchor timestamps, not generated timestamps",
        "label_semantics": "inherited generation class, not independently re-detected v4 anomalies",
        "nonpositive_values": "retained unchanged", "validation": "real values, source rows/times, labels, fit membership, finite arrays",
        "versions": {"numpy": np.__version__, "pandas": pd.__version__, "matplotlib": matplotlib.__version__},
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# v4 生成数据可视化（来源时间映射）", "",
             f"输入实验：`{args.run.resolve()}`；特征配置：`{protocol.get('feature_set', 'base')}`。", "",
             f"本次绘制 **{len(summaries)} 个配置**，种子 {args.seeds}；每个配置覆盖全部真实设备（无生成样本也保留面板）。", "",
             "## 图的含义", "",
             "- `real_vs_generated_by_device.png`：灰色真实 Measure；蓝色生成正常类、红色生成异常类；浅红背景是原始 v4 候选区间。",
             "- `anomaly_localization_by_device.png`：灰色真实曲线与全部红色正类生成窗口；底部蓝/红时间条分别表示正常/正类来源窗口覆盖。重复、重叠时间条取并集，统计仍计入全部样本。",
             "- `local_window_comparison_by_device.png`（启用 `--details` 时）：每设备优先选一个正类窗口，无正类则选正常；按 source_id 排序取中间样本，与其真实来源四点直接对比。不是挑选最佳效果。", "",
             "## 必须注意", "",
             "1. **生成数据是独立的长度 4 窗口，不是连续设备时间序列。** 每个窗口单独连线；横轴使用 `source_ids → source_rows → 原始 Timestamp` 的四个精确来源时间，不用等间隔插值。",
             "2. **红色标签是生成时继承/赋予的正类标签，不是对生成值重新执行 v4 判据的检测结果。** 来源设备是锚点设备；peer_group 生成不代表设备级独立建模，也不保证生成设备身份一致。",
             "3. 来源仅位于每设备时间顺序训练部分的 fit 子集（通常为训练段前 80%）；后续时间没有生成覆盖，不能解释为后续时间无异常。",
             "4. 生成值保留原始逆缩放值，包括 `Measure <= 0`；坐标范围包含全部绘制值，不做截断、正值修正或质量筛选。统计以窗口内点计数，重叠来源不去重。",
             "5. 一些条件生成器可能保留观测上下文（例如 CSDI 的部分来源点），不应将四点全部视为新生成点。v4 标签本身也只是统计候选，不是确认故障。",
             f"6. 对比图每设备每类上限：{args.max_windows_per_class}（0 表示全部）。定位图始终展示全部正类窗口；CSV 提供实际绘制数量及来源点覆盖率。", "",
             "## 校验与可追溯统计", "",
             "脚本校验真实 Measure 与来源行一致、点/窗口标签一致、四个来源时间与元数据起止一致、生成标签与来源一致、source_ids 属于 fit 子集、所有生成通道为有限值；异常数据会报错，不静默忽略。", "",
             "- `configuration_summary.csv`：各配置窗口数、正类数、非正 Measure 比例与图路径。",
             "- `device_generation_audit.csv`：每设备数量、独立来源数、绘制数量、来源覆盖、非正值统计。",
             "- `plot_protocol.json`：绘图参数、软件版本和语义约定。", "",
             "## 图索引", "", "| 比例（训练:测试） | 方法 | seed | 生成窗口 | 正类窗口 | 非正 Measure | 曲线对比 | 异常定位 |",
             "|---|---|---:|---:|---:|---:|---|---|"]
    for row in summaries:
        lines.append(f"| {row['ratio'].replace('to', ':')} | {row['method']} | {row['seed']} | {row['generated_windows']} | {row['positive_windows']} | {row['nonpositive_fraction']:.2%} | [图]({row['overlay']}) | [图]({row['localization']}) |")
    first = summaries[0]
    lines.extend(["", "## 首个配置预览", "", f"![真实与生成曲线]({first['overlay']})", "",
                  f"![异常窗口定位]({first['localization']})", "", "## 复现", "", "```bash",
                  "/home/test/miniconda3/envs/jyz_paper/bin/python plot_v4_generated_data.py \\",
                  f"  --run {args.run} \\", f"  --ratios {' '.join(args.ratios)} --methods {' '.join(args.methods)} \\",
                  f"  --seeds {' '.join(map(str, args.seeds))} --dpi {args.dpi} --max-windows-per-class {args.max_windows_per_class}" + (" --details" if args.details else ""),
                  "```", "", "绘图依赖参考：", "",
                  "- https://matplotlib.org/stable/gallery/shapes_and_collections/line_collection.html",
                  "- https://matplotlib.org/stable/api/dates_api.html#matplotlib.dates.date2num",
                  "- https://numpy.org/doc/stable/reference/generated/numpy.load.html",
                  "- https://pandas.pydata.org/docs/reference/api/pandas.read_csv.html", ""])
    (output / "GENERATED_DATA_VISUALIZATION.md").write_text("\n".join(lines), encoding="utf-8")


def main(args):
    require(args.dpi > 0 and args.max_windows_per_class >= 0, "Invalid DPI or window cap")
    ctx = load_context(args.run)
    args.ratios = list(dict.fromkeys(args.ratios or ctx.protocol["ratios"]))
    args.methods = list(dict.fromkeys(args.methods or ctx.protocol["methods"]))
    args.seeds = list(dict.fromkeys(args.seeds))
    require(set(args.ratios) <= set(ctx.protocol["ratios"]), "Unknown ratio")
    require(set(args.methods) <= set(ctx.protocol["methods"]), "Unknown method")
    require(set(args.seeds) <= set(ctx.protocol["seeds"]), "Seed absent from experiment protocol")
    jobs = []
    for ratio in args.ratios:
        for method in args.methods:
            safe = method.replace("/", "_").replace("-", "_")
            for seed in args.seeds:
                path = args.run / f"raw_{ratio}_{safe}_seed{seed}.npz"
                require(path.is_file(), f"Missing generated file {path}")
                jobs.append((ratio, method, safe, seed, path))
    output = args.output or args.run / "figures/generated_data"
    output.mkdir(parents=True, exist_ok=True)
    summaries, audit = [], []
    for i, (ratio, method, safe, seed, path) in enumerate(jobs, 1):
        values, labels, sources = load_generated(path, ctx, ratio)
        folder = Path(ratio) / f"{safe}_seed{seed}"
        destination = output / folder
        destination.mkdir(parents=True, exist_ok=True)
        title = f"v4 {ctx.protocol.get('feature_set', 'base')} | {method} | train:test={ratio.replace('to', ':')} | seed={seed}"
        rows = plot_grids(ctx, values, labels, sources, title, destination,
                          args.max_windows_per_class, args.dpi, args.details)
        audit.extend([dict(ratio=ratio, method=method, seed=seed, **row) for row in rows])
        summaries.append({"ratio": ratio, "method": method, "seed": seed,
                          "generated_windows": len(values), "positive_windows": int(labels.sum()),
                          "nonpositive_measure_points": int((values <= 0).sum()), "measure_points": values.size,
                          "nonpositive_fraction": float((values <= 0).mean()) if values.size else 0.,
                          "overlay": (folder / "real_vs_generated_by_device.png").as_posix(),
                          "localization": (folder / "anomaly_localization_by_device.png").as_posix()})
        print(f"[{i}/{len(jobs)}] {ratio} {method} seed={seed}: {len(values)} windows, +={labels.sum()}", flush=True)
    write_report(output, summaries, audit, args, ctx.protocol)
    print("Wrote", output, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--ratios", nargs="+")
    parser.add_argument("--methods", nargs="+")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--max-windows-per-class", type=int, default=0,
                        help="Overlay cap per device per class; 0=all; localization always uses all positives")
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument("--details", action="store_true", help="Also plot one local four-point pair per device")
    main(parser.parse_args())
