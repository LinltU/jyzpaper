# -*- coding: utf-8 -*-
"""Independent timestamp audit and a direct two-device peer comparison."""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", required=True, type=Path)
parser.add_argument("--previous-windows", type=Path, help="Optional CSV from an earlier run")
args = parser.parse_args()
out = args.output
d = pd.read_csv(out / "device_points_v4.csv", dtype={"device_id": str}, low_memory=False)
m = pd.read_csv(out / "peer_matches_v4.csv", dtype={"device_id": str, "peer_device_id": str})
w = pd.read_csv(out / "window_features_v4.csv", dtype={"device_id": str})
prev = pd.read_csv(args.previous_windows) if args.previous_windows else None
d["Timestamp"] = pd.to_datetime(d.Timestamp)
actual = (pd.to_datetime(m.nearest_peer_timestamp) - pd.to_datetime(m.Timestamp)).dt.total_seconds() / 60
assert np.allclose(actual, m.offset_minutes, atol=1e-9, rtol=0)
assert np.array_equal((actual.abs() <= m.tolerance_minutes).astype(int), m.accepted)
assert np.all(m.loc[m.accepted == 0, "peer_measure"].isna())
lookup = d.set_index(["device_id", "Timestamp"]).Measure
a = m.loc[m.accepted == 1]
values = lookup.reindex(pd.MultiIndex.from_arrays([a.peer_device_id, pd.to_datetime(a.nearest_peer_timestamp)]))
assert np.allclose(values, a.peer_measure, atol=1e-12, rtol=0)

ids = ["863644079748878", "863644079759867"]
stats = []
for device in ids:
    f, g, v = d.loc[d.device_id == device], m.loc[m.device_id == device], w.loc[w.device_id == device]
    assert set(g.peer_device_id) == set(ids) - {device}
    stats.append(dict(
        device_id=device, location=f.location.iloc[0], peer_device_id=g.peer_device_id.iloc[0],
        observed_points=len(f), matched_points=int(g.accepted.sum()),
        rejected_points=int(g.accepted.eq(0).sum()), matched_fraction=float(g.accepted.mean()),
        median_accepted_offset_seconds=float(g.loc[g.accepted == 1, "offset_minutes"].abs().median() * 60),
        maximum_accepted_offset_minutes=float(g.loc[g.accepted == 1, "offset_minutes"].abs().max()),
        peer_delta_median=float(f.peer_delta.median()), peer_jump_z_max=float(f.peer_delta_jump_z.max()),
        count_jump_z_ge8=int(f.peer_delta_jump_z.ge(8).sum()),
        window_peer_z_mean_z_max=float(v.peer_z_mean_z.max()), windows=len(v),
        candidate_windows=int(v.is_candidate.sum()),
    ))
pd.DataFrame(stats).to_csv(out / "110kv_peer_summary.csv", index=False, encoding="utf-8-sig")
m.loc[m.device_id.isin(ids)].to_csv(out / "110kv_peer_matches.csv", index=False, encoding="utf-8-sig")
changed_count = None
if prev is not None:
    current, previous = w.set_index("window_id"), prev.set_index("window_id")
    assert set(current.index) == set(previous.index), "Window IDs differ; compare like-for-like runs."
    changed = current.is_candidate.ne(previous.is_candidate.reindex(current.index))
    changes = current.loc[changed, ["device_id", "start", "end", "is_candidate"]].reset_index()
    changes["previous_is_candidate"] = changes.window_id.map(previous.is_candidate)
    changes.to_csv(out / "changed_windows_after_time_fix.csv", index=False, encoding="utf-8-sig")
    changed_count = int(changed.sum())

plt.rcParams.update({"font.family": "Microsoft YaHei", "font.size": 10, "axes.unicode_minus": False})
fig, axes = plt.subplots(3, 1, figsize=(14, 9), dpi=160, sharex=True)
colors = ["#4176a6", "#c78a43"]
for device, color in zip(ids, colors):
    f = d.loc[d.device_id == device]
    label = f"{device[-4:]}（第{'三~四' if device == ids[0] else '六~七'}片）"
    axes[0].plot(f.Timestamp, f.Measure, lw=.65, alpha=.85, color=color, label=label)
    axes[2].plot(f.Timestamp, f.peer_delta_jump_z, lw=.7, color=color, alpha=.8, label=label)
f = d.loc[d.device_id == ids[0]]
axes[1].plot(f.Timestamp, f.peer_delta, color=colors[0], lw=.75, label="8878 − 时间匹配后的 9867")
axes[1].axhline(0, lw=.7, color="#6d7885", linestyle="--")
axes[1].axhline(f.peer_delta.median(), lw=.9, color="#9c6a39", linestyle=":",
                label=f"匹配差值中位数 {f.peer_delta.median():.4f}")
axes[2].axhline(8, color="#b9555f", lw=1, linestyle="--", label="B：同窗至少 2 点 ≥8")
axes[2].axhline(12, color="#923743", lw=1, linestyle=":", label="C：至少 1 点 ≥12")
axes[2].set_ylim(-.3, max(13.3, max(s["peer_jump_z_max"] for s in stats) * 1.05))
axes[0].set_title("同一串两台设备的原始观测", loc="left")
axes[1].set_title("同串差值：超出时间容差的点不参与比较，缺失处断开", loc="left")
axes[2].set_title("同串差值的不同步跳变："
                 + "；".join(f"{s['device_id'][-4:]} 最大值 {s['peer_jump_z_max']:.2f}" for s in stats),
                 loc="left")
for ax, ylabel in zip(axes, ["Measure", "8878 − 9867", "跳变 z"]):
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", color="#e0e5ec", lw=.6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="upper left", frameon=False, fontsize=9, ncol=2)
axes[2].legend(loc="upper left", frameon=True, facecolor="white", edgecolor="white",
               framealpha=1, fontsize=9, ncol=2)
axes[2].set_xlabel("原始观测时间")
axes[2].xaxis.set_major_locator(mdates.MonthLocator(interval=2))
axes[2].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
fig.suptitle("110kV 玉中线串1：8878 与 9867 的直接比对（时间单位已修正）", fontsize=16, y=.985)
fig.text(.5, .015, "；".join(f"{s['device_id'][-4:]}：{s['matched_points']}/{s['observed_points']} 点匹配" for s in stats) + "。"
         "本图阈值对应跳变规则 B/C；持续偏离另按湿度分层与环境残差联合判定。", ha="center", fontsize=9, color="#596675")
fig.tight_layout(rect=[.02, .04, .99, .96], h_pad=1.5)
fig.savefig(out / "110kv_peer_comparison.png")
plt.close(fig)
result = dict(timestamp_units="independent arithmetic verified",
              attempted_pairs=len(m), accepted_pairs=int(m.accepted.sum()),
              rejected_pairs=int(m.accepted.eq(0).sum()),
              changed_window_count=changed_count, devices_110kv=stats)
(out / "time_fix_check.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(result, ensure_ascii=True))
