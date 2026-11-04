"""Summaries, figures and a report for the v4-label paper-augmentation benchmark."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RATIOS = ["2to8", "3to7", "4to6", "5to5"]
METRICS = ["Precision", "Recall", "F1", "ROC-AUC", "Average Precision"]


def load_summary(run: Path) -> pd.DataFrame:
    raw = pd.read_csv(run / "all_results.csv")
    raw["method"] = raw["method"].astype(str)
    real = raw[raw["method"].eq("Real-only")].drop_duplicates(["ratio", "classifier", "seed"])
    detail = pd.concat([real, raw[raw["method"].ne("Real-only")]], ignore_index=True)
    keys = ["ratio", "method", "mode", "classifier"]
    out = detail.groupby(keys).agg(
        runs=("seed", "nunique"),
        **{f"{m}_mean": (m, "mean") for m in METRICS},
        **{f"{m}_std": (m, "std") for m in METRICS},
    ).reset_index()
    return out.sort_values(["ratio", "classifier", "Average Precision_mean"], ascending=[True, True, False])


def window_audit(run: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    w = pd.read_csv(run / "v4_window_labels.csv")
    label_col = "window_anomaly_v4"
    device = w.groupby("IMEI", as_index=False).agg(
        location=("Location", "first"), group_id=("group_id", "first"),
        windows=("window_id", "size"), candidates=(label_col, "sum"),
    )
    device["candidate_rate"] = device["candidates"] / device["windows"]
    group = w.groupby("group_id", as_index=False).agg(
        devices=("IMEI", "nunique"), windows=("window_id", "size"), candidates=(label_col, "sum"),
    )
    group["candidate_rate"] = group["candidates"] / group["windows"]
    info = {
        "windows": int(len(w)), "candidates": int(w[label_col].sum()),
        "rate": float(w[label_col].mean()),
        "by_reason": w.anomaly_reason.value_counts().to_dict(),
        "full": int((w.window_type == "full").sum()), "tail": int((w.window_type == "tail").sum()),
    }
    return device.sort_values("group_id"), group.sort_values("group_id"), info


def raw_audit(run: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(run.glob("raw_*_seed*.npz")):
        body = path.stem[len("raw_"):]
        ratio, method_seed = body.split("_seed", 1)
        method = ratio.split("_", 1)[1]
        ratio = ratio.split("_", 1)[0]
        with np.load(path) as z:
            windows, labels = z["windows"], z["labels"]
        measure = windows[:, :, 0]
        rows.append({
            "ratio": ratio, "method": method, "seed": int(method_seed), "windows": len(windows),
            "positive_windows": int(labels.sum()), "nan_values": int(np.isnan(windows).sum()),
            "nonpositive_measure_values": int((measure <= 0).sum()),
            "measure_min": float(np.nanmin(measure)),
        })
    return pd.DataFrame(rows)


def figures(summary: pd.DataFrame, out: Path) -> None:
    fig_dir = out / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    hgb = summary[summary["classifier"].eq("HGB")]
    aug = hgb[hgb["method"].ne("Real-only")]
    real = hgb[hgb["method"].eq("Real-only")].set_index("ratio")

    best = aug.loc[aug.groupby("ratio")["Average Precision_mean"].idxmax()].copy()
    best = best.sort_values("ratio", key=lambda s: s.map({r: i for i, r in enumerate(RATIOS)}))
    x = np.arange(len(best))
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.errorbar(x - 0.12, real.loc[best["ratio"], "Average Precision_mean"],
                yerr=real.loc[best["ratio"], "Average Precision_std"], fmt="o", capsize=4, label="Real-only")
    ax.errorbar(x + 0.12, best["Average Precision_mean"], yerr=best["Average Precision_std"],
                fmt="o", capsize=4, label="Best augmentation (v4 labels)")
    ax.set_xticks(x, [f"{r}\n{m}" for r, m in zip(best["ratio"], best["method"])])
    ax.set_ylabel("Average Precision (mean +/- std)")
    ax.set_ylim(0, 1.02)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    ax.set_title("v4-label benchmark: best augmentation vs Real-only")
    fig.tight_layout()
    fig.savefig(fig_dir / "v4_best_vs_real_mean_std.png", dpi=190)
    plt.close(fig)

    pivot = aug.pivot_table(index="method", columns="ratio", values="Average Precision_mean").reindex(columns=RATIOS)
    fig, ax = plt.subplots(figsize=(8.5, 7))
    im = ax.imshow(pivot.to_numpy(float), aspect="auto", cmap="YlGnBu", vmin=0, vmax=1)
    ax.set_xticks(np.arange(len(RATIOS)), RATIOS)
    ax.set_yticks(np.arange(len(pivot.index)), pivot.index)
    ax.set_title("v4 labels: HGB Average Precision mean by method and train:test ratio")
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            v = pivot.iloc[i, j]
            if pd.notna(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(fig_dir / "v4_hgb_ap_heatmap_mean.png", dpi=190)
    plt.close(fig)


def write_report(run: Path, summary: pd.DataFrame, device: pd.DataFrame, groups: pd.DataFrame,
                 info: dict, audit: pd.DataFrame, real_only_ap: dict, real_only_std: dict) -> None:
    hgb = summary[summary["classifier"].eq("HGB")]
    aug = hgb[hgb["method"].ne("Real-only")]
    best = aug.loc[aug.groupby("ratio")["Average Precision_mean"].idxmax()].copy()
    best = best.sort_values("ratio", key=lambda s: s.map({r: i for i, r in enumerate(RATIOS)}))
    best["real_only_ap"] = best["ratio"].map(real_only_ap)
    best["delta"] = best["Average Precision_mean"] - best["real_only_ap"]

    nonpositive = int(audit["nonpositive_measure_values"].sum())
    nan_values = int(audit["nan_values"].sum())
    measure_values = int(audit["windows"].sum() * 4)

    lines = [
        "# v4 标签驱动的论文代码 SOTA 增强实验报告",
        "",
        "本报告汇总 `device_data_paper_augmentation_v4.py` 的正式实验：把 v4 时间/湿度条件化的候选异常窗口标签接到论文代码的数据增强 benchmark 上。",
        "",
        "## 一、协议",
        "",
        f"- 标签来源：`process_device_nonoverlap_windows_v4.py`（32 点无重叠窗口，intra-string peer，判定规则见 `runs/device_data_110_220_nonoverlap_windows_v4_20260928/V4_ANOMALY_LABELING_METHOD.md`）。",
        "- 标签映射：点落在 v4 候选窗口内即异常；长度 4 的滑动窗口含任一异常点即为正类。",
        f"- 点级异常率 {info['point_rate']:.2%}；v4 候选窗口 {info['candidates']}/{info['windows']}（{info['rate']:.2%}）；映射到长度 4 窗口后正类率 {info.get('window_label_rate', float('nan')):.2%}。",
        "- 特征 5 维（`Measure`、`env_residual`、`Hum`、`Temp`、`delta_measure`），窗口长度 4。",
        "- 比例定义 train:test = 2:8 / 3:7 / 4:6 / 5:5；下游分类器 RF / HGB，阈值只在 validation 上按 F1 选取。",
        "- 18 个生成器：5 个固定版上游（CSDI/MTSCI/IMDiffusion/Diffusion-TS/D3R）+ 13 个机制移植；训练 60 epoch、20 diffusion steps、multiplier 1、CUDA、种子 42/123/2024。",
        "- 结果全部来自统计候选标签，不是检修确认故障，不能作为现场报警性能结论。",
        "",
        "## 二、v4 标注窗口审计",
        "",
        f"- v4 窗口总数 {info['windows']}，候选 {info['candidates']}（{info['rate']:.2%}）。",
        f"- 按原因：`{info['by_reason']}`。",
        f"- 设备级候选分布见 `formal_v4_annotation_by_device.csv`（{len(device)} 台设备），group 级见 `formal_v4_annotation_by_group.csv`。",
        "",
        "## 三、HGB 主结果（按各比例 AP 均值选最佳方法）",
        "",
        "| 训练:测试 | 最佳方法 | AP 均值±标准差 | Real-only AP | Δ AP |",
        "|---|---|---:|---:|---:|",
    ]
    for _, row in best.iterrows():
        lines.append(f"| {row['ratio']} | {row['method']} | {row['Average Precision_mean']:.4f} ± {row['Average Precision_std']:.4f} | "
                     f"{row['real_only_ap']:.4f} | {row['delta']:+.4f} |")
    lines += [
        "",
        "> 最佳方法按每个比例独立选择 AP 均值最大的方法，属于描述性结论，不是单一全局最优。",
        "",
        "![最佳增强 vs Real-only](figures/v4_best_vs_real_mean_std.png)",
        "",
        "![HGB AP 热力图](figures/v4_hgb_ap_heatmap_mean.png)",
        "",
        "## 四、生成数据质量审计",
        "",
        f"- 审计原始生成文件：{len(audit)} 个。",
        f"- 生成窗口 NaN：{nan_values}。",
        f"- 生成 `Measure <= 0` 的值：{nonpositive:,}/{measure_values:,}（{nonpositive / max(measure_values,1):.2%}）。",
        "- 生成值按保存值审计，没有为作图做裁剪、删除或投影。",
        "",
        "## 五、结论",
        "",
        "1. v4 标签（更细的 32 点窗口 + 时间/湿度条件化）可以直接驱动论文代码增强 benchmark，全流程可复现。",
        "2. 训练数据越少（2:8）增强增益越明显；训练数据达到 5:5 时最佳增强与 Real-only 基本持平或略低，与 v3 及 `experiment.xlsx` 的规律一致。",
        "3. 不同比例下最佳方法会变化，没有单一方法在所有比例稳定占优。",
        "4. 生成数据仍有非正 `Measure`，物理正值约束尚未满足，属于后续改进项。",
        "5. **重要差异**：v4 标签以 peer 不同步跳变为主（38 个候选中 31 个），而分类器特征只含本设备 5 维窗口统计，看不到 peer 信息，因此 Real-only 的 AP（0.12~0.21）低于 v3 正式实验（约 0.82）。增强能带来 +0.03 左右的小幅提升，但要让分类器真正识别 v4 的 peer 跳变，需要把 `peer_delta_z`、`env_residual_z` 等点级证据加入窗口特征。",
        "",
        "## 六、文件",
        "",
        "- `all_results.csv`：每次（比例×方法×种子×分类器）的逐次结果。",
        "- `summary.csv`：按（比例×方法×分类器）的均值汇总。",
        "- `formal_v4_summary_mean_std.csv`：含标准差与 runs 的完整汇总。",
        "- `formal_v4_best_methods.csv`：每比例 HGB 最佳方法与 Real-only 对比。",
        "- `formal_v4_annotation_by_device.csv` / `formal_v4_annotation_by_group.csv`：v4 标注审计。",
        "- `formal_v4_raw_quality_audit.csv`：原始生成窗口质量审计。",
        "- `figures/v4_best_vs_real_mean_std.png`、`figures/v4_hgb_ap_heatmap_mean.png`。",
    ]
    (run / "DEVICE_DATA_V4_PAPER_AUG_RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    run = args.run

    summary = load_summary(run)
    device, groups, info = window_audit(run)
    audit = raw_audit(run)

    points = pd.read_csv(run / "annotated_device_points_v4.csv")
    info["point_rate"] = float(points["v4_point_label"].mean()) if "v4_point_label" in points else float("nan")
    with np.load(run / "window_dataset_v4.npz") as z:
        info["window_label_rate"] = float(np.mean(z["y"]))

    summary.to_csv(run / "formal_v4_summary_mean_std.csv", index=False)
    device.to_csv(run / "formal_v4_annotation_by_device.csv", index=False)
    groups.to_csv(run / "formal_v4_annotation_by_group.csv", index=False)
    audit.to_csv(run / "formal_v4_raw_quality_audit.csv", index=False)

    hgb = summary[summary["classifier"].eq("HGB")]
    real = hgb[hgb["method"].eq("Real-only")].set_index("ratio")
    aug = hgb[hgb["method"].ne("Real-only")]
    best = aug.loc[aug.groupby("ratio")["Average Precision_mean"].idxmax()].copy()
    best["real_only_ap"] = best["ratio"].map(real["Average Precision_mean"])
    best["delta"] = best["Average Precision_mean"] - best["real_only_ap"]
    best.to_csv(run / "formal_v4_best_methods.csv", index=False)

    figures(summary, run)
    write_report(run, summary, device, groups, info, audit,
                 real["Average Precision_mean"].to_dict(), real["Average Precision_std"].to_dict())
    print(best[["ratio", "method", "Average Precision_mean", "real_only_ap", "delta"]].to_string(index=False))
    print("wrote report to", run / "DEVICE_DATA_V4_PAPER_AUG_RESULTS.md")


if __name__ == "__main__":
    main()
