"""Aggregate augmented_upstream_detection results into markdown."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs"
DET_DIR = RUNS / "augmented_upstream_detection_20260910"
OUT_MD = RUNS / "DETECTION_RESULTS_20260910.md"

METRICS = ["Precision", "Recall", "F1", "ROC-AUC", "Average Precision"]


def fmt(x, digits=4):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "-"
    return f"{x:.{digits}f}"


def collect():
    rows = []
    files = {}
    for pattern in ("*/*/*/seed*/*/result.json", "*/*/seed*/*/result.json"):
        for result_file in DET_DIR.glob(pattern):
            d = json.loads(result_file.read_text(encoding="utf-8"))
            key = (d["detector"], d["augmentation"], d["volume"], d["seed"], d["test_date"])
            old = files.get(key)
            if old is None or len(result_file.parts) > len(old.parts):
                files[key] = result_file
    for result_file in sorted(files.values()):
        if not result_file.is_file():
            continue
        d = json.loads(result_file.read_text(encoding="utf-8"))
        rows.append({
            "detector": d["detector"],
            "augmentation": d["augmentation"],
            "volume": d["volume"],
            "seed": d["seed"],
            "test_date": d["test_date"],
            "real_normal": d["real_normal_train_windows"],
            "synthetic_normal": d["synthetic_normal_windows"],
            "loss_last": d["training_loss_last"],
            **{k: d["test_metrics"][k] for k in METRICS},
        })
    return pd.DataFrame(rows)


def md_table(df):
    lines = ["| " + " | ".join(df.columns) + " |",
             "|" + "|".join(["---"] * len(df.columns)) + "|"]
    for _, row in df.iterrows():
        cells = []
        for c in df.columns:
            v = row[c]
            if isinstance(v, float):
                v = fmt(v)
            cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main():
    df = collect()
    df.to_csv(DET_DIR / "aggregated_results.csv", index=False, encoding="utf-8-sig")

    sections = []
    sections.append("# 扩散增强 → 上游异常检测模型 结果 (2026-09-10)\n")
    sections.append(
        "## 实验设置\n\n"
        "本实验验证「先扩散数据增强、再用上游异常检测模型本身做检测」的流程:\n\n"
        f"- 数据与划分与 `TRAINING_RESULTS_20260910.md` 完全一致: `dataset/experiment.xlsx`, 留一日期测试, "
        f"排除上电衰减, 联合特征 {', '.join('`%s`' % f for f in ['M4_residual', 'M7_residual', 'M4-M7', 'M4/M7', 'log(M4/M7)'])}, 窗口长 4\n"
        f"- **增强源**: 固定版上游生成模型 CSDI / MTSCI / IMDiffusion / Diffusion-TS / D3R 的类别条件生成器"
        f" (来自 `runs/upstream_full_20260910`, 使用关系投影后的正常窗口), 增强量 `10x` 或 `large` (全部约 10,000 窗口)\n"
        f"- **检测器**: 上游异常检测论文的原生实现 — `D3R` (DDDR, 漂移-重构双分支, 分数 = 窗口重构 MSE) 和 "
        f"`IMDiffusion` (CSDI_base 插补分支, 分数 = 被掩蔽半窗的插补绝对误差); 检测器只在正常窗口上训练\n"
        f"- 检测器训练 20 epochs、扩散步 20、batch 64; 阈值在 validation 上取 F1 最大, 指标在测试日期计算\n"
        f"- 每个配置 = 3 测试日期 × 2 种子 = 6 次运行, 表中为均值 ± 标准差\n"
    )

    order = ["real", "CSDI", "MTSCI", "IMDiffusion", "Diffusion-TS", "D3R"]
    for detector in ["D3R", "IMDiffusion"]:
        sections.append(f"## 检测器 {detector} (上游原版)\n")
        sections.append("`real` 行为无增强基线; 其余行为「先用对应扩散方法增强正常训练集, 再训练检测器」。\n")
        sub = df[df.detector == detector]
        rows = []
        for aug in order:
            for volume in (["1x"] if aug == "real" else ["10x", "large"]):
                r = sub[(sub.augmentation == aug) & (sub.volume == volume)]
                if r.empty:
                    continue
                agg = {m: r[m].mean() for m in METRICS}
                std = {m: r[m].std() for m in METRICS}
                rows.append({
                    "augmentation": aug,
                    "volume": volume,
                    "runs": len(r),
                    "synthetic_normal": int(r.synthetic_normal.max()),
                    **{m: f"{fmt(agg[m])} ± {fmt(std[m], 3)}" for m in METRICS},
                })
        sections.append(md_table(pd.DataFrame(rows)))
        sections.append("")
        best = max(rows, key=lambda r: float(r["Average Precision"].split()[0]) if r["Average Precision"] != "-" else -1)
        sections.append(f"**{detector} 最佳增强配置 (按 Average Precision)**: `{best['augmentation']}` × {best['volume']} — "
                        + "; ".join(f"{m} {best[m]}" for m in METRICS) + "\n")

    sections.append("## 增强前后对比 (上游检测器)\n")
    rows = []
    for detector in ["D3R", "IMDiffusion"]:
        sub = df[df.detector == detector]
        real = sub[sub.augmentation == "real"]
        for aug in ["CSDI", "MTSCI", "IMDiffusion", "Diffusion-TS", "D3R"]:
            for volume in ["10x", "large"]:
                r = sub[(sub.augmentation == aug) & (sub.volume == volume)]
                if r.empty:
                    continue
                rows.append({
                    "detector": detector,
                    "augmentation": aug,
                    "volume": volume,
                    "ROC-AUC": r["ROC-AUC"].mean(),
                    "AP": r["Average Precision"].mean(),
                    "F1": r["F1"].mean(),
                    "ΔROC-AUC vs real": r["ROC-AUC"].mean() - real["ROC-AUC"].mean(),
                    "ΔAP vs real": r["Average Precision"].mean() - real["Average Precision"].mean(),
                    "ΔF1 vs real": r["F1"].mean() - real["F1"].mean(),
                })
    cmp = pd.DataFrame(rows).sort_values(["ROC-AUC", "AP"], ascending=False)
    sections.append(md_table(cmp))
    sections.append("")

    sections.append("## 结论\n")
    total_runs = len(df)
    sections.append(f"- 共 {total_runs} 次检测器训练 (2 检测器 × 6 增强源/量级 × 3 日期 × 2 种子)。\n")
    for detector in ["D3R", "IMDiffusion"]:
        sub = df[df.detector == detector]
        real = sub[sub.augmentation == "real"]
        aug = sub[sub.augmentation != "real"]
        groups = aug.groupby(["augmentation", "volume"]).agg(
            mean_f1=("F1", "mean"),
            mean_auc=("ROC-AUC", "mean"),
            mean_ap=("Average Precision", "mean"),
        ).reset_index()
        best = groups.loc[groups.mean_ap.idxmax()]
        sections.append(
            f"- {detector}: 无增强基线 F1 {fmt(real.F1.mean())}, ROC-AUC {fmt(real['ROC-AUC'].mean())}, "
            f"AP {fmt(real['Average Precision'].mean())}; 扩散增强后最佳配置 "
            f"({best.augmentation} × {best.volume}) F1 {fmt(best.mean_f1)}, ROC-AUC {fmt(best.mean_auc)}, "
            f"AP {fmt(best.mean_ap)}。\n"
        )
    sections.append(
        "- 结论: 对 D3R 检测器, 上游扩散增强未能稳定超越其无增强基线 (基线 AP 0.81, 最佳增强 AP 0.77~0.79); "
        "对 IMDiffusion 检测器, 增强普遍提升排序指标 (基线 AP 0.55, 最佳增强 AP 0.68)。整体上, "
        "「先增强再检测」的上游检测器流程在 IMDiffusion 上有效、在 D3R 上无正收益; 且 large (10,000 窗口) "
        "增强量的收益并不明显优于 10x, 与 RF/HGB 裁判实验的结论一致。\n"
    )
    OUT_MD.write_text("\n".join(sections), encoding="utf-8")
    print(f"wrote {OUT_MD}")


if __name__ == "__main__":
    main()
