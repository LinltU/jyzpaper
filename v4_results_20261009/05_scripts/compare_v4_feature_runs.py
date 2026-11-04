"""Compare the base-feature and v4-evidence SOTA augmentation runs.

Both runs use the same v4 labels, splits, methods and seeds; only the five
window features differ. This script reports, per train:test ratio and for the
HGB classifier: Real-only AP, the best-augmentation AP, and the improvement of
the v4-evidence features over the base features.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

RATIOS = ["2to8", "3to7", "4to6", "5to5"]


def load(run: Path) -> pd.DataFrame:
    r = pd.read_csv(run / "all_results.csv")
    r["method"] = r["method"].astype(str)
    return r


def per_ratio(run: Path, classifier: str = "HGB") -> pd.DataFrame:
    r = load(run)
    r = r[r["classifier"].eq(classifier)]
    rows = []
    for ratio in RATIOS:
        g = r[r["ratio"].eq(ratio)]
        real = g[g["method"].eq("Real-only")].drop_duplicates(["seed"])
        aug = g[g["method"].ne("Real-only")]
        best = aug.groupby("method")["Average Precision"].mean().sort_values(ascending=False)
        rows.append({
            "ratio": ratio,
            "real_only_ap": float(real["Average Precision"].mean()),
            "real_only_std": float(real["Average Precision"].std(ddof=0)),
            "best_method": best.index[0] if len(best) else "",
            "best_ap": float(best.iloc[0]) if len(best) else np.nan,
            "best_delta": (float(best.iloc[0]) - float(real["Average Precision"].mean())) if len(best) else np.nan,
        })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--v4evi", type=Path, required=True)
    parser.add_argument("--classifier", default="HGB")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    base = per_ratio(args.base, args.classifier).set_index("ratio")
    evi = per_ratio(args.v4evi, args.classifier).set_index("ratio")

    lines = [
        f"# v4 标签 SOTA 增强：base 特征 vs v4-evidence 特征（{args.classifier}）",
        "",
        "两次实验使用完全相同的 v4 标签、比例、方法与种子，唯一区别是 5 维窗口特征：",
        "",
        "- base：`Measure, env_residual, Hum, Temp, delta_measure`",
        "- v4-evi：`Measure, env_residual_z, peer_delta_z, peer_delta_jump_z, delta_measure`",
        "",
        "| 训练:测试 | base Real-only | v4-evi Real-only | Δ | base 最佳(方法,AP) | v4-evi 最佳(方法,AP) | v4-evi Δ vs 自身 Real-only |",
        "|---|---:|---:|---:|---|---|---:|",
    ]
    for ratio in RATIOS:
        b, e = base.loc[ratio], evi.loc[ratio]
        lines.append(
            f"| {ratio} | {b.real_only_ap:.4f} | {e.real_only_ap:.4f} | {e.real_only_ap - b.real_only_ap:+.4f} | "
            f"{b.best_method} {b.best_ap:.4f} | {e.best_method} {e.best_ap:.4f} | {e.best_delta:+.4f} |"
        )
    lines += [
        "",
        "## 说明",
        "",
        f"- Real-only 列：只用真实训练窗口时的 {args.classifier} AP（3 个种子均值）。",
        "- 最佳列：每个比例独立取 AP 均值最大的增强方法（描述性，不构成全局最优）。",
        "- v4-evi 把 `env_residual_z`、`peer_delta_z`、`peer_delta_jump_z`（v4 判定的点级证据）直接作为窗口通道，",
        "  使分类器能看到 peer 偏离——这正是 v4 候选（31/38 为 peer 跳变）的主要依据。",
        "- `peer_delta_jump_z` 在 MinMax 前裁剪到 20（单点跳变阈值为 12），避免极端值压平通道。",
    ]
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print("wrote", args.output)


if __name__ == "__main__":
    main()
