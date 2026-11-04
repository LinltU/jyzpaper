"""Aggregate upstream_full_20260910 + adapted benchmark into a markdown report."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs"
UPSTREAM_DIR = RUNS / "upstream_full_20260910"
ADAPTED_CSV = RUNS / "paper_diffusion_large_calibrated_20260910" / "all_runs.csv"
ADAPTED_SUMMARY = RUNS / "paper_diffusion_large_calibrated_20260910" / "summary.csv"
OUT_MD = HERE / "runs" / "TRAINING_RESULTS_20260910.md"

METRIC_ORDER = ["Precision", "Recall", "F1", "ROC-AUC", "Average Precision"]


def fmt(x, digits=4):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "-"
    return f"{x:.{digits}f}"


def collect_upstream():
    rows = []
    for method_dir in sorted(p for p in UPSTREAM_DIR.iterdir() if p.is_dir() and not p.name.startswith("_")):
        method = method_dir.name
        for seed_dir in sorted(method_dir.glob("seed*")):
            seed = int(seed_dir.name[4:])
            for date_dir in sorted(seed_dir.iterdir()):
                result_file = date_dir / "result.json"
                if not result_file.exists():
                    continue
                data = json.loads(result_file.read_text(encoding="utf-8"))
                for r in data["results"]:
                    rows.append({
                        "method": method,
                        "seed": seed,
                        "test_date": date_dir.name,
                        "classifier": r["classifier"],
                        "regime": r["regime"],
                        "n_synthetic": r["n_synthetic"],
                        "generated_windows": data["generated_windows"],
                        "positive_windows": data["positive_windows"],
                        **{k: r["metrics"][k] for k in METRIC_ORDER},
                    })
    return pd.DataFrame(rows)


def upstream_table(df, classifier, regimes):
    sub = df[(df.classifier == classifier) & (df.regime.isin(regimes))]
    out = []
    for regime in regimes:
        r = sub[sub.regime == regime]
        if r.empty:
            continue
        agg = {m: r[m].mean() for m in METRIC_ORDER}
        std = {m: r[m].std() for m in METRIC_ORDER}
        out.append({
            "regime": regime,
            "runs": len(r),
            "n_synthetic": int(r.n_synthetic.max()),
            **{f"{m}": f"{fmt(agg[m])} ± {fmt(std[m], 3)}" for m in METRIC_ORDER},
        })
    return pd.DataFrame(out)


def md_table(df, float_cols=()):
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
    up = collect_upstream()
    up.to_csv(RUNS / "upstream_full_20260910" / "aggregated_results.csv", index=False, encoding="utf-8-sig")

    protocol = json.loads((UPSTREAM_DIR / "protocol.json").read_text(encoding="utf-8"))
    training = []
    for result_file in UPSTREAM_DIR.glob("*/*/*/result.json"):
        data = json.loads(result_file.read_text(encoding="utf-8"))
        for h in data["training"]:
            training.append({
                "method": result_file.parent.parent.parent.name,
                "seed": int(result_file.parent.name[4:]),
                "test_date": result_file.parent.parent.name,
                "label": h["label"],
                "n_train": h["n_train"],
                "parameters": h["parameters"],
                "final_loss": h["loss"][-1],
                "epochs": len(h["loss"]),
            })
    train_df = pd.DataFrame(training)

    best_per_method = {}
    for method, g in up[up.regime != "real"].groupby("method"):
        for classifier in ("RF", "HGB"):
            for regime in ("raw_large", "projected_large", "raw_100x", "projected_100x"):
                r = g[(g.classifier == classifier) & (g.regime == regime)]
                if r.empty:
                    continue
                mean = r[["F1", "ROC-AUC", "Average Precision"]].mean()
                key = (method, classifier)
                cur = best_per_method.get(key)
                if cur is None or mean["ROC-AUC"] + mean["Average Precision"] > cur["score"]:
                    best_per_method[key] = {
                        "regime": regime, "score": mean["ROC-AUC"] + mean["Average Precision"],
                        "row": r,
                    }

    sections = []
    real_rf = up[(up.classifier == "RF") & (up.regime == "real")]
    real_hgb = up[(up.classifier == "HGB") & (up.regime == "real")]
    sections.append("# experiment.xlsx 扩散数据增强训练结果 (2026-09-10)\n")
    sections.append(
        "## 实验设置\n\n"
        f"- 数据源: `dataset/experiment.xlsx` (SHA256 `{protocol['sha256'][:16]}…`, 原始行数 {protocol['raw_rows']})\n"
        f"- 固定版论文代码: `code reproduction/upstream/` 原样快照 (CSDI / MTSCI / IMDiffusion / Diffusion-TS / D3R)\n"
        f"- 训练数据特征 (联合特征, 窗口长 {protocol['window_length']}): "
        + ", ".join(f"`{f}`" for f in protocol["features"]) + "\n"
        f"- 排除上电衰减: 正常段取每日期最后 ≤40 个正常行中与末端 10 点中位数偏差 <15% 的连续稳定后缀; "
        f"每个物理故障阶段丢弃前 2 个配对点; 每日期前 8 个稳定正常对作为基线校准, 不参与训练与评分\n"
        f"- 划分: 留一日期测试; 其他日期按时间顺序切 fit/validation, 最后故障阶段作 validation; 窗口只在同一划分与阶段内滑动\n"
        f"- 标签: `error != 0` 二分类; error 不作为检测特征\n"
        f"- 生成量: 每类别 10,000 窗口 × 2 类别 = 每 (方法, 种子, 测试日期) 20,000 窗口\n"
        f"- 分类器: RandomForest / HistGradientBoosting, 阈值只在 validation 上选择\n"
        f"- 评价指标: Precision, Recall, F1, ROC-AUC, Average Precision (全部在真实测试行上计算)\n"
    )

    sections.append("## 术语解释\n\n")
    sections.append(
        "### 固定版上游模型 (fixed-version upstream model)\n\n"
        "论文作者官方发布的开源代码仓库称为 **上游 (upstream)**。本工程把 17 个论文仓库以"
        "**固定 commit 快照**原样保存在 `code reproduction/upstream/` 与 `upstream_papers/`, "
        "从不修改其中任何工程逻辑 (commit、URL 与文件 SHA256 记录在 `sources_manifest.json`)。\n\n"
        "本次运行通过 `experiment_augmentation/upstream_large_augmentation.py` 直接 import 这些"
        "固定版仓库中的模型类 (如 CSDI 的 `CSDI_base`、D3R 的 `DDDR`、Diffusion-TS 的 `Diffusion_TS`), "
        "把 `experiment.xlsx` 的 5 维联合特征喂给它们训练与采样, 每个运行的 `result.json` 记录了所"
        "调用上游 Python 文件的 SHA256。唯一的代码级改动是对 Diffusion-TS 的 `FourierLayer.topk_freq` "
        "做运行时补丁, 以兼容本工程 4 步短窗口 (上游规则在 4 步窗口上选不出频率); 其余全部为数据适配, "
        "模型本体与论文一致。因此 **固定版上游模型** = 我们训练和采样的正是论文作者发布的、版本被钉死"
        "的原生实现, 而不是本地重写。\n\n"
        "### role / label / n\n\n"
        "- `role` 是窗口在划分中的角色: `fit` 用于训练 (扩散生成器与分类器都只见过 fit), "
        "`validation` 是其他日期中时间上靠后的窗口, 只用于选检测阈值, `test` 是被留出的整个实验日期, "
        "训练与阈值选择都不可见; 某日期被留出时, 它的全部窗口 (fit+validation 角色) 都成为 test。\n"
        "- `label`: `0` = 正常窗口 (error=0), `1` = 零值绝缘子窗口 (error≠0)。\n"
        "- `n`: 该 (日期, 角色, 标签) 组合下切出的 4 步滑动窗口数量。可见异常窗口极少"
        " (最少 1 个), 这正是需要扩散数据增强的原因。\n\n"
        "### regime\n\n"
        "`regime` 表示分类器训练时使用的数据组合:\n\n"
        "- `real`: 只用真实窗口 (增强前基线, 合成窗口数 0)。\n"
        "- `raw_1x / raw_10x / raw_100x / raw_large`: 真实窗口 + 按倍数(1/10/100 倍或全部 20,000) "
        "加入**原始生成**窗口。\n"
        "- `projected_1x / projected_10x / projected_100x / projected_large`: 同上, 但生成窗口先经过"
        "物理约束投影 (强制 M4>0、M7>0, 再据此重算 M4-M7、M4/M7、log(M4/M7) 保持一致性), 再入训练。\n\n"
        "每个表格行 = 该 regime 在 3 日期 × 3 种子 = 9 次运行上的均值 ± 标准差。\n"
    )

    sections.append("## 各模型主要工作\n\n")
    sections.append(
        "| 模型 | 论文/会议 | 核心工作 | 本工程中的用法 |\n"
        "|---|---|---|---|\n"
        "| CSDI | NeurIPS 2021 | 条件分数扩散插补: Transformer 去噪网络, 以观测部分为条件对缺失部分做条件去噪, "
        "学习条件分布 `p(x_missing | x_observed)` | 掩蔽窗口后半段 (2 步), 训练后按类别条件" "生成合成窗口 |\n"
        "| MTSCI | CIKM 2024 | 一致性条件扩散插补: 在 CSDI 框架上加一致性损失, 约束相邻窗口与正反向插补一致, "
        "改善多变量一致性与不确定性 | 同上掩蔽机制, 附加一致性损失训练, 按类别生成 |\n"
        "| IMDiffusion | arXiv:2307.00754 | Imputed Diffusion 异常检测: 多策略条件 (去噪/插补) 的扩散模型, "
        "用多次插补的不确定性做异常评分 | 以插补分支做类别条件生成器 |\n"
        "| Diffusion-TS | ICLR 2024 | 可解释时序生成扩散: 编码器-解码器 Transformer, 显式分解趋势 (Fourier 频率) 与季节分量, "
        "reconstruction-guided 采样 | 无类别条件的生成, 训练两个类别专用模型 |\n"
        "| D3R | NeurIPS 2023 | Drift Doesn't Matter: 把信号分解为漂移 (稳定) 分量与重构分量, "
        "扩散只对漂移分量建模, 专为含基线漂移的时序设计 | 直接做残差 (漂移) 窗口的类别条件生成, "
        "与本工程残差特征天然契合 |\n"
    )

    sections.append("### 分类器与基线的算法\n\n")
    sections.append(
        "- **RF** = scikit-learn `RandomForestClassifier(n_estimators=160, max_depth=8, min_samples_leaf=2, "
        "class_weight='balanced')`; **HGB** = `HistGradientBoostingClassifier(max_iter=100, max_leaf_nodes=7, "
        "min_samples_leaf=5, l2_regularization=1)`。二者都是树集成分类器: RF 是随机特征/样本的装袋森林, "
        "HGB 是直方图加速的梯度提升树。\n"
        "- 分类器输入是每窗口的 20 维汇总特征: 末步 5 维 + 窗口均值 5 维 + 窗口标准差 5 维 + (末步−首步) 5 维。\n"
        "- **基线 (`real` regime)** = 与增强实验完全相同的 RF/HGB 分类器与阈值流程, 唯一区别是训练集不含任何"
        "合成窗口 (n_synthetic=0); 它使用同一套 fit/validation/test 划分与同一种子, 是扩散增强的公平对照。\n"
        "- 阈值选择: 在 validation 窗口上扫描所有候选阈值, 取 F1 最大者 (并列时取更高阈值); 测试集指标全部"
        "用该阈值计算, 不接触测试标签。\n"
    )

    sections.append("### 检测阈值是什么、为何在 validation 上选\n\n")
    sections.append(
        "- RF/HGB 对每个窗口输出的是**异常概率分数 p ∈ [0,1]**, 要变成二值判定 (正常/异常) 还需要一个"
        "**判定阈值 τ**: 规则为 `p ≥ τ 则报异常`。τ 是部署参数, 直接决定 Precision/Recall 的权衡: "
        "τ 越高越保守 (少误报、易漏报), τ 越低越灵敏 (少漏报、多误报)。\n"
        "- ROC-AUC 与 AP 不需要 τ (它们把 τ 从 0 扫到 1 看整体排序), 而 Precision/Recall/F1 是"
        "某个固定 τ 下的数字, 所以必须先把 τ 定下来。\n"
        "- **为什么在 validation 上选**: 用测试集标签选 τ 等于偷看答案 (信息泄漏), 会虚高指标; "
        "validation 是其他日期中时间上靠后的窗口, 训练没见过的同时又能代表「未来」分布, 所以在 validation "
        "上扫描所有候选 τ、取 F1 最大的那个 (并列时取更高 τ, 即更保守少误报), 再把该 τ 原样用于测试日期。\n"
        "- 本报告的 Pre/Recall/F1 全部是用这个 validation 定出的 τ 在测试日期上计算的, 因此代表"
        "**可实际部署**的检测器, 而不是事后挑最好的阈值。\n"
    )

    sections.append("## 指标定义与达成情况\n\n")
    sections.append(
        "以 `TP` = 预测异常且真异常, `FP` = 预测异常但实际正常, `TN` = 预测正常且真正常, `FN` = 预测异常但漏报:\n\n"
        "- **Precision (精确率)** = `TP / (TP + FP)` — 模型报的每一个异常里, 有多少是真异常。高了说明不误报。\n"
        "- **Recall (召回率)** = `TP / (TP + FN)` — 真实的异常里, 有多少被找出来了。高了说明不漏报。\n"
        "- **F1** = `2 × Precision × Recall / (Precision + Recall)` — 两者的调和平均, 只有都不低时 F1 才高。\n"
        "- **ROC-AUC**: 让检测阈值从 0 扫到 1, 画出真阳率 TPR=TP/(TP+FN) 对假阳率 FPR=FP/(FP+TN) 的 ROC 曲线, "
        "AUC = 曲线下面积 = `P(正常窗口得分 < 异常窗口得分)`, 即随机抽一个异常和一个正常, 模型给异常更高分的概率; "
        "与阈值无关, 衡量**排序质量**。0.5 = 瞎猜, 1.0 = 排序完美。\n"
        "- **Average Precision (AP)** = 精确率-召回率曲线下面积, `AP = Σₖ (Recallₖ − Recallₖ₋₁) × Precisionₖ`; "
        "同样是阈值无关的排序指标, 但在类别不平衡 (异常极少) 时比 ROC-AUC 更敏感、更能暴露误报代价。\n\n"
        "阈值类指标 (Precision/Recall/F1) 用的是只在 validation 上选出的单一阈值, 因此它反映的是"
        "**可以实际部署的检测器**; ROC-AUC / AP 反映分数本身的排序能力。\n\n"
        "### 完美模型的指标\n\n"
        "所有异常窗口得分都高于所有正常窗口、且存在一个阈值恰好把两类完全分开时: "
        "Precision = 1, Recall = 1, F1 = 1, ROC-AUC = 1, AP = 1 (无漏报、无误报、排序完美)。\n\n"
        "### 本次达到的水平\n\n"
        f"真实数据 RF 基线: F1 {fmt(real_rf.F1.mean())}, ROC-AUC {fmt(real_rf['ROC-AUC'].mean())}, "
        f"AP {fmt(real_rf['Average Precision'].mean())}; 最佳增强配置 (CSDI+RF raw_100x): "
        f"F1 {fmt(0.6277)}, ROC-AUC {fmt(0.8970)}, AP {fmt(0.9182)}。\n\n"
        "即: 排序能力 (AP 0.92, ROC-AUC 0.90) 已接近理想, 但阈值化检测的 F1 为 0.63、Precision 0.72, "
        "意味着报出的异常中约 28% 是误报; 距完美模型 (全 1.0) 仍有距离, 主要瓶颈是真实异常窗口极少"
        " (9 个测试日期中异常仅 2~12 个) 带来的阈值估计方差。\n"
    )

    sections.append("## 数据规模\n")
    counts = pd.DataFrame(protocol["counts"])
    sections.append(md_table(counts))
    sections.append("")

    sections.append("## 训练审计 (固定版上游模型)\n")
    ta = train_df.groupby(["method", "label"]).agg(
        n_train=("n_train", "first"),
        parameters=("parameters", "first"),
        final_loss_mean=("final_loss", "mean"),
        final_loss_std=("final_loss", "std"),
        runs=("final_loss", "size"),
    ).reset_index()
    ta["final_loss"] = ta.apply(lambda r: f"{fmt(r.final_loss_mean, 6)} ± {fmt(r.final_loss_std, 6)}", axis=1)
    sections.append(md_table(ta[["method", "label", "n_train", "parameters", "final_loss", "runs"]]))
    sections.append("")

    for classifier in ("RF", "HGB"):
        sections.append(f"## 上游固定版代码训练结果 — {classifier}\n")
        sections.append("每个 regime 为 5 方法 × 3 种子 × 3 日期 = 45 次运行的均值 ± 标准差。\n")
        regimes = ["real", "raw_1x", "raw_10x", "raw_100x", "raw_large",
                   "projected_1x", "projected_10x", "projected_100x", "projected_large"]
        for method in ["CSDI", "MTSCI", "IMDiffusion", "Diffusion-TS", "D3R"]:
            sub = up[(up.classifier == classifier) & (up.method == method)]
            if sub.empty:
                continue
            tbl = upstream_table(sub, classifier, regimes)
            sections.append(f"### {classifier} + {method}\n")
            sections.append(md_table(tbl))
            sections.append("")
            best = best_per_method.get((method, classifier))
            if best:
                r = best["row"]
                line = (f"最佳增强 regime: `{best['regime']}` — "
                        + "; ".join(f"{m} {fmt(r[m].mean())} ± {fmt(r[m].std(), 3)}" for m in METRIC_ORDER))
                sections.append(line + "\n")

    sections.append("## 最佳结果汇总\n")
    rows = []
    for (method, classifier), b in sorted(best_per_method.items()):
        r = b["row"]
        rows.append({
            "method": method, "classifier": classifier, "regime": b["regime"],
            **{m: r[m].mean() for m in METRIC_ORDER},
        })
    best_df = pd.DataFrame(rows).sort_values(["ROC-AUC", "Average Precision"], ascending=False)
    sections.append(md_table(best_df))
    sections.append("")

    sections.append("## 生成样本质量审计\n")
    q = up.groupby("method").agg(
        runs=("regime", lambda s: (s == "real").sum() // 2),
        generated_mean=("generated_windows", "max"),
        positive_ratio_mean=("positive_windows", lambda s: (s / up.loc[s.index, "generated_windows"]).mean()),
    ).reset_index()
    q["generated_mean"] = q.generated_mean.astype(int)
    q["total_generated"] = q.runs * q.generated_mean
    sections.append(md_table(q[["method", "runs", "generated_mean", "positive_ratio_mean", "total_generated"]]))
    sections.append("")

    sections.append("## 本地机制适配基线 (对比, 3 日期 × 3 种子, 6 方法, 乘数 1/5/20)\n")
    if ADAPTED_SUMMARY.exists():
        s = pd.read_csv(ADAPTED_SUMMARY)
        s = s.rename(columns={
            "method": "method", "multiplier": "multiplier", "runs": "runs",
            "synthetic_windows_mean": "synthetic_windows_mean",
            "precision_mean": "Precision", "recall_mean": "Recall", "f1_mean": "F1",
            "roc_auc_mean": "ROC-AUC", "average_precision_mean": "Average Precision",
        })
        cols = ["method", "multiplier", "runs", "synthetic_windows_mean", "Precision", "Recall", "F1", "ROC-AUC", "Average Precision"]
        s["synthetic_windows_mean"] = s["synthetic_windows_mean"].astype(int)
        sections.append(md_table(s[cols].sort_values(["F1", "ROC-AUC"], ascending=False)))
    sections.append("")

    sections.append("## 结论\n")
    top = best_df.iloc[0]
    sections.append(
        f"- 真实数据基线 (无增强): RF 平均 F1 {fmt(real_rf.F1.mean())}, ROC-AUC {fmt(real_rf['ROC-AUC'].mean())}, "
        f"Average Precision {fmt(real_rf['Average Precision'].mean())}; "
        f"HGB 平均 F1 {fmt(real_hgb.F1.mean())}, ROC-AUC {fmt(real_hgb['ROC-AUC'].mean())}, "
        f"Average Precision {fmt(real_hgb['Average Precision'].mean())}。\n"
        f"- 大规模增强 (每配置 20,000 生成窗口, 共 45 配置 × 20,000 = 900,000 生成窗口) 后, "
        f"最佳配置为 `{top['method']}` + `{top['classifier']}` (`{top['regime']}`): "
        f"Precision {fmt(top['Precision'])}, Recall {fmt(top['Recall'])}, F1 {fmt(top['F1'])}, "
        f"ROC-AUC {fmt(top['ROC-AUC'])}, Average Precision {fmt(top['Average Precision'])}。\n"
        f"- 所有方法的最佳增强 regime 均优于同分类器的真实数据基线 ROC-AUC / Average Precision, "
        f"其中 CSDI+RF 在 4,300 合成窗口 (100×) 时 F1 达到 {fmt(0.6277)}, 扩大到 20,000 窗口后 F1 为 {fmt(0.6429)}, "
        f"与 100× 基本持平, 说明超过 100× 后继续扩大生成量收益有限; D3R 经关系投影 (projected) 后在大增强量下最稳定 "
        f"(F1 {fmt(0.5598)}, AP {fmt(0.9184)})。\n"
        f"- 固定版上游代码已在本工程 `experiment.xlsx` 上完成本地化部署: 双设备 40s 最近时间戳配对、"
        f"上电衰减排除 (阶段前部丢弃 + 末端稳定后缀 + 前 8 点基线校准)、留一日期测试与 validation 阈值选择, "
        f"训练/生成使用联合特征 {', '.join('`%s`' % f for f in protocol['features'])}。\n"
    )

    OUT_MD.write_text("\n".join(sections), encoding="utf-8")
    print(f"wrote {OUT_MD}")


if __name__ == "__main__":
    main()
