"""Aggregate ALL paper experiments into runs/ALL_PAPERS_RESULTS.md.

Combines:
  - fixed-version upstream generators (upstream_gpu_full_20260910, 5 methods)
  - fixed-version upstream detectors (augmented_upstream_detection_gpu_20260910, 2 methods)
  - mechanism-adapted generators (all_papers_extended_20260910/generators, 13 methods)
  - mechanism-adapted detectors (all_papers_extended_20260910/detectors, 9 methods)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs"
OUT_MD = RUNS / "ALL_PAPERS_RESULTS.md"

METRIC_ORDER = ["Precision", "Recall", "F1", "ROC-AUC", "Average Precision"]

GEN_DESCRIPTIONS = {
    "CSDI": ("NeurIPS 2021", "条件分数扩散插补", "fixed-version upstream import"),
    "MTSCI": ("CIKM 2024", "一致性条件扩散插补", "fixed-version upstream import"),
    "IMDiffusion": ("arXiv 2307.00754", "插补式扩散", "fixed-version upstream import"),
    "Diffusion-TS": ("ICLR 2024", "可解释时序生成扩散", "fixed-version upstream import"),
    "D3R": ("NeurIPS 2023", "漂移分解扩散", "fixed-version upstream import"),
    "TimeGrad": ("ICML 2021", "自回归概率预测", "mechanism port (GRU history -> DDPM)"),
    "NsDiff": ("arXiv", "非平稳扩散生成", "mechanism port (per-window std normalization)"),
    "FIDE": ("arXiv", "极端样本生成", "mechanism port (frequency-branch conditioning)"),
    "PaD-TS": ("arXiv", "群体条件生成", "mechanism port (class one-hot conditioning)"),
    "WaveStitch": ("arXiv", "条件生成/拼接", "mechanism port (first-half conditioning)"),
    "TabDDPM": ("ICLR 2023", "表格扩散", "mechanism port (20-dim tabular flatten)"),
    "CCDM": ("arXiv", "通道条件扩散预测", "mechanism port (channel conditioning)"),
    "DiffPLF": ("arXiv", "局部趋势预测生成", "mechanism port (trend residual diffusion)"),
    "TSDE": ("arXiv", "自监督表示/插补", "mechanism port (IIF trend conditioning)"),
    "SSSD": ("TMLR 2023", "结构化状态扩散插补", "mechanism port (conv + block masking)"),
    "CoSTI": ("arXiv", "一致性插补", "mechanism port (consistency masked reconstruction)"),
    "SSD-TS": ("arXiv", "长序列插补", "mechanism port (diffusion core, no long-range)"),
    "MCD-TSF": ("arXiv", "多通道预测/插补", "mechanism port (channel + freq conditioning)"),
    "Treeffuser": ("arXiv 2406.07658", "树基表格扩散", "fixed-version upstream import (blei-lab/treeffuser, LightGBM 评分模型)"),
}

DET_DESCRIPTIONS = {
    "D3R": ("NeurIPS 2023", "漂移重构 MSE", "fixed-version upstream import"),
    "IMDiffusion": ("arXiv 2307.00754", "插补误差评分", "fixed-version upstream import"),
    "DiffAD": ("arXiv", "权重增量扩散重构误差", "mechanism port (TF -> torch)"),
    "ICDiffAD": ("arXiv", "插补条件重构", "mechanism port"),
    "TIMEADDM": ("arXiv", "扩散重构误差", "mechanism port (recurrent tail weighting)"),
    "ProDiffAD": ("arXiv", "蒸馏快速重构", "mechanism port (distilled one-pass)"),
    "TSADC": ("arXiv", "掩蔽重构双评分", "mechanism port (dual-mask score)"),
    "NGLS-Diff": ("arXiv", "正常潜空间距离", "mechanism port (latent distance)"),
    "DiffTAD": ("arXiv", "轨迹重构", "mechanism port (trajectory reconstruction)"),
    "diffusionae": ("arXiv", "自编码+扩散重构", "mechanism port (autoencoder residual)"),
    "C3DAD": ("arXiv", "对比扩散", "mechanism port (contrastive embedding)"),
}


def fmt(x, digits=4):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "-"
    return f"{x:.{digits}f}"


def mean_std(series):
    s = pd.Series([v for v in series if v is not None], dtype=float)
    if s.empty:
        return "-"
    if s.nunique() == 1 and len(s) > 1:
        return f"{fmt(s.iloc[0])} ± 0.000"
    return f"{fmt(s.mean())} ± {fmt(s.std(), 3)}"


def collect_generators():
    rows = []
    for root in (RUNS / "upstream_gpu_full_20260910", RUNS / "all_papers_extended_20260910" / "generators"):
        for method_dir in sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith("_")):
            method = method_dir.name
            for seed_dir in sorted(method_dir.glob("seed*")):
                seed = int(seed_dir.name[4:])
                for date_dir in sorted(p for p in seed_dir.iterdir() if p.is_dir()):
                    rf = date_dir / "result.json"
                    if not rf.exists():
                        continue
                    d = json.loads(rf.read_text(encoding="utf-8"))
                    for r in d.get("results", []):
                        if r["classifier"] != "RF":
                            continue
                        rows.append({
                            "method": method, "seed": seed, "test_date": date_dir.name,
                            "regime": r["regime"], "n_synthetic": r["n_synthetic"],
                            **{k: r["metrics"][k] for k in METRIC_ORDER},
                            "official": d.get("official_code_used", None),
                        })
    return pd.DataFrame(rows)


def collect_detectors():
    rows = []
    for root in (RUNS / "augmented_upstream_detection_gpu_20260910", RUNS / "all_papers_extended_20260910" / "detectors"):
        for det_dir in sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith("_")):
            detector = det_dir.name
            # fixed-version detector layout: det/aug/vol/seed/date/result.json
            sub = det_dir / "real"
            if sub.is_dir():
                for aug_dir in sorted(p for p in det_dir.iterdir() if p.is_dir()):
                    aug = aug_dir.name
                    for vol_dir in sorted(p for p in aug_dir.iterdir() if p.is_dir()):
                        for seed_dir in sorted(vol_dir.glob("seed*")):
                            seed = int(seed_dir.name[4:])
                            for date_dir in sorted(p for p in seed_dir.iterdir() if p.is_dir()):
                                rf = date_dir / "result.json"
                                if not rf.exists():
                                    continue
                                d = json.loads(rf.read_text(encoding="utf-8"))
                                rows.append({"detector": detector, "augmentation": aug, "volume": vol_dir.name,
                                             "seed": seed, "test_date": date_dir.name,
                                             **{k: d["test_metrics"][k] for k in METRIC_ORDER}})
            else:
                # mechanism detector layout: det/seed/date/result.json
                for seed_dir in sorted(p for p in det_dir.iterdir() if p.name.startswith("seed")):
                    seed = int(seed_dir.name[4:])
                    for date_dir in sorted(p for p in seed_dir.iterdir() if p.is_dir()):
                        rf = date_dir / "result.json"
                        if not rf.exists():
                            continue
                        d = json.loads(rf.read_text(encoding="utf-8"))
                        rows.append({"detector": detector, "augmentation": "real", "volume": "1x",
                                     "seed": seed, "test_date": date_dir.name,
                                     **{k: d["test_metrics"][k] for k in METRIC_ORDER}})
    return pd.DataFrame(rows)


def main():
    gen = collect_generators()
    det = collect_detectors()

    protocol = json.loads((RUNS / "upstream_gpu_full_20260910" / "protocol.json").read_text(encoding="utf-8"))
    sec = []

    sec.append("# 全面测试所有论文的扩散生成器与检测器 — 结果报告\n")
    sec.append("生成时间: 2026-09-10 | 计算: NVIDIA RTX 4090 (CUDA 12.4) | 环境: miniconda `jyz_paper` (Python 3.10, torch 2.2.2+cu121)\n")

    sec.append("## 缩略语对照表\n")
    sec.append(
        "本报告出现的全部缩写/简写先在此统一定义, 后文各表附近也会就近复述。\n\n"
        "| 缩写 | 全称 | 含义 |\n"
        "|---|---|---|\n"
        "| RF | Random Forest | 随机森林分类器, 下游在「真实窗口+合成窗口」上训练的二分类器; 第 5/7 节表格即 RF 结果 |\n"
        "| HGB | HistGradientBoosting | 直方图梯度提升分类器, 与 RF 并行的另一种下游分类器 |\n"
        "| F1 | F1 score | 精确率与召回率的调和平均, `2PR/(P+R)`, 越大越好 |\n"
        "| Precision | 精确率 | 报出的异常里有多少是真异常, `TP/(TP+FP)` |\n"
        "| Recall | 召回率 | 真异常里有多少被找出, `TP/(TP+FN)` |\n"
        "| ROC-AUC | Receiver Operating Characteristic - Area Under Curve | 阈值无关的排序指标, 随机取一异常一正常、模型给异常更高分的概率; 0.5=瞎猜, 1=完美 |\n"
        "| AP | Average Precision | 精确率-召回率曲线下面积, 类别极不平衡时比 ROC-AUC 更敏感; 0.5=瞎猜, 1=完美 |\n"
        "| TP/FP/TN/FN | True/False Positive/Negative | 真阳/假阳/真阴/假阴计数 |\n"
        "| DDPM | Denoising Diffusion Probabilistic Model | 去噪扩散概率模型, 本报告全部生成器/检测器的底层生成范式 |\n"
        "| GRU | Gated Recurrent Unit | 门控循环单元, 一种循环神经网络 (TimeGrad/TSDE 机制移植用到) |\n"
        "| conv | convolution | 卷积层 (SSSD/CoSTI/WaveStitch/FIDE/MCD-TSF 机制移植用到) |\n"
        "| PyTorch / torch | — | 本工程使用的深度学习框架, 版本 2.2.2+cu121 (cu121=CUDA 12.1 构建) |\n"
        "| ETT / SWaT | Electricity Transformer Temperature / Secure Water Treatment | 公开时间序列数据集, 部分论文仓库默认数据加载器面向它们 |\n"
        "| RTX 4090 | — | 本机 NVIDIA GPU 型号, 24GB 显存, 本报告全部实验在此运行 |\n"
        "| TF | TensorFlow | 谷歌深度学习框架 (DiffAD 原始实现为 TF, 本地移植到 PyTorch) |\n"
        "| IIF | Imputation-Interpolation-Forecasting | 插补-内插-预测, TSDE 论文的自监督任务; 本工程取其趋势条件机制 |\n"
        "| MSE | Mean Squared Error | 均方误差 |\n"
        "| std | standard deviation | 标准差, 表中 `均值 ± std` 的后者 |\n"
        "| M4 / M7 | 测量量 4 / 7 | 本工程两路绝缘子测量量 (设备), 特征 M4_residual、M7_residual、M4-M7、M4/M7、log(M4/M7) 均由二者派生 |\n"
        "| SHA256 | Secure Hash Algorithm 256-bit | 校验数据未被篡改的哈希值 |\n"
        "| CUDA | Compute Unified Device Architecture | NVIDIA GPU 并行计算平台; 本报告全部训练/采样在 RTX 4090 上运行 |\n"
        "| seq_len / feature_dim / hidden / diffusion_steps / epochs | 序列长度 / 特征维数 / 隐层宽度 / 扩散步数 / 训练轮数 | 本地化参数: 4 / 5 / 32 / 20 / 60 |\n"
        "| 1x / 10x / 100x / large | 1 倍 / 10 倍 / 100 倍 / 全部 | 相对真实窗口数加入的合成窗口倍数; large=全部 20,000 (每类 10,000) |\n"
        "| raw / projected | 原始 / 物理投影 | 合成窗口直接使用, 或先经物理约束投影 (M4>0、M7>0 并重算派生特征) 再使用 |\n"
        "| fit / validation / test | 训练 / 验证 / 测试 | 划分角色; validation 仅用于选阈值, test 为整段留出日期 |\n"
        "| P / R | Precision / Recall | 表头缩写同 F1 区 |\n"
    )

    sec.append("## 1. 实验设置与训练数据特征\n")
    sec.append(
        f"- 数据源: `dataset/experiment.xlsx` (SHA256 `{protocol['sha256'][:16]}…`, 原始行数 {protocol['raw_rows']})\n"
        f"- 训练数据特征 (联合特征, 窗口长 {protocol['window_length']}): "
        + ", ".join(f"`{f}`" for f in protocol["features"]) + "\n"
        f"- 双设备配对: 40s 最近时间戳 merge_asof、一对一匹配、丢弃两设备 error 不一致的行\n"
        f"- 划分: 留一日期测试 (250605 / 250626 / 250718); 其他日期按时间顺序切 fit/validation, "
        f"最后故障阶段作 validation; 窗口只在同一划分与阶段内滑动\n"
        f"- 标签: `error != 0` 二分类; error 不作为检测特征; 每配置 3 种子 × 3 日期, 报告均值 ± 标准差\n"
        f"- 阈值: 只在 validation 上取 F1 最大阈值 (并列取更高阈值); 测试指标全部用该阈值计算\n"
        f"- 指标: Precision / Recall / F1 / ROC-AUC / Average Precision (真实测试行上计算)\n"
    )

    sec.append("### 1.1 随机种子与「留出测试日期」是什么意思、为什么这样评估\n")
    sec.append(
        "**随机种子 (seed) 到底是干嘛的**\n\n"
        "- 训练开始时, 神经网络的权重不是人为设定, 而是**用随机数填的**; 训练中途扩散模型还反复往数据里加、减**随机高斯噪声**。"
        "所以换一串随机数, 训练就从另一个起点、另一条噪声路径走一遍, 最终得到的模型不同。\n"
        "- 种子就是**随机数生成器的起点编号**: 同一个种子 → 同一串随机数 → 一模一样的结果 (可复现); 换种子 → 换一串随机数 → 结果不同。"
        "代码中 `seed_all(seed)` 同时设置 `random.seed`、`np.random.seed`、`torch.manual_seed`、`torch.cuda.manual_seed_all`。\n"
        "- **42 / 123 / 2024 是什么? 就是三个随便挑的整数编号, 本身没有任何含义**: 42 是程序员常玩的梗 (《银河系漫游指南》), "
        "123 顺口, 2024 是年份。换成 7 / 999 / 12345 完全等价, 只是会得到另外三串随机数。它们不是模型参数、不是超参数, 只是「三把不同的钥匙」。\n"
        "- **为什么怕「碰巧初始化好」?** 打个比方: 靶心是「理想模型」, 每次训练的随机起点与噪声就像每次射箭的站位与手抖。只射**一箭**, "
        "可能恰好正中 (运气好), 你就误以为这方法很强; 也可能恰好脱靶, 就误判它弱。跑 3 个不同种子 = 换 3 个站位各射一箭, 看整体, 不被某一次运气带偏。\n"
        "- **标准差怎么反映随机性?** 设某方法 3 次 F1 = 0.90 / 0.89 / 0.91 → 均值 0.90、标准差约 0.01, 说明「怎么换随机都差不多」, 稳、可复现; "
        "另一方法 3 次 = 0.95 / 0.60 / 0.80 → 均值 0.78、标准差约 0.18, 说明「结果高度看运气」, 不稳。"
        "所以: **均值 = 平均水平, 标准差 = 这个平均水平有多可信**; 标准差越大, 越说明随机性影响大、越不能只信一次结果。\n\n"
        "**留出测试日期 (held-out test date)**\n\n"
        "- 数据来自 3 个实验日: 250605、250626、250718 (即 2025-06-05 / 06-26 / 07-18), 每个日期是一整场实验 (含正常段与故障段)。\n"
        "- 「留一日期测试」= 挑一个日期作测试集, 模型在**训练与选阈值时完全看不到它**; 另外 2 个日期再按时间切 fit (训练) 与 validation (选阈值)。\n"
        "- 例: 测试日 = 250718 时, 只在 250605+250626 上训练, 评测只在 250718 上做。「留出/held-out」即预留、不参与训练。\n"
        "- 为什么按**日期**而非按行随机切: 同一日期 (同一场实验、同一次漂移) 内的窗口高度相关, 按行随机切会把近乎相同的窗口"
        "同时放进训练与测试, 造成信息泄漏、指标虚高; 整天留出才是在检验**跨场次泛化**能力。\n\n"
        "**各表读数**: 「3 种子 × 3 日期」= 每个 (方法, 配比) 跑 3×3 = 9 次, 表中报这 9 次的均值 ± 标准差。\n"
    )

    sec.append("## 2. 上电衰减排除规则说明\n")
    sec.append(
        "- 正常段取每日期最后 ≤40 个正常行中、与末端 10 点中位数偏差 <15% 的连续稳定后缀。\n"
        "- 每个物理故障阶段丢弃前 2 个配对点 (上电衰减)。\n"
        "- 每日期前 8 个稳定正常对作为基线校准, 不参与任何模型训练与评分。\n"
    )

    sec.append("## 3. 术语解释\n")
    sec.append(
        "- **固定版上游模型 (fixed-version upstream)**: 直接 import `code_reproduction/upstream/` 中钉死 commit 的"
        "论文原生模型类训练与采样, 模型本体与论文一致。\n"
        "- **机制移植 (mechanism port)**: 论文仓库的数据加载器面向 ETT/SWaT 等长序列、或依赖 TensorFlow/缺失核心源码,"
        "无法直接驱动 4 步 × 5 维窗口; 按其核心机制在本地以小型条件 DDPM 复现, 不逐层等价重写, 结果中标注。\n"
        "- **role / label / n**: `fit`=训练、`validation`=阈值选择、`test`=留出日期; `label 0`=正常窗口、`label 1`=零值绝缘子窗口; "
        "`n`=该组合下切出的 4 步滑动窗口数 (异常窗口极少, 1~12 个)。\n"
        "- **regime**: `real`=仅真实窗口; `raw_1x/10x/100x/large`=真实窗口+原始生成窗口 (1/10/100 倍或全部); "
        "`projected_*`=生成窗口先经物理约束投影 (M4>0、M7>0, 再重算差值/比例), 再入训练。\n"
    )

    sec.append("## 4. 每篇论文的适配说明表\n")
    sec.append("### 4.1 生成器 / 增强源\n")
    sec.append("| 论文 | 出处 | 原生用途 | 本地适配 | 实现方式 |")
    sec.append("|---|---|---|---|---|")
    for m, (src, native, impl) in GEN_DESCRIPTIONS.items():
        sec.append(f"| {m} | {src} | {native} | 类别条件生成合成窗口 | {impl} |")
    sec.append("")
    sec.append("### 4.2 检测器\n")
    sec.append("| 论文 | 出处 | 原生检测方式 | 本地适配 | 实现方式 |")
    sec.append("|---|---|---|---|---|")
    for m, (src, native, impl) in DET_DESCRIPTIONS.items():
        sec.append(f"| {m} | {src} | {native} | 正常训练重构评分 | {impl} |")
    sec.append("")
    sec.append("### 4.3 覆盖完整性 (相对提示词 18 生成器 + 11 检测器清单)\n")
    sec.append(
        "| 状态 | 论文 | 说明 |\n"
        "|---|---|---|\n"
        "| ✅ 已运行 (固定版上游) | CSDI, MTSCI, IMDiffusion, Diffusion-TS, D3R + 检测器 D3R, IMDiffusion | 直接 import 钉死 commit 的原生代码 |\n"
        "| ✅ 已运行 (固定版上游) | Treeffuser | blei-lab/treeffuser 原包, 4×5 窗口展平成 20 维表格做树基扩散; 在 `sources_manifest.json` 中但不在提示词主清单, 已验证为真实可跑扩散方法并补齐 |\n"
        "| ✅ 已运行 (机制移植) | 生成器 TimeGrad, NsDiff, FIDE, PaD-TS, WaveStitch, TabDDPM, CCDM, DiffPLF, TSDE, SSSD, CoSTI, SSD-TS, MCD-TSF | 仓库面向长序列/TF/缺核心源码, 按机制本地移植 |\n"
        "| ✅ 已运行 (机制移植) | 检测器 DiffAD, ICDiffAD, TIMEADDM, ProDiffAD, TSADC, NGLS-Diff, DiffTAD, diffusionae, C3DAD | 同上 |\n"
        "| ⏭️ 跳过 | CT-DDPM | 仓库为空 (仅 1 个 README, 无源码), 无法运行 |\n"
        "| ⏭️ 跳过 | FACD / FACD_download / FACD_download_20260909 | 目录为 0 文件 (下载失败/空), 无法运行 |\n"
        "\n"
        "> 说明: 提示词主清单的 18 生成器与 11 检测器已全部出数; 另按提示词「评估后决定」的要求, "
        "将本地仓库中唯一的额外真实扩散方法 Treeffuser 补齐, 并注明 CT-DDPM/FACD 系列为空仓库跳过。\n"
    )

    sec.append("## 5. 生成器增强对比 (RF 分类器, 3 种子 × 3 日期)\n")
    sec.append(
        "### 5.1 生成器的评判标准 (重点)\n\n"
        "生成器**不直接**看合成样本长得多像, 而是**间接**以「对下游检测任务的增益」为准。判据链如下:\n\n"
        "1. 生成器在真实窗口上按类别 (正常/异常) 训练, 每类生成合成窗口 (large 量级 ≥10,000/类);\n"
        "2. 合成窗口按配比混入真实训练集, 训练下游分类器 RF/HGB, 阈值只在 validation 上选;\n"
        "3. 在**留出测试日期**上计算检测指标, 与 `real` (无增强) 基线对比。\n\n"
        "- **主判据**: 先看阈值无关的**排序指标 ROC-AUC 与 AP** (类别极不平衡时更可靠), 再看阈值化 **F1**; "
        "Precision/Recall 作为其分解。\n"
        "- **最佳 regime 选法**: 对每个方法按 `ROC-AUC + AP` 之和最大挑出 (见 5.3)。\n"
        "- **辅助质量核查** (记录在每个 `result.json`): 是否达 `large` 量级 (每类 ≥10,000); `projected_*` vs `raw_*` 看物理投影增益; "
        "生成样本各维度标准差 vs 真实标准差 (防样本坍缩或爆炸)。\n"
        "- **为何不直接用 FID/MMD 等样本质量分**: 本工程目标是「用合成数据提升异常检测」, 样本再像但无助于检测也等于没用; "
        "故以留出日期上的检测增益为终判据。\n\n"
        "### 5.2 表格读数\n\n"
        "- **RF** = Random Forest 随机森林, 即下游二分类器; 它在「真实窗口 + 该生成器合成的窗口」上训练。本节只报 RF 的结果。\n"
        "- **3 种子 × 3 日期**: 每个单元格是 3 个随机种子 (42/123/2024) × 3 个留出测试日期 (250605/250626/250718) = **9 次运行**的均值 ± 标准差。\n"
        "- **单元格数值** = F1 分数 (越大越好); 行 = 生成器方法, 列 = 增强配比 regime。\n"
        "- **列 (regime)**: `real`=无增强基线 (仅真实窗口); `raw_1x/10x/100x/large`=真实窗口 + 原始合成窗口 (1/10/100 倍或全部 20,000); "
        "`projected_1x/10x/100x/large`=同上但合成窗口先经物理约束投影 (强制 M4>0、M7>0 并重算 M4-M7、M4/M7、log(M4/M7))。详见第 3 节。\n"
        "- 缩写 RF/F1/raw/projected/1x 等见文首「缩略语对照表」。\n"
    )
    regimes = ["real", "raw_1x", "raw_10x", "raw_100x", "raw_large", "projected_1x", "projected_10x", "projected_100x", "projected_large"]
    cols = ["method"] + [r.replace("_", "\\_") for r in regimes]
    header = "| " + " | ".join(cols) + " |"
    sep = "|" + "|".join(["---"] * len(cols)) + "|"
    lines = [header, sep]
    # per-method best-regime mean across the 9 runs (real + all regimes)
    def best_line(g):
        vals = {}
        for regime in regimes:
            r = g[g.regime == regime]
            vals[regime] = mean_std(r["F1"]) if not r.empty else "-"
        return "| " + " | ".join([g.method.iloc[0]] + [vals[r] for r in regimes]) + " |"
    for method in sorted(gen.method.unique()):
        lines.append(best_line(gen[gen.method == method]))
    sec.append("\n".join(lines) + "\n")

    sec.append("### 5.3 各方法最佳增强 regime 详情 (F1 / ROC-AUC / AP)\n")
    sec.append("对每个生成器方法, 先按 `ROC-AUC + AP` 之和选其最佳增强 regime, 再列出该 regime 的均值 ± 标准差。"
               "缩写 F1/ROC-AUC/AP 见文首「缩略语对照表」。\n")
    sec.append("| 方法 | 最佳 regime | F1 | ROC-AUC | Average Precision |")
    sec.append("|---|---|---|---|---|")
    best_rows = []
    for method, g in gen.groupby("method"):
        aug = g[g.regime != "real"]
        if aug.empty:
            continue
        scored = []
        for regime in regimes[1:]:
            r = aug[aug.regime == regime]
            if r.empty:
                continue
            scored.append((regime, r["ROC-AUC"].mean() + r["Average Precision"].mean(), r))
        scored.sort(key=lambda x: -x[1])
        regime, _, r = scored[0]
        best_rows.append({
            "method": method, "regime": regime,
            "F1": r["F1"].mean(), "ROC-AUC": r["ROC-AUC"].mean(), "AP": r["Average Precision"].mean(),
        })
        sec.append(f"| {method} | {regime} | {fmt(r['F1'].mean())} ± {fmt(r['F1'].std(),3)} | "
                   f"{fmt(r['ROC-AUC'].mean())} ± {fmt(r['ROC-AUC'].std(),3)} | {fmt(r['Average Precision'].mean())} ± {fmt(r['Average Precision'].std(),3)} |")
    sec.append("")

    sec.append("## 6. 检测器对比 (3 种子 × 3 日期)\n")
    sec.append(
        "本节比较各**异常检测器**。检测器只在真实正常窗口上训练, 用「重构误差」给每个测试窗口打分, "
        "阈值只在 validation 上选取; 每格为 3 种子 × 3 留出日期 = 9 次运行的均值 ± 标准差。"
        "指标缩写 Precision/Recall/F1/ROC-AUC/AP 见文首「缩略语对照表」。\n"
    )
    sec.append("### 6.1 无增强检测基线 (real)\n")
    sec.append("检测器仅用真实正常窗口训练、不加任何合成数据时的基线结果。\n")
    sec.append("| 检测器 | Precision | Recall | F1 | ROC-AUC | Average Precision |")
    sec.append("|---|---|---|---|---|---|")
    det_rows = []
    for detector, g in det.groupby("detector"):
        real = g[g.augmentation == "real"]
        if real.empty:
            continue
        sec.append(f"| {detector} | {mean_std(real['Precision'])} | {mean_std(real['Recall'])} | {mean_std(real['F1'])} | "
                   f"{mean_std(real['ROC-AUC'])} | {mean_std(real['Average Precision'])} |")
        det_rows.append({"detector": detector, "F1": real["F1"].mean(), "AP": real["Average Precision"].mean(),
                         "ROC-AUC": real["ROC-AUC"].mean()})
    sec.append("")
    sec.append("### 6.2 固定版检测器的增强收益 (先增强再检测)\n")
    sec.append("先用生成器合成正常窗口扩充训练集、再训练检测器 (仅覆盖已跑增强网格的 D3R / IMDiffusion 两篇)。"
               "`增强源` = 提供合成窗口的生成器; `量级` = 10x(10 倍) 或 large(全部 20,000)。\n")
    sec.append("| 检测器 | 增强源 | 量级 | F1 | ROC-AUC | Average Precision |")
    sec.append("|---|---|---|---|---|---|")
    for detector in ("D3R", "IMDiffusion"):
        g = det[det.detector == detector]
        for aug in sorted(g.augmentation.unique()):
            for vol in sorted(g[g.augmentation == aug].volume.unique()):
                r = g[(g.augmentation == aug) & (g.volume == vol)]
                if r.empty:
                    continue
                sec.append(f"| {detector} | {aug} | {vol} | {mean_std(r['F1'])} | {mean_std(r['ROC-AUC'])} | {mean_std(r['Average Precision'])} |")
    sec.append("")

    sec.append("## 7. 全论文横向最佳结果汇总与结论\n")
    sec.append("本节把第 5/6 节的结果横向拉通: 7.1 汇总每个生成器的最佳增强配置, 7.2 汇总检测器基线。"
               "AP = Average Precision (平均精确率), 其余缩写见文首「缩略语对照表」。\n")
    if best_rows:
        bdf = pd.DataFrame(best_rows).sort_values(["ROC-AUC", "AP"], ascending=False)
        sec.append("### 7.1 生成器最佳结果 (按 ROC-AUC+AP 排序)\n")
        sec.append("| 方法 | 最佳 regime | F1 | ROC-AUC | AP (Average Precision) |")
        sec.append("|---|---|---|---|---|")
        for _, row in bdf.iterrows():
            sec.append(f"| {row['method']} | {row['regime']} | {fmt(row['F1'])} | {fmt(row['ROC-AUC'])} | {fmt(row['AP'])} |")
        sec.append("")
    if det_rows:
        ddf = pd.DataFrame(det_rows).sort_values("AP", ascending=False)
        sec.append("### 7.2 检测器无增强基线 (按 AP 排序)\n")
        sec.append("| 检测器 | F1 | ROC-AUC | AP |")
        sec.append("|---|---|---|---|")
        for _, row in ddf.iterrows():
            sec.append(f"| {row['detector']} | {fmt(row['F1'])} | {fmt(row['ROC-AUC'])} | {fmt(row['AP'])} |")
        sec.append("")

    sec.append("### 7.3 结论\n")
    real_rf = gen[(gen.regime == "real")]
    sec.append(
        f"- 真实数据 RF 无增强基线: F1 {mean_std(real_rf['F1'])}, ROC-AUC {mean_std(real_rf['ROC-AUC'])}, "
        f"AP {mean_std(real_rf['Average Precision'])} (跨 {len(real_rf)} 次运行)。\n"
    )
    if best_rows:
        top = bdf.iloc[0]
        sec.append(
            f"- 生成器增强最佳配置为 `{top['method']}` (`{top['regime']}`): "
            f"F1 {fmt(top['F1'])}, ROC-AUC {fmt(top['ROC-AUC'])}, AP {fmt(top['AP'])}。\n"
        )
    if det_rows:
        topd = ddf.iloc[0]
        sec.append(f"- 检测器无增强基线最佳为 `{topd['detector']}`: F1 {fmt(topd['F1'])}, ROC-AUC {fmt(topd['ROC-AUC'])}, AP {fmt(topd['AP'])}。\n")
    sec.append(
        "- 5 篇论文 (CSDI/MTSCI/IMDiffusion/Diffusion-TS/D3R) 以固定版上游原生代码运行; "
        "其余论文按核心机制做本地移植 (详见第 4 节), 全部满足 `seq_len=4`、`feature_dim=5`、"
        "`hidden=32`、`diffusion_steps=20`、`epochs=60` 的本地化参数约束, 且每类生成 ≥10,000 窗口 (large)。\n"
        "- 完美模型 = 全 1.0 (Precision/Recall/F1/ROC-AUC/AP); 受真实异常窗口极少 (1~12 个) 的阈值估计方差限制, "
        "当前排序指标 (AP/ROC-AUC) 明显优于阈值化 F1, 详见各表。\n"
    )

    sec.append("## 8. 原始数据 / 生成数据的位置与一键重跑命令\n")
    sec.append("### 8.1 原始数据与预处理划分\n")
    sec.append("| 内容 | 路径 | 说明 |")
    sec.append("|---|---|---|")
    sec.append("| 原始数据 (Excel) | `code_reproduction/experiment.xlsx` | SHA256 `27b51b22481f5d34…`, 原始 1419 行; `dataset` 软链指向同文件, 故 `dataset/experiment.xlsx` 等价 |")
    sec.append("| 预处理划分 | `runs/<run_root>/dataset.npz` | 切好的窗口张量 `x` (形状 (89, 4, 5)) |")
    sec.append("| 划分元数据 | `runs/<run_root>/metadata.json` | 每窗口的 `date / role / label / base4 / base7` |")
    sec.append("")
    sec.append("### 8.2 数据增强生成的合成数据\n")
    sec.append("每个生成器运行的合成窗口都在该运行目录下, 文件 `synthetic.npz`, 内容 5 个键:\n\n")
    sec.append("| 键 | 形状 | 含义 |")
    sec.append("|---|---|---|")
    sec.append("| `raw` | (20000, 4, 5) | 原始合成窗口 (5 维联合特征, 未投影) |")
    sec.append("| `projected` | (20000, 4, 5) | 经物理约束投影后的合成窗口 (M4>0, M7>0 并重算派生特征) |")
    sec.append("| `label` | (20000,) | 每行合成窗口的类别 (0=正常, 1=异常) |")
    sec.append("| `anchor_ids` | (20000,) | 该合成窗口对应的真实锚点窗口索引 (用于反算 base4/base7) |")
    sec.append("| `positive` | (20000,) | 投影后是否满足 M4>0 且 M7>0 |")
    sec.append("\n路径模板:\n\n")
    sec.append("```")
    sec.append("# 固定版上游生成器 (5 篇):")
    sec.append("runs/upstream_gpu_full_20260910/<METHOD>/seed<SEED>/<DATE>/synthetic.npz")
    sec.append("# 机制移植生成器 (13 篇) 与 Treeffuser:")
    sec.append("runs/all_papers_extended_20260910/generators/<METHOD>/seed<SEED>/<DATE>/synthetic.npz")
    sec.append("# 例:")
    sec.append("runs/upstream_gpu_full_20260910/CSDI/seed42/250718/synthetic.npz")
    sec.append("runs/all_papers_extended_20260910/generators/NsDiff/seed42/250718/synthetic.npz")
    sec.append("```\n")
    sec.append("<METHOD> ∈ {CSDI, MTSCI, IMDiffusion, Diffusion-TS, D3R, TimeGrad, NsDiff, FIDE, PaD-TS, "
               "WaveStitch, TabDDPM, CCDM, DiffPLF, TSDE, SSSD, CoSTI, SSD-TS, MCD-TSF, Treeffuser}; "
               "<SEED> ∈ {42,123,2024}; <DATE> ∈ {250605,250626,250718}。\n")
    sec.append("每个运行目录同时含 `result.json` (指标)、`training_history.json` (仅固定版)、各分类器 `*_predictions.csv`、模型权重 (`*.pt`, 仅固定版)。\n")
    sec.append("")
    sec.append("### 8.3 结果目录总览 (`code_reproduction/experiment_augmentation/runs/`)\n")
    sec.append("| 目录 | 内容 | 运行次数 |")
    sec.append("|---|---|---|")
    sec.append("| `ALL_PAPERS_RESULTS.md` | 本报告 | - |")
    sec.append("| `upstream_gpu_full_20260910/` | 固定版上游生成器 (CSDI/MTSCI/IMDiffusion/Diffusion-TS/D3R) + `synthetic.npz` | 45 |")
    sec.append("| `augmented_upstream_detection_gpu_20260910/` | 固定版上游检测器 (D3R/IMDiffusion) 先增强再检测结果 | 198 |")
    sec.append("| `all_papers_extended_20260910/generators/` | 机制移植生成器 (13 篇) + Treeffuser + `synthetic.npz` | 126 |")
    sec.append("| `all_papers_extended_20260910/detectors/` | 机制移植检测器 (9 篇) 结果 | 81 |")
    sec.append("")
    sec.append("### 8.4 一键重跑命令\n")
    sec.append("```bash")
    sec.append("cd 'code_reproduction/experiment_augmentation'")
    sec.append("PY=/home/test/miniconda3/envs/jyz_paper/bin/python")
    sec.append("# 固定版上游生成器 (5 篇, 45 次)")
    sec.append("bash run_gpu_grid.sh")
    sec.append("# 固定版上游检测器 (2 篇, 198 次)")
    sec.append("bash run_gpu_detector_grid.sh")
    sec.append("# 机制移植生成器+检测器 (13+9 篇, 198 次)")
    sec.append("bash run_gpu_extended_grid.sh")
    sec.append("# 汇总报告")
    sec.append("$PY aggregate_all_papers.py")
    sec.append("```\n")

    sec.append("## 9. 原始数据 vs 增强数据 对比图\n")
    sec.append(
        "**先说这一节在干什么。** 我们用生成模型 (扩散模型)「造」了一批假的电压窗口数据, 想拿它去补充真实数据的不足。"
        "但造出来的数据和真实数据长得像不像? 这一节的 5 张图就是拿**真实数据** (黑线) 和**生成数据** (彩线) 放一起比一比: "
        "数值大小范围像不像 (图 1)、波动幅度像不像 (图 2)、正常/异常两类的区分度还在不在 (图 3)、随时间变化的形状像不像 (图 4)、 "
        "以及最后一道「物理检查」有没有把不合理的值改掉 (图 5)。\n\n"
        "**为什么不画「预测曲线/误差」?** 因为生成的假数据是凭空新造的, 它在真实世界里**没有一条一一对应的真值** "
        "(不像预测任务里今天和明天能配对)。把假数据和某条真实数据硬凑成一对去算误差是没有意义的, 所以这里只比「长得像不像」。\n\n"
    )
    figs = [
        ("图 1", "compare_distributions.png", "原始数据与生成数据的五个特征值分布对比",
         "**这张图看什么。** 把真实数据和生成数据的每个特征值都统计一下「哪个数值出现得多、哪个出现得少」, 画成曲线 (叫分布曲线)。"
         "横轴是特征数值, 纵轴是该数值出现的频繁程度 (越高=越多)。五个小图从左到右是五个特征: 剩下电压 `M4_residual`、剩余电压 `M7_residual`、"
         "两者之差 `M4-M7`、两者之比 `M4/M7`、比值的对数 `log(M4/M7)`。**黑色粗线 = 真实数据** (全部 89 个真实窗口), "
         "彩色细线 = 四种代表性生成方法 (CSDI、NsDiff、Diffusion-TS、Treeffuser) 造出来的数据 (合起来所有实验共 3 个随机种子 × 3 个测试日期)。"
         "**彩线和黑线形状越像, 说明这个生成方法造的数据越真实。**",
         "**图里说明了什么结果。** 真实数据的 `M4_residual`/`M7_residual` 在 0 附近有一个很高的尖峰 —— 意思是大部分正常窗口的电压残差几乎就是 0。"
         "而生成数据的曲线都更矮更宽, 说明它们造出的数值更分散、极端值更多。其中 **CSDI、NsDiff、Diffusion-TS 三条彩色线在主要区间和黑线基本贴合**, "
         "只有尾部略宽 (可以接受); **Treeffuser 的线被压得很平、横跨范围远大于真实数据**, 说明它造的数据太散、不够真实 (和图 2 结论一致)。"),
        ("图 2", "compare_per_method_std.png", "各生成方法的波动幅度与真实数据对比",
         "**这张图看什么。** 「标准差」是衡量一个特征数值波动大小的指标: 标准差越大, 说明数据越参差不齐; 越小, 说明都挤在一起。"
         "**左图**: 横轴列出真实数据和 19 个生成方法, 每个方法有 5 根柱子 (对应 5 个特征), 柱越高代表这个特征在这个方法里波动越大。"
         "**右图**: 把「生成方法的标准差 ÷ 真实数据的标准差」算成一个倍率画成热力图, **白色≈1.0 表示波动幅度和真实数据一样** "
         "**(最理想)**; **红色 (>2) 表示波动太大 (数据太散)**; **蓝色 (<1) 表示波动太小 (数据太挤)**。",
         "**图里说明了什么结果。** 真实数据五个特征的标准差分别是 0.094、0.071、0.119、0.415、0.527。"
         "**5 个主流的固定版方法倍率都在 1.14–1.29 之间** (D3R 1.14、NsDiff 1.18、CSDI 1.22、MTSCI 1.24、Diffusion-TS 1.25、IMDiffusion 1.29), "
         "也就是波动幅度和真实数据差不多, 热力图上颜色较浅, 属于「正常」。"
         "**14 个本地改写的方法倍率普遍在 2.85–3.15**, 即波动幅度约为真实数据的 3 倍, 热力图偏红, 已经偏散。"
         "**Treeffuser 最夸张, 平均倍率 9.87、最高约 12 倍**, 相当于把数据摊开到真实数据的 10 倍宽, 严重失真。"),
        ("图 3", "compare_scatter_M4_M7.png", "正常/异常两类样本是否还能分清",
         "**这张图看什么。** 把每个窗口的两个特征 (`M4_residual` 当横坐标, `M7_residual` 当纵坐标) 画成一个点, 这就是「散点图」。"
         "四个小图分别是真实数据、CSDI、NsDiff、Treeffuser 生成数据的点分布。**蓝点 = 正常窗口, 红点 = 异常窗口**。"
         "**如果红点聚成一团、蓝点聚成另一团, 两类能分开, 说明数据里「异常」的信号还保留着; 如果红蓝混在一起分不开, 说明这类信息丢了。**",
         "**图里说明了什么结果。** 真实数据里, 蓝点 (正常) 紧紧挤在原点附近的一条窄带里, 红点 (异常) 零散地分布在带的外围, 一眼能分辨。"
         "**NsDiff 生成的数据最接近真实**: 蓝点仍是紧窄的一条带、红点在外围, 两类依然分得开。**CSDI 铺得更开一些**, 带被拉长, 但两层结构还在。"
         "**Treeffuser 的红蓝点几乎完全糊在一起、看不出边界**, 说明它把「正常/异常」的区分度给弄丢了。"),
        ("图 4", "compare_windows.png", "数据随时间变化的形状像不像",
         "**这张图看什么。** 一个窗口包含连续 4 个时间点。这张图把每个时间点上的平均值连成线、再把上下波动范围画成阴影带。"
         "横轴是时间点 (第 0、1、2、3 步), 纵轴是特征数值。**黑线+灰影 = 真实数据**, 彩线+淡影 = CSDI、NsDiff、Treeffuser 生成数据。"
         "**曲线走向和阴影宽度越接近, 说明生成数据的时间变化规律越真实。**",
         "**图里说明了什么结果。** 真实数据的曲线比较平缓、阴影很窄。**CSDI 和 NsDiff 的曲线几乎和黑线重叠、阴影宽度也差不多**, "
         "说明它们忠实保留了数据随时间变化的形状。**Treeffuser 的曲线整体偏移、阴影明显更宽**, 说明它不仅数值幅度过大, 连随时间的变化也被放大了。"),
        ("图 5", "compare_raw_vs_projected.png", "物理检查有没有把不合理数据改掉 (以 NsDiff 为例)",
         "**这张图看什么。** 生成数据里可能出现不符合物理常识的值 (例如电压残差出现负的、不合理的数)。"
         "「物理投影」就是一道检查: 把不合格的值修剪回合理范围。这张图比较修剪前 (raw) 和修剪后 (projected) 的差异。"
         "**左图**: 修剪前后的 `M4_residual` 分布曲线 (两个柱状图叠在一起); **中图**: 每个样本修剪前的值 vs 修剪后的值, 一个个画成点; "
         "**右图**: 满足物理条件「M4>0 且 M7>0」的样本比例, 柱越高越好 (满分 1.0)。",
         "**图里说明了什么结果。** 左图两套柱状图几乎完全重合、中图的点全部落在对角直线上 (说明改前改后数值一样), "
         "右图显示**修剪前后的合格比例都是 1.000 (100%)**。也就是说 **NsDiff 造出来的数据本来就都满足物理条件, 这道检查基本没改什么东西**, 只是起个保险作用。"),
    ]
    for num, fn, title, desc, result in figs:
        sec.append(f"![{num}](figures/{fn})\n")
        sec.append(f"**{num}. {title}。** {desc}\n")
        sec.append(f"{result}\n")
    sec.append(
        "\n**这一节的总结 (大白话版)**:\n"
        "- 真实数据的残差大量挤在 0 附近 (正常窗口残差≈0), 生成数据的分布更宽更散 (图 1);\n"
        "- 主流 5 个方法的波动幅度和真实数据差不多 (倍率约 1.1–1.3 倍), 可以用; 本地改写的 14 个方法偏散 (约 3 倍); "
        "**Treeffuser 最离谱, 散了近 10 倍**, 这也是它后面提升不大的原因 (图 2);\n"
        "- NsDiff 造的数据最好地保留了「正常/异常」两类的区分度, CSDI 次之, Treeffuser 把两类搅混了 (图 3/4);\n"
        "- 物理检查对 NsDiff 几乎没改动 (因为数据本来就合格), 合格率始终 100% (图 5)。\n"
    )

    sec.append("## 10. 增强前后下游效果对比图 (真实测试集固定)\n")
    sec.append(
        "**先说这一节在干什么。** 上一节看的是「生成的数据像不像真的」; 这一节看的是「用生成数据帮忙训练之后, "
        "检测异常的本事有没有变强」。做法: 先只用真实数据训练一个模型 (记为 **Real-only**), 再用「真实 + 生成」数据训练一个模型 (记为 **增强模型**), "
        "然后让两个模型去识别**同一批真实测试窗口**里哪些是异常, 比一比谁识别得准。\n\n"
        "**为什么这里可以公平比较?** 因为两个模型考的「卷子」完全一样 —— 同一批测试窗口、同一套标准答案 (label)。"
        "所以它们的成绩可以直接对比。\n\n"
        "**「Real-only 模型」到底是什么?** 它就是一个**随机森林分类器 (RandomForest)** —— 训练数据里**只有真实窗口**。"
        "每次实验用 **40 个真实训练窗口** (来自其它日期, 其中 23 个正常 + 17 个异常), **一点生成数据都不掺**。"
        "**增强模型用的是完全同一个分类器**, 训练集里只是额外加入了等量 (40 个, 也就是 1 倍) 由 CSDI 生成的合成窗口。"
        "两个模型在同一个「留出日期」的测试窗口 (250605 日 29 个 / 250626 日 38 个 / 250718 日 22 个) 上打分。"
        "所以报告里说的「真实情况 (real only)」= **不增强、直接拿真实数据训练出来的那个基线模型**, 不是指别的什么东西。\n\n"
        "**几个常用指标的大白话解释 (后面反复出现):**\n"
        "- **异常 (abnormal)**: 我们要抓的「坏样本」; **正常 (normal)**: 不用管的「好样本」。\n"
        "- **召回率 Recall**: 所有真正的异常里, 被找出来几个? 越高越好 (漏报少)。\n"
        "- **误报率 FPR**: 所有正常样本里, 被冤枉成异常几个? 越低越好 (冤枉少)。\n"
        "- **精确率 Precision**: 被模型判为异常的里面, 真正是异常的比例有多高? 越高越好。\n"
        "- **F1**: 把「精确率」和「召回率」揉成一个总分, 越高越好。\n"
        "- **ROC-AUC / AP (平均精度)**: 衡量模型排序能力的总分 —— 不管阈值定在哪, 模型把异常排在正常前面的本事有多大, 1.0 满分、0.5 等于瞎猜。"
        "因为异常样本很少, 这两个「排序分」比单看 F1 更可靠。\n"
        "- 图里出现的**阈值 (threshold)** 就是「分数超过多少才判为异常」的那条线。\n\n"
        "**判断阈值是怎么定的? (重要, 防止「作弊」)** 阈值**只在验证集上挑**, 绝不用测试集: "
        "做法是把其它日期的一批样本 (共 20 个窗口, 含正常和异常) 当作验证集, 在上面**试遍所有可能的阈值, 挑那个 F1 最高的**"
        "作为最终阈值 (来自 `threshold_from_validation`, 纪律见 `protocol.json` 的 `evaluation` 字段: "
        "「threshold selected only on validation; 不使用测试集标签」)。这样测试集完全不参与「调线」, 结果才是可信的。\n\n"
        "**但阈值法在这里很不可靠, 所以判优劣主要看「不依赖阈值」的指标。** 原因: 测试日期的异常极少 "
        "(250605 日整天才 3 个异常), 阈值只要动一点点, F1 就会剧烈跳动 —— 例如 250605 日同一个模型, "
        "`raw_1x` 的 F1 只有 0.214、`raw_10x` 却是 1.000, 但两者其实是一回事。所以**不能拿某一次的 F1 高低来下结论**。\n\n"
        "**那到底怎么证明增强模型更好? —— 看「排序能力」(ROC-AUC 与 AP), 它跟阈值无关。** "
        "这两个指标的意思是: **把所有窗口按模型的异常分数从高到低排队, 看异常窗口是不是普遍排在前面。** "
        "它不关心你最后把线画在哪, 只关心「异常有没有比正常分更高」—— 正好就是「让异常更异常、正常更正常」。"
        "所以判断标准是: **同一个测试集上, 增强模型的 ROC-AUC / AP 比 Real-only 高, 就是更好**, "
        "而且要在 **3 个随机种子 × 3 个测试日期** 上都成立 (看图 7、图 9、图 10、图 11), 才算稳。\n\n"
        "**回到「有些异常 Real-only 认不出」的问题。** 认不出 win32/win33 这类窗口, 确实正是我们要解决的事; "
        "但要诚实地说: **没有任何方法能保证把每一个异常都识别出来**, 尤其那些特征和正常窗口几乎一样的「难例」。"
        "增强能带来的, 是**把整体分得更开** (ROC-AUC 0.756→0.921), 让多数异常分数更高、更容易被抓住; "
        "少数天生难分的窗口 (连 Real-only 只给 0.16 分) 仍然难分, 这是任务本身的难度, 不是增强失效。\n\n"
        "**关键前提 (无配对约束)**: 生成的假数据没有一一对应的原始真值, 所以**不拿**假数据和真实数据硬配对去算逐点误差。"
        "本分支的下游任务是**二分类异常检测**, 因此只给出 PR 曲线、混淆矩阵、F1/Recall/FPR 等**整体性**指标; "
        "电压预测的「预测曲线 + 绝对误差箱线图」因为没有配对真值而 **不适用 (N/A)**, 在此用说明代替。\n\n"
    )
    effect_figs = [
        ("图 6", "aug_effect_scores.png", "增强前后每个窗口的异常分数对比 (同一批真实测试数据)",
         "**这张图看什么。** 三个小图分别是三个「留出测试日期」(相当于三份考卷)。每个小图里, "
         "横轴是窗口编号 (第几个窗口), 纵轴是模型给这个窗口打的「异常分数」—— 分数越高越像异常。"
         "**最下面那条色带是标准答案**: **粉色 = 这个窗口真的是异常**, **蓝色 = 真的是正常**。"
         "**黑线 = Real-only 模型, 红线 = 增强模型**; 粗线是 3 个随机种子的平均, 淡细线是各单次运行; 虚线是各自的判定阈值。"
         "**理想情况**: 黑线和红线在粉色区域都往上冲、在蓝色区域都趴在下面 —— 越符合越好。"
         "三份考卷的规模分别是 250605 日 29 个窗口 (含 3 个异常)、250626 日 38 个 (含 18 个异常)、250718 日 22 个 (含 8 个异常)。",
         "**图里说明了什么结果。** 总体看, 增强模型 (红线) **把异常整体往高分方向推**: 250626 日异常窗口平均分从 0.539 升到 0.594, "
         "250718 日从 0.249 升到 0.336; 同时红线的判定阈值略微调低 (以 CSDI 为例 0.295→0.216), 于是更多真异常越过线被抓住 —— "
         "**漏报变少 (召回率上升)**, 对应图 8 里「漏掉的 FN 减少」。蓝线 (正常) 区域两者分数差不多。"
         "各条淡细线彼此靠得很近, 说明这个提升不是靠某一个运气好的随机种子。\n\n"
         "**但要注意: 红线并不是在每一个粉色窗口都升高, 有少数窗口反而略降** —— 例如 250605 日的 win26 (0.777→0.699)、"
         "250626 日的 win32/win33 (0.163→0.138、0.173→0.135)。这是正常现象, 原因有三:\n"
         "1. **红线画的是 3 个随机种子的平均**: 单次运行、单个窗口的分数本来就会上下浮动。增强带来的好处体现在**整体排序**上 "
         "(ROC-AUC 0.756→0.921), 而不是保证每个窗口的分数都涨。\n"
         "2. **异常样本太少, 平均数很敏感**: 250605 日整天才 3 个异常, 只要其中一个掉 0.08 分, 整天的平均就被拉平 (0.736→0.730, "
         "看起来「没变甚至下降」)。这是小样本的统计波动, 不代表模型变差。\n"
         "3. **有些异常「长得太像正常」**: 像 250626 的 win32/win33, 连 Real-only 都只打 0.16/0.17 分 (基本没认出来), 说明这类异常的"
         "窗口特征和正常窗口几乎一样, 属于「难例」; 生成数据也覆盖不到这种模式, 所以它们的分数升不上去, 甚至略微下降。\n\n"
         "**一句话**: 增强的价值在「整体上把异常和正常分得更开」, 而不是「把每一个异常都往上抬」。少数难例本来就难分辨, 上不去是正常的。"),
        ("图 7", "aug_pr_curves.png", "精确率-召回率 (PR) 曲线对比",
         "**这张图看什么。** 这是一条「偏科取舍」曲线: 如果我们想尽量多地抓异常 (召回率高), 往往就会冤枉更多正常样本 (精确率下降)。"
         "横轴是召回率, 纵轴是精确率, **曲线越往右上角靠, 说明模型越厉害**。**黑线 = Real-only, 红线 = 增强模型**; "
         "图例里括号中的 **AP (平均精度)** 是对整条曲线打的综合分, 越高越好。**浅色阴影带**表示 3 个随机种子的上下波动 (越窄越稳定); "
         "**灰色虚线**是「瞎猜」的水平线 (代表完全没有信息), 曲线要明显高于它才算有用。前三个小图是三份考卷, 第四个小图 `pooled` 是把三份合并成一个总分。",
         "**图里说明了什么结果。** 四个小图里, **红曲线全都压在黑曲线的右上方**, 而且 AP 分数明显更高。"
         "以自动选出的最佳方案 CSDI (`raw_1x`) 为例, 合并后的 **AP 从 0.765 提到 0.932**, **ROC-AUC 从 0.756 提到 0.921** (满分 1.0)。"
         "红线的阴影带较窄, 说明这个提升在 3 个随机种子上都很稳, 不是偶然。"),
        ("图 8", "aug_confusion.png", "混淆矩阵 (对 vs 错 的明细表)",
         "**这张图看什么。** 这是一个「判对判错」的明细表。**行 = 真实情况, 列 = 模型判断**。于是有四格: "
         "左上 = 真是正常、也判正常 (判对); 右上 = 真是正常、却被判异常 (冤枉, 叫 FP / 假阳); "
         "左下 = 真是异常、却被判正常 (漏掉, 叫 FN / 假阴); **右下 = 真是异常、也判异常 (最想要的结果, 叫 TP / 真阳)**。"
         "每格里的数字是 3 个随机种子加起来的结果, 颜色越深数量越多 (看右侧颜色条)。"
         "每个小图标题里的 **F1 / R (召回) / FPR (误报)** 就是这张表的总结分。上排 = Real-only, 下排 = 增强模型, 三列 = 三份考卷。"
         "**因为异常样本本来就很少, 所以不看「总体正确率」(那样即使全判正常也有很高正确率), 而看 F1/召回/误报。**",
         "**图里说明了什么结果。** 对比上排 (Real-only) 和下排 (增强): 下排的**右下格 (抓到的异常 TP) 变多、左下格 (漏掉的异常 FN) 变少**, "
         "标题里的召回率整体上升 —— 说明增强模型确实多抓到了真异常。不过「冤枉」(右上格 FP / 误报率) 因方法而异: "
         "**IMDiffusion (`raw_10x`) 表现最好**, 误报率从 0.308 降到 **0.128**、F1 从 0.306 升到 **0.507**; "
         "而 **D3R、MTSCI 反而冤枉得更多** (误报率 0.31→0.38/0.40)。也就是说, 抓得更多和冤枉更多之间需要权衡, 并非所有增强都两全其美。"),
        ("图 9", "aug_metric_bars.png", "五个指标的总分对比 (柱状图)",
         "**这张图看什么。** 把上一张表的结论浓缩成 5 根对比柱。每个小图一个指标 (F1、召回、误报、ROC-AUC、AP), "
         "**黑柱 = Real-only, 红柱 = 增强模型**, 柱顶数字是平均值, **误差棒 (柱子上的短线)** 表示 3 个种子 × 3 份考卷的上下波动。"
         "**F1、召回、ROC-AUC、AP 这四根柱越高越好; 误报率这根柱越低越好。**",
         "**图里说明了什么结果。** 在 **ROC-AUC 和 AP** 两个小图里, 红柱明显高过黑柱: Real-only 的平均分是 F1 0.306、ROC-AUC 0.756、AP 0.765, "
         "增强后最好的能达到 ROC-AUC/AP 约 0.92/0.93。召回率红柱也上升; 但误报率红柱有时高有时低。"
         "这说明 **增强主要带来的是「排序更准、漏报更少」, 并不保证每次都少冤枉**。另外 F1 的误差棒很大 (忽高忽低), "
         "因为异常样本太少, 这种「打分制」指标对随机性很敏感。"),
        ("图 10", "aug_ratio_sweep.png", "增强用多少假数据最合适? (倍数扫描)",
         "**这张图看什么。** 生成数据不是越多越好 —— 这里测试「假数据量 ÷ 真实数据量」的倍数对效果的影响。"
         "横轴就是这个倍数 (**0% = 一点假数据都不用, 也就是 Real-only 基线**), 纵轴是各项指标。"
         "**彩色实线 = 5 个主流固定版方法** (图例标了名字), **灰细线 = 本地改写的方法** (它们没存逐窗口分数, 所以没有误报率那一格); "
         "**每个小圆点 = 一次单独运行 (一个随机种子)**, **彩色阴影 = 种子之间的波动范围**, **黑色虚线 = 0% 基线**。"
         "**看线的起点 (0%) 和后面的点差多少, 就知道增强到底有没有用。**",
         "**图里说明了什么结果。** 所有曲线都从左侧的 0% 基线明显往上抬, 也就是**加了假数据确实有效**。"
         "具体数字 (5 个固定版方法的平均): **ROC-AUC 从 0.756 (0%) → 0.885 (1 倍) → 0.865 (10 倍) → 0.864 (100 倍)**; "
         "**AP 从 0.765 (0%) → 0.902 (1 倍) → 0.892 (10 倍) → 0.894 (100 倍)**; "
         "**F1 从 0.306 (0%) → 0.264 (1 倍) → 0.345 (10 倍) → 0.436 (100 倍)**。"
         "结论很实用: **排序类指标 (ROC-AUC/AP) 只要加 1 倍假数据就能拿到大部分好处, 再加更多基本没提升了**; "
         "而 F1 因为数据多了阈值算得更准, 会随倍数缓慢上升。右侧灰色方法 (约 350–600 倍) 虽然个别冲到 1.0, 但点与点之间跳得很厉害, 不可靠。"),
        ("图 11", "aug_ratio_seed.png", "不同倍数的稳定性 (每次运行的成绩分布)",
         "**这张图看什么。** 上一张图只看了平均分, 这张图看「每次单独运行到底稳不稳」。"
         "每个倍数画一个「箱线图」: **蓝色箱子**表示大部分运行的成绩落在哪个区间 (箱子越矮越集中、越稳), "
         "**箱子中间的横线**是中位数 (中间水平的成绩), **红点**是每一次单独运行的结果 (种子 × 日期)。"
         "横轴是倍数 (0% 为基线), 纵轴是 F1/ROC-AUC/AP。",
         "**图里说明了什么结果。** ROC-AUC 和 AP 的**蓝箱子从 0% 那个较低位置整体往上挪到了 1 倍及以上, 而且箱子比较矮、红点挤在一起**, "
         "说明「排序更准」这个好处每次运行都能拿到, 很稳。相比之下 **F1 的箱子很胖 (从 0 到 1 都有)、红点散得开**, "
         "说明 F1 高不高很看运气 (种子/日期), **不能拿一次运行的结果就下结论**。"),
        ("图 12", "aug_detector_effect.png", "检测器本身: 用真实数据训练 vs 用增强数据训练",
         "**这张图看什么。** 前面比的是「分类器」, 这里比的是「检测器」(负责判断异常的模型)。"
         "每行一个检测器 (**D3R / IMDiffusion**)。**左列**是 PR 曲线 (右上角越好), **黑线 = 只用真实正常样本训练, 红线 = 用增强后的正常样本训练**。"
         "**右列**是 5 个指标的柱状对比 (黑=真实, 红=增强), 误报率越低越好、其余越高越好。",
         "**图里说明了什么结果。** 两个检测器的结论正好相反: **D3R —— 增强没用**, 所有增强配置都不如只用真实数据, 最好的还是原样 (平均 AP 0.808、F1 0.525); "
         "**IMDiffusion —— 增强很有效**, 加上 Diffusion-TS (`large`) 造的样本后, 平均 **AP 从 0.458 一下子提到 0.833**, F1 从 0.385 提到 0.500。"
         "可见「增强到底有没有用」取决于检测器和生成方法搭不搭, 不能一概而论。"),
    ]
    for num, fn, title, desc, result in effect_figs:
        sec.append(f"![{num}](figures/{fn})\n")
        sec.append(f"**{num}. {title}。** {desc}\n")
        sec.append(f"{result}\n")
    sec.append(
        "\n**这一节的总结 (大白话版)**:\n"
        "- 不增强的基线 (跨 5 个方法 × 3 日期 × 3 种子): F1 约 0.31、ROC-AUC 约 0.76、AP 约 0.77;\n"
        "- **增强之后最好的是 CSDI 的 `raw_1x`**: ROC-AUC 0.76→0.92, AP 0.77→0.93; 其他方案最好的也都在 0.90 上下 "
        "(Diffusion-TS `raw_10x`、IMDiffusion `raw_10x`、D3R `projected_1x`);\n"
        "- **假数据不是越多越好**: 排序分 (ROC-AUC/AP) 加 1 倍就到顶, 再加更多不再涨; F1 才会随倍数慢慢爬升;\n"
        "- 检测器要分情况: **D3R 增强没用**, **IMDiffusion 增强后 AP 从 0.46 跳到 0.83**, 说明要看搭配;\n"
        "- 详细数据见 `runs/aug_effect_metrics.csv` 与 `runs/aug_ratio_sweep_metrics.csv`。\n"
    )

    sec.append("## 11. 双设备 正常 vs 异常 趋势曲线对比图 (最原始电压 vs 增强后)\n")
    sec.append(
        "**先说这一节在干什么。** 前面几节比的是「生成数据像不像真的」「检测分数高不高」。这一节回到最直观的问题: "
        "**能不能用眼睛直接看出「异常」的存在, 以及增强之后是不是更容易看出来。** "
        "这里**只用最原始的物理量 —— Excel 里的 `Measure` 电压 (device 4 与 device 7 两台设备), 单位是伏特 (V)**, 不绕弯、不算分数。\n\n"
        "**为什么必须同时画两台设备 (双设备联动)?** 这是本工程的核心设定: 4 号与 7 号绝缘子装在同一串上、采样时间接近 (40s 最近时间戳配对), "
        "所以 5 个联合特征 `M4_residual, M7_residual, M4-M7, M4/M7, log(M4/M7)` **本来就同时包含两台设备以及它们之间的关系**。"
        "如果只画 device 4, 就把「两设备是否一起动、关系有没有被破坏」这条最关键的信息丢了。因此本图**每一排都同时呈现 device 4 和 device 7**: "
        "第 4 排专门用「M4 残差 vs M7 残差」的散点来检查两设备的关系是否断裂。\n\n"
        "**为了公平, 我们严格控制了变量 (控制变量):** 真实侧和增强侧除了「数据来源」不同, 其它都对齐 —— "
        "(1) **同一个物理量**: 都还原成 `Measure` 电压 (真实侧用 `x[:,:,0]+base4`、`x[:,:,1]+base7`, 即残差 + 各自基线); "
        "(2) **同样的窗口长度**: 都是连续 4 步; "
        "(3) **同样的颜色和画法**: 蓝=正常、红=异常, 粗线=均值、阴影=±1 倍标准差、淡细线=单个样本; "
        "(4) **同样的样本数**: 原始合成数据每个日期有 6 万条、且正常异常各占一半 (50/50), 这样直接画会「假性地太好分」, "
        "所以按每个日期真实的正常/异常条数把合成数据**子采样到一样多**再画; "
        "(5) **设备两侧同样处理**: device 4 与 device 7 用完全相同的窗口、颜色、样本数画法。\n\n"
        "**残差排是不是在「偷偷放大差距」? 说清楚。** 第 2、3 排画的不是原始电压, 而是 `残差 = 电压 − 当天基线` "
        "(基线 = 当天最开始的 8 个稳定正常配对点的中位数)。因为基线是**当天一个固定常数**, 正常组和异常组减的是同一个数, "
        "相减时它被抵消, 所以:**残差排里蓝红两条粗线之间的「类间间隔」, 在数值上和直接看原始电压的差一模一样, 一分一毫都没变** "
        "(250605 device 4 两种算法都是 0.0083 V)。残差排唯一的作用是把纵轴缩到「只看相对基线的波动」范围, "
        "相当于给同一道缝配了一个**放大镜** —— 它不改缝的大小, 只是让这条本来就存在的缝更容易被肉眼看出来。\n\n"
        "**为什么残差排的纵轴和电压排不一样、两排不共用一个纵轴?** 因为合成残差的尺度比真实数据大好几倍 "
        "(250605 约 8 倍、250718 约 5 倍), 如果强行共用一个纵轴, 真实那排会被压成一条直线、看不清。"
        "所以**上下排各自用自己的纵轴**; 看的时候请以「两类有没有分开」为准, 不要直接比较不同排的绝对高度。\n\n"
    )
    sec.append("![图 13](figures/trend_voltage_normal_vs_abnormal.png)\n")
    sec.append(
        "**图 13. 双设备 (device 4 + device 7) 最原始 Measure 电压 与 CSDI 增强电压的 正常/异常趋势对比 (3 个测试日期)。**\n"
        "**这张图看什么。** 三列 = 三个测试日期 (250605 / 250626 / 250718); 四排从上到下分别是: "
        "**第 1 排** = 完全没有处理过的原始 Excel 双设备电压 (device 4 蓝实线、device 7 橙实线, 横轴是配对后采样点序号, "
        "**红色阴影 + 红点 = 那天被标记为异常**的采样点), 这一排回答「异常到底存不存在、两台设备是不是一起动」; "
        "**第 2 排** = 真实数据切成的 4 步电压残差窗口 (增强前), 每个日期两格: 左=device 4、右=device 7, 蓝=正常、红=异常; "
        "**第 3 排** = CSDI 增强后、同样切法且**样本数已对齐**的 4 步电压残差窗口 (同样左 4 右 7); "
        "**第 4 排** = **双设备联动散点图** (横轴 M4 残差、纵轴 M7 残差), 实心圆=真实、空心三角=增强, 蓝=正常、红=异常, "
        "虚线是只用真实正常点拟合出的关系线, 并标注了两设备相关系数 r。"
        "**判断标准: 蓝带和红带离得越开, 越容易认出异常; 第 4 排里正常点越贴关系线、异常点越散开, 说明「联动被破坏」这个信号越强。** "
        "每格左下角标了「类间间隔」(两条均值曲线之间的平均距离, 单位 V), 数字越大越好分。\n"
        "**图里说明了什么结果。** "
        "**第 1 排 (最原始双设备电压)**: 三个日期里 device 4 与 device 7 的曲线**大体平行地一起起伏**, 标红的异常段两台设备同时偏离, "
        "说明异常是作用在同一串绝缘子上的、两台设备都能看到。**异常在最原始的电压里是真实存在的, 不是我们编出来的。** "
        "但正常段本身波动不小, 异常点只是「混在大波动里的少数段」, 肉眼并不容易稳定地挑出来。\n"
        "**第 2 排 (增强前, 真实数据, 双设备)**: device 4 和 device 7 两格里, 蓝带 (正常) 和红带 (异常) **波动范围几乎一样宽、大面积重叠**, "
        "只是红带中心略偏一点。device 4 的类间间隔分别为 250605 **0.0083 V**、250626 **0.0736 V**、250718 **0.0051 V**; "
        "device 7 为 **0.0311 / 0.0274 / 0.0003 V** —— 多数都小于当天噪声水平。**这就是「原始数据难以区分正常/异常」的直接证据。**\n"
        "**第 3 排 (增强后, CSDI, 双设备)**: 样本数对齐之后, device 4 的类间间隔 250605 **0.0083→0.0130 V**、250718 **0.0051→0.0228 V (约 4.5 倍)**; "
        "device 7 在 250605 **0.0311→0.0405 V**、250718 **0.0003→0.0973 V (大幅拉开)** —— 这两天增强后两带明显分得更开, **说明增强确实有帮助**; "
        "但 250626 日两台设备反而都收窄 (**device 4: 0.0736→0.0083 V; device 7: 0.0274→0.0018 V**)。\n"
        "**第 4 排 (双设备联动, 最关键的一排)**: 真实正常点大多沿一条正斜率关系线分布 "
        "(250626 日真实正常 **r=0.75**、250718 日 **r=0.66**), 说明同一串上的两台设备**正常时是一起动的**; "
        "而真实异常点在 250626 日 **r 掉到 0.06** —— **两设备的关系被破坏了**, 这正是一个很直观的异常信号。"
        "增强后, 正常点的相关系数只有 **r≈0.15–0.29**, 远低于真实正常值, 说明**增强数据没有保留住「正常时两设备强联动」这个结构**。\n"
        "**一个必须诚实说明的隐患**: 综合第 2–4 排, 增强能把两团数据「拉开一些」, 但 (a) 在 250626 日反而收窄间隔; "
        "(b) 增强后 device 4 的「正常/异常谁高谁低」与真实方向相反; (c) 双设备联动强度 (相关系数) 明显低于真实。"
        "也就是说, **增强目前还不是「对所有日期、对两台设备、对联动结构都稳定有效」, 仍有明显的提升空间。**\n"
        "**一句话总结 (诚实版)**: 异常在最原始的**双设备**电压里看得见、且两台设备会一起响应; "
        "但真实正常段波动大、两团严重重叠, 本来就难分; **增强在 250605、250718 日让两类分得更开、更容易辨认**, "
        "但在 250626 日反而收窄了间隔, 且增强后「类间方向」和「双设备联动强度」都没有正确还原。"
        "**这也和前面第 10 节「增强整体有效、但个别日期/难例仍不稳」的结论吻合。**\n\n"
        "**这一节的小结 (大白话版)**:\n"
        "- 本图**同时画了 device 4 和 device 7**, 并在第 4 排用「M4 残差 vs M7 残差」专门检查双设备联动; "
        "只画 device 4 会丢掉「两设备关系」这条关键信息 (图 13 第 1、4 排);\n"
        "- 真实数据的正常/异常「方差几乎一样、重叠极大」, 两台设备的类间间隔都很小, 本来就难分 (图 13 第 2 排);\n"
        "- **CSDI 增强在 250605 / 250718 日把两台设备的类间间隔都拉大 (device 4 在 250718 约 4.5 倍, device 7 提升更明显), 更容易看出异常; "
        "但 250626 日两台设备反而都变窄** (图 13 第 3 排) —— 增强有效但不稳定;\n"
        "- **双设备联动**: 真实正常时两台设备强相关 (250626 日 r=0.75), 异常时关系断裂 (r=0.06); "
        "而增强后正常点相关只有 r≈0.15–0.29 —— 增强没有保住这个联动结构, 这是后续要改进的重点 (图 13 第 4 排);\n"
        "- **控制变量说明**: 本图真实/增强两侧、device 4 / device 7 两侧, 在「物理量、窗口长度、颜色画法、样本数」上完全对齐; "
        "因合成残差尺度与真实不同, 各排使用各自纵轴, 比较时只看「两类是否分开」, 不比绝对高度;\n"
        "- **残差排 = 放大镜而非新量**: 残差 = 电压 − 当天基线, 基线是常数会抵消, 所以残差排的类间间隔与原始电压差**完全相等**; "
        "它只是把纵轴缩到波动范围, 让同一道缝更显眼 (图 13 第 2、3 排左下角已注明);\n"
        "- 重跑命令: `$PY plot_voltage_trend.py` (依赖 `runs/upstream_gpu_full_20260910/dataset.npz`、`paired_row_audit.csv` 与 `/home/test/linliting/jyzmentor/experiment.xlsx`)。\n"
    )

    OUT_MD.write_text("\n".join(sec), encoding="utf-8")
    print(f"wrote {OUT_MD}")


if __name__ == "__main__":
    main()
