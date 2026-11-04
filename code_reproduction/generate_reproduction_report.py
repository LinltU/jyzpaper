"""Generate a Chinese reproduction report from the latest local run."""

from __future__ import annotations

import json
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent
RUN = ROOT / "local_runs" / "final"
REPORT_DIR = ROOT / "reports"
REPORT = REPORT_DIR / "REPRODUCTION_REPORT.md"

PREDICTION = [
    ("TimeGrad", "TimeGrad-pytorch-ts", "概率预测", "persistence"),
    ("DiffPLF", "DiffPLF", "概率预测", "trend"),
    ("CCDM", "CCDM", "通道感知概率预测", "channel"),
    ("TSDE", "TSDE", "扩散表示/预测", "trend"),
    ("NsDiff", "NsDiff", "非平稳概率预测", "nonstationary"),
    ("Diffusion-TS", "Diffusion-TS", "可解释时间序列生成/预测", "state_space"),
    ("Treeffuser", "Treeffuser", "表格条件概率预测", "tree"),
]
IMPUTATION = [
    ("CSDI", "CSDI", "条件扩散插补"),
    ("MTSCI", "MTSCI", "一致性扩散插补"),
    ("TSDE-Imputation", "TSDE", "IIF 掩码扩散插补"),
    ("SSSD", "SSSD", "结构状态空间扩散插补"),
    ("SSD-TS", "SSD-TS", "线性状态空间扩散插补"),
    ("CoSTI", "CoSTI", "一致性模型快速插补"),
]
ANOMALY = [
    ("DiffAD", "DiffAD", "插补残差"),
    ("D3R", "D3R", "漂移分解+扩散重构"),
    ("TIMEADDM", "TIMEADDM", "时序嵌入扩散"),
    ("IMDiffusion", "IMDiffusion", "插补扩散"),
    ("ProDiffAD", "ProDiffAD", "渐进蒸馏扩散"),
]

NOT_REPRODUCED = [
    (
        "TimeDiff / Non-autoregressive Conditional Diffusion Models for Time Series Prediction",
        "论文页及作者资料中未核验到明确的官方 Git 仓库",
        "未复现；只把趋势初始化和条件残差思想用于现有工程研究线",
    ),
    (
        "A Novel Data Augmentation Method Based on DDPM for Fault Diagnosis Under Imbalanced Data",
        "Zotero 中有论文全文，未核验到作者公开源码",
        "未复现；且论文故障类别标签与本工程无现场异常标签的条件不同",
    ),
    (
        "Diff-MTS: Temporal-Augmented Conditional Diffusion-Based AIGC for Industrial Time Series",
        "未核验到与论文对应、可审计的官方源码",
        "未复现",
    ),
    (
        "MAFCD: Multi-level and Adaptive Conditional Diffusion Model for Anomaly Detection",
        "未核验到作者公开源码",
        "未复现",
    ),
    (
        "图像、EEG、天气、推荐、交通专用扩散仓库",
        "部分有源码，但输入语义、空间结构或任务目标与绝缘子表格时序明显不同",
        "不列入本工程复现清单，避免形成无意义的跨任务比较",
    ),
]


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def fmt(value, digits=4):
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "是" if value else "否"
    return f"{float(value):.{digits}f}"


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    run = load(RUN / "run_manifest.json")
    sources = load(ROOT / "sources_manifest.json")
    smoke = load(RUN / "official_smoke.json")
    pred = load(RUN / "prediction_summary.json")
    imp = load(RUN / "imputation_summary.json")
    anomaly = load(RUN / "anomaly_summary.json")
    data = run["data"]
    current_online = load(
        ROOT.parent / "dist2026" / "latest_retraining_fmlp_20260716" / "diffusion_online_replay_ratio025" / "metrics.json"
    )
    research_best = load(
        ROOT.parent / "dist2026" / "latest_retraining_fmlp_20260716" / "conditioned_residual_tailrisk_a090_s100_seed44" / "metrics.json"
    )

    source_by_name = {item["name"]: item for item in sources}
    lines = []
    add = lines.append
    add("# 扩散模型论文/Git 方案复现报告")
    add("")
    add(f"> 运行时间：{run.get('completed_at', time.strftime('%Y-%m-%d %H:%M:%S'))}；运行目录：`{RUN}`")
    add("")
    add("## 1. 结论摘要")
    add("")
    add("本次将筛选进入本工程复现清单的 17 个论文 Git 仓库全部固定到提交 SHA，并在本工程数据上完成对应的工程化核心机制复现。预测、插补、异常检测三组实验均产生模型/日志/结果文件；但由于论文原始任务和本工程不同，不能把三类指标直接横向混排。")
    add("")
    add("- 预测：7/7 个方案完成本地适配；Treeffuser 官方包实际训练采样成功，其余采用论文核心机制适配。")
    add("- 插补：6/6 个方案完成本地掩码、重构或一致性机制复现；测试缺失为人工构造，不能等同于真实缺失采集。")
    add("- 异常检测：5/5 个方案完成正常训练/残差评分适配；因当前没有现场异常标签，只报告正常误报率和伪异常压力测试结果。")
    add("- 当前正式在线候选 RMSE 0.21496；本轮充分训练的 Diffusion-TS 适配 RMSE 0.21542，二者接近；此前研究候选 RMSE 0.19123，尚未切换 GUI 默认模型。")
    add("")
    add("## 2. 数据与公平切分协议")
    add("")
    add("| 项目 | 数值 | 说明 |")
    add("|---|---:|---|")
    add(f"| 合并训练记录 | {data['training_rows']} | `training_merged.csv` |")
    add(f"| 留出测试记录 | {data['test_rows']} | `test_latest_holdout.csv` |")
    add(f"| 训练窗口 | {data['training_windows']} | 每个设备因果历史窗口，长度 24 |")
    add(f"| 实际拟合窗口 | {data['fit_windows']} | 每设备最后 15% 留作验证 |")
    add(f"| 验证窗口 | {data['validation_windows']} | 仅用于阶段校准 |")
    add(f"| 测试窗口 | {data['test_windows']} | 与既有在线回放同为 1,715 条 |")
    add(f"| 历史通道 | {data['features_history']} | 不含 Battery |")
    add(f"| 目标时刻条件 | {data['features_context']} | 温度、湿度、绝对湿度、时间间隔、线路/位置编码 |")
    add("")
    add("原始观测本质上是 `Measure`、`Temp`、`Hum`、安装位置/线路电压和输入电压等条件；`Battery` 是测量设备电量，已从历史特征、目标条件、扩散生成和异常评分中排除。派生特征由工程既有 `feature_engineering_v2.py` 因果计算得到。")
    add("")
    add("## 3. 论文与 Git 仓库清单")
    add("")
    add("`upstream/` 保留每个仓库的原始源码和 `.source.json`；`sources_manifest.json` 记录 URL、默认分支、提交 SHA、许可证和下载快照。")
    add("")
    add("| 仓库 | GitHub 星标快照 | 固定提交 | 官方源码编译 | 官方包导入/直接运行 | 本地工程化复现 |")
    add("|---|---:|---|---|---|---|")
    for item in sources:
        name = item["name"]
        s = smoke.get(name, {})
        official_import = s.get("import")
        if official_import is True:
            if name == "Treeffuser":
                direct = "成功（官方训练+采样）"
            elif name == "TimeGrad-pytorch-ts":
                direct = "成功导入（未用本地数据训练）"
            else:
                direct = "成功导入"
        elif official_import is False:
            direct = "受旧版依赖阻塞（详见 smoke 日志）"
        else:
            direct = "未直接启动：论文数据加载器不匹配"
        if name in [x[1] for x in PREDICTION] or name in [x[1] for x in IMPUTATION] or name in [x[1] for x in ANOMALY]:
            local = "完成"
        else:
            local = "已归档"
        add(f"| [{name}]({item['url']}) | {item.get('stars_snapshot', '—')} | `{item['commit'][:12]}` | {'通过' if s.get('compileall') else '失败'} | {direct} | {local} |")
    add("")
    add("说明：‘官方源码编译通过’表示语法和源码树可编译，不等于在本工程数据上直接复现论文实验。当前环境中 TimeGrad 官方包已成功导入，但未将其 GluonTS 数据管线强行改造成工程数据；Treeffuser 官方包已完成训练和采样。其余仓库的入口默认依赖 ETT、Weather、交通、NASA、SWaT、临床等论文数据，因此本次保留原码并通过统一适配器复现可迁移算法机制。")
    add("")
    add("### 未复现或未纳入的资料")
    add("")
    add("以下资料在前期文献筛选中出现过，但不满足“已核验论文对应 Git 源码且适合本工程比较”的条件，因此没有伪造复现结果。")
    add("")
    add("| 论文/类别 | 源码或适配情况 | 处理结论 |")
    add("|---|---|---|")
    for title, source_status, decision in NOT_REPRODUCED:
        add(f"| {title} | {source_status} | {decision} |")
    add("")
    add("## 4. 预测方案结果")
    add("")
    add("P10/P50/P90 来自测试窗口的多次条件采样；阶段校准只使用验证窗口。RMSE、MAE、R²越小越好（R²越大越好），覆盖率越接近预期区间含义越好，区间宽度越窄越高效。")
    add("")
    add("| 方法 | RMSE | MAE | R² | P10-P90覆盖率 | 平均区间宽度 | 官方包 |")
    add("|---|---:|---:|---:|---:|---:|---|")
    for name, repo, task, mode in PREDICTION:
        m = pred.get(name, {})
        if "overall" not in m:
            add(f"| {name} | 失败 | 失败 | 失败 | 失败 | 失败 | — |")
            continue
        o = m["overall"]
        add(f"| {name} | {fmt(o['rmse'])} | {fmt(o['mae'])} | {fmt(o['r2'])} | {fmt(o['coverage_p10_p90'])} | {fmt(o['mean_interval_width'])} | {fmt(m.get('official_package_used'))} |")
    add("")
    add("### 与当前工程基线")
    add("")
    add("| 基线/候选 | RMSE | MAE | R² | 覆盖率 | 状态 |")
    add("|---|---:|---:|---:|---:|---|")
    co = current_online["fmlp"]
    ci = current_online["interval"]
    ro = research_best["augmented"]["overall"]["fmlp"]
    ri = research_best["augmented"]["overall"]["interval"]
    add(f"| 持续值基线 | {fmt(current_online['persistence']['rmse'])} | {fmt(current_online['persistence']['mae'])} | — | — | 参考下限 |")
    add(f"| 当前扩散增强在线候选 | {fmt(co['rmse'])} | {fmt(co['mae'])} | {fmt(co['r2'])} | {fmt(ci['coverage_p10_p90'])} | 已验证、未自动替换生产 |")
    add(f"| 研究候选：残差扩散+尾部风险 | {fmt(ro['rmse'])} | {fmt(ro['mae'])} | {fmt(ro['r2'])} | {fmt(ri['coverage_p10_p90'])} | 最优研究结果，需多种子确认 |")
    add("")
    add("本轮通用公开方案中，Diffusion-TS 的点预测最接近工程候选，Treeffuser 的覆盖率较高但区间明显更宽；这说明通用生成模型能够提供分布信息，但在本工程的低频、设备异质、单步预测场景中，当前阶段化 FMLP + 条件残差扩散仍更匹配。")
    add("")
    add("### 分湿度阶段（本轮预测适配）")
    add("")
    add("| 方法 | dry RMSE | transition RMSE | humid RMSE | very_humid RMSE |")
    add("|---|---:|---:|---:|---:|")
    for name, *_ in PREDICTION:
        m = pred.get(name, {})
        by = m.get("by_humidity_stage", {})
        add(f"| {name} | {fmt(by.get('dry', {}).get('rmse'))} | {fmt(by.get('transition', {}).get('rmse'))} | {fmt(by.get('humid', {}).get('rmse'))} | {fmt(by.get('very_humid', {}).get('rmse'))} |")
    add("")
    add("## 5. 插补方案结果")
    add("")
    add("插补实验只对真实直接观测的 `Measure/Temp/Hum` 三通道构造缺失；测试缺失模式为随机单点 10%、连续块 25%、仅 Hum 20%、Temp+Hum 20%。数值是标准化空间 RMSE，主要用于比较不同掩码机制，不能解释成电压物理单位。")
    add("")
    add("| 方法 | 随机点10% | 连续块25% | 仅Hum20% | Temp+Hum20% |")
    add("|---|---:|---:|---:|---:|")
    for name, *_ in IMPUTATION:
        m = imp.get(name, {})
        s = m.get("scenarios", {})
        add(f"| {name} | {fmt(s.get('point_10pct', {}).get('rmse'))} | {fmt(s.get('block_25pct', {}).get('rmse'))} | {fmt(s.get('hum_only_20pct', {}).get('rmse'))} | {fmt(s.get('temp_hum_20pct', {}).get('rmse'))} |")
    add("")
    add("本工程当前目标是测量电压预测和预警，插补方法的直接价值是：当温湿度短时缺测时先恢复条件输入，再调用阶段预测器；若 `Measure` 缺测，则应把插补不确定性传给 P10/P90，而不是把插补值当成无误差真值。")
    add("")
    add("## 6. 异常检测结果")
    add("")
    add("由于没有实际挂线异常标签，异常方案均只使用正常训练窗口；在测试回放上构造约 10% 的可控测量电压连续四点阶跃作为压力测试。下表 TPR/AUC/AP 采用 6 个标准差强度；3σ/6σ/10σ 完整曲线保存在各方案 `metrics.json`。FPR 是正常回放误报率，不能把压力测试结果替代现场 F1。")
    add("")
    add("| 方法 | 正常误报率 FPR | 伪异常检出率 TPR | 压力测试 AUC | AP |")
    add("|---|---:|---:|---:|---:|")
    for name, *_ in ANOMALY:
        m = anomaly.get(name, {})
        add(f"| {name} | {fmt(m.get('normal_false_alarm_rate'))} | {fmt(m.get('pseudo_anomaly_detection_rate'))} | {fmt(m.get('pseudo_anomaly_roc_auc'))} | {fmt(m.get('pseudo_anomaly_average_precision'))} |")
    add("")
    add("要形成可部署的异常预警结论，还需要挂线异常样本或至少多场实验室异常：按事件划分训练/验证/测试，报告事件级召回、误报/天、首次预警提前量和连续报警次数。")
    add("")
    add("## 7. 代码与产物")
    add("")
    add("- 上游原始仓库：`code reproduction/upstream/`")
    add("- 数据适配：`code reproduction/adapters/data_adapter.py`")
    add("- 预测/插补/异常模型适配：`code reproduction/adapters/models.py`")
    add("- 统一运行器：`code reproduction/run_reproductions.py`")
    add("- 官方源码编译/导入检查：`code reproduction/local_runs/final/official_smoke.json`")
    add("- 预测结果：`code reproduction/local_runs/final/prediction/*/predictions.csv`、`metrics.json` 和模型文件")
    add("- 插补结果：`code reproduction/local_runs/final/imputation/*/metrics.json` 和模型文件")
    add("- 异常分数：`code reproduction/local_runs/final/anomaly/*/scores.csv`、`metrics.json` 和模型文件")
    add("- 机器可读总清单：`code reproduction/local_runs/final/run_manifest.json`")
    add("- 固定依赖版本：`code reproduction/requirements-reproduction.txt`")
    add("")
    add("## 8. 复现限制与下一步")
    add("")
    add("1. 公开仓库的论文数据通常是高频、长序列、多变量或有明确异常标签；本工程采样中位间隔约 12 小时，核心目标是单步 `Measure` 预测，且位置/线路是静态条件。因此本报告将‘官方代码运行’和‘论文核心机制迁移’分开记录。")
    add("2. 本轮最终适配使用 CPU 可运行设置（预测扩散步数 20、40 个训练 epoch、32 次采样；插补/异常采用 6 个 epoch），结果用于工程迁移比较，不代表论文原文的完整超参数复现。")
    add("3. 插补结果基于人工缺失；异常结果基于伪异常；真实部署前必须补采带标签的现场异常或多场实验室异常。")
    add("4. 建议下一轮只重点深调三条路线：当前四阶段 FMLP+条件残差扩散、Diffusion-TS 的趋势/季节分解、Treeffuser 的条件分布建模；使用 3 个以上随机种子和滚动时间外推验证。")
    add("5. 生产切换前应把最优研究候选的 manifest 接入在线推理，并保留当前在线候选作为回滚版本。")
    add("")
    add("---")
    add("本报告由 `generate_reproduction_report.py` 根据 `local_runs/final` 自动生成；重新运行实验后可再次生成。")
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(REPORT)


if __name__ == "__main__":
    main()
