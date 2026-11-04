# 扩散模型论文/Git 方案复现报告

> 运行时间：2026-09-02T20:35:12+0800；运行目录：`C:\Project\JYZmentor\code reproduction\local_runs\final`

> 口径说明：本文件保留的是早期总复现档案，包含 `training.csv/test.csv` 的预测、插补和 6σ 压力测试内容。当前用户指定的正式测试目标已经切换为 `dataset/experiment.xlsx`；请以 `DIFFUSION_EXPERIMENT_FULL_REVIEW_20260909.md` 和 `experiment_augmentation/runs/within_date_stable_20260909_v2/REPORT.md` 为准。`experiment.xlsx` 的 `error!=0` 是真实受控实验异常标签，不是人工注入伪异常。

## 1. 结论摘要

本次将筛选进入本工程复现清单的 17 个论文 Git 仓库全部固定到提交 SHA，并在本工程数据上完成对应的工程化核心机制复现。预测、插补、异常检测三组实验均产生模型/日志/结果文件；同时又将 TimeGrad、DiffPLF、CCDM、TSDE、NsDiff 从原来的共享适配器拆成独立模型和独立训练循环。由于论文原始任务和本工程不同，不能把三类指标直接横向混排。

- 预测：7/7 个方案完成统一适配；其中 TimeGrad、DiffPLF、CCDM、TSDE、NsDiff 已另行完成独立分支复现，Treeffuser 官方包实际训练采样成功，Diffusion-TS 有完整 Transformer/Trend/Fourier 分支。
- 插补：6/6 个方案完成本地掩码、重构或一致性机制复现；测试缺失为人工构造，不能等同于真实缺失采集。
- 异常检测（早期主工程档案）：5/5 个方案完成正常训练/残差评分适配；该历史段使用人工压力测试，不代表当前 `experiment.xlsx` 结果。
- 当前正式在线候选 RMSE 0.21496；此前研究候选 RMSE 0.19123，尚未切换 GUI 默认模型。独立分支同预算复核中，NsDiff 最好（RMSE 0.22978），但暂未超过调参后的 Diffusion-TS 适配器（RMSE 约 0.2086）。

## 2. 数据与公平切分协议

| 项目 | 数值 | 说明 |
|---|---:|---|
| 合并训练记录 | 9520 | `training_merged.csv` |
| 留出测试记录 | 1715 | `test_latest_holdout.csv` |
| 训练窗口 | 8920 | 每个设备因果历史窗口，长度 24 |
| 实际拟合窗口 | 7569 | 每设备最后 15% 留作验证 |
| 验证窗口 | 1351 | 仅用于阶段校准 |
| 测试窗口 | 1715 | 与既有在线回放同为 1,715 条 |
| 历史通道 | 26 | 不含 Battery |
| 目标时刻条件 | 13 | 温度、湿度、绝对湿度、时间间隔、线路/位置编码 |

原始观测本质上是 `Measure`、`Temp`、`Hum`、安装位置/线路电压和输入电压等条件；`Battery` 是测量设备电量，已从历史特征、目标条件、扩散生成和异常评分中排除。派生特征由工程既有 `feature_engineering_v2.py` 因果计算得到。

## 3. 论文与 Git 仓库清单

`upstream/` 保留每个仓库的原始源码和 `.source.json`；`sources_manifest.json` 记录 URL、默认分支、提交 SHA、许可证和下载快照。

| 仓库 | GitHub 星标快照 | 固定提交 | 官方源码编译 | 官方包导入/直接运行 | 本地工程化复现 |
|---|---:|---|---|---|---|
| [TimeGrad-pytorch-ts](https://github.com/zalandoresearch/pytorch-ts) | 1370 | `7860c9693d55` | 通过 | 成功导入（未用本地数据训练） | 完成 |
| [DiffPLF](https://github.com/LSY-Cython/DiffPLF) | 41 | `3e23cc76cbba` | 通过 | 未直接启动：论文数据加载器不匹配 | 完成 |
| [CCDM](https://github.com/LSY-Cython/CCDM) | 7 | `a9a4998d3978` | 通过 | 未直接启动：论文数据加载器不匹配 | 完成 |
| [TSDE](https://github.com/EQTPartners/TSDE) | 31 | `67c1e9621697` | 通过 | 未直接启动：论文数据加载器不匹配 | 完成 |
| [NsDiff](https://github.com/wwy155/NsDiff) | 65 | `dd7b97ae0c47` | 通过 | 未直接启动：论文数据加载器不匹配 | 完成 |
| [Treeffuser](https://github.com/blei-lab/treeffuser) | 57 | `6be21acdea92` | 通过 | 成功（官方训练+采样） | 完成 |
| [CSDI](https://github.com/ermongroup/CSDI) | 461 | `7f24a436f08d` | 通过 | 未直接启动：论文数据加载器不匹配 | 完成 |
| [SSSD](https://github.com/ai4healthuol/sssd) | 335 | `4d3b7a51c54b` | 通过 | 未直接启动：论文数据加载器不匹配 | 完成 |
| [MTSCI](https://github.com/JeremyChou28/MTSCI) | 33 | `fa54d6496c0f` | 通过 | 未直接启动：论文数据加载器不匹配 | 完成 |
| [SSD-TS](https://github.com/decisionintelligence/SSD-TS) | 8 | `81f6ac381f70` | 通过 | 未直接启动：论文数据加载器不匹配 | 完成 |
| [CoSTI](https://github.com/javiersgjavi/CoSTI) | 7 | `362192650eef` | 通过 | 未直接启动：论文数据加载器不匹配 | 完成 |
| [DiffAD](https://github.com/ChunjingXiao/DiffAD) | 125 | `a05f764dea8c` | 通过 | 未直接启动：论文数据加载器不匹配 | 完成 |
| [D3R](https://github.com/ForestsKing/D3R) | 96 | `9a812ce92452` | 通过 | 未直接启动：论文数据加载器不匹配 | 完成 |
| [TIMEADDM](https://github.com/Hurongyao/TIMEADDM) | 20 | `ccaf03837f5d` | 通过 | 未直接启动：论文数据加载器不匹配 | 完成 |
| [IMDiffusion](https://github.com/17000cyh/IMDiffusion) | 115 | `b6e07685a0d1` | 通过 | 未直接启动：论文数据加载器不匹配 | 完成 |
| [ProDiffAD](https://github.com/AiShare-WHU/ProDiffAD) | 2 | `7aa2a56567b0` | 通过 | 未直接启动：论文数据加载器不匹配 | 完成 |
| [Diffusion-TS](https://github.com/Y-debug-sys/Diffusion-TS) | 481 | `566307e6cf2d` | 通过 | 未直接启动：论文数据加载器不匹配 | 完成 |

说明：‘官方源码编译通过’表示语法和源码树可编译，不等于在本工程数据上直接复现论文实验。当前环境中 TimeGrad 官方包已成功导入，但未将其 GluonTS 数据管线强行改造成工程数据；Treeffuser 官方包已完成训练和采样。其余仓库的入口默认依赖 ETT、Weather、交通、NASA、SWaT、临床等论文数据，因此本次保留原码并通过统一适配器复现可迁移算法机制。

### 独立论文分支（新增）

为避免“同一个扩散模型只切换 `mode`”造成误解，新增目录
`code reproduction/individual_paper_branches/`，每个方案各自拥有独立编码器、条件变量和训练循环：

| 方案 | 独立结构 | 本地实现口径 |
|---|---|---|
| TimeGrad | GRU 历史状态 + 条件 DDPM | 论文核心机制适配 |
| DiffPLF | 局部线性趋势 + 局部残差尺度 | 论文核心机制适配 |
| CCDM | Measure/Temp/Hum 分通道编码 | 论文核心机制适配 |
| TSDE | IIF 插补/插值/预测掩码 + GRU | 论文核心机制适配 |
| NsDiff | 窗口局部均值/尺度归一化 | 论文核心机制适配 |

这些分支仍不是官方代码逐行复制：官方仓库的数据加载器和任务目标分别面向 EV 充电、Time-Series-Library、ETT/电力/天气等公开数据，本地分支只迁移论文中与当前单步 `Measure` 预测相关的算法思想。完整结果和数据差异见
`reports/INDEPENDENT_PAPER_BRANCHES_20260907.md`。

## 3.1 全部方案算法模型与本工程适配度

下表把本工程实际使用过的模型全部列出。适配度评价针对当前任务：使用过去 24 步，预测下一步 `Measure`，输入包含温湿度、派生统计量、线路电压和绝缘子安装位置，采样间隔不完全规则，并且没有足够的真实异常标签。

“高”表示模型结构与当前单步预测/高湿度非平稳问题直接匹配；“中”表示需要较多任务改造或依赖假设；“低”表示主要用于对照或其原始优势在当前数据上难以发挥。适配度不是论文原始排名。

### 预测和数据增强模型

| 方案/模型 | 当前本地模型结构 | 核心特点 | 本地代码/来源 | 对本工程适配度 |
|---|---|---|---|---|
| 四阶段 FMLP | 共享 FMLP + 四个湿度阶段残差专家 + P10/P50/P90 分位数输出 | 按 `dry/transition/humid/very_humid` 建模，直接处理低湿趋势和高湿波动差异 | `dist2026/model/`、`dist2026/experiments/` | **高**：与湿度阶段差异和单步预测直接对应 |
| DDPM 增强 FMLP | 条件 DDPM 生成 `Measure-persistence` 残差，再与真实窗口混合训练 FMLP | 数据增强，不直接替代预测器；用条件温湿度和湿度趋势控制生成 | `dist2026/experiments/evaluate_diffusion_augmented_high_expert.py` | **高**：适合真实异常样本少、需要增强高湿专家的场景 |
| TimeGrad | 独立 GRU 历史编码器 + 条件残差 DDPM | 通过循环状态进行概率预测，适合多步自回归生成 | `code reproduction/individual_paper_branches/models.py`；上游 `TimeGrad-pytorch-ts` | 中高：历史依赖适合，但当前只预测一步，未完全发挥自回归优势 |
| DiffPLF | 局部线性趋势基准 + 局部残差尺度 + DDPM | 先预测局部趋势，再生成趋势残差 | `individual_paper_branches/models.py`；上游 `DiffPLF` | 中：高湿强波动时局部线性趋势容易失效 |
| CCDM | Measure/Temp/Hum 分通道投影 + 其他派生特征融合 + DDPM | 保留通道独立性，再做条件概率建模 | `individual_paper_branches/models.py`；上游 `CCDM` | 中高：直接观测通道有意义，但本工程派生/位置特征不是同质传感器通道 |
| TSDE 预测 | IIF 掩码训练的双向 GRU + 条件 DDPM | 联合插补、插值、预测学习时序表示 | `individual_paper_branches/models.py`；上游 `TSDE` | 中：IIF 对缺失和多任务有帮助，当前在线任务主要是单步预测 |
| NsDiff | 每窗口局部均值/尺度归一化 + 非平稳残差 DDPM | 对湿度变化和设备差异造成的局部分布漂移更稳健 | `individual_paper_branches/models.py`；上游 `NsDiff` | **高**：与高湿度波动和设备异质性最匹配，当前独立分支同预算最好 |
| Diffusion-TS | Transformer Encoder/Decoder + TrendBlock + FourierLayer，直接重建 `x0` | 将趋势、周期和细节分解，使用频域损失 | `full_diffusion_ts/model.py`；上游 `Diffusion-TS` | 中高：表达能力强，但本工程为短窗口、单步、不规则采样，Fourier 优势受限 |
| Treeffuser | LightGBM score model + diffusion sampler | 表格条件分布预测，容易处理混合数值和位置条件 | `adapters/models.py` 中 `run_treeffuser`；上游 `Treeffuser` | 中高：适合表格条件，但时间依赖和区间宽度需要额外控制 |

### 插补模型

| 方案/模型 | 当前本地模型结构 | 核心特点 | 本地代码/来源 | 对本工程适配度 |
|---|---|---|---|---|
| CSDI | MLP 掩码重构适配器 | 条件 score-based diffusion 插补 | `adapters/models.py`；上游 `CSDI` | 中：适合窗口内随机缺测；当前不是整段时间戳缺失插补 |
| MTSCI | MLP + 互补掩码一致性损失 | 用不同缺失视图保持重构一致 | `adapters/models.py`；上游 `MTSCI` | 中高：温湿度缺测时有价值，但没有真实缺测标签 |
| TSDE-Imputation | 双向 GRU 掩码重构 | IIF 掩码同时覆盖插补、插值和预测 | `adapters/models.py`；上游 `TSDE` | 中高：适合短时传感器缺测，仍需补充真实离线段测试 |
| SSSD | 1D Conv 掩码重构 | 结构化状态空间思想和扩散插补 | `adapters/models.py`；上游 `SSSD` | 中：连续块缺失较匹配，但本地未使用官方 S4 |
| SSD-TS | GRU 掩码重构 | 线性状态空间模型用于扩散插补 | `adapters/models.py`；上游 `SSD-TS` | 中：可处理连续缺测，但只是 GRU 近似 |
| CoSTI | 1D Conv + 较高一致性权重 | consistency model 目标是减少采样步数、提高插补速度 | `adapters/models.py`；上游 `CoSTI` | 中高：在线补值速度有吸引力，但本地未搬入官方时空/Mamba 模块 |

### 异常检测模型

| 方案/模型 | 当前本地模型结构 | 核心特点 | 本地代码/来源 | 对本工程适配度 |
|---|---|---|---|---|
| DiffAD | MLP 掩码重构 + Measure/Temp/Hum 通道加权 | 插补误差作为异常分数，并强化不同通道权重 | `run_reproductions.py::run_anomaly_method`；上游 `DiffAD` | 中高：适合发现电压与环境不一致，但需要真实故障校准 |
| D3R | 1D Conv，先移动平均去趋势再重构 | 动态分解处理非平稳漂移，减少趋势导致的误报 | `run_anomaly_method`；上游 `D3R` | **高**：高湿阶段存在漂移时有价值，但当前现场标签不足 |
| TIMEADDM | GRU 重构评分 | 循环表示积累全局时间依赖后进行扩散检测 | `run_anomaly_method`；上游 `TIMEADDM` | 中高：适合序列相关异常，当前窗口短且只做本地适配 |
| IMDiffusion | MLP 多次插补集成 | 同时使用重构误差和插补不确定性 | `run_anomaly_method`；上游 `IMDiffusion` | **高**：能把测量不确定性纳入预警，但需要正常运行分布更充分 |
| ProDiffAD | 1D Conv 快速评分 | 通过教师到学生的渐进蒸馏降低检测开销 | `run_anomaly_method`；上游 `ProDiffAD` | 中高：适合在线部署，但本地只复现评分思想，未执行完整蒸馏 |

### 统一实现边界

- `upstream/` 保存固定 commit 的论文公开仓库，用于溯源和源码编译检查。
- `adapters/models.py` 是本工程的共享插补/异常适配器；这些结果属于论文机制迁移，不是官方网络逐层复现。
- `individual_paper_branches/models.py` 是预测方案的独立实现，每个方案拥有独立编码器和训练循环。
- Treeffuser 使用了官方包训练和采样；其他大多数论文方案由于原始数据加载器面向 ETT、Weather、交通、NASA、SWaT、临床等数据，使用本地数据适配。
- 插补当前模拟窗口内的值缺失；异常检测当前使用正常数据和人工注入的伪异常，二者都不能替代真实时间段缺失或真实挂线故障验证。

### 未复现或未纳入的资料

以下资料在前期文献筛选中出现过，但不满足“已核验论文对应 Git 源码且适合本工程比较”的条件，因此没有伪造复现结果。

| 论文/类别 | 源码或适配情况 | 处理结论 |
|---|---|---|
| TimeDiff / Non-autoregressive Conditional Diffusion Models for Time Series Prediction | 论文页及作者资料中未核验到明确的官方 Git 仓库 | 未复现；只把趋势初始化和条件残差思想用于现有工程研究线 |
| A Novel Data Augmentation Method Based on DDPM for Fault Diagnosis Under Imbalanced Data | Zotero 中有论文全文，未核验到作者公开源码 | 未复现；且论文故障类别标签与本工程无现场异常标签的条件不同 |
| Diff-MTS: Temporal-Augmented Conditional Diffusion-Based AIGC for Industrial Time Series | 未核验到与论文对应、可审计的官方源码 | 未复现 |
| MAFCD: Multi-level and Adaptive Conditional Diffusion Model for Anomaly Detection | 未核验到作者公开源码 | 未复现 |
| 图像、EEG、天气、推荐、交通专用扩散仓库 | 部分有源码，但输入语义、空间结构或任务目标与绝缘子表格时序明显不同 | 不列入本工程复现清单，避免形成无意义的跨任务比较 |

## 4. 预测方案结果

P10/P50/P90 来自测试窗口的多次条件采样；阶段校准只使用验证窗口。RMSE、MAE、R²越小越好（R²越大越好），覆盖率越接近预期区间含义越好，区间宽度越窄越高效。

| 方法 | RMSE | MAE | R² | P10-P90覆盖率 | 平均区间宽度 | 官方包 |
|---|---:|---:|---:|---:|---:|---|
| TimeGrad | 0.2391 | 0.1343 | 0.2870 | 0.6892 | 0.2610 | 否 |
| DiffPLF | 0.2495 | 0.1432 | 0.2239 | 0.7160 | 0.2969 | 否 |
| CCDM | 0.2431 | 0.1371 | 0.2632 | 0.7131 | 0.2764 | 否 |
| TSDE | 0.2495 | 0.1432 | 0.2239 | 0.7160 | 0.2969 | 否 |
| NsDiff | 0.2443 | 0.1390 | 0.2558 | 0.7184 | 0.2836 | 否 |
| Diffusion-TS | 0.2154 | 0.1220 | 0.4214 | 0.6787 | 0.2319 | 否 |
| Treeffuser | 0.2440 | 0.1474 | 0.2576 | 0.9382 | 0.7712 | 是 |

### 与当前工程基线

| 基线/候选 | RMSE | MAE | R² | 覆盖率 | 状态 |
|---|---:|---:|---:|---:|---|
| 持续值基线 | 0.3019 | 0.1891 | — | — | 参考下限 |
| 当前扩散增强在线候选 | 0.2150 | 0.1210 | 0.4239 | 0.8146 | 已验证、未自动替换生产 |
| 研究候选：残差扩散+尾部风险 | 0.1912 | 0.1019 | 0.5441 | 0.8076 | 最优研究结果，需多种子确认 |

### 独立分支同预算结果（2026-09-07）

该组实验使用 12 epochs、hidden=64、8 个扩散步、16 次采样，训练/验证/测试窗口仍为
7569/1351/1715。结果来自 `local_runs/individual_final_20260907/prediction_summary.json`。

| 方法 | RMSE | MAE | R² | P10-P90覆盖率 | 平均区间宽度 |
|---|---:|---:|---:|---:|---:|
| TimeGrad（独立） | 0.28100 | 0.17402 | 0.01557 | 74.17% | 0.40243 |
| DiffPLF（独立） | 0.31283 | 0.20141 | -0.22013 | 72.48% | 0.45878 |
| CCDM（独立） | 0.28768 | 0.18267 | -0.03178 | 74.81% | 0.45199 |
| TSDE（独立） | 0.29008 | 0.18329 | -0.04910 | 74.99% | 0.45625 |
| NsDiff（独立） | **0.22978** | **0.14229** | **0.34171** | 74.99% | **0.33827** |

独立分支中 NsDiff 最好，但不能据此宣称它优于所有论文方案：完整 Diffusion-TS 和
四阶段 FMLP 的训练预算、模型结构及调参过程不同，必须在完全同预算、多随机种子下再作最终排名。

随后已完成同预算、多随机种子复核。8 epochs、batch=512、hidden/d_model=64、8 个扩散步、
16 次采样、学习率 8e-4，seed=42/43/44 下：NsDiff 的 RMSE 为 `0.24069 ± 0.00074`，
完整 Diffusion-TS 为 `0.25486 ± 0.01488`。NsDiff 的点预测更好且稳定性更高；Diffusion-TS
平均 MAE 略低、区间更窄，但随机种子波动明显。详细记录见
`local_runs/fair_nsdiff_diffusion_ts_20260907_v2/FAIR_COMPARISON.md`。

本轮通用公开方案中，Diffusion-TS 的点预测最接近工程候选，Treeffuser 的覆盖率较高但区间明显更宽；这说明通用生成模型能够提供分布信息，但在本工程的低频、设备异质、单步预测场景中，当前阶段化 FMLP + 条件残差扩散仍更匹配。

### 分湿度阶段（本轮预测适配）

| 方法 | dry RMSE | transition RMSE | humid RMSE | very_humid RMSE |
|---|---:|---:|---:|---:|
| TimeGrad | 0.0434 | 0.0924 | 0.1344 | 0.3028 |
| DiffPLF | 0.0336 | 0.0952 | 0.1480 | 0.3139 |
| CCDM | 0.0340 | 0.0988 | 0.1459 | 0.3050 |
| TSDE | 0.0336 | 0.0952 | 0.1480 | 0.3139 |
| NsDiff | 0.0325 | 0.1002 | 0.1492 | 0.3058 |
| Diffusion-TS | 0.0336 | 0.1019 | 0.1471 | 0.2638 |
| Treeffuser | 0.0215 | 0.1122 | 0.2011 | 0.2866 |

## 5. 插补方案结果

插补实验只对真实直接观测的 `Measure/Temp/Hum` 三通道构造缺失；测试缺失模式为随机单点 10%、连续块 25%、仅 Hum 20%、Temp+Hum 20%。数值是标准化空间 RMSE，主要用于比较不同掩码机制，不能解释成电压物理单位。

这里的“连续块 25%”是指：在一个已经存在的 24 步历史窗口中，随机选择约 6 个连续时间位置并把三通道值遮挡，再让模型恢复这 6 个位置。它不是从原始 CSV 中找到一段完全没有记录的时间区间，也不是预测未来时间段。当前代码没有对原始数据中缺少的行自动补齐时间戳；如果某个设备在某段时间完全没有采集记录，该段不会被自动生成成插补样本。具体测试方式如下：

| 场景 | 实际遮挡方式 | 是否代表原始时间段缺失 |
|---|---|---|
| `point_10pct` | 24×3 窗口内随机遮挡约 10% 单元格 | 否，模拟零散传感器缺测 |
| `block_25pct` | 每个窗口内遮挡约 6 个连续时间位置的三通道 | 部分模拟连续缺测，但时间行仍然存在 |
| `hum_only_20pct` | 只遮挡 Hum 通道 | 否，模拟单传感器通道缺测 |
| `temp_hum_20pct` | 遮挡 Temp 和 Hum 通道，Measure 保留 | 否，模拟环境传感器缺测 |

因此，这些结果可以回答“已有时间轴中部分传感器值缺失时能否恢复”，不能直接回答“设备离线数小时/数天、原始时间戳和整行记录都缺失时能否补齐”。后者需要先按设备建立规则时间网格，识别缺失时间段，再使用时间间隔、前后观测和设备条件构造整行缺失任务，单独评估时间段级 MAE、连续段恢复误差和不确定性。

| 方法 | 随机点10% | 连续块25% | 仅Hum20% | Temp+Hum20% |
|---|---:|---:|---:|---:|
| CSDI | 1.1363 | 0.9132 | 0.8417 | 0.9631 |
| MTSCI | 0.7859 | 0.7549 | 0.6897 | 0.6694 |
| TSDE-Imputation | 0.8016 | 0.8404 | 0.8606 | 0.8380 |
| SSSD | 0.7775 | 0.8675 | 0.6724 | 0.8482 |
| SSD-TS | 0.8133 | 0.8546 | 0.8702 | 0.8544 |
| CoSTI | 0.6318 | 0.9350 | 0.6214 | 0.6925 |

本工程当前目标是测量电压预测和预警，插补方法的直接价值是：当温湿度短时缺测时先恢复条件输入，再调用阶段预测器；若 `Measure` 缺测，则应把插补不确定性传给 P10/P90，而不是把插补值当成无误差真值。

### 插补方案、论文和代码对应关系

这里的“6/6 完成”表示 6 个论文方向都完成了本地任务适配和测试，不表示 6 个官方网络都已逐层搬入本工程。当前本地运行器使用统一的 `ImputationAdapter`，通过不同的网络骨架、掩码方式和一致性/平滑损失表达论文中与本工程相关的机制。

| 本地方案 | 论文/公开代码 | 论文核心 | 本地对应模型与代码 | 本地复现口径 |
|---|---|---|---|---|
| CSDI | [CSDI: Conditional Score-based Diffusion Models for Probabilistic Time Series Imputation](https://arxiv.org/abs/2107.03502)；[ermongroup/CSDI](https://github.com/ermongroup/CSDI) | 条件 score-based diffusion、随机掩码插补 | `ImputationAdapter` + MLP；`mask=random`；`run_reproductions.py` 中 `CSDI` 配置 | 论文机制适配，不是官方 CSDI 网络逐层运行 |
| MTSCI | [MTSCI: A Conditional Diffusion Model for Consistent Imputation in Incomplete Time Series](https://dl.acm.org/doi/10.1145/3627673.3679532)；[JeremyChou28/MTSCI](https://github.com/JeremyChou28/MTSCI) | 多视图/互补掩码一致性 | `ImputationAdapter` + MLP；`mask=random`；`consistency=0.10` | 本地一致性机制适配 |
| TSDE-Imputation | [Self-Supervised Learning of Time Series Representation via Diffusion Process and Imputation-Interpolation-Forecasting Mask](https://arxiv.org/abs/2405.05959)；[EQTPartners/TSDE](https://github.com/EQTPartners/TSDE) | IIF 掩码：插补、插值、预测联合训练 | `ImputationAdapter` + 双向 GRU；`mask=iif`；加一致性和平滑项 | 本地 IIF 掩码适配；不是官方双正交 Transformer |
| SSSD | [Diffusion-based Time Series Imputation and Forecasting with Structured State Space Models](https://openreview.net/forum?id=hHiIbk7ApW)；[ai4healthuol/sssd](https://github.com/AI4HealthUOL/SSSD) | S4/结构化状态空间 + 条件扩散 | `ImputationAdapter` + 1D Conv；`mask=block`；加一致性和平滑项 | 用卷积近似长依赖机制；未使用官方 S4 |
| SSD-TS | [SSD-TS: Exploring the potential of linear state space models for diffusion models in time series imputation](https://github.com/decisionintelligence/SSD-TS)；[decisionintelligence/SSD-TS](https://github.com/decisionintelligence/SSD-TS) | 线性状态空间模型 + diffusion imputer | `ImputationAdapter` + GRU；`mask=block`；加一致性和平滑项 | 状态空间思想的 GRU 本地适配；未使用官方 Mamba/SSD 算子 |
| CoSTI | [CoSTI: Consistency Models for (a faster) Spatio-Temporal Imputation](https://arxiv.org/abs/2501.19364)；[javiersgjavi/CoSTI](https://github.com/javiersgjavi/CoSTI) | consistency model、时空特征提取、快速确定性插补 | `ImputationAdapter` + 1D Conv；`mask=random`；`consistency=0.15`、平滑项 | 一致性训练思想适配；未使用官方 STFEM/Mamba |

本地公共代码位置：`code reproduction/adapters/models.py` 中的 `MaskedConsistencyImputer` 和 `ImputationAdapter`；具体方案配置位于 `code reproduction/run_reproductions.py` 的 `IMPUTATION_METHODS`。

## 6. 异常检测结果

由于没有实际挂线异常标签，异常方案均只使用正常训练窗口；在测试回放上构造约 10% 的可控测量电压连续四点阶跃作为压力测试。下表 TPR/AUC/AP 采用 6 个标准差强度；3σ/6σ/10σ 完整曲线保存在各方案 `metrics.json`。FPR 是正常回放误报率，不能把压力测试结果替代现场 F1。

| 方法 | 正常误报率 FPR | 伪异常检出率 TPR | 压力测试 AUC | AP |
|---|---:|---:|---:|---:|
| DiffAD | 0.0052 | 0.9551 | 0.9979 | 0.9869 |
| D3R | 0.0052 | 0.1026 | 0.9533 | 0.5824 |
| TIMEADDM | 0.0052 | 0.8462 | 0.9977 | 0.9771 |
| IMDiffusion | 0.0052 | 0.6987 | 0.9836 | 0.9035 |
| ProDiffAD | 0.0052 | 0.9167 | 0.9913 | 0.9674 |

要形成可部署的异常预警结论，还需要挂线异常样本或至少多场实验室异常：按事件划分训练/验证/测试，报告事件级召回、误报/天、首次预警提前量和连续报警次数。

### 异常检测方案、论文和代码对应关系

当前 5 个异常方案都使用“正常训练窗口 + 掩码重构误差”的统一本地适配框架，再按论文方向改变去趋势、评分加权、递归时间权重、随机插补集成或蒸馏式评分。它们没有使用现场异常标签，也没有把人工压力测试当成真实故障验证。

| 本地方案 | 论文/公开代码 | 论文核心 | 本地对应模型与代码 | 本地复现口径 |
|---|---|---|---|---|
| DiffAD | [Imputation-based Time-Series Anomaly Detection with Conditional Weight-Incremental Diffusion Models](https://dl.acm.org/doi/10.1145/3580305.3599391)；[ChunjingXiao/DiffAD](https://github.com/ChunjingXiao/DiffAD) | 扩散插补 + weight-incremental 异常评分 | `ImputationAdapter` + MLP；`score=weighted_incremental`，Measure/Temp/Hum 加权 | 本地残差重构评分适配 |
| D3R | [Drift doesn't Matter: Dynamic Decomposition with Diffusion Reconstruction for Unstable Multivariate Time Series Anomaly Detection](https://proceedings.neurips.cc/paper_files/paper/2023/hash/22f5d8e689d2a011cd8ead552ed59052-Abstract-Conference.html)；[ForestsKing/D3R](https://github.com/ForestsKing/D3R) | 动态分解、去漂移、扩散重构 | `ImputationAdapter` + 1D Conv；先移动平均去趋势，`score=detrended` | 去趋势和重构机制适配 |
| TIMEADDM | [Unsupervised Anomaly Detection for Multivariate Time Series Using Diffusion Model](https://github.com/Hurongyao/TIMEADDM)；[Hurongyao/TIMEADDM](https://github.com/Hurongyao/TIMEADDM) | 循环表示累积全局时间相关，再做扩散 | `ImputationAdapter` + GRU；`score=recurrent`，最近时间点加权 | 循环时间权重近似，本地未运行官方扩散网络 |
| IMDiffusion | [IMDiffusion: Imputed Diffusion Models for Multivariate Time Series Anomaly Detection](https://github.com/17000cyh/IMDiffusion)；[17000cyh/IMDiffusion](https://github.com/17000cyh/IMDiffusion) | 多次插补/扩散样本的不确定性异常评分 | `ImputationAdapter` + MLP；`score=imputation_ensemble`，重构误差加不确定性 | 插补集成评分适配 |
| ProDiffAD | [ProDiffAD: Progressively Distilled Diffusion Models for Multivariate Time Series Anomaly Detection](https://github.com/AiShare-WHU/ProDiffAD)；[AiShare-WHU/ProDiffAD](https://github.com/AiShare-WHU/ProDiffAD) | 扩散教师到快速学生的渐进蒸馏 | `ImputationAdapter` + 1D Conv；`score=distilled`，加权平均误差和最大误差 | 快速学生评分思想适配，未执行官方教师-学生蒸馏训练 |

本地公共代码位置：`code reproduction/run_reproductions.py` 的 `run_anomaly_method`；模型骨架和重构逻辑来自 `code reproduction/adapters/models.py` 的 `ImputationAdapter`。因此这些异常方案当前是“论文评分机制的本地可运行适配”，而不是官方仓库的完整逐层复现。

## 7. 代码与产物

### 7.1 `experiment.xlsx` 两设备条件扩散增强

针对实验室 9 片、两个测量设备的数据，新增了 `code reproduction/experiment_augmentation/` 分支。该分支先按日期将第4片和第7片设备按最近时间戳配对，再在同一个 `error` 连续阶段内切出 6 步窗口，避免把断电上电和更换零值绝缘子造成的阶段混合到一个样本中。

每个时间点使用 `M4、M7、Temp4、Hum4、Temp7、Hum7、M4-M7、M4/M7、log(M4/M7)、温差、湿差、elapsed_min` 12 个特征。`error` 和实验日期只作为条件扩散标签；在线检测时不能使用真实 `error`。实现了直接标准化 DDPM 和带窗口局部均值/尺度归一化的 NsDiff 风格 DDPM，并输出合成窗口与元数据。

本地入口：`code reproduction/experiment_augmentation/run_experiment_augmentation.py`；结果：`code reproduction/experiment_augmentation/runs/experiment_ddpm_20260908_v4/`。在当前数据量和 80 轮训练下，直接 DDPM 增强后的异常位置分类 balanced accuracy 从 `0.9869` 降至 `0.8168`；NsDiff 风格增强为 `0.9528`，未超过只用真实数据的 `0.9869`。因此当前扩散增强已完成工程尝试，但生成样本仍存在分布收缩，暂不建议直接用于生产训练。详细特征和限制见该目录 `README.md`。

随后增加了共享环境特征复核：因为第4片和第7片设备位于同一绝缘子串且相邻采样时刻温湿度差异很小，新增 `Temp_shared=(Temp4+Temp7)/2`、`Hum_shared=(Hum4+Hum7)/2`，同时保留 `M4、M7、M4-M7、M4/M7、logratio、elapsed_min`。结果位于 `code reproduction/experiment_augmentation/runs/experiment_shared_env_20260909/`：只用真实数据 balanced accuracy 为 `0.9830`，直接 DDPM 扩增为 `0.8168`，NsDiff 风格扩增为 `0.8558`。共享温湿度特征更符合物理假设，但当前大量平衡生成仍未带来提升，后续应减少合成比例并显式约束上电衰减。

进一步完成了“只用后期稳定正常段 + 双设备残差”的专项验证：`code reproduction/experiment_augmentation/stable_residual_diffusion.py`。按每个实验日期正常窗口的 M4/M7 斜率和波动筛选最低 40% 作为稳定窗口，共 217 个；正常基线使用共享温湿度和 elapsed_min 预测 M4、M7、M4-M7、logratio，扩散模型只生成稳定正常残差，数量限制为稳定正常样本的 20%。结果位于 `code reproduction/experiment_augmentation/runs/stable_residual_diffusion_20260909/`。跨日期留出时，稳定 Mahalanobis AUROC 约 `0.295`，加入正常残差 DDPM 后约 `0.287`，说明当前三个实验批次的稳态水平仍有明显批次漂移，必须增加按批次的稳态校准/自适应阈值；当前不应把生成样本直接用于生产。

DiffAD 相关的论文与公开代码清单已补充到 `code reproduction/reports/DIFFAD_RELATED_PAPER_CODE_MAP_20260909.md`，包含 DiffAD、D3R、CSDI、TimeGrad、Diffusion-TS、NsDiff、IMDiffusion、ProDiffAD、TIMEADDM，以及 TSADC、CT-DDPM、C3DAD、FACD 等候选。清单明确区分上游官方代码、论文机制本地适配和待复现仓库，并说明它们与本工程双设备、共享温湿度、稳定段和上电衰减数据协议的差异。

- 上游原始仓库：`code reproduction/upstream/`
- 数据适配：`code reproduction/adapters/data_adapter.py`
- 预测/插补/异常模型适配：`code reproduction/adapters/models.py`
- 独立论文预测分支：`code reproduction/individual_paper_branches/models.py`
- 独立分支说明与运行边界：`code reproduction/individual_paper_branches/README.md`
- 统一运行器：`code reproduction/run_reproductions.py`
- 官方源码编译/导入检查：`code reproduction/local_runs/final/official_smoke.json`
- 预测结果：`code reproduction/local_runs/final/prediction/*/predictions.csv`、`metrics.json` 和模型文件
- 插补结果：`code reproduction/local_runs/final/imputation/*/metrics.json` 和模型文件
- 异常分数：`code reproduction/local_runs/final/anomaly/*/scores.csv`、`metrics.json` 和模型文件
- 机器可读总清单：`code reproduction/local_runs/final/run_manifest.json`
- 固定依赖版本：`code reproduction/requirements-reproduction.txt`

## 8. 复现限制与下一步

1. 公开仓库的论文数据通常是高频、长序列、多变量或有明确异常标签；本工程采样中位间隔约 12 小时，核心目标是单步 `Measure` 预测，且位置/线路是静态条件。因此本报告将‘官方代码运行’、‘共享适配器’和‘独立论文机制迁移’分开记录。
2. 本轮最终适配使用 CPU 可运行设置（预测扩散步数 20、40 个训练 epoch、32 次采样；插补/异常采用 6 个 epoch），结果用于工程迁移比较，不代表论文原文的完整超参数复现。
3. 插补结果基于人工缺失；异常结果基于伪异常；真实部署前必须补采带标签的现场异常或多场实验室异常。
4. 建议下一轮只重点深调四条路线：当前四阶段 FMLP+条件残差扩散、NsDiff 独立分支、Diffusion-TS 的趋势/季节分解、Treeffuser 的条件分布建模；使用 3 个以上随机种子和滚动时间外推验证。
5. 生产切换前应把最优研究候选的 manifest 接入在线推理，并保留当前在线候选作为回滚版本。

---
基础复现表由 `generate_reproduction_report.py` 根据 `local_runs/final` 生成；独立论文分支章节由本次独立实验结果补充，重新生成基础报告时应保留该章节。
