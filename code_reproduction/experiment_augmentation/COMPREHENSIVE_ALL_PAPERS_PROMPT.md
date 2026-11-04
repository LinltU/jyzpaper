# 全面测试所有论文的扩散生成器与检测器（本地化适配 + 大规模增强 + Markdown 报告）

## 任务总目标

对 `C:/project/JYZmentor/dataset/experiment.xlsx`（WSL 路径 `/mnt/c/project/JYZmentor/dataset/experiment.xlsx`），把 `code reproduction/upstream/` 与 `code reproduction/upstream_papers/` 中**全部**论文代码做本地化部署适配，并完成两套测试：

1. **生成器测试**：每篇论文的生成/插补模型适配为「类别条件生成器」→ 生成大量合成窗口做扩散数据增强 → 用统一分类器（RF/HGB）评估增强收益。
2. **检测器测试**：每篇异常检测论文的原生模型适配为「检测器」→ 在「先扩散增强、再检测」流程下训练与评分。

最后产出**训练结果 Markdown 报告**，对比指标：**Precision、Recall、F1、ROC-AUC、Average Precision**。

## 必须遵守的数据与协议约束（不可更改）

- **数据源**：`dataset/experiment.xlsx`（SHA256 `27b51b22481f5d3413da6f2b191163932e0e76eb20b3abff37e75fb6f71ab49b`，1419 行）。每日期两张表：`YYMMDD_80kv_9p_设备号(4).csv` / `(7).csv`（第 4、7 片绝缘子双设备）。
- **排除上电衰减**：正常段取每日期最后 ≤40 个正常行中与末端 10 点中位数偏差 <15% 的连续稳定后缀；每个物理故障阶段丢弃前 2 个配对点；每日期前 8 个稳定正常对作为基线校准，不参与任何模型训练与评分。
- **双设备配对**：40s 最近时间戳 merge_asof、一对一匹配、丢弃两设备 error 不一致的行。
- **联合特征（训练数据特征，报告必须标明）**：`M4_residual`、`M7_residual`、`M4-M7`、`M4/M7`、`log(M4/M7)`，共 5 维，窗口长 4。残差 = 对每日期正常稳定段按 `Measure ~ 1 + Temp4 + Hum4 + elapsed` 最小二乘回归后的残差（或按现有 `upstream_large_augmentation.py` 的基线定义）。
- **划分**：留一日期测试（250605 / 250626 / 250718）；其他日期按时间顺序切 fit/validation（最后故障阶段作 validation）；窗口只在同一划分与阶段内滑动。标签 `error != 0` 二分类，error 不作为检测特征。
- **阈值**：只在 validation 上取 F1 最大阈值（并列取更高阈值），测试集指标必须用该阈值计算。
- **指标**：Precision、Recall、F1、ROC-AUC、Average Precision（真实测试行上计算，每配置 = 3 日期 × 多种子，报告均值 ± 标准差）。
- **种子**：42、123、2024（至少 3 个）。**增强量**：1x / 10x / 100x / large（每类 ≥ 10,000 窗口），即“大范围尝试、大数据量生成”。

## 运行环境（必须按此执行，否则无法本地运行）

- 解释器：`/mnt/c/project/JYZmentor/code reproduction/runtime/python.exe`（Windows Python 3.10.21；torch 2.2.2+cpu、sklearn 1.5.2、pandas 2.1.4、numpy 1.26.4）。**给 python.exe 传参必须用 Windows 路径**（如 `C:/project/JYZmentor/...`），bash 内做文件存在性检查用 `/mnt/c/...` 路径。
- **无 TensorFlow**：依赖 TF 的仓库（如 DiffAD 的 tf_dataset 分支）必须用 torch 移植其机制，或改用其 torch 实现文件；报告注明。
- CPU only，14 核。后台任务用 `setsid nohup ... &` 会被工具会话终止杀死；**大规模训练请用前台分批 + wait**（参考现有 `run_detection_batch.sh`：每批 6 个并行、单次 bash 调用内 `wait` 完成，超时设置 ≥ 600s），或用独立 tmux/screen。

## 参考的现有集成（先读这些文件再动手，保持口径一致）

- `code reproduction/experiment_augmentation/upstream_large_augmentation.py`：已集成 CSDI / MTSCI / IMDiffusion / Diffusion-TS / D3R 的 `Upstream` 类（含模型加载、损失、采样、Diffusion-TS FourierLayer 短窗口补丁）。**新增论文照此模式集成**。
- `code reproduction/experiment_augmentation/augmented_upstream_detection.py`：D3R（重构 MSE 评分）与 IMDiffusion（插补误差评分）检测器，含增强→检测流程。新增检测器照此模式。
- `code reproduction/experiment_augmentation/aggregate_upstream_markdown.py` / `aggregate_detection_markdown.py`：结果聚合与 Markdown 生成（含术语解释、指标公式、完美模型对照、结论）。扩展为新报告模板。
- `code reproduction/sources_manifest.json`：仓库 URL 与固定 commit 清单。

## 论文清单与适配任务（全部论文）

### A. 生成器 / 增强源（适配为类别条件生成器，训练两个类别专用模型）

| 论文 | 上游路径 | 原生用途 | 本地适配要求（修改为本地可执行参数） |
|---|---|---|---|
| CSDI | `upstream/CSDI` | 插补 | 已集成；掩蔽后半窗生成 |
| MTSCI | `upstream/MTSCI` | 一致性插补 | 已集成；加一致性损失 |
| IMDiffusion | `upstream/IMDiffusion` | 插补/检测 | 已集成；strategy_type=0 生成 |
| Diffusion-TS | `upstream/Diffusion-TS` | 生成 | 已集成；含 Fourier 短窗口补丁 |
| D3R | `upstream/D3R` | 漂移重构 | 已集成；残差窗口生成 |
| TimeGrad | `upstream/TimeGrad-pytorch-ts` | 自回归生成 | 改用本地 seq_len=4、feature=5、CPU、小 hidden，输出窗口生成 |
| NsDiff | `upstream/NsDiff` | 非平稳生成 | 按窗口均值/尺度归一化 + DDPM；本地小参数 |
| FIDE | `upstream_papers/FIDE` | 极端样本生成 | 适配 4 步短窗；频率分支截断到短窗可用频率 |
| PaD-TS | `upstream_papers/PaD-TS` | 群体条件生成 | 类别条件 = 标签 one-hot + 日期 one-hot |
| WaveStitch | `upstream_papers/WaveStitch` | 条件生成 | 短窗生成，本地 batch/step 参数 |
| TabDDPM | `upstream_papers/TabDDPM` | 表格生成 | 窗口展平为 20 维表格特征做生成，再还原 4×5 |
| CCDM | `upstream/CCDM` | 预测生成 | 无条件生成正常/异常两类窗口 |
| DiffPLF | `upstream/DiffPLF` | 预测生成 | 同上 |
| TSDE | `upstream/TSDE` | 表示/插补 | 用插补分支当生成器 |
| SSSD | `upstream/SSSD` | 插补 | 掩蔽后半窗生成 |
| CoSTI | `upstream/CoSTI` | 一致性插补 | 掩蔽后半窗生成 |
| SSD-TS | `upstream/SSD-TS` | 长序列插补 | 4 步短窗下关掉长程结构模块，保留扩散核心 |
| MCD-TSF | `upstream_papers/MCD-TSF` | 多通道预测/插补 | 类别条件生成窗口 |

### B. 检测器（适配为「先增强再检测」流程的异常评分器）

| 论文 | 上游路径 | 原生检测方式 | 本地适配要求 |
|---|---|---|---|
| D3R | `upstream/D3R` | 重构 MSE | 已集成 |
| IMDiffusion | `upstream/IMDiffusion` | 多次插补一致性评分 | 已集成（单次插补误差评分） |
| DiffAD | `upstream/DiffAD` | 权重增量扩散重构误差 | TF 分支不可用 → 移植其 torch 机制：正常重构误差评分 |
| ICDiffAD | `upstream/ICDiffAD` | 插补条件重构 | 用其评分函数，本地 pkl 装载改为 npz |
| TIMEADDM | `upstream/TIMEADDM` | 扩散重构误差 | 本地窗口化、feature=5、CPU |
| ProDiffAD | `upstream/ProDiffAD` | 蒸馏快速重构 | 本地步数 4~10，正常训练重构评分 |
| TSADC | `upstream/TSADC` | 掩蔽重构 MSE 双评分 | 直接复用其 score1/score2 逻辑 |
| NGLS-Diff | `upstream_papers/NGLS-Diff` | 正常潜空间距离 | 潜空间维度降为本地规模 |
| DiffTAD | `upstream_papers/DiffTAD` | 轨迹重构 | 4 步窗适配 |
| diffusionae | `upstream_papers/diffusionae` | 自编码+扩散重构 | 小 latent、CPU |
| C3DAD | `upstream/C3DAD` | 对比扩散 | 对比分支适配双设备 5 维特征 |

### C. 排除项（报告注明原因）

- `CT-DDPM`：仓库为空（仅 README），跳过。
- `FACD / FACD_download / Treeffuser / SSD-TS`（若本地依赖缺失或与扩散增强无关）：评估后决定，跳过须在报告中写明理由。
- 依赖 TensorFlow 且无 torch 文件的仓库：只做机制移植，标注「非官方代码运行，为机制适配」。

## 参数修改要求（“修改成本地论文可执行的参数”）

每集成一篇论文，必须：
1. 在适配代码中显式设置本地参数：`seq_len=4`、`feature_dim=5`、`hidden=32~64`、`batch=32~128`、`diffusion_steps=20~30`、`epochs=60~80`、`device='cpu'`，替换论文默认的 ETT/SWaT 长序列参数。
2. 在 result.json 记录 `adapted_parameters`（改了哪些参数、原值/新值）与所用上游文件 SHA256。
3. 不修改 `upstream/`、`upstream_papers/` 内任何源文件；适配代码全部放在 `experiment_augmentation/` 新脚本中。
4. 生成数据必须满足物理约束：M4>0、M7>0，并据此重算差值/比例保持特征一致性（projected 模式）；同时保留 raw 模式对照。

## 执行规模（大范围、大数据量）

- 每个生成器：3 种子 × 3 日期 × {1x, 10x, 100x, large(≥10,000/类)} 增强量，全跑。
- 每个检测器：真实基线 + 5 种增强源 × {10x, large}，3 种子 × 3 日期。
- 每次运行的 result.json 必须含：方法、种子、日期、生成窗口数、训练 loss 曲线、指标、参数修改记录、耗时。
- 全部结果 CSV 聚合，防止新旧路径重复计数（按 (detector, aug, volume, seed, date) 去重）。

## 报告要求（Markdown，路径 `code reproduction/experiment_augmentation/runs/ALL_PAPERS_RESULTS.md`）

必须包含以下章节：
1. 实验设置与**训练数据特征**（明确列出 `M4_residual、M7_residual、M4-M7、M4/M7、log(M4/M7)`）；
2. 上电衰减排除规则说明；
3. 术语解释（上游固定版、role/label/n、regime、阈值）；
4. **每篇论文的适配说明表**（原生用途、改了什么参数、是否官方代码运行还是机制移植）；
5. 生成器增强对比表：方法 × 增强量 × Precision/Recall/F1/ROC-AUC/AP（均值 ± 标准差），含真实数据基线；
6. 检测器对比表：检测器 × 增强源 × 各指标；
7. 全论文横向最佳结果汇总与结论（完美模型=全 1.0 的对照、当前达成水平、哪篇论文在哪个环节有效）。

## 验收标准

- 所有可运行的生成器与检测器都已适配并出数；不能运行的给出书面原因。
- 每个配置至少 3 种子、报告含标准差；增强量覆盖到 ≥10,000 窗口/类。
- 报告中标明训练数据特征、参数修改记录、上游代码 SHA256 审计。
- 全流程可复现：给出可一键重跑的命令清单（附在报告末尾）。
