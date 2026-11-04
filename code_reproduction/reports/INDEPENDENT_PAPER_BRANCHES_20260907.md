# 独立论文方案复现与数据集差异说明

日期：2026-09-07  
工程：JYZmentor 绝缘子测量电压预测

## 1. 本次完成内容

此前的 TimeGrad、DiffPLF、CCDM、TSDE、NsDiff 使用同一个
`ConditionalResidualDiffusion`，只改变 `mode`，只能称为“统一适配器对比”。
本次新增 `code reproduction/individual_paper_branches/models.py`，将 5 个方案拆为
5 个独立类和独立训练循环：

| 方案 | 独立实现的核心 | 是否使用官方仓库代码 |
|---|---|---|
| TimeGrad | GRU 历史状态 + 条件 DDPM 残差采样 | 否；保留上游仓库并按本工程窗口复现论文机制 |
| DiffPLF | 局部线性趋势 + 局部残差尺度 + 扩散残差 | 否；论文机制本地适配 |
| CCDM | Measure/Temp/Hum 独立通道投影 + 通道融合 | 否；论文机制本地适配 |
| TSDE | IIF（插补/插值/预测）掩码 + 双向 GRU 条件扩散 | 否；论文机制本地适配 |
| NsDiff | 每窗口均值/尺度归一化 + 非平稳残差扩散 | 否；论文机制本地适配 |
| Diffusion-TS | 已有完整分支：Transformer、TrendBlock、FourierLayer | 否；基于论文结构重写本地分支 |
| Treeffuser | 官方 Treeffuser 包入口 | 是，官方包可运行时才使用 |

说明：上表中的“独立实现”意味着每个方案拥有不同的编码器、条件变量和训练步骤，
不是通过同一个模型传入不同字符串参数。由于论文官方代码的数据加载器、任务定义和
输入形状都针对各自公开数据集，不能直接把原仓库训练入口套到本工程，因此本次属于
“上游机制对齐 + 本地数据适配”，不是声称逐行复现论文代码。

## 2. 统一本工程数据协议

- 训练文件：`dist2026/latest_retraining_data_20260716/training_merged.csv`
- 测试文件：`dist2026/latest_retraining_data_20260716/test_latest_holdout.csv`
- 训练记录 9,520，测试记录 1,715，合计 11,235。
- 训练窗口 8,920；实际拟合 7,569；验证 1,351；最终测试 1,715。
- 每个样本使用过去 24 步，预测当前时刻第 25 步 `Measure`。
- 历史输入 26 个特征，目标时刻已知条件 13 个特征。
- `Battery` 已排除，它是测量设备电量，不是绝缘子测量电压状态。
- 采用按设备的时间后段验证切分；测试集标签只用于最终报告，不参与调参。
- 所有方案输出样本分布，再计算 P10/P50/P90，并使用验证集进行分阶段校准。

## 3. 独立分支训练结果

以下为同一 CPU 配置下的机制验证/公平对比：12 epoch、hidden=64、8 个扩散步、
16 个样本、相同训练/验证/测试窗口。结果文件位于：
`code reproduction/local_runs/individual_final_20260907/`。

| 方案 | RMSE | MAE | R² | P10-P90覆盖率 | 平均区间宽度 |
|---|---:|---:|---:|---:|---:|
| TimeGrad（独立） | 0.28100 | 0.17402 | 0.01557 | 74.17% | 0.40243 |
| DiffPLF（独立） | 0.31283 | 0.20141 | -0.22013 | 72.48% | 0.45878 |
| CCDM（独立） | 0.28768 | 0.18267 | -0.03178 | 74.81% | 0.45199 |
| TSDE（独立） | 0.29008 | 0.18329 | -0.04910 | 74.99% | 0.45625 |
| NsDiff（独立） | **0.22978** | **0.14229** | **0.34171** | 74.99% | **0.33827** |

这组结果表明，在当前数据和当前训练预算下，独立 NsDiff 分支最好；但它仍未超过
此前调参后的完整 Diffusion-TS 适配器（RMSE 约 0.2086）。因此不能把本表结论写成
“NsDiff 优于所有论文方案”，只能写成“在同一低成本独立分支配置下，NsDiff 最好”。
完整 Diffusion-TS 的 Transformer/Fourier 分支使用了不同训练预算，应单独进行同预算
多随机种子复核后再做最终排名。

## 4. 论文原始数据与本工程数据差异

| 方案 | 原论文常见任务/数据 | 本工程任务 | 关键差异及影响 |
|---|---|---|---|
| TimeGrad | 概率多步时间序列预测，常见电力、交通、Solar 等规则序列 | 多设备单步 `Measure` 预测 | 原论文通常连续多步生成；本工程只预测一步，GRU 自回归优势未完全发挥；本工程还有设备位置条件 |
| DiffPLF | 条件概率预测/残差扩散基准时序 | 温湿度和位置条件下的电压残差 | 论文数据多为均匀采样公共序列；本工程采样间隔约 12 小时且不严格规则，局部趋势和尺度含义不同 |
| CCDM | 多变量通道概率预测 | 26 维历史 + 13 维目标条件，目标只有 Measure | 原论文通道通常是同一序列的多个观测变量；本工程既有直接传感器量，也有派生统计量和位置编码，通道相关性不同 |
| TSDE | 自监督插补/插值/预测，使用 ETT、天气等掩码任务 | 在线单步预测 | 原论文训练目标包含缺失值恢复；本工程没有真实缺失标签，IIF 掩码是机制迁移，不能等同于原论文插补指标 |
| NsDiff | 非平稳时间序列预测，重点处理分布随时间变化 | 湿度阶段和设备差异引起的非平稳 Measure | 本工程非平稳性主要来自湿度、设备位置和环境变化，而不是论文中的长周期金融/交通分布漂移；局部归一化更贴合当前窗口 |
| Diffusion-TS | 完整序列生成/预测，ETTh、Energy、Stocks、fMRI、EEG 等 | 24 步到第 25 步单步预测 | 原任务通常生成整段序列并利用趋势/季节结构；本工程窗口短、采样不规则、目标单变量，Fourier 优势可能被削弱 |
| Treeffuser | 表格条件分布预测，LightGBM score model + diffusion sampler | 历史统计量、位置和环境条件预测 Measure | Treeffuser 更接近表格回归；本工程的时间依赖比一般表格任务强，必须保留历史窗口特征 |

共同差异：论文公开数据通常规则采样、单一数据源、异常或缺失协议明确；本工程是
25 个设备的实验/现场混合数据，真实挂线异常标签不足，且只有温度、湿度、测量电压、
线路位置和输入电压等有限原始量。由此，论文中的 CRPS、插补 MAE、异常 F1 等指标
不能直接与本工程的单步 RMSE、MAE、R²、区间覆盖率混为一谈。

## 5. 源代码与运行方式

独立实现：

```text
C:\Project\JYZmentor\code reproduction\individual_paper_branches\models.py
```

统一入口：

```text
C:\Project\JYZmentor\code reproduction\run_reproductions.py
```

运行示例（使用工程自带 Python 运行时）：

```powershell
& 'C:\Project\JYZmentor\code reproduction\runtime\python.exe' `
  'C:\Project\JYZmentor\code reproduction\run_reproductions.py' `
  --only prediction `
  --methods TimeGrad,DiffPLF,CCDM,TSDE,NsDiff `
  --output 'C:\Project\JYZmentor\code reproduction\local_runs\individual_final_20260907' `
  --epochs 12 --batch-size 512 --hidden-dim 64 --diffusion-steps 8 --samples 16 --skip-smoke
```

结果：

- 每个方案：`prediction/<方法名>/training_history.json`
- 测试预测表：`prediction/<方法名>/predictions.csv`
- 指标：`prediction/<方法名>/metrics.json`
- 汇总：`prediction_summary.json`

上游仓库仍保存在 `code reproduction/upstream/`，来源和 commit 记录在
`code reproduction/sources_manifest.json`。这些文件用于溯源和官方代码编译检查，
本次本地训练没有擅自把论文公开数据混入绝缘子数据。

## 6. 结论与下一步

1. 5 个剩余预测方案已经完成独立本地实现并跑通统一测试协议。
2. 当前独立分支中 NsDiff 最适合继续优化，原因是本工程高湿阶段存在明显局部尺度变化。
3. Diffusion-TS 仍是当前总体精度候选，但必须用与独立分支相同 epoch、扩散步数和样本数
   重新复核，才能形成严格公平排名。
4. 由于缺少真实挂线异常标签，当前区间越界只能作为“无监督/弱监督预警候选”，不能直接
   宣称已经验证真实故障预警能力。
5. 下一步建议：对 NsDiff、完整 Diffusion-TS 和 25% DDPM-FMLP 做三随机种子同预算
   复核，并按四个湿度阶段分别报告误差和区间覆盖率。

补充：当前工程的插补实验是在已有 24 步窗口中人工遮挡点或连续位置，恢复
`Measure/Temp/Hum` 单元格；它不等同于原始采集文件中整行时间记录缺失。原始数据若存在
设备离线造成的时间段空洞，目前不会自动补齐时间戳，需要另建规则时间网格和时间段级插补评估。

## 7. 同预算多种子复核与调参结果（2026-09-07）

为验证“NsDiff 独立实现最好、Diffusion-TS 是整体候选”的结论，新增公平运行器
`code reproduction/fair_compare_nsdiff_diffusion_ts.py`，统一使用：8 epochs、batch=512、
hidden/d_model=64、扩散步数/时间步数=8、16 次采样、学习率 8e-4，随机种子 42/43/44。
两者仍保留各自的论文结构，因此参数量不强行相等，但数据、窗口、训练轮数和采样预算一致。

| 方法 | RMSE均值±标准差 | MAE均值±标准差 | R²均值±标准差 | 覆盖率均值±标准差 | 区间宽度 |
|---|---:|---:|---:|---:|---:|
| NsDiff | **0.24069 ± 0.00074** | 0.15645 ± 0.00133 | **0.27772 ± 0.00446** | **79.61% ± 0.59%** | 0.45757 |
| Diffusion-TS | 0.25486 ± 0.01488 | **0.15178 ± 0.01120** | 0.18741 ± 0.09453 | 71.25% ± 2.77% | **0.30753** |

公平结果说明：NsDiff 的点预测 RMSE 平均低 0.01417，随机波动也更小；Diffusion-TS 的
平均 MAE略低、区间更窄，但 seed=43 的点预测明显回退，稳定性不足。当前证据支持先
保留 NsDiff 作为独立论文方案候选，同时继续优化 Diffusion-TS 的目标位置权重、设备条件和
时间间隔编码。

### 调参试验

在 seed=42 上先用验证集观察配置，再查看测试集最终结果：

| 方案 | 关键调参 | 验证RMSE | 测试RMSE | 测试覆盖率 |
|---|---|---:|---:|---:|
| NsDiff基线 | 8轮，64维，8步，lr=8e-4 | 0.14488 | 0.24164 | 79.01% |
| NsDiff | lr=1.2e-3，8轮 | 0.13636 | 0.23694 | 77.90% |
| NsDiff | lr=1.2e-3，12轮 | **0.11747** | **0.22821** | 64.90% |
| NsDiff | hidden=96 | 0.14429 | 0.24213 | 78.66% |
| Diffusion-TS | 目标位置权重=8 | 0.13329 | 0.24895 | 68.05% |

NsDiff 增大学习率并训练更久明显提升点预测，但区间覆盖率下降，不能直接用于预警部署。
因此需要把点预测调参和 P10/P90 区间校准分开优化，例如使用独立校准集提高覆盖率，避免
通过扩大或缩小区间影响对点预测模型的判断。

公平复核产物：

- `local_runs/fair_nsdiff_diffusion_ts_20260907_v2/seed_results.csv`
- `local_runs/fair_nsdiff_diffusion_ts_20260907_v2/summary.csv`
- `local_runs/fair_nsdiff_diffusion_ts_20260907_v2/FAIR_COMPARISON.md`
- `local_runs/fair_nsdiff_diffusion_ts_20260907_v2/TUNING_NOTES.md`
