# 论文预测方案调参与训练过程说明

更新时间：2026-09-07

## 1. 实验结论

本轮重新训练并调参了 7 个预测方案：TimeGrad、DiffPLF、CCDM、TSDE、NsDiff、Diffusion-TS 和 Treeffuser。

公开论文方案适配组中，调参后的 Diffusion-TS 点预测最好：RMSE=0.20863、MAE=0.11698、R²=0.45734。它不是直接加载论文预训练权重，而是使用固定 Git 源码中的核心思想适配本工程数据；Treeffuser 是本轮唯一实际调用官方包的方案。

当前应区分两个目标：

- 点预测：调参后的 Diffusion-TS 更好；
- 区间预测和异常预警：25% DDPM 增强 FMLP 更稳妥。

## Diffusion-TS论文核心原理

### 原始论文核心机制

Diffusion-TS 的论文题目是 *Diffusion-TS: Interpretable Diffusion for General Time Series Generation*，发表于 ICLR 2024。它学习的不是单一确定值，而是：

```text
给定历史或条件 → 生成多个合理的时间序列样本
```

### 2A.1 前向加噪和反向去噪

训练阶段将真实时间序列逐步加入高斯噪声：

```text
x0（真实序列） → x1 → ... → xT（接近随机噪声）
```

模型学习每个扩散时间步的还原过程。生成阶段从随机噪声开始反向去噪：

```text
xT（随机噪声） → ... → x1 → x0（生成序列）
```

### 2A.2 直接重建和可解释分解

与许多直接预测噪声的 DDPM 实现不同，Diffusion-TS 论文实现训练模型直接重建当前目标样本，并使用编码器-解码器 Transformer 提取时间结构。

其解码器把序列表示为：

```text
时间序列 = Trend趋势部分 + Season季节/周期部分
```

趋势部分使用多项式回归和分块均值表示；季节部分使用 Fourier/傅里叶表示，学习周期性波动。

### 2A.3 Fourier频域损失

论文除时间域重建损失外，还比较真实序列和重建序列的 Fourier 频谱：

```text
总损失 = 时间域重建损失 + λ × Fourier频域损失
```

它可以约束生成序列的频率结构，减少数值接近但波动节奏不合理的问题。

### 2A.4 条件预测

Diffusion-TS 支持条件生成、预测和插补。已知历史或已观测部分作为条件，模型只生成未知的未来或缺失部分。因此它输出的是条件分布，可以通过多次采样获得预测区间。

### 本工程对Diffusion-TS的改造

论文原始任务主要是生成一段多变量时间序列；本工程是非均匀采样、带设备和安装位置条件的单步 `Measure` 预测。因此没有直接使用论文原始数据加载器和完整 Transformer 解码器，而是保留可迁移的核心思想：条件扩散、趋势/状态基线、残差生成和多样本分位数输出。

本工程实际流程为：

```text
24步历史特征
→ 最近8步Measure加权状态基线
→ 目标Measure残差
→ 条件扩散去噪
→ 生成32个残差样本
→ 恢复32个Measure预测值
→ P10/P50/P90
```

因此当前结果应表述为：

> Diffusion-TS 论文核心机制在本工程绝缘子测量电压单步预测任务上的适配结果。

而不是直接使用 Diffusion-TS 原始预训练模型的结果。

论文原始代码位于 `code reproduction/upstream/Diffusion-TS/`；本工程适配实现位于 `code reproduction/adapters/models.py` 和 `code reproduction/run_reproductions.py`。

### 为什么前一轮没有逐层接入论文模型

这不是因为本地无法实现，而是因为论文原始代码和本工程任务存在明显接口差异：

1. **论文模型生成完整序列，本工程主要预测一个下一时刻值**。原始模型输入和输出都是 `[batch, sequence_length, feature_size]` 的完整序列；本工程的目标是使用 24 步历史预测第 25 个时刻的 `Measure`。
2. **论文默认均匀时间索引，本工程采样间隔不完全均匀**。原始 Fourier 约束按序列位置计算频率，而本工程还存在 `DeltaHours`。如果直接使用 Fourier 损失，它会把“不规则时间间隔”误当成规则周期结构。
3. **论文原始数据加载器面向 ETTh、Energy、Stocks、fMRI 等数据**。它的 `feature_size`、mask、预测长度和归一化方式需要改成设备分组、位置条件、湿度阶段和本工程的训练/验证/测试切分。
4. **本工程需要避免目标泄漏**。预测目标时刻的 `Measure` 派生特征不能输入模型，因此需要重新设计 future mask 和条件输入。
5. **完整 Transformer 版本计算量和调参成本更高**。前一轮需要同时比较 7 个论文方案，先采用统一轻量适配器是为了保证训练预算、数据切分和指标口径一致。

### 本地是否可以实现完整版本

可以。上游 Diffusion-TS 的 Transformer、TrendBlock、FourierLayer 和 Fourier loss 都已经保存在：

```text
code reproduction/upstream/Diffusion-TS/
```

建议增加一个独立的完整复现分支，而不是覆盖当前适配器，具体改造为：

```text
24步历史 + 1个待预测位置 → Transformer Diffusion-TS
```

训练时将第 25 个位置的 `Measure` 设为 mask 或噪声目标；历史 24 步和目标时刻已知的温度、湿度、位置条件作为条件；采样后只读取第 25 个位置的 `Measure`，再计算 P10/P50/P90。

对于 Fourier loss，需要做两组对照：

- `use_ff=False`：适合作为不规则采样的保守版本；
- `use_ff=True`：仅在按实际时间间隔重采样或加入时间感知频率编码后使用。

这样才能判断提升来自 Transformer 趋势/季节解码器，还是仅来自更大的模型容量。完整版本应与当前适配器使用相同的 7,569 个训练窗口、1,351 个验证窗口、1,715 个测试窗口和 3 个随机种子重新比较。

另外需要准确修正：当前 `adapters/models.py` 中的 `ConditionalResidualDiffusion` 仍然采用“预测加入的噪声”的轻量 DDPM 训练目标；它没有实现论文原始的“直接重建样本 + Fourier 损失”。因此当前结果是 Diffusion-TS 思想的条件残差扩散适配结果，不应标记为完整 Diffusion-TS 原模型结果。

特别说明：本地论文方案适配器使用轻量条件残差去噪网络，当前没有逐层复现论文的 Transformer 趋势/季节解码器和 Fourier 损失。因此本地结果用于评估 Diffusion-TS 思想在本工程上的可迁移性，不应称为论文原始模型的逐层等价复现。

## 2. 数据量和切分

数据文件：`dist2026/latest_retraining_data_20260716/training_merged.csv` 和 `test_latest_holdout.csv`。

| 数据部分 | 数量 | 用途 |
|---|---:|---|
| 合并后真实记录 | 11,235 | 特征计算 |
| 训练记录 | 9,520 | 训练和验证 |
| 真实留出测试记录 | 1,715 | 最终测试 |
| 训练窗口总数 | 8,920 | 由训练记录构造 |
| 实际拟合窗口 | 7,569 | 更新模型参数 |
| 验证窗口 | 1,351 | 选参和校准 |
| 测试窗口 | 1,715 | 最终评价 |

每台设备按时间排序，最后约 15% 窗口作为验证集，前 85% 作为训练集。任务是单步预测：目标时刻之前的 24 步历史预测目标时刻 `Measure`。

## 3. 当前使用的特征

代码位置：`code reproduction/adapters/data_adapter.py`。

### 3.1 24步历史输入的26个特征

`Measure`、`Temp`、`Hum`、`AH`、`DeltaHours`、`MeasureRate`、`TempRate`、`HumRate`、`MeasureEMA3`、`HumEMA3`、`MeasureMAD6`、`HumMAD6`、`HoursAbove65`、`HoursAbove80`、`HumRising`、`HumFalling`、`HumInvalid`、`LineVoltageKV`、`Loc_SideCode`、`Loc_PhaseCode`、`Loc_SizeCode`、`Loc_StrNum`、`Loc_PlateCount`、`Loc_PlateStart`、`Loc_PlateEnd`、`Loc_PlatePositionRatio`。

这些特征分别描述：原始测量、温湿度、绝对湿度、采样间隔、变化率、指数平均、局部波动、湿度持续时间、湿度趋势、线路电压以及绝缘子安装位置。

单个历史输入形状为 `[24, 26]`。

### 3.2 目标时刻的13个条件特征

`Temp`、`Hum`、`AH`、`DeltaHours`、`LineVoltageKV`、`Loc_SideCode`、`Loc_PhaseCode`、`Loc_SizeCode`、`Loc_StrNum`、`Loc_PlateCount`、`Loc_PlateStart`、`Loc_PlateEnd`、`Loc_PlatePositionRatio`。

`Battery` 是测量设备电量，与绝缘子测量电压无关，因此不进入历史特征、目标条件、扩散模型和异常评分。

## 4. 湿度阶段

| 阶段 | 条件 |
|---|---|
| `dry` | Hum < 50 |
| `transition` | 50 ≤ Hum < 65 |
| `humid` | 65 ≤ Hum < 80 |
| `very_humid` | Hum ≥ 80 |

阶段转成 4 维 one-hot 条件。测试集数量为：dry 15、transition 211、humid 547、very_humid 942。

## 5. 调参后Diffusion-TS模型尺寸

条件维度为：24×26=624维历史特征 + 13维目标条件 + 4维湿度阶段 + 25维设备身份 = 666维。再加入 32 维扩散时间编码和 1 维带噪残差，去噪网络输入为 699 维。

最优配置：

| 参数 | 值 |
|---|---:|
| `hidden_dim` | 160 |
| `epochs` | 60 |
| `diffusion_steps` | 24 |
| `learning_rate` | 5e-4 |
| `batch_size` | 512 |
| `samples` | 32 |

网络结构：`Linear(699→160) → SiLU → Linear(160→160) → SiLU → Linear(160→80) → SiLU → Linear(80→1)`。

可训练参数约 150,721 个。对比配置：`hidden_dim=96` 约 81,217 个参数，`hidden_dim=128` 约 114,433 个参数。

## 6. 训练过程

1. 对每台设备按时间排序，构造目标时刻之前的 24 步因果窗口。
2. 只用训练拟合窗口计算标准化统计量。
3. 根据论文方案计算基础预测。
4. 计算 `Residual = Target Measure - Base Prediction`。
5. 对残差随机加入扩散噪声。
6. 去噪网络根据带噪残差、时间编码、历史特征、目标环境、位置、湿度阶段和设备身份预测噪声。
7. 使用噪声均方误差训练。

Diffusion-TS 使用最近 8 个历史 `Measure` 的加权状态作为基础预测。模型学习的是历史趋势之外的创新残差，而不是直接拟合绝对电压。

## 7. 预测过程

从随机噪声开始，执行 24 次反向去噪，得到残差样本，再恢复：`Predicted Measure = Base + Generated Residual`。

每个目标窗口生成 32 个候选值，并计算：

- `P10`：10%分位数；
- `P50`：50%分位数，用作点预测；
- `P90`：90%分位数；
- `P10-P90`：预测不确定性区间。

注意：`samples=32` 表示每个目标时刻生成 32 个候选值，不是预测未来 32 个时间步。

## 8. 调参思路

每个方案测试 4 组配置，共 28 组实验：

| 配置 | Epoch | hidden_dim | 扩散步数 | 学习率 | samples |
|---|---:|---:|---:|---:|---:|
| baseline | 40 | 128 | 20 | 8e-4 | 32 |
| low_lr_long | 60 | 128 | 30 | 5e-4 | 32 |
| compact_fast | 60 | 96 | 16 | 5e-4 | 32 |
| wide | 60 | 160 | 24 | 5e-4 | 32 |

选参规则是：训练窗口拟合，验证集按 RMSE 排序，RMSE 相同时比较 MAE，最后固定参数后才查看测试集。

Diffusion-TS 最优配置为 `wide`。增加隐藏层和训练轮数、降低学习率后，对高维历史、位置、设备和湿度条件的表达更充分。

## 9. 训练和预测数量

单次调参后的 Diffusion-TS：

| 阶段 | 数量 |
|---|---:|
| 训练窗口 | 7,569 |
| 验证窗口 | 1,351 |
| 测试窗口 | 1,715 |
| 训练轮数 | 60 |
| Batch size | 512 |

每个 epoch 约 15 个 batch，60 个 epoch 约处理 `7,569×60=454,140` 次窗口级训练样本。

验证阶段生成 `1,351×32=43,232` 个随机预测样本；测试阶段生成 `1,715×32=54,880` 个随机预测样本，最终输出 1,715 条预测记录。

## 10. 多随机种子验证

固定验证集选出的配置后，使用 seed=42、43、44 重复：

| Seed | RMSE |
|---:|---:|
| 42 | 0.20863 |
| 43 | 0.21048 |
| 44 | 0.20962 |
| 平均 | 0.20958 |
| 标准差 | 0.00093 |

三个种子均低于 0.211，说明调参收益不是单个随机种子的偶然结果。

## 11. 结果提升

### 11.1 相比原始Diffusion-TS

| 指标 | 原始配置 | 调参后 | 提升 |
|---|---:|---:|---:|
| RMSE | 0.21542 | 0.20863 | 降低约3.15% |
| MAE | 0.12196 | 0.11698 | 降低约4.08% |
| R² | 0.42142 | 0.45734 | 提高约0.036 |
| 覆盖率 | 67.87% | 72.48% | 提高约4.61个百分点 |

### 11.2 与工程模型比较

| 模型 | RMSE | MAE | P10-P90覆盖率 |
|---|---:|---:|---:|
| 纯真实FMLP | 0.21740 | 0.12227 | 80.12% |
| 25% DDPM增强FMLP | 0.21496 | 0.12098 | 81.46% |
| 调参Diffusion-TS | **0.20863** | **0.11698** | 72.48% |

调参 Diffusion-TS 相比纯真实 FMLP 的 RMSE 降低约 4.04%、MAE 降低约 4.33%。相比 25% DDPM，RMSE 低约 2.95%、MAE 低约 3.31%，但区间覆盖率低约 8.98 个百分点。

## 12. 区间校准

Diffusion-TS 默认校准分位点为 0.80。将校准分位点提高到 0.90 后：

| 校准设置 | 覆盖率 | 平均区间宽度 |
|---|---:|---:|
| 0.80 | 72.48% | 0.2257 |
| 0.90 | 83.03% | 0.3156 |

提高校准分位点会扩大 P10-P90 区间，但不会明显改变 P50。正式上线前应使用独立校准集确定该参数，不能根据测试集结果直接固定。

## 13. 主要创新点

1. **因果历史窗口**：只使用目标时刻之前的数据，避免未来信息泄漏。
2. **残差扩散**：生成相对历史基线的 Measure 残差，而不是直接生成绝对电压。
3. **位置条件化**：将线路、左右侧、相位、大小号、串号、片数和相对位置编码为数值条件。
4. **湿度阶段条件化**：区分低湿、过渡、高湿和极高湿状态。
5. **验证集驱动调参**：训练集拟合、验证集选参、测试集只做最终评价。
6. **点预测与区间联合评价**：同时报告 RMSE、MAE、R²、覆盖率和区间宽度。

## 14. 代码和结果位置

| 内容 | 路径 |
|---|---|
| 数据适配 | `code reproduction/adapters/data_adapter.py` |
| 统一运行器 | `code reproduction/run_reproductions.py` |
| 调参脚本 | `code reproduction/tune_prediction_reproductions.py` |
| 多随机种子重复 | `code reproduction/run_best_prediction_seeds.py` |
| 调参报告 | `code reproduction/reports/TUNED_PREDICTION_REPORT_20260905.md` |
| 28组结果 | `code reproduction/local_runs/tuning_all_20260905/` |
| 三种子结果 | `code reproduction/local_runs/best_seed_repeats_20260905/` |
| 区间校准结果 | `code reproduction/local_runs/calibration_q090_20260905/` |
| 汇总CSV | `code reproduction/results/prediction_tuned_comparison_20260905.csv` |

## 15. 当前建议

建议保留两个模型版本：

- 点预测研究版：调参后的 Diffusion-TS，RMSE 约 0.209。
- 预测与预警部署版：25% DDPM 增强 FMLP，RMSE 约 0.215、P10-P90 覆盖率约 81.46%。
