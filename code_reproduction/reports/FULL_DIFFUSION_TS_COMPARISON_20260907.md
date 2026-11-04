# 完整 Diffusion-TS 分支实现与对比

更新时间：2026-09-07

## 1. 实现结论

本工程已经新增完整 Diffusion-TS Transformer 分支，按上游论文/仓库结构实现了：

- Transformer Encoder；
- Transformer Decoder；
- TrendBlock 多项式趋势分支；
- FourierLayer 周期分支；
- 直接重建 `x0` 的扩散训练方式；
- Fourier 频域损失。

本地实现了两个版本：

1. 完整 Transformer + 关闭 Fourier 损失；
2. 完整 Transformer + 开启 Fourier 损失。

两者使用相同数据切分和相同模型规模，用于单独判断 Fourier 损失是否有效。

说明：本轮公平复核时上游 `upstream/Diffusion-TS` 目录在当前 Windows 环境返回访问拒绝，
因此运行时采用 `full_diffusion_ts/model.py` 中的本地等价 Transformer、TrendBlock 和
FourierLayer 回退结构。该结构保持论文机制和输入输出协议，但不能表述为上游仓库代码直接运行；
上游仓库仍保留用于源码溯源和编译检查。

## 2. 与当前轻量适配器的区别

当前轻量适配器是：

```text
条件残差DDPM + MLP去噪器
```

完整分支是：

```text
上游Diffusion-TS Transformer
Encoder + Decoder + TrendBlock + FourierLayer
```

完整分支训练模型直接重建完整 25 位置序列，而不是只预测一维残差噪声。

## 3. 本工程中的完整分支任务定义

每个样本构造成长度 25 的序列：

```text
第1～24个位置：已观测历史窗口
第25个位置：目标时刻
```

第25个位置的 `Measure` 被设为未知，目标时刻的温度、湿度、时间间隔、线路电压和安装位置条件保留为已知信息。

模型只读取第25个位置的 `Measure` 作为最终预测：

```text
24步历史 + 目标环境/位置条件
→ 生成第25步完整状态
→ 读取第25步Measure
```

本实现使用 26 个序列通道：

- 26 个历史特征保持与论文适配实验一致；
- 目标时刻的 Measure 是待生成通道；
- 目标时刻的 Measure 派生特征不使用真实未来值，而采用最后历史状态作为因果占位。

## 4. 模型配置

本轮正式对比配置：

| 参数 | 值 |
|---|---:|
| 序列长度 | 25 |
| 历史长度 | 24 |
| 特征通道数 | 26 |
| Encoder层数 | 2 |
| Decoder层数 | 2 |
| Transformer隐藏维度 | 64 |
| 注意力头数 | 4 |
| 扩散步数 | 24 |
| 训练轮数 | 30 |
| Batch size | 128 |
| 学习率 | 5e-4 |
| 每个目标采样数 | 16 |
| 随机种子 | 42 |

该配置是 CPU 可运行的完整结构验证配置，不是论文原始 ETTh 配置的逐参数复制。

## 5. 训练过程

### 5.1 直接重建目标

上游 Diffusion-TS 不是只让网络预测噪声，而是让 Transformer 在每个扩散时间步直接重建当前时间序列：

```text
真实序列 x0
→ 加噪得到 xt
→ Transformer输出重建序列 x0_hat
→ 比较 x0_hat 与 x0
```

### 5.2 趋势和周期分解

Decoder 中包含两个分支：

```text
TrendBlock：多项式趋势
FourierLayer：频域周期结构
```

最终重建结果由趋势项、周期项和残差项组合得到。

### 5.3 Fourier损失

开启 Fourier 损失时：

```text
Loss = 时间域重建MSE + 0.05 × Measure通道Fourier损失
```

本工程只对 Measure 通道计算 Fourier 损失，未对位置编码和类别型派生通道计算频域约束。

## 6. 数据数量

本轮完整分支使用：

| 阶段 | 数量 |
|---|---:|
| 训练窗口 | 7,569 |
| 验证窗口 | 1,351 |
| 测试窗口 | 1,715 |
| 每个目标采样数 | 16 |

测试阶段生成：

```text
1,715 × 16 = 27,440个完整序列样本
```

最终从每个序列样本中读取目标位置的 Measure，并计算 P10/P50/P90。

## 7. 完整分支结果

### 7.1 关闭Fourier损失

```text
RMSE = 0.22639
MAE  = 0.12960
R²   = 0.36102
覆盖率 = 78.66%
```

### 7.2 开启Fourier损失

```text
RMSE = 0.22653
MAE  = 0.12922
R²   = 0.36022
覆盖率 = 78.83%
```

### 7.3 分湿度阶段结果

| 版本 | dry RMSE | transition RMSE | humid RMSE | very_humid RMSE |
|---|---:|---:|---:|---:|
| Fourier关闭 | 0.03573 | 0.06333 | 0.14901 | 0.28195 |
| Fourier开启 | 0.03545 | 0.06301 | 0.14844 | 0.28235 |

Fourier 损失对 dry、transition 和 humid 阶段有轻微改善，但 very_humid 阶段略有回退，整体 RMSE 基本没有改善。

## 8. 与已有模型比较

| 模型 | RMSE | MAE | R² | P10-P90覆盖率 |
|---|---:|---:|---:|---:|
| 完整 Transformer，Fourier关闭 | 0.22639 | 0.12960 | 0.36102 | 78.66% |
| 完整 Transformer，Fourier开启 | 0.22653 | 0.12922 | 0.36022 | 78.83% |
| 原始论文机制适配 Diffusion-TS | 0.21542 | 0.12196 | 0.42142 | 67.87% |
| 调参论文机制适配 Diffusion-TS | **0.20863** | **0.11698** | **0.45734** | 72.48% |
| 25% DDPM增强 FMLP | 0.21496 | 0.12098 | 0.42392 | **81.46%** |

## 9. 当前解释

完整 Transformer 分支当前没有超过轻量适配器，主要可能有以下原因：

1. 本轮完整分支只训练 30 epoch，而调参适配器训练 60 epoch；
2. 完整分支每个样本只采样16次，适配器采样32次；
3. 本工程是单步预测，完整序列生成结构的优势没有完全发挥；
4. 本工程采样间隔不规则，直接使用序列位置 Fourier 频率可能不完全符合物理时间；
5. 完整分支当前没有显式加入25维设备 one-hot 条件，而轻量适配器包含设备身份条件；
6. 目标位置的非 Measure 特征采用因果占位值，和论文完整序列生成场景不同。

因此，这次结果不能说明完整 Diffusion-TS 没有价值，只能说明：在当前轻量训练配置和单步任务改造下，尚未超过已经针对本工程优化过的残差适配器。

## 10. 文件位置

| 内容 | 路径 |
|---|---|
| 完整模型封装 | `code reproduction/full_diffusion_ts/model.py` |
| 完整训练入口 | `code reproduction/full_diffusion_ts/run_full_branch.py` |
| Fourier关闭结果 | `code reproduction/local_runs/full_diffusion_ts_strong_20260907/fourier_off/` |
| Fourier开启结果 | `code reproduction/local_runs/full_diffusion_ts_strong_20260907/fourier_on/` |
| 原始上游代码 | `code reproduction/upstream/Diffusion-TS/` |

每个结果目录包含：

- `model.pth`；
- `training_history.json`；
- `metrics.json`；
- `predictions.csv`。

## 11. 下一步建议

如果要继续深度优化完整论文模型，建议按以下顺序：

1. 将设备身份作为显式 embedding 加入完整 Transformer；
2. 将 `DeltaHours` 纳入时间位置编码，而不是只作为普通通道；
3. 比较真实时间间隔 Fourier 和序列索引 Fourier；
4. 将训练轮数提高到 60 或 100；
5. 将采样数提高到 32；
6. 使用 3 个随机种子重复；
7. 只在高湿专家上训练完整 Transformer，避免低湿样本稀释高湿目标；
8. 如果完整模型点预测仍不超过 0.209 左右，则保留它作为研究对照，不替换当前生产候选。
