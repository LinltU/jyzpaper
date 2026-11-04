# 湿度条件扩散数据增强实验报告

## 1. 目标

本实验验证扩散模型生成的高湿传感器窗口能否改善现有“四阶段共享 FMLP + 阶段残差专家 + 分位数输出”。

约束如下：

- 扩散模型只能访问下游真实训练切分。
- 验证集和测试集始终为真实数据，不加入合成样本。
- 生成原始动态通道后重新计算绝对湿度等工程特征。
- 不生成或扰动设备 one-hot、位置和时间周期等静态/确定性特征。

## 2. 可直接使用的公开模型判断

Diffusion-TS、CSDI、TimeGrad 和 SSSD 都有公开代码，但没有能直接处理本工程 46 个特征、25 台设备和四湿度阶段的通用预训练权重。直接套用公开 checkpoint 会产生字段、量纲和条件语义不一致。

因此本工程实现了一个 PyTorch 1.8 可运行的小型条件 DDPM：

```text
生成变量：
24 步 [Measure, Temp, Hum, Battery, DeltaHours]
+ 目标时刻 Temp、Hum
+ 下一步 Measure 残差

生成条件：
四湿度阶段 + 设备 ID
```

模型使用标准 DDPM 噪声预测目标，并参考 CSDI 的条件扩散思路。模型规模针对 1,144 个真实训练窗口进行了收缩，没有引入外部仓库或额外运行时依赖。

## 3. 数据规模

| 切分 | 窗口数 |
|---|---:|
| 真实训练 | 1,144 |
| 真实验证 | 219 |
| 真实测试 | 320 |
| 真实训练 `Hum >= 80` | 84 |
| 真实测试 `Hum >= 80` | 25 |

极高湿扩散条件实际只有 84 个训练窗口，这是本实验的主要限制。

## 4. 实现文件

- `model/humidity_conditional_ddpm.py`：阶段和设备条件 DDPM。
- `evaluate_diffusion_augmented_fmlp.py`：生成、质量诊断、FMLP 基线/增强对照和产物保存。
- `test_humidity_conditional_ddpm.py`：扩散前向、反向采样和梯度测试。
- `test_diffusion_augmentation.py`：生成向量、矩匹配和合成样本权重测试。

## 5. 正式实验结果

所有实验使用相同真实训练/验证/测试切分、相同 FMLP 配置和随机种子 42。

| 增强方式 | 合成高湿窗口 | 整体 RMSE | 整体 MAE | 整体 R2 | 极高湿 RMSE |
|---|---:|---:|---:|---:|---:|
| 真实数据基线 | 0 | **0.09216** | **0.04750** | **0.87592** | **0.28027** |
| 原始 DDPM 0.25x | 21 | 0.09678 | 0.05375 | 0.86319 | 0.28605 |
| 原始 DDPM 1.0x | 84 | 0.10031 | 0.05374 | 0.85300 | 0.30425 |
| DDPM 1.0x，合成损失权重 0.25 | 84 | 0.10920 | 0.06120 | 0.82581 | 0.32122 |
| 边缘矩校准 DDPM 0.5x | 42 | 0.10166 | 0.05526 | 0.84904 | 0.30655 |

当前所有扩散增强配置均未超过真实数据基线，因此不应把增强版替换为生产模型。

## 6. 生成质量诊断

原始 1.0x DDPM：

- 标准化维度中位标准差比为 0.55，说明生成分布明显偏窄。
- 合成目标残差标准差为 0.212，真实高湿训练数据为 0.403。
- 合成到真实的最近邻距离与真实到真实的最近邻距离比为 1.01。
- 没有合成样本落入真实最近邻距离 10% 以内，未观察到直接复制。

针对分布偏窄增加了只使用真实训练阶段统计量的边缘矩校准。校准后：

- 中位标准差比提升到 0.964。
- 合成目标残差标准差提升到 0.394，接近真实值 0.403。
- 合成目标均值和湿度均值也接近真实高湿训练集。
- 最近邻距离比为 1.28，仍没有直接复制迹象。

即使边缘分布得到校准，下游 RMSE 仍然变差。这说明问题不只是均值和方差，而是 84 个高湿窗口不足以稳定学习目标残差与多变量时序之间的联合分布。

## 7. 结论

本工程现在已经具备可复现的扩散生成和增强实验能力，但当前数据下的结论是否定的：

```text
扩散模型可以生成形式合理且没有明显复制的高湿窗口，
但这些窗口没有提高真实测试集上的 FMLP 精度。
```

现阶段继续使用真实数据训练的四阶段 FMLP。扩散生成数据可保留用于压力测试和可视化，不应默认用于训练生产模型。

建议在每台设备采集更长的连续高湿序列，将 `Hum >= 80` 的真实窗口从 84 增加到至少 500，再重新验证扩散增强。新增数据应覆盖湿度快速上升、持续高湿和回落三个过程，而不只是增加相似静态高湿点。

## 8. 复现命令

标准扩散增强实验：

```powershell
C:\ProgramData\miniconda3\envs\JYZmentor\python.exe `
  evaluate_diffusion_augmented_fmlp.py `
  --training training.csv `
  --test test.csv `
  --output-dir diffusion_augmented_fmlp_evaluation `
  --synthetic-ratio 1.0 `
  --synthetic-loss-weight 1.0 `
  --moment-match-strength 0.0 `
  --diffusion-epochs 120 `
  --diffusion-steps 50 `
  --seed 42
```

运行测试：

```powershell
C:\ProgramData\miniconda3\envs\JYZmentor\python.exe -m unittest -v `
  test_humidity_conditional_ddpm.py `
  test_diffusion_augmentation.py `
  tests/test_humidity_stage_fmlp.py
```

## 9. 产物

每个实验目录包含：

- `humidity_conditional_ddpm.pth`
- `baseline_humidity_stage_fmlp.pth`
- `augmented_humidity_stage_fmlp.pth`
- `synthetic_training_windows.csv`
- `synthetic_training_windows.npz`
- `metrics.json`
- `augmented_predictions.jsonl`
- 三个训练历史 JSON 文件

原始 1.0x 实验已归档至 `archive/comparison_experiments_202607/diffusion_augmented_fmlp_evaluation`，边缘矩校准实验已归档至 `archive/comparison_experiments_202607/diffusion_augmented_fmlp_calibrated`。
