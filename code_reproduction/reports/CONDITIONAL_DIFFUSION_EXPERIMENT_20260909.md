# 本工程条件残差扩散方案复测报告

## 1. 目标

本方案只针对：

```text
C:\Project\JYZmentor\dataset\experiment_sheets_csv
```

该目录是 `experiment.xlsx` 的工作表镜像。目标为二分类：

```text
error=0  -> 正常绝缘子串
error!=0 -> 存在零值绝缘子
```

不判断具体零值片号。

## 2. 论文思想与本工程调整

| 参考方案 | 原始思想 | 本工程调整 |
|---|---|---|
| TimeGrad | 使用历史状态作为条件，扩散生成未来/残差 | 用共享温湿度和时间状态作为条件，生成双设备正常残差 |
| NsDiff | 用局部归一化处理非平稳时间序列 | 每个实验日期独立标准化条件和残差，避免日期批次漂移 |
| DiffAD | 只用正常数据建立重构/异常评分 | 只用稳定 `error=0` 窗口训练，`error!=0` 仅用于评估 |

这不是上述论文官方网络的逐层复现，而是把三种机制调整到当前双设备实验数据协议后的本地模型。

## 3. 条件和生成目标

条件向量为：

```text
Temp_shared
Hum_shared
HumidityTrend
TempTrend
elapsed_min
```

其中：

```text
Temp_shared=(Temp4+Temp7)/2
Hum_shared=(Hum4+Hum7)/2
HumidityTrend=(Hum_last-Hum_first)/(L-1)
TempTrend=(Temp_last-Temp_first)/(L-1)
```

扩散生成目标不是原始 `Measure`，而是残差向量：

```text
[M4残差, M7残差, Mdiff残差, logratio残差]
```

残差先由温湿度和经过时间回归得到：

```text
Measure = f(Temp_shared, Hum_shared, elapsed_min) + residual
```

这样扩散模型学习的是“在相同环境条件下，正常双设备关系允许有多大波动”，而不是直接学习不同日期的绝对 Measure 水平。

## 4. 训练和评估流程

```text
按日期读取两台设备
        ↓
最近时间戳配对第4片/第7片设备
        ↓
切分长度6的窗口，不跨error阶段
        ↓
筛选稳定的error=0窗口
        ↓
每个日期单独训练条件残差DDPM
        ↓
用条件扩散重构误差或Mahalanobis残差距离评分
        ↓
用原始error!=0标签计算检测指标
```

运行代码：

```text
C:\Project\JYZmentor\code reproduction\experiment_augmentation\conditional_residual_diffusion_experiment.py
```

运行结果：

```text
C:\Project\JYZmentor\code reproduction\experiment_augmentation\runs\conditional_residual_diffusion_20260909_v2\results.json
```

训练参数：

| 参数 | 值 |
|---|---:|
| 窗口长度 | 6 |
| 扩散步数 | 24 |
| 隐藏层宽度 | 128 |
| 训练轮数 | 120 |
| 正常残差增强比例 | 20% |
| 日期校准 | 每个日期独立 |
| 随机种子 | 42、123、2024 |

## 5. 结果

### 5.1 条件扩散重构分数

该分数直接比较真实残差和条件 DDPM 去噪恢复残差之间的差异。

| seed | AUROC | AUPRC | 平衡准确率 | 正常误报率 |
|---:|---:|---:|---:|---:|
| 42 | 0.9973 | 0.9690 | 0.9762 | 0 |
| 123 | 0.9975 | 0.9699 | 0.9762 | 0 |
| 2024 | 0.9973 | 0.9690 | 0.9762 | 0 |
| 平均 | 0.9974 | 0.9693 | 0.9762 | 0 |

### 5.2 与其他 experiment 方案比较

| 方案 | AUROC | AUPRC | 平衡准确率 | 正常误报率 |
|---|---:|---:|---:|---:|
| 条件残差 DDPM 重构 | **0.9974** | **0.9693** | 0.9762 | 0 |
| 稳定正常 Mahalanobis | 0.9969 | 0.9675 | 0.9762 | 0 |
| DiffAD 风格重构 | 0.9454* | 0.6756* | 0.8662* | 0.0078 |
| IMDiffusion 风格重构 | 0.9212* | 0.6554* | 0.8655* | 0.0093 |
| D3R 风格去趋势 | 0.8413 | 0.4085 | 0.7632 | 0.1117 |

`*` DiffAD/IMDiffusion 为三个种子的平均值，条件残差 DDPM 为三次复测平均值。

### 5.3 数据增强效果

| 方法 | 不加入合成残差 | 加入20%正常残差 | 变化 |
|---|---:|---:|---:|
| 条件 DDPM Mahalanobis | AUROC 0.9969 | AUROC 0.9969 | 无变化 |
| 条件 DDPM 重构分数 | AUROC 0.9974 | 未用于重新训练分类器 | — |

条件扩散生成的少量正常残差没有提升 Mahalanobis，因为原始稳定正常残差分布已经足以建立可靠协方差。扩散模型的价值主要体现在条件重构评分，而不是扩大样本数量。

## 6. 结论

当前针对本工程最有效的条件扩散形式是：

```text
共享温湿度/趋势/时间作为条件
        +
只生成稳定正常双设备残差
        +
使用条件去噪重构误差作为异常分数
```

在 `experiment.xlsx` 的受控实验标签上，三随机种子平均 AUROC 为 `0.9974`，略高于稳定 Mahalanobis 的 `0.9969`，且正常误报率为 `0`。

但提升幅度很小，不能说明复杂扩散网络已经稳定优于 Mahalanobis。当前更合理的工程组合是：

```text
Mahalanobis 作为稳定主基线
条件残差 DDPM 作为辅助分数
两者分数融合后再进行报警
```
