# experiment.xlsx 扩散方案重新测试报告

## 1. 测试范围

本报告只使用：

```text
C:\Project\JYZmentor\dataset\experiment_sheets_csv
```

该目录是 `experiment.xlsx` 的工作表镜像。本次只使用该实验数据及其原始 `error` 标签。

标签定义：

- `error=0`：正常绝缘子串；
- `error!=0`：实验中实际替换零值绝缘子后的异常工况；
- 本报告把 `error!=0` 作为真实受控实验正类，只判断“一串绝缘子上是否存在零值”，不判断具体片号。

## 2. 数据处理与特征

两台设备分别安装在第4片和第7片绝缘子上，按最近时间戳配对，并且不跨越 `error` 阶段切分窗口。每个窗口长度为 6 个采样点。

正式稳定段特征为：

```text
M4, M7, Temp_shared, Hum_shared,
Mdiff=M4-M7,
Mratio=(M4+0.001)/(M7+0.001),
logratio=log((M4+ε)/(M7+ε))
```

其中：

```text
Temp_shared=(Temp4+Temp7)/2
Hum_shared=(Hum4+Hum7)/2
```

训练规则：每个实验日期单独建立正常基线，只使用该日期最早的后期稳定 `error=0` 窗口；测试使用该日期全部窗口，`error` 只用于离线评估，不输入检测器。

稳定窗口统计：

| 日期 | 稳定正常拟合窗口 | 全部测试窗口 | `error!=0` 测试窗口 |
|---|---:|---:|---:|
| 250605 | 64 | 236 | 7 |
| 250626 | 35 | 159 | 35 |
| 250718 | 52 | 199 | 11 |
| 合计 | 151 | 594 | 53 |

## 3. 重新测试结果：按日期稳定校准

运行目录：

```text
C:\Project\JYZmentor\code reproduction\experiment_augmentation\runs\experiment_only_20260909
```

### 3.1 单次复测（seed=42）

| 方案 | 核心工作 | AUROC | AUPRC | 平衡准确率 | 精确率 | 召回率 | 正常误报率 |
|---|---|---:|---:|---:|---:|---:|---:|
| 稳定正常 Mahalanobis | 双设备联合残差的稳健距离 | 0.9969 | 0.9675 | 0.9762 | 1.0000 | 0.9524 | 0 |
| D3R 风格 | 去趋势后计算动态残差 | 0.8413 | 0.4085 | 0.7632 | 0.7162 | 0.6381 | 11.17% |
| DiffAD 风格 | 正常窗口掩码/重构误差评分 | 0.9491 | 0.6763 | 0.8665 | 0.8485 | 0.7403 | 0.73% |
| IMDiffusion 风格 | 多次掩码重构不确定性评分 | 0.9198 | 0.6610 | 0.8658 | 0.8333 | 0.7403 | 0.87% |
| 稳定正常 + 20% 残差 DDPM | 只生成正常残差并加入正常分布 | 0.9969 | 0.9675 | 0.9762 | 1.0000 | 0.9524 | 0 |

DiffAD/IMDiffusion 的神经重构部分存在随机初始化影响，因此不能只看一次 seed。Mahalanobis 和 DDPM 增强结果在本实验中稳定一致。

### 3.2 三个随机种子复核

复测目录：

```text
runs/experiment_only_20260909
runs/experiment_only_seed123
runs/experiment_only_seed2024
```

| 方案 | AUROC seed=42 | seed=123 | seed=2024 | 观察 |
|---|---:|---:|---:|---|
| 稳定正常 Mahalanobis | 0.9969 | 0.9969 | 0.9969 | 无随机性，最稳定 |
| D3R 风格 | 0.8413 | 0.8413 | 0.8413 | 无随机性 |
| DiffAD 风格 | 0.9491 | 0.9498 | 0.9372 | 小样本神经重构有波动 |
| IMDiffusion 风格 | 0.9198 | 0.9236 | 0.9202 | 相对稳定但低于 Mahalanobis |
| 稳定正常 + 残差 DDPM | 0.9969 | 0.9969 | 0.9969 | 当前增强没有改变检测排序 |

## 4. 数据增强方案复测

运行目录：

```text
C:\Project\JYZmentor\code reproduction\experiment_augmentation\runs\experiment_only_augmentation_20260909
```

使用共享温湿度、6 步窗口、二分类标签、80 个训练 epoch。训练窗口 470，测试窗口 124，每种方法生成 395 个合成窗口。

| 增强方案 | 真实数据训练平衡准确率 | 加入合成数据后 | 真实数据 Macro-F1 | 加入合成数据后 |
|---|---:|---:|---:|---:|
| 直接 DDPM | 0.9495 | 0.7844 | 0.8393 | 0.5574 |
| NsDiff 风格 DDPM | 0.9495 | 0.7569 | 0.8393 | 0.5201 |

结论：当前按异常标签条件生成并大量加入合成窗口会降低分类效果。原因是每个异常阶段样本很少，生成窗口出现分布收缩，不能直接当作真实实验样本扩充。稳定正常残差只生成 20% 时，对 Mahalanobis 指标没有提升，也没有破坏原有结果。

## 5. 跨日期测试对照

### 5.0 环境因素与日期漂移的关系

稳定正常窗口的环境统计如下：

| 日期 | Hum_shared 范围 | Temp_shared 范围 | M4稳定均值 | M7稳定均值 |
|---|---:|---:|---:|---:|
| 250605 | 45.20–53.51 | 29.61–31.75 | 0.032 | 0.061 |
| 250626 | 66.50–69.91 | 30.91–32.47 | 0.822 | 0.703 |
| 250718 | 45.28–54.65 | 34.01–34.94 | 0.038 | 0.075 |

可以看到，250626 的湿度明显高于另外两次实验，同时 M4/M7 的稳定水平也明显不同。稳定窗口内 `Hum_shared` 与 Measure 的相关性很高：

| 日期 | corr(Hum_shared, M4) | corr(Hum_shared, M7) |
|---|---:|---:|
| 250605 | 0.987 | 0.982 |
| 250626 | 0.990 | 0.992 |
| 250718 | 0.947 | 0.962 |

这说明“日期批次漂移”很可能包含明显的温湿度环境效应，不能简单解释为设备批次差异。但三个日期的湿度区间几乎没有重叠，日期与环境变量发生了混杂：模型没有足够数据回答“同一湿度下不同日期是否仍有偏置”。

当前条件残差方法已加入 `Temp_shared`、`Hum_shared`、温湿度趋势和经过时间，但跨日期的简单条件残差基线 AUROC 仍只有约 `0.306`。这说明仅使用线性温湿度校正还不够，可能存在湿度阈值、非线性、滞后效应或上电状态影响。

因此，正式改进方向应是：

```text
Measure = f(Temp_shared, Hum_shared, TempTrend, HumidityTrend, elapsed, humidity regime) + residual
```

并增加湿度阶段/非线性项，例如低湿、中湿、高湿分段，或使用样条/轻量树模型拟合环境基线。只有在相近温湿度条件下增加多次实验，才能把环境效应与日期效应真正分开。

运行目录：

```text
C:\Project\JYZmentor\code reproduction\experiment_augmentation\runs\experiment_only_crossdate_20260909
```

该实验将某一日期留出，只用其他日期的正常窗口建模，用于检验实验批次漂移影响。

| 方案 | AUROC | AUPRC | 平衡准确率 | 精确率 | 召回率 | 正常误报率 |
|---|---:|---:|---:|---:|---:|---:|
| Mahalanobis 正常分布 | 0.7640 | 0.7085 | 0.8095 | 0.7400 | 0.9524 | 33.33% |
| 条件正常残差 | 0.3060 | 0.1439 | 0.6328 | 0.2649 | 0.7524 | 48.68% |
| D3R 去趋势 | 0.7281 | 0.6803 | 0.8095 | 0.7400 | 0.9524 | 33.33% |
| DiffAD 重构 | 0.1903 | 0.1518 | 0.5429 | 0.3616 | 0.7524 | 66.67% |
| IMDiffusion 掩码重构 | 0.1993 | 0.1578 | 0.5414 | 0.1460 | 0.8381 | 75.54% |
| NsDiff 正常增强 | 0.1969 | 0.1488 | 0.5429 | 0.3616 | 0.7524 | 66.67% |

跨日期结果明显变差，说明 250605、250626、250718 的稳定 Measure 基线存在批次漂移。当前正式方案必须按实验日期校准，不能直接把三个日期合并为一个全局阈值。

## 6. 论文方案与代码状态

| 方案 | 公开代码 | 针对 experiment 的测试状态 | 适配说明 |
|---|---|---|---|
| DiffAD | `ChunjingXiao/DiffAD` | 已完成本地重构适配并得到指标 | 用稳定正常双设备窗口训练，重构误差判断 `error!=0` |
| D3R | `ForestsKing/D3R` | 已完成本地去趋势适配并得到指标 | 适合处理上电衰减/漂移，但稳定段效果低于 Mahalanobis |
| IMDiffusion | `17000cyh/IMDiffusion` | 已完成本地多次掩码重构适配并得到指标 | 用重构不确定性辅助判断异常 |
| NsDiff | `wwy155/NsDiff` | 已完成正常残差增强和数据增强复测 | 局部归一化没有解决跨日期批次漂移 |
| C3DAD | `Tian-siyuan/C3DAD` | Lite pipeline 已用 experiment 双设备数据运行 | 输入形状 `[201,6]`，输出 condition、lag context、side info；完整扩散骨干未公开 |
| TSADC | `Armanfard-Lab/TSADC` | 源码编译通过，未完成 experiment 训练 | 需要 `torch_geometric` 和专用数据加载器，未伪造指标 |
| ICDiffAD | `ZhangFan21/ICDiffAD` | 源码编译通过，未完成 experiment 训练 | 需要 `rich/hydra` 和专用公开数据配置，未伪造指标 |
| CT-DDPM | `4Aver/CT-DDPM` | 未完成 experiment 训练 | 当前公开仓库仅 README，无可执行 Python 源码 |
| FACD | `ShenLi0127/FACD` | 未完成 experiment 训练 | 本轮源码包下载不完整，未引用官方结果 |

所有上游仓库固定版本和审计结果见：

```text
C:\Project\JYZmentor\code reproduction\sources_manifest.json
C:\Project\JYZmentor\code reproduction\local_runs\final\official_smoke.json
```

## 7. 最终结论

1. 本次所有正式数值测试均针对 `experiment.xlsx`，使用真实的 `error` 实验标签。
2. 当前最优方案是“按实验日期后期稳定正常数据校准 + 双设备联合 Mahalanobis”。
3. DiffAD 风格重构是最有价值的扩散辅助方案，但当前低于 Mahalanobis。
4. D3R 能处理动态趋势，但在只看后期稳定段时优势不明显。
5. 大量 DDPM/NsDiff 合成数据会降低分类效果；少量正常残差增强目前没有带来提升。
6. 跨日期直接合并会因批次漂移导致误报明显上升，因此日期/实验批次必须作为校准边界。
7. C3DAD 已完成输入路径验证，但其完整论文扩散模型尚未公开，不能列入完整指标排名。

## 8. 新增条件残差扩散分支

本轮新增本工程定制的条件残差 DDPM，参考 TimeGrad 的条件扩散、NsDiff 的非平稳条件处理以及 DiffAD 的正常数据边界。条件为 `Temp_shared`、`Hum_shared`、湿度趋势、温度趋势和经过时间；生成目标为 `[M4残差,M7残差,Mdiff残差,logratio残差]`，不生成原始 Measure，也不把 `error` 作为条件。

三次随机种子复测平均结果：

| 方案 | AUROC | AUPRC | 平衡准确率 | 正常误报率 |
|---|---:|---:|---:|---:|
| 条件残差 DDPM 重构分数 | **0.9974** | **0.9693** | 0.9762 | 0 |
| 稳定正常 Mahalanobis | 0.9969 | 0.9675 | 0.9762 | 0 |

代码和结果：

```text
experiment_augmentation/conditional_residual_diffusion_experiment.py
experiment_augmentation/runs/conditional_residual_diffusion_20260909_v2/
```

完整说明见 `reports/CONDITIONAL_DIFFUSION_EXPERIMENT_20260909.md`。

当前建议的实验部署顺序：

```text
稳定正常 Mahalanobis 主检测器
        +
DiffAD 风格重构辅助分数
        +
按实验日期独立校准阈值
```
