# P10/P50/P90 环境区间 + 双设备 Mahalanobis 组合复测

## 1. 组合逻辑

本次只使用 `experiment.xlsx` 数据。两个分数分别表示：

```text
环境区间分数：当前 M4/M7 残差超出条件 P10-P90 区间的程度
联合关系分数：M4、M7、M4-M7、logratio 残差的 Mahalanobis 距离
```

先对两个分数分别进行稳健尺度归一化，再组合：

```text
S = w × S_environment + (1-w) × S_mahalanobis
```

测试了：

```text
w=0.25、0.50、0.75
```

其中环境区间使用共享温度、共享湿度、温湿度趋势和经过时间条件，并按每个日期正常稳定湿度的三分位点划分低/中/高湿阶段。

代码（已增加详细注释和逐窗口分数输出）：

```text
C:\Project\JYZmentor\code reproduction\experiment_augmentation\combined_environment_mahalanobis_benchmark_v2.py
```

结果：

```text
C:\Project\JYZmentor\code reproduction\experiment_augmentation\runs\combined_environment_mahalanobis_v2_20260909\results.json
```

## 2. 结果

| 方案 | AUROC | AUPRC | 平衡准确率 | 精确率 | 召回率 | 正常误报率 |
|---|---:|---:|---:|---:|---:|---:|
| 环境区间单独评分 | 0.9393 | 0.5907 | 0.8607 | 0.7719 | 0.7403 | 1.89% |
| Mahalanobis 单独评分 | 0.9969 | 0.9675 | 0.9762 | 1.0000 | 0.9524 | 0 |
| 组合 w=0.25 | **0.9969** | **0.9675** | **0.9762** | **1.0000** | **0.9524** | **0** |
| 组合 w=0.50 | **0.9969** | **0.9675** | **0.9762** | **1.0000** | **0.9524** | **0** |
| 组合 w=0.75 | 0.9967 | 0.9589 | 0.9746 | 0.9246 | 0.9524 | 0.32% |

## 3. 结果解释

当前实验中，Mahalanobis 已经能够很好地区分零值异常，因此加入环境区间分数后没有明显提升。`w=0.25` 和 `w=0.50` 的结果与 Mahalanobis 完全一致，说明联合双设备关系分数在当前数据中占主导；当环境分数权重提高到 `0.75` 时，误报略有增加。

这并不表示温湿度没有作用。环境区间单独评分已经达到：

```text
精确率 77.19%
召回率 74.03%
正常误报率 1.89%
```

它更适合作为：

- 对 Mahalanobis 报警的环境解释；
- 高湿/低湿阶段的辅助置信度；
- 跨日期时判断“是否只是环境变化”的参考分数。

## 4. 当前推荐权重

对于当前 `experiment.xlsx`：

```text
S_final = 0.25 × S_environment + 0.75 × S_mahalanobis
```

但从本次指标看，`w=0.25` 与 Mahalanobis 单独评分相同，因此工程上可以先使用：

```text
Mahalanobis 主分数
环境 P10/P90 分数作为解释和辅助报警条件
```

不建议当前直接使用 `w=0.75`，因为它没有提高召回率，反而引入少量正常误报。

## 5. 为什么没有明显提升

1. 目前三个日期的湿度区间差异很大，环境与日期批次混杂；
2. 稳定正常窗口数量较少，分湿度阶段后每阶段样本更少；
3. P10/P90 是单设备边界，Mahalanobis 使用了双设备、差值和比例的联合协方差；
4. 当前异常样本在双设备关系上已经很明显，环境分数提供的是重复信息。

后续如果补充相同温湿度下的正常/异常实验，环境分数才更可能帮助识别“温湿度造成的变化”和“零值绝缘子造成的额外偏离”。
逐窗口分数文件：

```text
C:\Project\JYZmentor\code reproduction\experiment_augmentation\runs\combined_environment_mahalanobis_v2_20260909\window_scores.csv
```

该 CSV 包含 `environment_score`、`mahalanobis_score`、两个归一化分数以及 `combined_w25/50/75`，可以直接检查每个窗口到底是哪一部分触发了报警。
