# Measure 小数精度与 ×1000 处理分析

## 结论

可以把 `Measure` 表示成整数毫单位：

```text
Measure_milli = round(Measure, 3) × 1000
```

但这只是数值表示方式变化，不会增加传感器真实精度。由于当前数据第四位小数没有可靠物理意义，推荐：

1. 先将原始 `Measure` 量化到 0.001：`round(Measure, 3)`；
2. 可以在内部使用 `Measure_milli` 做整数/浮点计算；
3. 所有测量值派生特征必须同步使用毫单位；
4. 模型输出和报告再除以 1000 恢复原单位；
5. 温度、湿度、时间和比例特征不乘 1000。

## 派生特征的正确缩放

如果使用毫单位，必须采用：

```text
M4m = round(M4, 3) × 1000
M7m = round(M7, 3) × 1000
Mdiffm = M4m - M7m
Mratio = (M4m + 1) / (M7m + 1)
logratio = log(Mratio)
```

这里的 `+1` 对应原单位的 `+0.001`。不能把 `M4/M7` 乘 1000 后仍然使用原来的 `+1e-3`，否则比例特征定义会发生变化。

残差扩散和 Mahalanobis 的输入也要保持一致：`M4m`、`M7m`、`Mdiffm` 的残差均以毫单位计算；协方差和标准化使用同一套单位时，理论上距离排序基本不变。

## 已完成的对照

### 原始数据

使用完整浮点 `Measure`，结果见：

```text
experiment_augmentation/runs/experiment_only_20260909/
```

稳定正常 Mahalanobis：AUROC `0.9969`。

### 先保留 3 位小数

对输入先执行 `round(Measure, 3)`，但仍在原单位中构造特征，结果见：

```text
experiment_augmentation/runs/experiment_quantized3_benchmark_20260909/
```

稳定正常 Mahalanobis：AUROC `0.9969`；扩散增强后 AUROC `0.9975`。与原始结果基本一致，说明保留 3 位小数不会损失当前异常区分信息。

### 直接把原始输入乘 1000

该试验结果见：

```text
experiment_augmentation/runs/experiment_measure_x1000_benchmark_20260909/
```

结果明显变化，但该试验不能作为有效结论，因为旧代码仍使用原单位的 `+1e-3` 构造比例特征，缩放前后特征定义不一致。因此它证明的是“不能只改输入数值而不改派生特征公式”。

## 工程建议

当前工程不建议为了“增加计算精度”简单乘 1000。推荐将 `Measure` 处理为：

```text
原始值 → round(3) → 可选转换为毫单位 → 统一构造 Mdiff/Mratio/logratio → 标准化 → 模型
```

如果后续需要部署整数化设备或避免极小浮点差异，可以采用毫单位存储；如果只是训练模型，直接保留 3 位小数并做标准化即可，模型效果不会因为乘 1000 自动提升。
