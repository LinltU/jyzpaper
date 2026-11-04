# experiment.xlsx 单论文扩散方案对比说明

## 重要更正

此前将 `TimeGrad + NsDiff + DiffAD` 合并后得到的“条件残差 DDPM”称为一个最好算法是不严谨的。它是本地组合适配，不能作为任何单篇论文的结果。本报告将单论文方案与组合方案分开。

## 单独方案结果

| 单独方案 | 对应论文/代码 | 本地任务 | 指标 | 是否可作为单论文结论 |
|---|---|---|---|---|
| 原始条件 DDPM | TimeGrad 思路；[pytorch-ts](https://github.com/zalandoresearch/pytorch-ts) | 生成窗口并增强分类器 | 真实测试 BA `0.7844`，Macro-F1 `0.5574` | 可以，属于 TimeGrad/DDPM 适配，不含 NsDiff/DiffAD |
| 局部归一化 DDPM | NsDiff；[NsDiff](https://github.com/wwy155/NsDiff) | 局部归一化后生成窗口并增强分类器 | 真实测试 BA `0.7569`，Macro-F1 `0.5201` | 可以，属于 NsDiff 风格适配 |
| DiffAD | [DiffAD](https://github.com/ChunjingXiao/DiffAD) | 正常训练、掩码插补重构异常检测 | 旧结果来自其他数据协议，不能并入本次 experiment-only 表 | 需要按本报告协议重新运行 |
| D3R | [D3R](https://github.com/ForestsKing/D3R) | 动态分解后扩散重构 | 旧结果来自其他数据协议，不能并入本次 experiment-only 表 | 需要按本报告协议重新运行 |
| IMDiffusion | [IMDiffusion](https://github.com/17000cyh/IMDiffusion) | 掩码插补不确定性异常检测 | 旧结果来自其他数据协议，不能并入本次 experiment-only 表 | 需要按本报告协议重新运行 |
| 双设备关系 DDPM | 无单一官方论文原样对应；本地新实现 | 生成 `[M4残差,M7残差,Mdiff残差,logratio残差]` | AUROC `0.7439`，AUPRC `0.4456`，BA `0.6010` | 可作为本地关系扩散探索，不应标成某篇论文结果 |

## 已确认的单独运行结果

### TimeGrad / 原始条件 DDPM 适配

训练 470 个窗口，测试 124 个真实窗口，生成 395 个窗口。只用真实测试集评估：

```text
Baseline BA       0.9495
Augmented BA      0.7844
Baseline Macro-F1 0.8393
Augmented Macro-F1 0.5574
```

结论：单独原始条件 DDPM 生成的数据没有提升分类效果。

### NsDiff 单独适配

训练和测试划分相同，生成 395 个窗口：

```text
Baseline BA       0.9495
Augmented BA      0.7569
Baseline Macro-F1 0.8393
Augmented Macro-F1 0.5201
```

结论：在当前小样本和异常类别极不平衡条件下，局部归一化生成数据反而改变了类别边界。

## 当前不能继续沿用的结果

旧目录中的 D3R、DiffAD、IMDiffusion、TIMEADDM、ProDiffAD 指标使用的是旧的正常训练/人工 6σ 伪异常协议，不能与 `experiment.xlsx` 的真实 `error` 标签结果混合。因此此前类似“DiffAD AUROC 0.9979”的数值不能回答本次 experiment-only 单论文比较。

## 结论

当前严格单论文口径下，已得到的结果不能支持“某个论文算法绝对最好”。目前只能确认：

1. 原始 TimeGrad/DDPM 适配优于 NsDiff 适配，但两者作为数据增强都没有超过真实数据基线；
2. 组合型条件残差 DDPM 的 `AUROC 0.9973` 不能归属于单一论文；
3. DiffAD、D3R、IMDiffusion 必须用同一 `experiment.xlsx` 真实标签协议重新运行后，才能进行公平排序。

