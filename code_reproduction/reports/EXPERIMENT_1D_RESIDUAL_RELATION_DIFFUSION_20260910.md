# experiment.xlsx：一维条件残差扩散与一维双设备关系扩散对比

## 1. 数据和复现范围

本次只测试 `C:\Project\JYZmentor\dataset\experiment.xlsx` 的真实实验窗口；`error=0` 为正常，`error!=0` 为替换零值绝缘子后的异常。第 4、7 片设备按最近时间配对，温湿度使用两台设备平均值作为共享环境条件。

参考来源包括 Zotero diffusion 清单中的 TimeGrad、NsDiff、DiffAD，以及 GitHub 开源实现：

- [TimeGrad / pytorch-ts](https://github.com/zalandoresearch/pytorch-ts)
- [NsDiff](https://github.com/wwy155/NsDiff)
- [DiffAD](https://github.com/ChunjingXiao/DiffAD)
- [D3R](https://github.com/ForestsKing/D3R)
- [IMDiffusion](https://github.com/17000cyh/IMDiffusion)

由于这些仓库原始输入格式、窗口长度和数据规模与本实验不同，本地代码是保持核心机制的工程适配，不是原仓库数据集上的逐文件复现。

## 2. 两个目标方案

### 2.1 一维条件残差扩散

将每个短时间窗口表示为一维多变量序列，扩散目标为环境校正后的残差：

```text
r4 = M4 - 正常环境基线
r7 = M7 - 正常环境基线
rdiff = (M4-M7) - 正常关系基线
rratio = log(M4/M7) - 正常比例基线
```

条件为共享温度、共享湿度、温湿度趋势和经过时间。只用每个日期后期稳定正常窗口校准，并通过扩散重构误差评分。

代码：`experiment_augmentation\conditional_residual_diffusion_experiment.py`

### 2.2 一维双设备关系扩散

单独将关系向量组成序列：

```text
[r4, r7, rdiff, rratio]
```

模型学习两个设备共同变化以及差值/比例关系，而不是只重构单台 Measure。实现文件：

`experiment_augmentation\dual_device_relation_diffusion.py`

## 3. 重新测试结果

### 3.1 一维条件残差扩散

结果文件：

`experiment_augmentation\runs\conditional_residual_diffusion_retest_20260909\results.json`

| 方法 | AUROC | AUPRC | 平衡准确率 | 正常误报率 |
|---|---:|---:|---:|---:|
| 条件残差 DDPM 重构 | 0.9973 | 0.9690 | 0.9762 | 0 |
| 条件残差 DDPM + Mahalanobis | 0.9969 | 0.9675 | 0.9762 | 0 |
| 稳定正常 Mahalanobis | 0.9969 | 0.9675 | 0.9762 | 0 |

### 3.2 一维双设备关系扩散

结果文件：

`experiment_augmentation\runs\dual_device_relation_diffusion_20260910\results.json`

本次采用 151 个稳定正常训练窗口，594 个真实实验窗口测试，120 个 epoch：

| 方法 | AUROC | AUPRC | 平衡准确率 | 正常误报率 |
|---|---:|---:|---:|---:|
| 双设备关系 DDPM 重构 | 0.7439 | 0.4456 | 0.6010 | 0.55% |

关系扩散当前明显低于条件残差 DDPM 和 Mahalanobis。原因主要是实验样本很少，关系残差中仍包含日期批次漂移；当前实现尚未加入分日期关系基线、关系协方差白化和多随机种子集成。因此它是有效的独立对照分支，但不是当前主方案。

## 4. 结论

当前排序为：

1. 条件残差 DDPM 重构：当前扩散方案最好；
2. 稳定正常 Mahalanobis：结果接近且更简单、更稳定；
3. 双设备关系 DDPM：已实现，但当前版本需要关系基线和物理一致性约束后再判断是否有价值。

不能仅凭“显式使用双设备关系”就认定性能更好。下一步若用于论文，应重点改进关系扩散：按日期拟合 `r4/r7/rdiff/rratio` 基线，加入协方差白化、M4/M7 比例一致性损失，以及 3 个以上随机种子和跨日期验证。

