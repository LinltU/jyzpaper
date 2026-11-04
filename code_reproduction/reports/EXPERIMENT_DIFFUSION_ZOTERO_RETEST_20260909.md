# experiment.xlsx 扩散论文方案重新测试报告

## 1. 测试范围

本次只使用 `C:\Project\JYZmentor\dataset\experiment.xlsx`（运行时使用其 CSV 镜像 `dataset\experiment_sheets_csv`），不使用现场 `training.csv/test.csv`，也不使用人工 6σ 伪异常。

标签定义：`error=0` 为正常绝缘子串，`error!=0` 为实验中实际替换零值绝缘子后的异常窗口。第 4、7 片设备按最近时间配对，并使用统一环境变量 `Temp_shared`、`Hum_shared`。

## 2. Zotero diffusion 文献的本地适配

| 论文/代码方案 | 原始任务 | 本工程适配方式 | 本次是否作为主测试 |
|---|---|---|---|
| TimeGrad / 条件 DDPM | 概率时间序列预测 | 条件生成双设备窗口/残差 | 是 |
| NsDiff | 非平稳时间序列扩散 | 对窗口做局部归一化后条件生成 | 是 |
| DiffAD | 扩散重构异常检测 | 仅用稳定正常窗口训练，按重构误差报警 | 已有本地适配结果 |
| D3R | 动态分解+扩散重构 | 对 Measure 去趋势后重构评分 | 已有本地适配结果 |
| IMDiffusion | 掩码插补/不确定性异常检测 | 人工掩码双设备窗口，再用重构不确定性评分 | 已有本地适配结果 |
| CSDI、MTSCI | 时间序列插补 | 适用于缺失值/缺失时间点，不等同于当前零值异常识别 | 不作为异常主模型 |
| Diffusion-TS | 可解释时间序列预测/生成 | 可作预测对照，但目标不是本实验异常分类 | 不纳入本次异常增强主表 |

原论文的数据通常是长时间、多变量、连续采样数据；本实验只有 3 个日期、双设备、短实验阶段和真实 `error` 标签。因此本地实现采用短窗口、双设备联合特征和按日期切分，不能宣称是原论文数据集上的完全复现。

## 3. 本次重新运行的条件扩散

运行代码：

`C:\Project\JYZmentor\code reproduction\experiment_augmentation\run_experiment_augmentation.py`

运行结果：

`C:\Project\JYZmentor\code reproduction\experiment_augmentation\runs\experiment_only_diffusion_retest_20260909_v2\results.json`

协议：6 步窗口；训练 470 个真实窗口，测试 124 个真实窗口；按每个日期和 error 阶段前 80% 训练、后 20% 测试；80 个 epoch；共享温湿度特征为 `M4、M7、Temp_shared、Hum_shared、Mdiff、Mratio、logratio、elapsed_min`。

| 方案 | 合成窗口 | 真实测试平衡准确率 | 真实测试 Macro-F1 |
|---|---:|---:|---:|
| 仅真实数据 RF 基线 | 0 | 0.9495 | 0.8393 |
| 原始条件 DDPM 增强 | 395 | 0.7844 | 0.5574 |
| NsDiff 风格局部归一化 DDPM 增强 | 395 | 0.7569 | 0.5201 |

这里的合成窗口仅用于训练分类器，最终评估完全使用未参与生成训练的真实实验窗口。两种大比例类别增强均降低固定分类效果，说明当前小样本下生成异常边界还不够可信，不能用合成异常替代真实异常。

## 4. 正常稳定残差条件扩散

代码：

`C:\Project\JYZmentor\code reproduction\experiment_augmentation\conditional_residual_diffusion_experiment.py`

目标不是直接生成原始 Measure，而是生成：`M4 残差、M7 残差、M4-M7 残差、logratio 残差`。条件为共享温度、共享湿度、温湿度趋势和经过时间；只用每个日期后期稳定的正常窗口校准。

| 方案 | AUROC | AUPRC | 平衡准确率 | 正常误报率 |
|---|---:|---:|---:|---:|
| 条件残差 DDPM 重构评分 | 0.9973 | 0.9690 | 0.9762 | 0 |
| 条件残差 DDPM + Mahalanobis | 0.9969 | 0.9675 | 0.9762 | 0 |
| 稳定正常 Mahalanobis 基线 | 0.9969 | 0.9675 | 0.9762 | 0 |

仅生成约 20% 的正常残差并加入 Mahalanobis 参考分布没有增益；扩散重构评分略高，但差异很小，仍需更多日期和随机种子验证。

## 5. 结论

当前针对 `experiment.xlsx` 最可靠的顺序是：

1. 按实验日期用后期稳定正常数据校准双设备 Mahalanobis；
2. 使用条件残差 DDPM 作为独立重构评分或辅助增强；
3. 不建议大规模生成异常窗口直接训练分类器；
4. CSDI/MTSCI 只适合插补对照，不应解释为零值绝缘子异常检测；
5. 目前结果不能证明扩散增强已提升主模型，真实稳定数据基线仍是首选。

