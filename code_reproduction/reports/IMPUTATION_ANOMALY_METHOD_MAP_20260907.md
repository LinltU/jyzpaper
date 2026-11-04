# 插补与异常检测方案对应表

本文档回答两个问题：当前运行的模型是什么，以及对应的论文/公开代码在哪里。

## 插补

当前 6 个方案全部基于本地 `ImputationAdapter` 运行。公共实现位于：

`C:\Project\JYZmentor\code reproduction\adapters\models.py`

方案注册和参数位于：

`C:\Project\JYZmentor\code reproduction\run_reproductions.py` 的 `IMPUTATION_METHODS`

| 方案 | 论文/代码 | 本地模型骨架 | 本地机制 |
|---|---|---|---|
| CSDI | [论文](https://arxiv.org/abs/2107.03502) / [GitHub](https://github.com/ermongroup/CSDI) | MLP | 随机掩码、条件重构 |
| MTSCI | [论文](https://dl.acm.org/doi/10.1145/3627673.3679532) / [GitHub](https://github.com/JeremyChou28/MTSCI) | MLP | 互补掩码、一致性损失 |
| TSDE-Imputation | [论文](https://arxiv.org/abs/2405.05959) / [GitHub](https://github.com/EQTPartners/TSDE) | 双向 GRU | IIF 掩码：插补、插值、预测 |
| SSSD | [论文](https://openreview.net/forum?id=hHiIbk7ApW) / [GitHub](https://github.com/AI4HealthUOL/SSSD) | 1D Conv | 连续块掩码、平滑和一致性 |
| SSD-TS | 论文/代码见 [GitHub](https://github.com/decisionintelligence/SSD-TS) | GRU | 连续块掩码、状态序列重构 |
| CoSTI | [论文](https://arxiv.org/abs/2501.19364) / [GitHub](https://github.com/javiersgjavi/CoSTI) | 1D Conv | 一致性权重、随机掩码 |

这些模型输入已有的 24 步窗口，只对 `Measure/Temp/Hum` 三个直接观测通道进行人工遮挡和恢复。测试缺失不是从原始 CSV 的空时间段中读取的。

## 异常检测

当前 5 个方案都使用正常训练数据训练重构模型，再计算掩码重构误差作为异常分数。公共实现位于：

`C:\Project\JYZmentor\code reproduction\run_reproductions.py` 的 `run_anomaly_method`

| 方案 | 论文/代码 | 本地模型骨架 | 本地评分机制 |
|---|---|---|---|
| DiffAD | [论文/代码](https://github.com/ChunjingXiao/DiffAD) | MLP | 通道加权的增量重构误差 |
| D3R | [论文](https://proceedings.neurips.cc/paper_files/paper/2023/hash/22f5d8e689d2a011cd8ead552ed59052-Abstract-Conference.html) / [代码](https://github.com/ForestsKing/D3R) | 1D Conv | 移动平均去趋势后计算重构误差 |
| TIMEADDM | [论文/代码](https://github.com/Hurongyao/TIMEADDM) | GRU | 最近时间位置加权的重构误差 |
| IMDiffusion | [论文/代码](https://github.com/17000cyh/IMDiffusion) | MLP | 多次插补的误差加不确定性 |
| ProDiffAD | [论文/代码](https://github.com/AiShare-WHU/ProDiffAD) | 1D Conv | 快速蒸馏式组合误差评分 |

异常测试包含两部分：

- 正常测试窗口上的阈值和误报率；
- 人工注入最近四个时间点的 `Measure` 阶跃，幅度为 3σ、6σ、10σ 的伪异常压力测试。

当前没有现场异常标签，因此结果不能解释成真实故障召回率，也不能证明已经完成现场故障预警验证。

## 复现结果位置

- 插补：`code reproduction/local_runs/final/imputation/<方法名>/metrics.json`
- 异常检测：`code reproduction/local_runs/final/anomaly/<方法名>/metrics.json`
- 异常分数表：`code reproduction/local_runs/final/anomaly/<方法名>/scores.csv`
- 上游固定仓库：`code reproduction/upstream/`
- 仓库 commit 和星标快照：`code reproduction/sources_manifest.json`
