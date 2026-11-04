# 独立论文方案分支

此目录中的实现用于把论文中的可迁移机制独立适配到绝缘子数据，而不是把各论文的公开数据加载器强行改成同一份代码。

## 分支

- `TimeGradBranch`：GRU 编码 24 步历史，最后隐状态与目标环境条件共同作为 DDPM 条件。
- `DiffPLFBranch`：用最近 6 步局部线性趋势作为基准，扩散模型只学习趋势残差，并保留局部尺度项。
- `CCDMBranch`：Measure、Temp、Hum 三个直接测量通道分别投影，再与派生特征融合。
- `TSDEBranch`：训练时随机生成点掩码、区间掩码和尾部预测掩码，对应 IIF（Imputation-Interpolation-Forecasting）自监督思想。
- `NsDiffBranch`：对每个窗口计算局部均值和标准差，再在局部标准化空间学习残差扩散。

统一约定：目标是标准化后的 `Measure - persistence`；输出是多个样本，外层脚本统一计算 P10/P50/P90 和验证集分阶段校准。Battery 不进入模型。

## 重要边界

这些分支不是官方仓库逐行复制。官方仓库保存在上级 `upstream/`，但其数据形状和任务分别针对 EV 充电负荷、Time-Series-Library、ETT/电力/天气等数据。本目录是保持论文核心机制、并适配本工程单步预测协议的独立实现。详见：

`../reports/INDEPENDENT_PAPER_BRANCHES_20260907.md`
